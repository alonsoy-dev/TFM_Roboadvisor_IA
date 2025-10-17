import os
import pandas as pd
import torch as th
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Importaciones de Stable Baselines 3 ---
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

# --- Importaciones del proyecto ---
from src.environment.data_loader import load_time_series_csv, load_all_exposures, precompute_indicators
from src.utils.config_file import configYaml
from src.agent.profile_sampler import ProfileSampler
from src.agent.env_gym_wrapper import PortfolioGymWrapper


# ==============================================================================
# FUNCIÓN DE CONFIGURACIÓN DEL ENTORNO
# ==============================================================================
def setup_data_and_sampler():
    """
    Carga todos los datos necesarios desde los paths del fichero de configuración
    y prepara el ProfileSampler.
    """
    times_series = configYaml.paths.time_series
    ex_country = configYaml.paths.exposures_country
    ex_sector = configYaml.paths.exposures_sector

    prices, aum, _ = load_time_series_csv(times_series)
    exposures = load_all_exposures(ex_country, ex_sector, prices)

    print("Precalculando indicadores técnicos...")
    indicators = precompute_indicators(prices)

    # Instanciar el ProfileSampler para la domain randomization
    sampler = ProfileSampler(seed=123)

    print(f"Histórico de datos completo cargado con éxito.")
    return prices, aum, exposures, sampler, indicators


# Warm-up durante una fracción del aprendizaje del fold 1 (warm_frac=0.2 -> 20% del fold)
def make_lr_schedule(base=3e-4, warm=1e-5, warm_frac=0.2):
    def schedule(progress_remaining):
        t = 1.0 - progress_remaining  # 0 -> 1
        if t < warm_frac:
            return warm + (base - warm) * (t / warm_frac)
        return base

    return schedule


def make_clip_schedule(start=0.1, end=0.2, warm_frac=0.2):
    def schedule(progress_remaining):
        t = 1.0 - progress_remaining
        if t < warm_frac:
            return start + (end - start) * (t / warm_frac)
        return end

    return schedule

class RewardMetricsTB(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if not infos:
            return True
        for info in infos:
            m = info.get("reward_metrics")
            if not m:
                continue
            for k, v in m.items():
                try:
                    self.logger.record(f"reward/{k}", float(v))
                except Exception:
                    pass
        return True


# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN CON WALK-FORWARD
# ==============================================================================
if __name__ == '__main__':
    # --- CONFIGURACIÓN GENERAL ---
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    LOGS_DIR = configYaml.paths.logs_dir
    RUNS_DIR = os.path.join(configYaml.paths.runs_dir, f"WalkForward_{TIMESTAMP}")
    os.makedirs(RUNS_DIR, exist_ok=True)

    N_ENVS = 8

    # --- PARÁMETROS DEL WALK-FORWARD ---
    INITIAL_TRAIN_YEARS = 6  # Empezamos entrenando con 6 años de datos
    START_YEAR = 2010
    END_TRAIN_YEAR = 2022  # Último año que se incluirá en un set de entrenamiento
    VALIDATION_MONTHS = 12  # Cada fold valida sobre los siguientes 12 meses
    TIMESTEPS_PER_FOLD = 500_000  # Pasos de entrenamiento en CADA fold

    # --- CARGA DE DATOS COMPLETOS ---
    all_prices, all_aum, exposures, sampler, all_indicators = setup_data_and_sampler()

    # --- BUCLE PRINCIPAL DE WALK-FORWARD ---
    model = None  # El modelo se inicializa a None y se reutiliza en cada fold
    start_date = pd.Timestamp(f'{START_YEAR}-09-01')

    for current_year in range(START_YEAR + INITIAL_TRAIN_YEARS, END_TRAIN_YEAR + 1):

        # DIVISIÓN DINÁMICA DE DATOS PARA EL FOLD ACTUAL
        train_end_date = pd.Timestamp(f'{current_year}-09-01') - pd.Timedelta(days=1)
        eval_end_date = train_end_date + relativedelta(months=VALIDATION_MONTHS)

        print("\n" + "=" * 50)
        print(f" INICIANDO FOLD: {current_year}")
        print(f"   - Datos de Entrenamiento: {start_date.year} -> {train_end_date.year}")
        print(f"   - Datos de Validación:    {train_end_date.year + 1}")
        print("=" * 50 + "\n")

        # Filtramos los DataFrames para este fold específico
        fold_prices = all_prices.loc[start_date:eval_end_date]
        fold_aum = all_aum.loc[start_date:eval_end_date]

        # CREACIÓN DE ENTORNOS PARA EL FOLD ACTUAL
        env_kwargs = {
            'prices': fold_prices,
            'aum': fold_aum,
            'exposures': exposures,
            'sampler': sampler,
            'indicators': all_indicators
        }

        # Entorno de entrenamiento con datos hasta train_end_date
        train_env_kwargs = env_kwargs.copy()
        train_env_kwargs['end_date_limit'] = train_end_date  # Pasamos el límite al wrapper
        # Creación del entorno vectorizado y paralelizado
        # SubprocVecEnv ejecuta cada entorno en un núcleo de CPU distinto, acelerando drásticamente el proceso.
        vec_env = make_vec_env(
            PortfolioGymWrapper,
            n_envs=N_ENVS,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=train_env_kwargs
        )
        # VecMonitor es un wrapper esencial para que TensorBoard pueda registrar las recompensas de entornos paralelos.
        vec_env = VecMonitor(vec_env)

        # Entorno de validación con datos DESDE train_end_date
        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs['start_date_limit'] = train_end_date
        eval_env_kwargs['end_date_limit'] = eval_end_date
        # Entorno de evaluación (un solo entorno, no paralelizado)
        # Es crucial para tener una medida objetiva del rendimiento del agente.
        eval_env = make_vec_env(
            PortfolioGymWrapper,
            n_envs=1,
            env_kwargs=eval_env_kwargs
        )
        eval_env = VecMonitor(eval_env)

        # CREACIÓN O CARGA DEL MODELO (TRANSFER LEARNING)
        fold_model_name = f"PPO_Fold_{current_year}"

        RESUME_CHECKPOINT = configYaml.paths.run_checkpoint

        if model is None:
            if os.path.exists(RESUME_CHECKPOINT):
                print(f"Reanudando desde checkpoint inicial: {RESUME_CHECKPOINT}")
                model = PPO.load(RESUME_CHECKPOINT, env=vec_env, device="cuda")
                model.learning_rate = make_lr_schedule(1e-4, 1e-5, 0.2)
                if hasattr(model, "clip_range"):
                    model.clip_range = make_clip_schedule(0.1, 0.2, 0.2)
                model.target_kl = None
            else:
                print("Primer fold: Creando un nuevo modelo PPO...")
                # Arquitectura de la red neuronal
                policy_kwargs = dict(
                    activation_fn=th.nn.Tanh,
                    # Se usa Tanh para suavizar la salida de las neuronas, útil en entornos ruidosos.
                    net_arch=dict(
                        pi=[256, 256],
                        vf=[256, 256]
                    )
                )
                # Instanciación del modelo PPO
                model = PPO(
                    "MlpPolicy",  # Política estándar para datos vectoriales
                    vec_env,  # Entorno de entrenamiento paralelizado
                    policy_kwargs=policy_kwargs,
                    n_steps=4096,  # Pasos por entorno antes de actualizar la red
                    batch_size=256,  # Tamaño del lote para la actualización del gradiente
                    n_epochs=4,  # Veces que se recorre el buffer de datos en cada actualización
                    gamma=0.995,  # Factor de descuento (importancia de recompensas futuras)
                    gae_lambda=0.95,  # Parámetro del estimador de ventaja
                    ent_coef=0.02,  # Coeficiente de entropía (fuerza de la exploración)
                    learning_rate=make_lr_schedule(1e-4, 1e-5, 0.2 ),  # Tasa de aprendizaje
                    clip_range=make_clip_schedule(0.1, 0.2, 0.2),
                    target_kl=None,  # corta updates cuando KL sube demasiado
                    verbose=1,  # Nivel de detalle en la consola (1 = muestra progreso)
                    tensorboard_log=LOGS_DIR,  # Directorio para guardar los logs de TensorBoard
                    device='cuda' # Explicitamente usa la GPU NVIDIA
                )
        else:  # Folds siguientes: cargamos el modelo anterior y lo adaptamos (fine-tuning)
            print("Cargando el modelo del fold anterior para continuar el entrenamiento (fine-tuning)...")
            # Reutilizamos los pesos aprendidos, pero con el nuevo entorno de datos expandido
            model.set_env(vec_env)
            # Para el fine-tuning, es buena práctica reducir aún más la tasa de aprendizaje.
            model.learning_rate = lambda progress_remaining: 1e-5
            if hasattr(model, "clip_range"):
                model.clip_range = lambda progress_remaining: 0.2
            model.target_kl = None

        # CONFIGURACIÓN DE CALLBACKS PARA EL FOLD ACTUAL
        # El EvalCallback guarda el mejor modelo encontrado durante las evaluaciones periódicas.
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(RUNS_DIR, fold_model_name),
            log_path=os.path.join(RUNS_DIR, fold_model_name),
            eval_freq=max(15_000 // N_ENVS, 1),
            deterministic=True, render=False
        )

        callback = CallbackList([RewardMetricsTB(), eval_callback])

        try:
            # ENTRENAMIENTO DEL FOLD
            model.learn(
                total_timesteps=TIMESTEPS_PER_FOLD,
                callback=callback,
                tb_log_name=f"WalkForward_{current_year}",
                reset_num_timesteps=False  # Importante para que el log de TensorBoard sea continuo
            )
        finally:
            # Guardamos el último modelo del fold para cargarlo en la siguiente iteración
            final_model_path = os.path.join(RUNS_DIR, f"{fold_model_name}_final_interrupt.zip")
            model.save(final_model_path)
            print(f"Modelo de interrupción guardado en: {final_model_path}")

        # Cargamos el mejor modelo de este fold (guardado por EvalCallback) para la siguiente iteración
        best_model_path = os.path.join(RUNS_DIR, fold_model_name, 'best_model.zip')
        if os.path.exists(best_model_path):
            print(f"Cargando el mejor modelo del fold ({best_model_path}) para la siguiente iteración.")
            model = PPO.load(best_model_path, env=vec_env, device='cuda')
        else:
            print("WARN: No se encontró un 'best_model'. Se continuará con el último modelo guardado.")

    print("Entrenamiento del agente PPO-INVESTOR completado.")
