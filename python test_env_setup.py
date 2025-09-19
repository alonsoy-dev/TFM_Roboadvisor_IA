import torch
import gymnasium as gym
from stable_baselines3 import PPO

print("="*50)
print("🔍 Verificando PyTorch")
print("="*50)
print("Versión Torch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU detectada:", torch.cuda.get_device_name(0))
else:
    print("⚠️  No se detecta GPU, correrá en CPU.")

print("\n" + "="*50)
print("Verificando Gymnasium")
print("="*50)
env = gym.make("CartPole-v1")
obs, info = env.reset()
print("Obs inicial shape:", obs.shape)
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

print("\n" + "="*50)
print("🔍 Verificando Stable-Baselines3 (PPO)")
print("="*50)
model = PPO("MlpPolicy", env, n_steps=32, batch_size=32, learning_rate=3e-4, n_epochs=1, verbose=0)
print("Entrenando modelo PPO rápido (100 pasos)...")
model.learn(total_timesteps=100)
print("✅ SB3 entrenó sin errores.")

print("\nTodo OK 🚀. Tu entorno está listo para empezar tu TFM.")