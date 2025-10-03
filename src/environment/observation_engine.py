from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from math import log10


# ============================================================
#   ObservationEngine
#   - Calcula todos los campos de la observación
#   - Usa solo datos hasta env.today (sin mirar el futuro)
#   - Devuelve ya normalizado/capado a [-1, 1] cuando aplica
# ============================================================

class ObservationEngine:
    # -------------------- Constantes / Capping--------------------
    _RCAP_1D   = 0.05   # ±5% -> /0.05
    _RCAP_5D   = 0.10   # ±10% -> /0.10
    _RCAP_21D  = 0.20   # ±20% -> /0.20
    _RCAP_126D = 0.50   # ±50% -> /0.50
    _GAP20_CAP  = 0.20  # ±20% -> /0.20
    _GAP120_CAP = 0.40  # ±40% -> /0.40
    _ATR_CAP    = 0.05  # ATR/Precio cap 0..5% -> /0.05 -> 0..1 -> 2x-1

    def __init__(self, env: "PortfolioEnv"):
        self.env = env

    # -------------------- API --------------------
    def build(self,
              done: bool = False,
              contribution_applied: float = 0.0,
              reward: float = 0.0) -> Dict[str, Any]:
        """Construye el diccionario de observación completo"""
        perfil = self._build_perfil()
        cartera = self._build_cartera(contribution_applied)
        mercado_global = self._build_mercado_global()
        mercado_por_activo = self._build_mercado_por_activo()

        return {
            "fecha": self.env.today,
            "perfil": perfil,
            "cartera": cartera,
            "mercado": {
                "global": mercado_global,
                "por_activo": mercado_por_activo,
            },
            "done": bool(done),
            "reward": float(reward),
        }

    # ============================================================
    #                         BLOQUES
    # ============================================================

    # -------------------- PERFIL --------------------
    def _build_perfil(self) -> Dict[str, Any]:
        e = self.env
        # riesgo: (1..7) => [-1,1]
        riesgo_pm1 = float((int(e.profile.riesgo) - 4) / 3.0)
        # horizonte restante (meses): meses_restantes/meses_totales -> 2x-1
        meses_totales = max(1, int(e.profile.horizonte_anios) * 12)
        meses_pasados = max(0, self._meses_transcurridos())
        meses_restantes = max(0, meses_totales - meses_pasados)
        h01 = meses_restantes / meses_totales
        horizonte_pm1 = self._to_pm1_from_01(h01)
        # aportación mensual relativa a liquidez inicial -> 2x-1
        liq0 = max(1.0, float(e.profile.liquidez_inicial_usd))
        aport01 = self._cap(float(e.profile.aportacion_mensual_usd) / liq0, 0.0, 1.0)
        aport_pm1 = self._to_pm1_from_01(aport01)

        return {
            "riesgo": float(self._cap(riesgo_pm1, -1.0, 1.0)),
            "horizonte_restante": float(self._cap(horizonte_pm1, -1.0, 1.0)),
            "aportacion_mensual": float(aport_pm1),
        }

    # -------------------- CARTERA --------------------
    def _build_cartera(self, contribution_applied: float) -> Dict[str, Any]:
        e = self.env

        w_dict, w_cash = self._weights_today()
        pesos_list = [{"ticker": t, "peso": self._to_pm1_from_01(w_dict.get(t, 0.0))}
                      for t in e.universe]

        liquidez_pm1 = self._to_pm1_from_01(self._cap(w_cash, 0.0, 1.0))

        # valor_cartera relativo a liquidez inicial -> cap (0..3x) -> 2x-1
        equity_rel = self._portfolio_equity_indexed()
        eq01 = self._cap(equity_rel / 3.0, 0.0, 1.0)
        valor_cartera_pm1 = self._to_pm1_from_01(eq01)

        div01 = self._diversificacion_from_w(w_dict)
        top3_01 = self._top3_concentration_from_w(w_dict)
        peso_rv01 = self._peso_rv_from_w(w_dict)

        # drawdown 63d de la cartera (aprox con pesos actuales)
        drawdown_pm1 = self._drawdown_cartera_63d_pm1(w_dict)

        # límite máx por fondo
        wmax = getattr(e, "max_weight_per_fund", None)
        if wmax is None:
            limite_pm1 = 0.0
        else:
            limite_pm1 = self._to_pm1_from_01(self._cap(float(wmax), 0.0, 1.0))

        # rentabilidad de ayer neta de aportación de ayer (cap ±5%) -> [-1,1]
        rentab_1d_lag_pm1 = self._rentab_cartera_1d_lag(contribution_applied)

        return {
            "pesos": pesos_list,
            "liquidez": float(liquidez_pm1),
            "valor_cartera": float(valor_cartera_pm1),
            "diversificacion": float(self._to_pm1_from_01(div01)),
            "concentracion_top3": float(self._to_pm1_from_01(top3_01)),
            "peso_renta_variable": float(self._to_pm1_from_01(peso_rv01)),
            "drawdown_3_meses": float(drawdown_pm1),
            "limite_max_por_fondo": float(limite_pm1),
            "rentabilidad_1d_lag": float(rentab_1d_lag_pm1),
        }

    # -------------------- MERCADO (GLOBAL) --------------------
    def _build_mercado_global(self) -> Dict[str, Any]:
        return {
            "volatilidad_global": float(self._vol_global_63d_pm1()),
            "amplitud_universo": float(self._amplitud_universo_ma120_pm1()),
            "correlacion_media_3m": float(self._correlacion_media_3m_pm1()),
        }

    # -------------------- MERCADO (POR ACTIVO) --------------------
    def _build_mercado_por_activo(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in self.env.universe:  # orden fijo
            out.append({
                "ticker": t,
                "subida_1_dia":        self._ret_k_cap_pm1(t, 1,   self._RCAP_1D),
                "subida_5_dias":       self._ret_k_cap_pm1(t, 5,   self._RCAP_5D),
                "subida_21_dias":      self._ret_k_cap_pm1(t, 21,  self._RCAP_21D),
                "tendencia_6_meses":   self._ret_k_cap_pm1(t, 126, self._RCAP_126D),
                "volatilidad_3_meses": self._vol_63d_pm1(t),
                "gap_ma20":            self._gap_ma_pm1(t, 20,  self._GAP20_CAP),
                "gap_ma120":           self._gap_ma_pm1(t, 120, self._GAP120_CAP),
                "rsi_14":              self._rsi_14_pm1(t),
                "macd_hist":           self._macd_hist_norm_pm1(t),
                "atr_14":              self._atr14_pm1(t),
                "aum":                 self._aum_pm1(t),
            })
        return out

    # ============================================================
    #                      HELPERS INTERNOS
    # ============================================================

    @staticmethod
    def _cap(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _to_pm1_from_01(x01: float) -> float:
        return 2.0 * float(x01) - 1.0

    @staticmethod
    def _safe_pct_change(series: pd.Series, periods: int) -> float:
        if len(series) <= periods:
            return 0.0
        px_t = float(series.iloc[-1])
        px_tp = float(series.iloc[-1 - periods])
        if not np.isfinite(px_t) or not np.isfinite(px_tp) or px_tp == 0.0:
            return 0.0
        return (px_t / px_tp) - 1.0

    @staticmethod
    def _rolling_std_pct(series: pd.Series, window: int) -> float:
        if series.shape[0] < window + 1:
            return 0.0
        rets = series.pct_change().dropna()
        if rets.shape[0] < window:
            return 0.0
        return float(rets.tail(window).std(ddof=1))

    @staticmethod
    def _sma(series: pd.Series, window: int) -> float:
        if series.shape[0] < window:
            return float(series.mean()) if series.shape[0] > 0 else 0.0
        return float(series.tail(window).mean())

    @staticmethod
    def _rsi_14_from_prices(series: pd.Series) -> float:
        if series.shape[0] < 15:
            return 50.0
        delta = series.diff().dropna()
        up = np.where(delta > 0, delta, 0.0)
        dn = np.where(delta < 0, -delta, 0.0)
        up = pd.Series(up, index=delta.index).rolling(14).mean()
        dn = pd.Series(dn, index=delta.index).rolling(14).mean()
        rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] != 0 else np.inf
        rsi = 100.0 - (100.0 / (1.0 + rs)) if np.isfinite(rs) else 100.0
        return float(ObservationEngine._cap(rsi, 0.0, 100.0))

    @staticmethod
    def _atr_14_from_close(close: pd.Series) -> float:
        if close.shape[0] < 15:
            return 0.0
        tr = close.diff().abs()
        return float(tr.rolling(14).mean().iloc[-1]) if tr.shape[0] >= 14 else 0.0

    @staticmethod
    def _macd_hist_from_close(close: pd.Series) -> float:
        if close.shape[0] < 35:
            return 0.0
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return float(hist.iloc[-1])

    # --- pesos y derivados ---
    def _weights_today(self) -> Tuple[Dict[str, float], float]:
        e = self.env
        e._recalc_portfolio_value()
        total = float(e.portfolio_value_usd)
        if total <= 0.0:
            return {}, 1.0
        vals: Dict[str, float] = {}
        for t, sh in e.positions_shares.items():
            if sh <= 0.0:
                continue
            p = e._price_today(t)
            if pd.notna(p) and p > 0.0:
                vals[t] = float(sh) * float(p)
        w_cash = max(0.0, float(e.cash_usd) / total)
        w = {t: (v / total) for t, v in vals.items()}
        s = sum(w.values()) + w_cash
        if s > 0 and abs(s - 1.0) > 1e-10:
            k = 1.0 / s
            w = {t: v * k for t, v in w.items()}
            w_cash *= k
        return w, w_cash

    @staticmethod
    def _diversificacion_from_w(w: Dict[str, float]) -> float:
        if not w:
            return 0.0
        ws = list(w.values())
        N = len(ws)
        if N == 0:
            return 0.0
        hhi = sum(v * v for v in ws)
        div_raw = max(0.0, 1.0 - hhi)
        div_max = 1.0 - 1.0 / N
        if div_max <= 0.0:
            return 0.0
        return float(ObservationEngine._cap(div_raw / div_max, 0.0, 1.0))

    @staticmethod
    def _top3_concentration_from_w(w: Dict[str, float]) -> float:
        if not w:
            return 0.0
        ws = sorted(w.values(), reverse=True)
        s3 = sum(ws[:3]) if len(ws) >= 3 else sum(ws)
        return float(ObservationEngine._cap(s3, 0.0, 1.0))

    def _peso_rv_from_w(self, w: Dict[str, float]) -> float:
        tipos = self.env.exposures["type_sector"]  # dict {ticker -> 'Equity'/'FixedIncome'}
        return sum(wi for t, wi in w.items() if tipos.get(t) == "Equity")

    # --- equity / drawdown / retornos ---
    def _portfolio_equity_indexed(self) -> float:
        e = self.env
        liq0 = max(1.0, float(e.profile.liquidez_inicial_usd))
        return float(e.portfolio_value_usd / liq0)

    @staticmethod
    def _drawdown_pct(series: pd.Series) -> float:
        if series.shape[0] < 2:
            return 0.0
        peak = series.cummax()
        dd = (series / peak) - 1.0
        return float(dd.min())

    def _drawdown_cartera_63d_pm1(self, w: Dict[str, float]) -> float:
        e = self.env
        px_hist = e._prices_ffill.loc[:e.today, e.universe].dropna(how='all')
        if px_hist.shape[0] < 2:
            return 0.0
        rets = px_hist.pct_change().dropna()
        w_vec = pd.Series({t: w.get(t, 0.0) for t in e.universe}, index=rets.columns).fillna(0.0)
        port = (rets * w_vec).sum(axis=1).tail(63)
        if port.shape[0] == 0:
            return 0.0
        curve = (1.0 + port).cumprod()
        dd = self._drawdown_pct(curve)
        dd_pos = self._cap(abs(dd), 0.0, 0.30) / 0.30  # cap 30%
        return self._to_pm1_from_01(dd_pos)

    def _rentab_cartera_1d_lag(self, contribution_applied: float) -> float:
        e = self.env
        pv_prev = e.prev_portfolio_value_usd
        if not pv_prev or pv_prev <= 0.0:
            return 0.0
        r = ((float(e.portfolio_value_usd) - float(contribution_applied)) / float(pv_prev)) - 1.0
        return float(self._cap(r / self._RCAP_1D, -1.0, 1.0))

    # --- indicadores por activo ---
    def _ret_k_cap_pm1(self, ticker: str, k: int, cap_abs: float) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, ticker].dropna()
        r = self._safe_pct_change(px, k)
        return float(self._cap(r / cap_abs, -1.0, 1.0))

    def _vol_63d_pm1(self, ticker: str) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, ticker].dropna()
        vol = self._rolling_std_pct(px, 63)
        x01 = self._cap(vol / 0.03, 0.0, 1.0)  # cap 3% diario
        return self._to_pm1_from_01(x01)

    def _gap_ma_pm1(self, ticker: str, n: int, cap_abs: float) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, ticker].dropna()
        if px.shape[0] < n + 1:
            return 0.0
        ma = self._sma(px, n)
        if ma <= 0.0:
            return 0.0
        gap = (float(px.iloc[-1]) - ma) / ma
        return float(self._cap(gap / cap_abs, -1.0, 1.0))

    def _rsi_14_pm1(self, ticker: str) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, ticker].dropna()
        rsi = self._rsi_14_from_prices(px)  # 0..100
        return float((rsi / 50.0) - 1.0)

    def _atr14_pm1(self, ticker: str) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, ticker].dropna()
        if px.shape[0] < 15:
            return 0.0
        atr = self._atr_14_from_close(px)
        price = float(px.iloc[-1]) if px.shape[0] > 0 else 0.0
        if price <= 0.0:
            return 0.0
        atr_rel = max(0.0, float(atr) / price)
        x01 = self._cap(atr_rel / self._ATR_CAP, 0.0, 1.0)
        return self._to_pm1_from_01(x01)

    def _macd_hist_norm_pm1(self, ticker: str) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, ticker].dropna()
        if px.shape[0] < 35:
            return 0.0
        hist = self._macd_hist_from_close(px)
        atr = self._atr_14_from_close(px)
        denom = atr if atr and atr > 0.0 else (0.01 * float(px.iloc[-1]))
        x = hist / denom
        return float(self._cap(x / 3.0, -1.0, 1.0))

    def _aum_pm1(self, ticker: str) -> float:
        a = self.env._aum_today(ticker)
        if a is None or not np.isfinite(a) or a <= 0.0:
            return 0.0
        # log10(AUM) cap [7,12] -> 0..1 -> [-1,1]
        x = self._cap(log10(a), 7.0, 12.0)
        x01 = (x - 7.0) / 5.0
        return self._to_pm1_from_01(x01)

    # --- globales ---
    def _vol_global_63d_pm1(self) -> float:
        vols = []
        for t in self.env.universe:
            px = self.env._prices_ffill.loc[:self.env.today, t].dropna()
            vols.append(self._rolling_std_pct(px, 63))
        if not vols:
            return 0.0
        mean_vol = float(np.nanmean(vols))
        x01 = self._cap(mean_vol / 0.03, 0.0, 1.0)  # cap 3% diario
        return self._to_pm1_from_01(x01)

    def _amplitud_universo_ma120_pm1(self) -> float:
        ups = 0
        tot = 0
        for t in self.env.universe:
            px = self.env._prices_ffill.loc[:self.env.today, t].dropna()
            if px.shape[0] < 121:
                continue
            ma = self._sma(px, 120)
            if ma <= 0.0:
                continue
            tot += 1
            if float(px.iloc[-1]) > ma:
                ups += 1
        raw = (ups / tot) if tot > 0 else 0.5  # neutral si faltan datos
        return self._to_pm1_from_01(self._cap(raw, 0.0, 1.0))

    def _correlacion_media_3m_pm1(self) -> float:
        px = self.env._prices_ffill.loc[:self.env.today, self.env.universe].dropna(how='all')
        if px.shape[0] < 64:
            return 0.0
        rets = px.pct_change().dropna().tail(63)
        if rets.shape[0] < 2 or rets.shape[1] < 2:
            return 0.0
        C = rets.corr().values
        n = C.shape[0]
        if n < 2:
            return 0.0
        vals = [C[i, j] for i in range(n) for j in range(i + 1, n)]
        m = float(np.nanmean(vals)) if vals else 0.0
        return float(self._cap(m, -1.0, 1.0))

    # --- horizonte ---
    def _meses_transcurridos(self) -> int:
        e = self.env
        if getattr(e, "_start_date_reset", None) is None or e.today is None:
            return 0
        a1, a2 = e._start_date_reset, e.today
        return (a2.year - a1.year) * 12 + (a2.month - a1.month)
