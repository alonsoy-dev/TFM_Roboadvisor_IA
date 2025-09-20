import os, sys, zipfile
import pandas as pd
from io import BytesIO

# ------------------ Config ------------------
START_DATE = pd.Timestamp("2010-09-01")
OUT_TIME_SERIES  = "data/processed/time_series.csv"
OUT_SECTOR       = "data/processed/exposures_sector.csv"
OUT_COUNTRY      = "data/processed/exposures_country.csv"

REQUIRED_HIST = {
    "date": "As Of",
    "nav":  "NAV per Share",
    "shares_outstanding": "Shares Outstanding",
}
REQUIRED_HOLD = {
    "sector":  "Sector",
    "country": "Location",
    "weight":  "Weight (%)",
}

# Columnas de salida
TIME_SERIES_COLS   = ["date", "ticker","name", "type", "nav", "shares_outstanding", "aum"]
SECTOR_EXPOSURE    = ["ticker", "type", "sector",  "weight"]
COUNTRY_EXPOSURE   = ["ticker", "type", "country", "weight"]
# ---------------------------------------------------

# ---------- Parsers ----------
def _num(x):
    """Formato europeo: '.' miles, ',' decimal -> float. Ej: '1.234,34' -> 1234.34"""
    if pd.isna(x): return None
    s = str(x).strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def _pct(x):
    """Porcentajes europeos -> fracción [0,1]. Ej: '12,34%' -> 0.1234"""
    if pd.isna(x): return None
    s = str(x).replace("%", "").strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s) / 100.0
    except:
        return None
# -----------------------------

def detect_ticker_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def classify_fund_type(input_path: str) -> str:
    name = os.path.basename(input_path).lower()
    if any(k in name for k in ["fixed","income", "fixedincome"]):
        return "FixedIncome"
    if any(k in name for k in ["equity"]):
        return "Equity"
    return "Unknown"

def read_excel_from_bytes(data: bytes, ext: str):
    ext = (ext or "").lower()
    if ext == ".xls":
        return pd.ExcelFile(BytesIO(data), engine="xlrd")
    return pd.ExcelFile(BytesIO(data), engine="openpyxl")

# ---------- Validación estricta ----------
def find_holdings_header_row(df: pd.DataFrame, max_row=200):
    for i in range(min(max_row, len(df))):
        row = df.iloc[i].astype(str).str.strip().tolist()
        if ("Sector" in row or "Location" in row) and any("Weight" in c for c in row):
            return i
    return None

def is_ticker_schema_valid(xls) -> bool:
    # Historical: columnas exactas
    if "Historical" not in xls.sheet_names: return False
    hist = pd.read_excel(xls, "Historical", nrows=0)
    hist_cols = [str(c).strip() for c in hist.columns]
    for req in REQUIRED_HIST.values():
        if req not in hist_cols:
            return False

    # Holdings: cabecera detectada + columnas exactas
    if "Holdings" not in xls.sheet_names: return False
    raw = pd.read_excel(xls, "Holdings")
    h = find_holdings_header_row(raw)
    if h is None: return False
    hold_cols = [str(c).strip() for c in raw.iloc[h].tolist()]
    for req in REQUIRED_HOLD.values():
        if req not in hold_cols:
            return False
    return True
# ----------------------------------------

# ---------- Normalizadores ----------
def normalize_historical(xls, ticker: str, fund_type: str, fund_name: str) -> pd.DataFrame:
    raw = pd.read_excel(xls, "Historical")

    # Construir columnas con índice
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[REQUIRED_HIST["date"]], errors="coerce"),
        "nav":  raw[REQUIRED_HIST["nav"]].apply(_num),
        "shares_outstanding": raw[REQUIRED_HIST["shares_outstanding"]].apply(_num),
    })

    # Recorte y orden
    out = out.dropna(subset=["date"]).loc[out["date"] >= START_DATE].sort_values("date")

    # Añadir constantes
    out["ticker"] = ticker
    out["name"]   = fund_name
    out["type"]   = fund_type

    # Calcular AUM
    out["aum"] = out["nav"] * out["shares_outstanding"]

    return out[TIME_SERIES_COLS]

def normalize_holdings(xls, ticker: str, fund_type: str):
    raw = pd.read_excel(xls, "Holdings")
    h = find_holdings_header_row(raw)
    if h is None:
        return (pd.DataFrame(columns=SECTOR_EXPOSURE), pd.DataFrame(columns=COUNTRY_EXPOSURE))

    df = raw.iloc[h+1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[h].tolist()]

    sec_col = REQUIRED_HOLD["sector"]
    cty_col = REQUIRED_HOLD["country"]
    w_col   = REQUIRED_HOLD["weight"]

    # filtrado y normalización inicial
    cols = [c for c in [sec_col, cty_col, w_col] if c in df.columns]
    df = df[cols].dropna(how="all")
    df["weight"] = df[w_col].apply(_pct)

    # sector
    if sec_col in df.columns:
        sec = (df.dropna(subset=[sec_col])
                 .groupby(sec_col, as_index=False)["weight"].sum())
        s = sec["weight"].sum()
        if s and s > 0: sec["weight"] = sec["weight"] / s
        sec = sec.rename(columns={sec_col: "sector"})
        sec["ticker"] = ticker; sec["type"] = fund_type
        sec = sec[SECTOR_EXPOSURE]
    else:
        sec = pd.DataFrame(columns=SECTOR_EXPOSURE)

    # country
    if cty_col in df.columns:
        cty = (df.dropna(subset=[cty_col])
                 .groupby(cty_col, as_index=False)["weight"].sum())
        s = cty["weight"].sum()
        if s and s > 0: cty["weight"] = cty["weight"] / s
        cty = cty.rename(columns={cty_col: "country"})
        cty["ticker"] = ticker; cty["type"] = fund_type
        cty = cty[COUNTRY_EXPOSURE]
    else:
        cty = pd.DataFrame(columns=COUNTRY_EXPOSURE)

    return sec, cty

def get_fund_name(xls) -> str:
    df = pd.read_excel(xls, sheet_name="Holdings", header=None, usecols=[0], nrows=2)
    name = str(df.iat[1, 0]).strip()
    if not name:
        raise ValueError("Holdings!A2 vacío")
    return name

# ---------- IO helpers ----------
def iter_items_from_path(input_path):
    """Yield (ticker, fund_type, ExcelFile) desde un ZIP/carpeta."""
    fund_type = classify_fund_type(input_path)
    if input_path.lower().endswith(".zip"):
        with zipfile.ZipFile(input_path, "r") as zf:
            for name in sorted(zf.namelist()):
                ext = os.path.splitext(name)[1].lower()
                if ext in (".xlsx", ".xls"):
                    data = zf.read(name)
                    yield detect_ticker_from_filename(name), fund_type, read_excel_from_bytes(data, ext)
    else:
        for f in sorted(os.listdir(input_path)):
            ext = os.path.splitext(f)[1].lower()
            if ext in (".xlsx", ".xls"):
                full = os.path.join(input_path, f)
                with open(full, "rb") as fh:
                    data = fh.read()
                yield detect_ticker_from_filename(f), fund_type, read_excel_from_bytes(data, ext)

# ---------- Clean global ----------
def clean_datasets(ts: pd.DataFrame, sec: pd.DataFrame, cty: pd.DataFrame, skipped: list):
    """
    Elimina de los 3 DataFrames cualquier ticker que tenga al menos un NaN
    en cualquiera de ellos.
    """
    bad = set()
    if not ts.empty:
        cols_check = [c for c in ts.columns if c != "name"]
        bad |= set(ts.loc[ts[cols_check].isna().any(axis=1), "ticker"].unique())
    if not sec.empty:
        bad |= set(sec.loc[sec.isna().any(axis=1), "ticker"].unique())
    if not cty.empty:
        bad |= set(cty.loc[cty.isna().any(axis=1), "ticker"].unique())

    if bad:
        print(f"[CLEAN] Eliminando {len(bad)} tickers con NaN: {sorted(bad)}")
        skipped.extend([(t, "NaN") for t in sorted(bad)])

    f = lambda d: (d[~d["ticker"].isin(bad)].copy() if not d.empty else d)
    return f(ts), f(sec), f(cty), skipped

# ------------------ Main ------------------
def main(input_paths):
    all_ts, all_sec, all_cty = [], [], []
    success_equity, success_fixed = 0, 0
    skipped = []

    for input_path in input_paths:
        for ticker, fund_type, xls in iter_items_from_path(input_path):
            if not is_ticker_schema_valid(xls):
                print(f"[SKIP] {ticker} [{fund_type}] -> columnas no coinciden. Ticker descartado.")
                skipped.append((ticker, fund_type))
                continue

            fund_name = get_fund_name(xls)
            ts = normalize_historical(xls, ticker, fund_type, fund_name)
            sec, cty = normalize_holdings(xls, ticker, fund_type)

            if len(ts):  all_ts.append(ts)
            if len(sec): all_sec.append(sec)
            if len(cty): all_cty.append(cty)

            if len(ts):
                if fund_type == "Equity":
                    success_equity += 1
                elif fund_type == "FixedIncome":
                    success_fixed += 1

            print(f"[OK] {ticker} [{fund_type}] -> {len(ts)} TS, {len(sec)} sec, {len(cty)} cty")

    if not all_ts:
        print("[WARN] No hay datos válidos tras la validación.")
        return

    # concat
    ts = pd.concat(all_ts, ignore_index=True).sort_values(["ticker", "date"])
    sec = pd.concat(all_sec, ignore_index=True) if all_sec else pd.DataFrame(columns=SECTOR_EXPOSURE)
    cty = pd.concat(all_cty, ignore_index=True) if all_cty else pd.DataFrame(columns=COUNTRY_EXPOSURE)

    # fechas comunes (sobre ts)
    sets = [set(g["date"]) for _, g in ts.groupby("ticker")]
    common_dates = sets[0].intersection(*sets[1:]) if sets else set()
    ts = ts[ts["date"].isin(common_dates)].sort_values(["ticker", "date"])

    # limpieza global por NaNs
    ts, sec, cty, skipped = clean_datasets(ts, sec, cty, skipped)

    # guardar CSVs
    os.makedirs(os.path.dirname(OUT_TIME_SERIES), exist_ok=True)
    ts.to_csv(OUT_TIME_SERIES, index=False)
    print(f"Guardado {OUT_TIME_SERIES} ({len(ts)} filas, {len(common_dates)} fechas comunes)")

    os.makedirs(os.path.dirname(OUT_SECTOR), exist_ok=True)
    sec.to_csv(OUT_SECTOR, index=False)
    print(f"Guardado {OUT_SECTOR} ({len(sec)} filas)")

    os.makedirs(os.path.dirname(OUT_COUNTRY), exist_ok=True)
    cty.to_csv(OUT_COUNTRY, index=False)
    print(f"Guardado {OUT_COUNTRY} ({len(cty)} filas)")

    # ---------- Logs summary ----------
    print(f"[SUMMARY] Fondos guardados -> Equity: {success_equity}, FixedIncome: {success_fixed}")
    print(f"[SUMMARY] Fondos no guardados -> {len(skipped)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python script.py <ruta1.zip> <ruta2.zip> ...")
        sys.exit(1)
    main(sys.argv[1:])
