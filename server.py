# bot.py — Bot SOLO para TRON (TRX). Timeframes: h, d, w, m, y
# Comandos:
#   /start
#   /info TRX h|d|w|m|y
#
# Mejoras:
# - Zona horaria fija (America/Mexico_City)
# - Bollinger: umbrales claros (>=0.80 sobrecompra, <=-0.80 sobreventa)
# - Mensajes claros (Precio → SMA/EMA → RSI → MACD → Bollinger → Prob.)
# - Probabilidad “capada” por horizonte para no exagerar en plazos largos
# - yfinance con threads=True
# - Aviso: Solo propósitos informativos (no asesoría)

import os, math
from datetime import datetime, timezone
import zoneinfo
import numpy as np
import pandas as pd
import yfinance as yf

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# -------------------- Config TRX-only --------------------
TRX_ALIASES = {"TRX", "TRX-USD", "TRXUSD", "TRX/USDT", "TRXUSDT"}
TRX_CANONICAL = "TRX-USD"  # símbolo en yfinance
VALID_TF = {"h","d","w","m","y"}  # sin "1"
MX_TZ = zoneinfo.ZoneInfo("America/Mexico_City")

def is_trx(t: str) -> bool:
    return (t or "").upper().strip() in TRX_ALIASES or (t or "").upper().strip() == "TRX-USD"

def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"{x:,.{nd}f}"
    except Exception:
        return str(x)

def ts_local_utc(now=None):
    now_utc = (now or datetime.now(timezone.utc))
    utc = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    local_dt = now_utc.astimezone(MX_TZ)
    local = local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return local, utc

# -------------------- Indicadores --------------------
def rsi_series(close: pd.Series, n=14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    down = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_series(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

# -------------------- Datos --------------------
def fetch_candles_for_tf(ticker: str, tf: str) -> pd.DataFrame:
    """Devuelve DataFrame según tf humano (h/d/w/m/y)."""
    if tf == "h":
        interval = "1h";  period = "60d"
    elif tf == "d":
        interval = "1d";  period = "2y"
    elif tf == "w":
        interval = "1wk"; period = "5y"
    elif tf == "m":
        interval = "1mo"; period = "20y"
    elif tf == "y":
        interval = "1mo"; period = "20y"
    else:
        raise ValueError("Timeframe no soportado")

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,   # ayuda a reducir latencia
        prepost=False,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    return df

def get_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if "Close" in df.columns and not isinstance(df["Close"], pd.DataFrame):
        return df["Close"].dropna().astype(float)

    if isinstance(df.columns, pd.MultiIndex):
        try:
            sub = df.xs("Close", axis=1, level=0)
        except KeyError:
            try:
                sub = df.xs("Close", axis=1, level=1)
            except KeyError:
                sub = None
        if sub is not None:
            if isinstance(sub, pd.DataFrame):
                s = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
            else:
                s = sub
            return s.dropna().astype(float)

    if "Adj Close" in df.columns:
        s = df["Adj Close"]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze()
        return s.dropna().astype(float)

    raise ValueError("No se encontró columna Close/Adj Close en el DataFrame")

# -------------------- Features --------------------
def ta_features(df: pd.DataFrame, ticker: str, tf: str) -> dict:
    if df is None or df.empty:
        raise ValueError("Sin datos")
    close = get_close_series(df, ticker)
    if close.empty:
        raise ValueError("Sin cierres válidos")

    last_price = float(close.iloc[-1])

    rsi = rsi_series(close, 14)
    rsi14 = float(rsi.iloc[-1]) if not rsi.empty else np.nan

    macd_line, macd_sig, macd_hist = macd_series(close)
    mline = float(macd_line.iloc[-1])
    msig  = float(macd_sig.iloc[-1])
    mhist = float(macd_hist.iloc[-1])

    # Bollinger 20,2
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std(ddof=0)
    if len(close) >= 20 and not np.isnan(sd20.iloc[-1]) and sd20.iloc[-1] != 0:
        m  = float(ma20.iloc[-1])
        s  = float(sd20.iloc[-1])
        k  = 2.0
        bb_pos = (last_price - m) / (k*s)
    else:
        bb_pos = np.nan

    def last_ma(series, n):
        return float(series.rolling(n).mean().iloc[-1]) if len(series) >= n else np.nan

    # Medias en unidades del TF
    sma50  = last_ma(close, 50)
    sma200 = last_ma(close, 200)
    ema20  = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(close) >= 20 else np.nan

    # Alternativas anuales (meses) para vista 'y'
    sma12m = last_ma(close, 12)
    sma24m = last_ma(close, 24)

    # Timestamps
    last_idx = close.index[-1]
    try:
        last_local = last_idx.to_pydatetime().astimezone(MX_TZ)
        last_utc   = last_idx.tz_convert("UTC").to_pydatetime()
    except Exception:
        last_local = pd.Timestamp(last_idx).to_pydatetime().astimezone(MX_TZ)
        last_utc   = pd.Timestamp(last_idx).to_pydatetime()

    return {
        "price": last_price,
        "rsi14": rsi14,
        "macd_line": mline,
        "macd_signal": msig,
        "macd_hist": mhist,
        "bb_pos": bb_pos,
        "sma50": sma50,
        "sma200": sma200,
        "ema20": ema20,
        "sma12m": sma12m,
        "sma24m": sma24m,
        "ts_local": last_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "ts_utc":   last_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

# -------------------- Etiquetas + Prob --------------------
def label_rsi(v):
    if np.isnan(v): return "—"
    if v >= 70: return "Sobrecompra"
    if v <= 30: return "Sobreventa"
    return "Neutro"

def label_macd(line, sig):
    if np.isnan(line) or np.isnan(sig): return "—"
    return "Alcista" if line > sig else "Bajista"

def label_trend(sma50, sma200):
    if np.isnan(sma50) or np.isnan(sma200): return "—"
    return "Alcista (50>200)" if sma50 > sma200 else "Bajista (50<=200)"

def label_bollinger(pos):
    if np.isnan(pos): return "—"
    if pos >=  0.8:  return "Sobrecompra"
    if pos <= -0.8:  return "Sobreventa"
    if -0.3 <= pos <= 0.3: return "Zona media"
    return "Intermedia"

def prob_model(feat: dict, tf: str) -> int:
    # Heurística simple + “capas” por horizonte
    score = 0
    p = feat["price"]; ema20 = feat["ema20"]
    if not (np.isnan(p) or np.isnan(ema20)):
        score += 1 if p > ema20 else -1
    s50, s200 = feat["sma50"], feat["sma200"]
    if not (np.isnan(s50) or np.isnan(s200)):
        score += 1 if s50 > s200 else -1
    if not np.isnan(feat["macd_hist"]):
        score += 1 if feat["macd_hist"] > 0 else -1
    r = feat["rsi14"]
    if not np.isnan(r):
        if r > 70: score -= 1
        elif r > 50: score += 1
        elif r < 30: score -= 1
    base = 50 + 10*score
    base = max(5, min(95, base))
    caps = {"h": (35, 65), "d": (40, 60), "w": (45, 55), "m": (47, 53), "y": (48, 52)}
    lo, hi = caps.get(tf, (40, 60))
    return int(min(hi, max(lo, base)))

# -------------------- Render --------------------
def render_one(tf: str) -> str:
    try:
        df = fetch_candles_for_tf(TRX_CANONICAL, tf)
        feat = ta_features(df, TRX_CANONICAL, tf)
        prob_up = prob_model(feat, tf)
        txt = []
        txt.append(f"TRX-USD [{tf}] — Última vela: {feat['ts_local']} ({feat['ts_utc']})")

        # 1) Precio
        txt.append(f"• Precio: {fmt(feat['price'], 4)}")

        # 2) Tendencia (SMA/EMA)
        if tf == "y":
            alt_trend = "—"
            if not np.isnan(feat.get("sma12m", np.nan)) and not np.isnan(feat.get("sma24m", np.nan)):
                alt_trend = "Alcista (12>24)" if feat["sma12m"] > feat["sma24m"] else "Bajista (12<=24)"
            txt.append(f"• SMA12/SMA24 (meses): {fmt(feat.get('sma12m'),4)} / {fmt(feat.get('sma24m'),4)} → {alt_trend}")
            if np.isnan(feat['sma200']):
                txt.append("• SMA200 (meses): —  · TRX aún no tiene ≥200 meses")
            else:
                txt.append(f"• SMA200 (meses): {fmt(feat['sma200'],4)}")
        else:
            txt.append(f"• SMA50/SMA200: {fmt(feat['sma50'],4)} / {fmt(feat['sma200'],4)} → {label_trend(feat['sma50'], feat['sma200'])}")

        if not (np.isnan(feat['ema20']) or np.isnan(feat['price'])):
            txt.append(f"• EMA20: {fmt(feat['ema20'],4)} → {'Precio>EMA20' if feat['price']>feat['ema20'] else 'Precio<EMA20'}")
        else:
            txt.append("• EMA20: —")

        # 3) RSI
        txt.append(f"• RSI14: {fmt(feat['rsi14'])} → {label_rsi(feat['rsi14'])}")

        # 4) MACD
        txt.append(f"• MACD: {label_macd(feat['macd_line'], feat['macd_signal'])}")

        # 5) Bollinger
        if not np.isnan(feat["bb_pos"]):
            txt.append(f"• Bollinger (20,2): pos={fmt(feat['bb_pos'],2)} → {label_bollinger(feat['bb_pos'])}")
        else:
            txt.append("• Bollinger (20,2): —")

        # 6) Probabilidad
        txt.append(f"• Prob. próxima vela: ⬆️ {prob_up}%  |  ⬇️ {100-prob_up}%")
        return "\n".join(txt)
    except Exception as e:
        return f"TRX-USD [{tf}]: no pude calcular ({e})"

# -------------------- Telegram --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Bienvenido al bot de análisis técnico de TRX (TRON).\n\n"
        "Comandos disponibles:\n"
        "/info TRX h   (intradía por horas)\n"
        "/info TRX d   (diario)\n"
        "/info TRX w   (semanal)\n"
        "/info TRX m   (mensual)\n"
        "/info TRX y   (anual con velas mensuales)\n\n"
        "Nota: Solo con propósitos informativos. No es asesoría financiera."
    )
    await update.message.reply_text(msg)

async def info_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if len(args) != 2:
        await update.message.reply_text("Formato: /info TRX TF (TF: h, d, w, m, y). Ej: /info TRX d")
        return

    raw_ticker, tf = args[0], args[1].lower()
    if not is_trx(raw_ticker):
        await update.message.reply_text("Este bot es SOLO para TRON (TRX). Usa: /info TRX d (por ejemplo).")
        return
    if tf not in VALID_TF:
        await update.message.reply_text("Timeframe no válido. Usa: h, d, w, m, y.")
        return

    local, utc = ts_local_utc()
    header = f"TRX-USD — análisis\nActualizado: {local} ({utc})\n"
    block = render_one(tf)
    footer = "\nSolo con propósitos informativos. No es asesoría financiera."
    await update.message.reply_text(header + "\n" + block + footer)

# -------------------- Main --------------------
def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise RuntimeError('Falta BOT_TOKEN (export BOT_TOKEN="TU_TOKEN")')
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info_handler))
    print("Bot corriendo… (Ctrl+C para salir)")
    app.run_polling(drop_pending_updates=True, poll_interval=0.1)

if __name__ == "__main__":
    main()
