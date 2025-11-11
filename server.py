# server.py — Bot TRX (FastAPI + Webhook Telegram)
# Corre en Render (Web Service Free) con:
# Start Command: uvicorn server:app --host 0.0.0.0 --port 10000

import os
import asyncio
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from telegram import Update
from telegram.ext import ApplicationBuilder, Application, CommandHandler, ContextTypes

# =========================
# Config
# =========================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").strip()  # ej. https://tu-servicio.onrender.com
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "trx-secret").strip()

if not BOT_TOKEN:
    raise RuntimeError("Falta BOT_TOKEN en variables de entorno.")

TICKER = "TRX-USD"

# Timeframes permitidos
TF_MAP = {
    "h": ("1h", "h"),
    "d": ("1d", "d"),
    "w": ("1wk", "w"),
    "m": ("1mo", "m"),
    "y": ("1mo", "y"),  # y = velas mensuales para promedios 12/24/200
}

# =========================
# Indicadores (versión segura)
# =========================
def ema(series: pd.Series, n: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = ema(up, n) / ema(down, n)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    series = pd.to_numeric(series, errors="coerce")
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line, signal

def bollinger_position(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    series = pd.to_numeric(series, errors="coerce")
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std(ddof=0).replace(0, np.nan)
    upper = ma + k * sd
    lower = ma - k * sd
    pos = ((series - ma) / (k * sd)).clip(-1, 1)
    return ma, upper, lower, pos

def last_float(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return np.nan
    v = pd.to_numeric(pd.Series(s).tail(1).squeeze(), errors="coerce")
    try:
        return float(v)
    except Exception:
        return np.nan

def fmt_price(x: float) -> str:
    return "—" if np.isnan(x) else f"{x:.4f}"

def label_rsi(x: float) -> str:
    if np.isnan(x): return "—"
    if x >= 70: return "Sobrecompra"
    if x <= 30: return "Sobreventa"
    return "Neutro"

def label_boll(pos: float) -> str:
    if np.isnan(pos): return "—"
    if pos >= 0.75: return "Alta"
    if pos <= -0.75: return "Baja"
    return "Intermedia"

def score_direction(price: float, ema20: float, rsi_v: float, macd_line: float, macd_sig: float, boll_pos: float) -> Tuple[float,float]:
    score = 0.0
    if not np.isnan(ema20):
        score += 1.0 if (not np.isnan(price) and price > ema20) else -1.0
    if not np.isnan(rsi_v):
        score += 0.5 if rsi_v > 60 else (-0.5 if rsi_v < 40 else 0.0)
    if not (np.isnan(macd_line) or np.isnan(macd_sig)):
        score += 1.0 if macd_line > macd_sig else -1.0
    if not np.isnan(boll_pos):
        score += 0.5 if boll_pos > 0.5 else (-0.5 if boll_pos < -0.5 else 0.0)
    p_up = max(10, min(90, 50 + 10 * score))
    return p_up, 100 - p_up

# =========================
# Cálculo por timeframe (seguro)
# =========================
def compute_tf(ticker: str, tf_key: str) -> str:
    if tf_key not in TF_MAP:
        return "Timeframe inválido."

    interval, key = TF_MAP[tf_key]
    period = "max" if key in ("w", "m", "y") else "1y"

    data = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False
    )

    if data is None or "Close" not in data.columns or len(data) < 60:
        return f"{ticker} [{key}] — No hay suficientes datos."

    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    if close.empty:
        return f"{ticker} [{key}] — No hay datos válidos."

    last_ts = close.index[-1]
    last_price = last_float(close)

    # Indicadores
    rsi_series = rsi(close, 14)
    macd_line, macd_sig = macd(close)
    _, _, _, bb_pos = bollinger_position(close, 20, 2.0)
    ema20s = ema(close, 20)
    sma50s = close.rolling(50).mean()
    sma200s = close.rolling(200).mean()

    rsi_v   = last_float(rsi_series)
    macd_l  = last_float(macd_line)
    macd_s  = last_float(macd_sig)
    bbp     = last_float(bb_pos)
    ema20_v = last_float(ema20s)

    # SMAs según tf
    if key in ("h", "d", "w", "m"):
        s50  = last_float(sma50s)
        s200 = last_float(sma200s)
        if np.isnan(s50) or np.isnan(s200):
            sma_txt = "— / — → —"
        else:
            trend = "Alcista (50>200)" if s50 > s200 else "Bajista (50<=200)"
            sma_txt = f"{fmt_price(s50)} / {fmt_price(s200)} → {trend}"
    else:
        # Anual: medias mensuales 12/24 y 200 si existe
        sma12 = close.rolling(12).mean()
        sma24 = close.rolling(24).mean()
        s12 = last_float(sma12); s24 = last_float(sma24)
        parts = []
        if np.isnan(s12) or np.isnan(s24):
            parts.append("SMA12/SMA24 (meses): — / — → —")
        else:
            trend = "Alcista (12>24)" if s12 > s24 else "Bajista (12<=24)"
            parts.append(f"SMA12/SMA24 (meses): {fmt_price(s12)} / {fmt_price(s24)} → {trend}")
        sma200m = close.rolling(200).mean()
        s200m = last_float(sma200m)
        if np.isnan(s200m):
            parts.append("SMA200 (meses): —  · Motivo: TRX aún no tiene ≥200 meses de historia")
        else:
            parts.append(f"SMA200 (meses): {fmt_price(s200m)}")
        sma_txt = "\n• " + "\n• ".join(parts)

    # Probabilidades
    p_up, p_dn = score_direction(last_price, ema20_v, rsi_v, macd_l, macd_s, bbp)

    # Labels
    rsi_label  = label_rsi(rsi_v)
    macd_label = "Alcista" if (not np.isnan(macd_l) and not np.isnan(macd_s) and macd_l > macd_s) else "Bajista"
    boll_label = label_boll(bbp)

    # Fechas
    utc_dt = last_ts.tz_convert("UTC") if getattr(last_ts, "tzinfo", None) else last_ts.tz_localize("UTC")
    utc_str = utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tf_human = {"h":"h","d":"d","w":"w","m":"m","y":"y"}[key]

    lines = [
        f"{TICKER} — análisis",
        f"Actualizado: {now_utc}",
        "",
        f"{TICKER} [{tf_human}] — Última vela: {utc_str}",
        f"• Precio: {fmt_price(last_price)}",
        f"• RSI14: {rsi_v:.2f} → {rsi_label}" if not np.isnan(rsi_v) else "• RSI14: —",
        f"• MACD: {macd_label}",
    ]
    lines.append("• Bollinger (20,2): —" if np.isnan(bbp) else f"• Bollinger (20,2): pos={bbp:.2f} → {boll_label}")
    if key == "y":
        lines.append(sma_txt)
    else:
        lines.append(f"• SMA50/SMA200: {sma_txt}")
    if np.isnan(ema20_v):
        lines.append("• EMA20: —")
    else:
        rel = "Precio>EMA20" if (not np.isnan(last_price) and last_price > ema20_v) else "Precio<EMA20"
        lines.append(f"• EMA20: {fmt_price(ema20_v)} → {rel}")
    lines.append(f"• Prob. próxima vela: ⬆️ {int(round(p_up))}%  |  ⬇️ {int(round(p_dn))}%")
    lines.append("Solo con propósitos informativos. No es asesoría financiera.")
    return "\n".join(lines)

# =========================
# Telegram Handlers
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Bienvenido al bot de análisis técnico de TRX (TRON).\n\n"
        "Comandos disponibles:\n"
        "/info TRX h   (intradía por horas)\n"
        "/info TRX d   (diario)\n"
        "/info TRX w   (semanal)\n"
        "/info TRX m   (mensual)\n"
        "/info TRX y   (anual con velas mensuales)\n\n"
        "Nota: Solo con propósitos informativos. No es asesoría financiera."
    )
    await update.message.reply_text(text)

async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Uso: /info TRX h|d|w|m|y
    args = context.args or []
    if len(args) != 2:
        await update.message.reply_text("Uso: /info TRX h|d|w|m|y")
        return

    sym = args[0].upper().strip()
    tf = args[1].lower().strip()

    if sym != "TRX":
        await update.message.reply_text("Este bot está centrado únicamente en TRX.")
        return
    if tf not in TF_MAP:
        await update.message.reply_text("Timeframe inválido. Usa h, d, w, m o y.")
        return

    try:
        msg = compute_tf(TICKER, tf)
    except Exception as e:
        msg = f"Error calculando indicadores: {e}"
    await update.message.reply_text(msg)

# =========================
# FastAPI + Webhook Telegram
# =========================
app = FastAPI()
application: Application | None = None

@app.on_event("startup")
async def on_startup():
    global application
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("info", cmd_info))

    await application.initialize()

    # Webhook auto (Render)
    if PUBLIC_URL:
        webhook_url = f"{PUBLIC_URL.rstrip('/')}/webhook/{WEBHOOK_SECRET}"
        await application.bot.set_webhook(url=webhook_url, drop_pending_updates=True)

    await application.start()

@app.on_event("shutdown")
async def on_shutdown():
    if application:
        await application.stop()
        await application.shutdown()

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="forbidden")
    if not application:
        raise HTTPException(status_code=503, detail="bot not ready")

    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return JSONResponse({"ok": True})
