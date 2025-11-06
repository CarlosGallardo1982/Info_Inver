# server.py — Web + Bot TRX

import os
import asyncio
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("trx-bot")

# =======================
# Utilidades TA (resumen)
# =======================
def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up, down = d.clip(lower=0), -d.clip(upper=0)
    ru = up.ewm(alpha=1/n, adjust=False).mean()
    rd = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ru / rd.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out

def macd(s, f=12, s_=26, sig=9):
    ema_f = ema(s, f)
    ema_s = ema(s, s_)
    line = ema_f - ema_s
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist

def bollinger_pos(s, n=20, k=2):
    m = sma(s, n)
    sd = s.rolling(n).std()
    upper = m + k*sd
    lower = m - k*sd
    denom = (upper - lower).replace(0, np.nan)
    pos = ((s - lower) / denom) * 2 - 1
    return pos

def label_boll(pos):
    if pos <= -0.7: return "Banda baja (sobreventa)"
    if pos >=  0.7: return "Banda alta (sobrecompra)"
    return "Intermedia"

def label_rsi(v):
    if v >= 70: return "Sobrecompra"
    if v <= 30: return "Sobreventa"
    return "Neutro"

# Normalización de símbolos conocidos
ALIASES = {
    "TRX": "TRX-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

# Cálculo por timeframe
def calc_tf(ticker: str, tf: str):
    # map de tf “corto” a yfinance
    tf_map = {"h":"1h","d":"1d","w":"1wk","m":"1mo","y":"1mo"}  # y=1mo (vela mensual)
    yf_tf  = tf_map[tf]

    period_map = {"h":"60d", "d":"2y", "w":"10y", "m":"max", "y":"max"}
    period = period_map[tf]

    df = yf.download(ticker, period=period, interval=yf_tf, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Sin datos de Yahoo para ese timeframe")

    close = df["Close"].copy()
    # indicadores
    rsi14 = rsi(close, 14)
    macd_line, macd_sig, _ = macd(close)
    bb_pos = bollinger_pos(close)
    ema20 = ema(close, 20)

    # promedios móviles
    if tf == "y":
        # promedios sobre velas mensuales (12/24); 200 meses requiere mucha historia
        sma_fast = sma(close, 12)
        sma_slow = sma(close, 24)
        sma200   = None
    else:
        sma_fast = sma(close, 50)
        sma_slow = sma(close, 200)
        sma200   = sma_slow

    last = df.index[-1]
    price = float(close.iloc[-1])

    out = {
        "last_label": f"{last} {'CST' if 'CST' in str(last) else ''}",
        "price": price,
        "rsi": float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else None,
        "macd": "Alcista" if macd_line.iloc[-1] > macd_sig.iloc[-1] else "Bajista",
        "bb_pos": float(bb_pos.iloc[-1]) if not np.isnan(bb_pos.iloc[-1]) else None,
        "ema20": float(ema20.iloc[-1]) if not np.isnan(ema20.iloc[-1]) else None,
        "sma_fast": float(sma_fast.iloc[-1]) if not np.isnan(sma_fast.iloc[-1]) else None,
        "sma_slow": float(sma_slow.iloc[-1]) if not np.isnan(sma_slow.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if sma200 is not None and not np.isnan(sma200.iloc[-1]) else None,
    }
    return out

def summarize(ticker, tf, feat):
    p = feat["price"]
    r = feat["rsi"]; rlab = label_rsi(r) if r is not None else "—"
    bb = feat["bb_pos"]; blab = label_boll(bb) if bb is not None else "—"
    macd_lbl = feat["macd"]
    ema20 = feat["ema20"]
    f = feat["sma_fast"]; s = feat["sma_slow"]; s200 = feat["sma200"]

    # tendencia por medias
    if tf == "y":
        tend = "Alcista (12>24)" if (f is not None and s is not None and f > s) else "Bajista (12≤24)"
        mm_line = f"SMA12/SMA24 (meses): {f:.4f} / {s:.4f} → {tend}" if (f and s) else "SMA12/SMA24 (meses): —"
        if s200 is None:
            mm_line += "\n• SMA200 (meses): —  · Motivo: insuficiente histórico"
    else:
        if (f is not None) and (s is not None):
            tend = "Alcista (50>200)" if f > s else "Bajista (50≤200)"
            mm_line = f"SMA50/SMA200: {f:.4f} / {s:.4f} → {tend}"
        else:
            mm_line = "SMA50/SMA200: —"

    ema_line = f"EMA20: {ema20:.4f} → {'Precio>EMA20' if (ema20 is not None and p>ema20) else 'Precio<EMA20'}" if ema20 else "EMA20: —"

    bb_line = f"Bollinger (20,2): pos={bb:.2f} → {blab}" if bb is not None else "Bollinger (20,2): —"

    # “probabilidad” heurística simple (igual que antes)
    score = 0
    if r is not None and 30 <= r <= 70: score += 1
    if macd_lbl == "Alcista": score += 1
    if bb is not None and abs(bb) < 0.7: score += 1
    if f and s and f > s: score += 1
    if ema20 and p > ema20: score += 1
    up = int(round(50 + (score - 2.5) * 10))
    up = max(10, min(90, up))
    down = 100 - up

    return (
        f"{ticker} [{tf}] — Última vela: {feat['last_label']}\n"
        f"• Precio: {p:.4f}\n"
        f"• RSI14: {r:.2f} → {rlab}\n"
        f"• MACD: {macd_lbl}\n"
        f"• {bb_line}\n"
        f"• {mm_line}\n"
        f"• {ema_line}\n"
        f"• Prob. próxima vela: ⬆️ {up}%  |  ⬇️ {down}%"
    )

# =======================
# Telegram bot
# =======================
BOT = None  # Application
RUN_TASK = None

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def info_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = (context.args or [])
    if len(args) == 0:
        return await update.message.reply_text("Uso: /info TRX [h|d|w|m|y]")

    sym = args[0].upper()
    tf = args[1].lower() if len(args) > 1 else "d"
    if sym in ALIASES: sym = ALIASES[sym]
    if tf not in {"h","d","w","m","y"}:
        return await update.message.reply_text("Timeframe inválido. Usa: h, d, w, m, y.")

    try:
        feat = calc_tf(sym, tf)
        msg = (
            f"{sym} — análisis\n"
            f"Actualizado: {datetime.now():%Y-%m-%d %H:%M:%S}  ({datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC)\n\n"
            + summarize(sym, tf, feat)
            + "\nSolo con propósitos informativos. No es asesoría financiera."
        )
    except Exception as e:
        msg = f"{sym} — análisis\nError: {e}"
    await update.message.reply_text(msg)

async def run_bot():
    global BOT
    token = os.environ.get("BOT_TOKEN")
    if not token:
        log.error("Falta BOT_TOKEN en variables de entorno.")
        return

    BOT = ApplicationBuilder().token(token).build()
    BOT.add_handler(CommandHandler("start", start_cmd))
    BOT.add_handler(CommandHandler("info", info_cmd))
    log.info("Iniciando polling del bot…")
    await BOT.initialize()
    await BOT.start()
    await BOT.updater.start_polling(drop_pending_updates=True)

# =======================
# FastAPI (para Render)
# =======================
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "name": "AT_TRX_bot"}

@app.get("/healthz")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.on_event("startup")
async def on_startup():
    # Lanza el bot en segundo plano
    global RUN_TASK
    loop = asyncio.get_event_loop()
    RUN_TASK = loop.create_task(run_bot())
    log.info("Startup: bot lanzado en tarea de fondo.")

@app.on_event("shutdown")
async def on_shutdown():
    global BOT, RUN_TASK
    try:
        if BOT:
            await BOT.updater.stop()
            await BOT.stop()
            await BOT.shutdown()
    finally:
        if RUN_TASK and not RUN_TASK.done():
            RUN_TASK.cancel()
