# server.py — Bot Telegram vía webhook en Render (gratis)
# Uso educativo. No es asesoría financiera.

import os, math, html, time, asyncio
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# =================== Config & Credenciales ===================
BOT_TOKEN   = os.environ.get("BOT_TOKEN")      # <-- OBLIGATORIO (ponerlo en Render)
PUBLIC_URL  = os.environ.get("PUBLIC_URL", "") # <-- Se pondrá después del primer deploy
APP_NAME    = "SignalInver"

ALLOWED_INTERVALS = {"1d","1h","30m","15m","5m"}     # velas
ALLOWED_HORIZONS  = {"1d","1w","1m","1y"}            # horizonte
DEFAULT_INTERVAL  = "1d"
DEFAULT_HORIZON   = "1d"
CACHE_TTL         = 60
CACHE = {}

def safe(x:str)->str: return html.escape(str(x))

# Alias de tickers comunes → yfinance
TICKER_MAP = {
    "BTC":"BTC-USD", "ETH":"ETH-USD",
    "SPX":"^GSPC", "SP500":"^GSPC", "S&P500":"^GSPC",
    "NDX":"^NDX", "DOW":"^DJI",
    "QQQ":"QQQ", "NVDA":"NVDA", "AAPL":"AAPL",
    "GOLD":"GC=F", "XAU":"GC=F",
    "OIL":"CL=F", "WTI":"CL=F", "BRENT":"BZ=F",
    "EURUSD":"EURUSD=X", "DXY":"DX-Y.NYB",
}
def norm_ticker(t:str)->str: return TICKER_MAP.get(t.upper().strip(), t.upper().strip())

def _get_cached(key, ttl): 
    x = CACHE.get(key)
    return x["data"] if x and (time.time()-x["ts"]<=ttl) else None
def _set_cached(key, data): 
    CACHE[key] = {"ts": time.time(), "data": data}

# =================== Indicadores ===================
def sma(s,n): return s.rolling(n).mean()
def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); down=-d.clip(upper=0)
    ru=up.ewm(alpha=1/n, adjust=False).mean()
    rd=down.ewm(alpha=1/n, adjust=False).mean()
    rs=ru/rd.replace(0, np.nan)
    return 100-(100/(1+rs))

def macd(s, fast=12, slow=26, signal=9):
    ef=s.ewm(span=fast, adjust=False).mean()
    es=s.ewm(span=slow, adjust=False).mean()
    line=ef-es
    sig=line.ewm(span=signal, adjust=False).mean()
    hist=line-sig
    return line, sig, hist

def bollinger(close, n=20, k=2):
    mid  = close.rolling(n).mean()
    sd   = close.rolling(n).std()
    up   = mid + k*sd
    low  = mid - k*sd
    return up, mid, low, sd

def tag_rsi(v):
    if v>=70: return "Sobrecompra"
    if v<=30: return "Sobreventa"
    return "Neutro"

def tag_macd(line, sig): return "Alcista" if line>sig else "Bajista"

def tag_boll(pos):
    if pos>=1:  return "Sobrecompra (arriba banda)"
    if pos<=-1: return "Sobreventa (abajo banda)"
    return "Dentro de bandas"

def trend_tag(s50, s200):
    return "Tendencia alcista (SMA50>SMA200)" if s50>s200 else "Tendencia bajista (SMA50<SMA200)"

# =================== Datos ===================
def load_prices(ticker, interval):
    key=("prices",ticker,interval)
    df=_get_cached(key, CACHE_TTL)
    if df is not None: return df
    if interval=="1d": period="3y"
    elif interval in {"1h","30m","15m"}: period="180d"
    else: period="60d"
    df=yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError("Sin datos para ese ticker/intervalo.")
    df["Close"]=df["Close"].astype(float)
    _set_cached(key, df)
    return df

def build_ta_summary(ticker, interval):
    df=load_prices(ticker, interval)
    c=df["Close"].dropna()
    if len(c)<220: raise ValueError("Pocos datos; prueba 1d u otro ticker.")
    rsi14=rsi(c,14)
    mline, msig, mhist = macd(c)
    up, mid, low, sd = bollinger(c,20,2)
    s50=sma(c,50); s200=sma(c,200); e20=ema(c,20)

    # timestamps
    idx = c.index[-1]
    ts = pd.Timestamp(idx)
    if ts.tzinfo is None:
        candle_utc = ts.tz_localize("UTC")
    else:
        candle_utc = ts.tz_convert("UTC")
    local_tz = datetime.now().astimezone().tzinfo
    candle_local = candle_utc.tz_convert(local_tz)
    now_local = datetime.now().astimezone()

    last=dict(
        price=float(c.iloc[-1]),
        rsi=float(rsi14.iloc[-1]),
        macd_line=float(mline.iloc[-1]),
        macd_sig=float(msig.iloc[-1]),
        macd_hist=float(mhist.iloc[-1]),
        bb_upper=float(up.iloc[-1]),
        bb_mid=float(mid.iloc[-1]),
        bb_lower=float(low.iloc[-1]),
        sma50=float(s50.iloc[-1]),
        sma200=float(s200.iloc[-1]),
        ema20=float(e20.iloc[-1]),
        band_pos=0.0,
        candle_local=candle_local,
        candle_utc=candle_utc,
        now_local=now_local
    )
    band_half=(last["bb_upper"]-last["bb_lower"])/2
    last["band_pos"]=0.0 if band_half==0 else (last["price"]-last["bb_mid"])/band_half

    tags=dict(
        rsi=tag_rsi(last["rsi"]),
        macd=tag_macd(last["macd_line"], last["macd_sig"]),
        boll=tag_boll(last["band_pos"]),
        trend=trend_tag(last["sma50"], last["sma200"]),
        ema20="Precio>EMA20" if last["price"]>last["ema20"] else "Precio<EMA20",
    )
    return last, tags

# =================== Probabilidad histórica ===================
def sigmoid(x): return 1/(1+math.exp(-x))

def _horizon_steps(h: str) -> int:
    return {"1d": 1, "1w": 5, "1m": 21, "1y": 252}.get(h, 1)

def _prep_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].astype(float).copy()
    rsi14 = rsi(c, 14)
    mline, msig, mhist = macd(c)
    up, mid, low, sd = bollinger(c, 20, 2)
    s50, s200 = sma(c, 50), sma(c, 200)
    e20 = ema(c, 20)
    feat = pd.DataFrame(index=c.index)
    feat["price"] = c
    feat["rsi"] = rsi14
    feat["macd_unit"] = (mhist / c.replace(0, np.nan)) / 0.005
    feat["trend"] = (s50 - s200) / s200.replace(0, np.nan)
    band_half = (up - low) / 2
    feat["band_pos"] = (c - mid) / band_half.replace(0, np.nan)
    feat["ema_flag"] = (c > e20).astype(int)
    return feat

def _wilson_interval(p: float, n: int, z: float = 1.96):
    if n == 0: return (0.0, 0.0)
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5)
    low = (centre - margin)/denominator
    high = (centre + margin)/denominator
    return (max(0.0, low), min(1.0, high))

def empirical_prob_now(ticker: str, interval: str, horizon: str):
    df_daily = load_prices(ticker, "1d")
    feats = _prep_feature_frame(df_daily).dropna().copy()
    if feats.empty: return {"mode":"heuristic"}

    # estado actual (en diario)
    last_daily, _ = build_ta_summary(ticker, "1d")

    steps = _horizon_steps(horizon)
    fut_ret = feats["price"].pct_change(steps).shift(-steps)
    feats["fut_win"] = (fut_ret > 0).astype(float)
    feats = feats.iloc[:-steps].dropna().copy()
    if feats.empty: return {"mode":"heuristic"}

    # tolerancias base
    tol = {"rsi":5.0, "macd_unit":0.5, "trend":0.02, "band_pos":0.30, "ema_flag":0}

    def filter_similar(center, scale):
        tols = {k: (v if k=="ema_flag" else v*scale) for k,v in tol.items()}
        m = (
            feats["rsi"].between(center["rsi"]-tols["rsi"], center["rsi"]+tols["rsi"]) &&
            feats["macd_unit"].between(center["macd_unit"]-tols["macd_unit"], center["macd_unit"]+tols["macd_unit"]) &&
            feats["trend"].between(center["trend"]-tols["trend"], center["trend"]+tols["trend"]) &&
            feats["band_pos"].between(center["band_pos"]-tols["band_pos"], center["band_pos"]+tols["band_pos"]) &&
            (feats["ema_flag"] == int(center["ema_flag"]))
        )
        return feats[m]

    center = {
        "rsi": last_daily["rsi"],
        "macd_unit": (last_daily["macd_hist"]/max(1e-8, last_daily["price"])) / 0.005,
        "trend": (last_daily["sma50"]-last_daily["sma200"])/max(1e-8, last_daily["sma200"]),
        "band_pos": last_daily["band_pos"],
        "ema_flag": 1 if last_daily["price"]>last_daily["ema20"] else 0,
    }

    selected = None
    for target_n in (50, 100, 200):
        for sc in (1.0, 1.5, 2.0, 3.0):
            cand = filter_similar(center, sc)
            if len(cand) >= target_n:
                selected = cand
                break
        if selected is not None:
            break
    if selected is None:
        selected = filter_similar(center, 3.0)

    n = int(len(selected))
    if n == 0: return {"mode":"heuristic"}
    p = float(selected["fut_win"].mean())
    lo, hi = _wilson_interval(p, n)
    return {"mode":"empirical", "n": n, "p_up": p, "p_dn": 1.0-p, "ci_low": lo, "ci_high": hi}

# =================== /info único ===================
def parse_args(args):
    """
    /info TICKER [tf]
    tf: velas -> 1d,1h,30m,15m,5m  |  horizonte -> 1d,1w,1m,1y
    """
    if not args: return None, None, None
    t = norm_ticker(args[0])
    if len(args)>=2:
        tf = args[1].lower().strip()
        if tf in ALLOWED_INTERVALS:
            return t, tf, "1d"
        if tf in ALLOWED_HORIZONS:
            return t, "1d", tf
    return t, DEFAULT_INTERVAL, DEFAULT_HORIZON

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg=(
        "Hola 👋 Soy un bot de análisis técnico educativo.\n\n"
        "Usa /info TICKER [tf]\n"
        "• tf velas: 1d, 1h, 30m, 15m, 5m  (intervalo)\n"
        "• tf horizonte: 1d, 1w, 1m, 1y (probabilidad histórica)\n\n"
        "Ejemplos: /info BTC 1d  •  /info BTC 1w  •  /info AAPL 1m\n"
        "<i>Alias normalizados (BTC→BTC-USD). Uso educativo; no asesoría.</i>"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def cmd_info(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        await update.message.reply_text(
            "Formato: /info TICKER [tf]\n"
            "Ejemplos: /info BTC 1d  |  /info BTC 1w  |  /info AAPL 1m\n"
            "tf velas: 1d, 1h, 30m, 15m, 5m  |  horizonte: 1d, 1w, 1m, 1y"
        ); return
    t, interval, horizon = parse_args(ctx.args)
    try:
        last, tags = build_ta_summary(t, interval)
        emp = empirical_prob_now(t, interval, horizon)
        if emp.get("mode")=="empirical":
            p_up, p_dn = emp["p_up"], emp["p_dn"]
            lo, hi, n = emp["ci_low"], emp["ci_high"], emp["n"]
            prob_line = (f"• Prob. histórica {horizon.upper()}: ⬆️ <b>{p_up*100:,.1f}%</b>  |  "
                         f"⬇️ <b>{p_dn*100:,.1f}%</b> (n={n}, IC95% {lo*100:,.1f}–{hi*100:,.1f}%)")
        else:
            # respaldo heurístico
            rsi_comp   = (50.0 - last["rsi"])/20.0
            macd_unit  = (last["macd_hist"]/max(1e-8,last["price"])) / 0.005
            trend_comp = (last["sma50"]-last["sma200"])/max(1e-8,last["sma200"])
            boll_comp  = - last["band_pos"]
            ema_comp   = 0.5 if last["price"]>last["ema20"] else -0.5
            S = 0.9*rsi_comp + 0.8*macd_unit + 1.0*trend_comp + 0.7*boll_comp + 0.6*ema_comp
            k = {"1d":0.9, "1w":0.7, "1m":0.5, "1y":0.3}[horizon]
            p_up = 1/(1+math.exp(-k*S)); p_dn = 1-p_up
            prob_line = f"• Prob. educativa {horizon.upper()}: ⬆️ <b>{p_up*100:,.1f}%</b>  |  ⬇️ <b>{p_dn*100:,.1f}%</b> (heurístico)"

        def fmt(dt): return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        txt=(
            f"<b>{safe(t)}</b> — velas [{safe(interval)}], horizonte [{safe(horizon)}]\n"
            f"• <b>Actualizado:</b> {safe(fmt(last['now_local']))}\n"
            f"• <b>Última vela:</b> {safe(fmt(last['candle_local']))} (UTC {safe(fmt(last['candle_utc']))})\n"
            f"• Precio: <b>{last['price']:,.2f}</b>\n"
            f"• RSI14: <b>{last['rsi']:.1f}</b> → <b>{safe(tags['rsi'])}</b>\n"
            f"• MACD: línea {'&gt;' if last['macd_line']>last['macd_sig'] else '&lt;'} señal → <b>{safe(tags['macd'])}</b>\n"
            f"• Bollinger (20,2): pos={last['band_pos']:+.2f} → <b>{safe(tags['boll'])}</b>\n"
            f"• SMA50/SMA200: {last['sma50']:,.2f} / {last['sma200']:,.2f} → <b>{safe(tags['trend'])}</b>\n"
            f"• EMA20: {last['ema20']:,.2f} → <b>{safe(tags['ema20'])}</b>\n"
            f"{prob_line}\n"
            f"<i>Uso educativo; no asesoría financiera.</i>"
        )
        await update.message.reply_text(txt, parse_mode=ParseMode.HTML)
    except Exception as e:
        try:
            await update.message.reply_text(f"No pude calcular /info para {t}. Detalle: {str(e)}")
        except Exception:
            pass

# =================== Telegram Application ===================
if not BOT_TOKEN:
    raise RuntimeError("Falta BOT_TOKEN (defínelo en Render).")

application = Application.builder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("start", cmd_start))
application.add_handler(CommandHandler("info",  cmd_info))

# =================== FastAPI + Webhook ===================
app = FastAPI()

@app.on_event("startup")
async def _startup():
    await application.initialize()
    await application.start()
    if PUBLIC_URL:
        await application.bot.set_webhook(url=f"{PUBLIC_URL}/webhook/{BOT_TOKEN}")

@app.on_event("shutdown")
async def _shutdown():
    await application.stop()
    await application.shutdown()

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido")
    data = await request.json()
    update = Update.de_json(data, application.bot)
    await application.update_queue.put(update)
    return PlainTextResponse("OK")

@app.get("/")
def root():
    return { "status": "ok", "app": APP_NAME }
