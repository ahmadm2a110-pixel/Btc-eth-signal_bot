import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from telegram import Bot
from telegram.constants import ParseMode

# ================== تنظیمات ==================
BOT_TOKEN = "8739764593:AAFxXMy__Ob84yZX8cCCYT66tGQGYEBVU0o"
CHAT_ID = 283870453

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "5m"
LIMIT = 100

# حداقل امتیاز برای ارسال سیگنال (۷ از ۱۰)
MIN_SCORE = 7

# جلوگیری از سیگنال تکراری (دقیقه)
COOLDOWN_MINUTES = 15

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
exchange = ccxt.binance({"enableRateLimit": True})

# ذخیره آخرین سیگنال
last_signal_time = {}

async def fetch_data(symbol: str):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def analyze(df: pd.DataFrame, symbol: str):
    if df is None or len(df) < 50:
        return None

    # اندیکاتورها
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["vol_sma"] = ta.sma(df["volume"], length=20)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = last["close"]
    ema20 = last["ema20"]
    ema50 = last["ema50"]
    rsi = last["rsi"]
    atr = last["atr"]
    vol_ratio = last["volume"] / last["vol_sma"] if last["vol_sma"] > 0 else 1

    score = 0
    reasons = []
    direction = None

    # ---------- لانگ ----------
    if close > ema20 > ema50:  # روند صعودی
        score += 3
        reasons.append("روند صعودی (EMA20 > EMA50)")

        if 40 <= rsi <= 62:
            score += 2
            reasons.append(f"RSI مناسب ({rsi:.1f})")

        if vol_ratio >= 1.25:
            score += 2
            reasons.append(f"حجم بالا ({vol_ratio:.2f}x)")

        if close > prev["close"] and last["close"] > last["open"]:
            score += 1
            reasons.append("کندل صعودی")

        if close > ema20 * 0.998:  # نزدیک یا بالای EMA20
            score += 1
            reasons.append("نزدیک EMA20")

        if score >= MIN_SCORE:
            direction = "LONG"

    # ---------- شورت ----------
    elif close < ema20 < ema50:  # روند نزولی
        score += 3
        reasons.append("روند نزولی (EMA20 < EMA50)")

        if 38 <= rsi <= 60:
            score += 2
            reasons.append(f"RSI مناسب ({rsi:.1f})")

        if vol_ratio >= 1.25:
            score += 2
            reasons.append(f"حجم بالا ({vol_ratio:.2f}x)")

        if close < prev["close"] and last["close"] < last["open"]:
            score += 1
            reasons.append("کندل نزولی")

        if close < ema20 * 1.002:
            score += 1
            reasons.append("نزدیک EMA20")

        if score >= MIN_SCORE:
            direction = "SHORT"

    if direction is None or score < MIN_SCORE:
        return None

    # محاسبه ورود و حد ضرر و سود
    entry = round(close, 2 if "BTC" in symbol else 3)
    atr_mult = 1.6

    if direction == "LONG":
        sl = round(entry - atr * atr_mult, 2 if "BTC" in symbol else 3)
        risk = entry - sl
        tp1 = round(entry + risk * 1.5, 2 if "BTC" in symbol else 3)
        tp2 = round(entry + risk * 2.8, 2 if "BTC" in symbol else 3)
        leverage = 7 if score >= 8 else 5
    else:
        sl = round(entry + atr * atr_mult, 2 if "BTC" in symbol else 3)
        risk = sl - entry
        tp1 = round(entry - risk * 1.5, 2 if "BTC" in symbol else 3)
        tp2 = round(entry - risk * 2.8, 2 if "BTC" in symbol else 3)
        leverage = 7 if score >= 8 else 5

    rr = round((tp1 - entry) / risk if direction == "LONG" else (entry - tp1) / risk, 2)

    return {
        "symbol": symbol.replace("/", ""),
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": score,
        "reasons": reasons,
        "leverage": leverage,
        "rr": rr,
        "rsi": round(rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "time": last["timestamp"].strftime("%H:%M")
    }

def format_signal(sig: dict) -> str:
    emoji = "🟢" if sig["direction"] == "LONG" else "🔴"
    dir_fa = "لانگ" if sig["direction"] == "LONG" else "شورت"

    text = f"""
{emoji} **سیگنال اسکلپ ۵ دقیقه** {emoji}

📌 **{sig['symbol']}** | {dir_fa}
⏰ زمان کندل: {sig['time']} UTC

🎯 **ورود**: `{sig['entry']}`
🛑 **حد ضرر**: `{sig['sl']}`
✅ **حد سود ۱**: `{sig['tp1']}` (R:R ≈ {sig['rr']})
✅ **حد سود ۲**: `{sig['tp2']}`

💪 قدرت سیگنال: **{sig['score']}/10**
📊 RSI: {sig['rsi']} | حجم: {sig['vol_ratio']}x
🔥 پیشنهاد لوریج: **{sig['leverage']}x تا ۱۰x**

📝 دلایل:
"""
    for r in sig["reasons"]:
        text += f"• {r}\n"

    text += "\n⚠️ مدیریت سرمایه جدی بگیر. فقط با پولی وارد شو که از دست دادنش مشکلی برات ایجاد نکنه."
    return text.strip()

async def check_and_send():
    global last_signal_time
    now = datetime.now(timezone.utc)

    for symbol in SYMBOLS:
        df = await fetch_data(symbol)
        sig = analyze(df, symbol)

        if not sig:
            continue

        key = f"{symbol}_{sig['direction']}"
        last_time = last_signal_time.get(key)

        if last_time and (now - last_time).total_seconds() < COOLDOWN_MINUTES * 60:
            continue

        message = format_signal(sig)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
            last_signal_time[key] = now
            logger.info(f"Signal sent: {symbol} {sig['direction']} score={sig['score']}")
        except Exception as e:
            logger.error(f"Send error: {e}")

async def main():
    logger.info("ربات سیگنال اسکلپ ۵ دقیقه‌ای شروع به کار کرد...")
    await bot.send_message(chat_id=CHAT_ID, text="✅ ربات سیگنال BTC و ETH (۵ دقیقه) فعال شد.\nمنتظر سیگنال‌های باکیفیت باش.")

    while True:
        try:
            await check_and_send()
        except Exception as e:
            logger.error(f"Loop error: {e}")
        await asyncio.sleep(45)  # هر ۴۵ ثانیه چک می‌کنه (روی بستن کندل حساس است)

if __name__ == "__main__":
    asyncio.run(main())
