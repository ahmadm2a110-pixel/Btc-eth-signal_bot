import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError


# ============================================================
# تنظیمات اصلی
# ============================================================

BOT_TOKEN = “8739764593:AAHxY0MrHCvcOjl2mIBfd1syIyZvyRXKsi0”
CHAT_ID = 8739764593

SYMBOLS = ["BTC/USDT", "ETH/USDT"]

TIMEFRAME = "5m"
LIMIT = 100

MIN_SCORE = 7
COOLDOWN_MINUTES = 15


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# اتصال‌ها
# ============================================================

bot = Bot(token=BOT_TOKEN)

exchange = ccxt.binance({
    "enableRateLimit": True
})


# جلوگیری از ارسال سیگنال تکراری
last_signal_time = {}


# ============================================================
# دریافت اطلاعات بازار
# ============================================================

async def fetch_data(symbol: str):

    try:
        ohlcv = await exchange.fetch_ohlcv(
            symbol,
            timeframe=TIMEFRAME,
            limit=LIMIT
        )

        df = pd.DataFrame(
            ohlcv,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        return df

    except Exception as e:

        logger.error(
            f"خطا در دریافت اطلاعات {symbol}: {e}"
        )

        return None


# ============================================================
# تحلیل
# ============================================================

def analyze(df: pd.DataFrame, symbol: str):

    if df is None or len(df) < 50:
        return None

    try:

        # اندیکاتورها
        df["ema20"] = ta.ema(
            df["close"],
            length=20
        )

        df["ema50"] = ta.ema(
            df["close"],
            length=50
        )

        df["rsi"] = ta.rsi(
            df["close"],
            length=14
        )

        df["atr"] = ta.atr(
            df["high"],
            df["low"],
            df["close"],
            length=14
        )

        df["vol_sma"] = ta.sma(
            df["volume"],
            length=20
        )

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["close"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        rsi = float(last["rsi"])
        atr = float(last["atr"])
        volume = float(last["volume"])
        vol_sma = float(last["vol_sma"])

        if pd.isna(ema20) or pd.isna(ema50):
            return None

        if pd.isna(rsi) or pd.isna(atr):
            return None

        if pd.isna(vol_sma) or vol_sma <= 0:
            vol_ratio = 1
        else:
            vol_ratio = volume / vol_sma

        score = 0
        reasons = []
        direction = None

        # ====================================================
        # LONG
        # ====================================================

        if close > ema20 > ema50:

            score += 3
            reasons.append(
                "روند صعودی: EMA20 بالاتر از EMA50"
            )

            if 40 <= rsi <= 62:

                score += 2
                reasons.append(
                    f"RSI مناسب: {rsi:.1f}"
                )

            if vol_ratio >= 1.25:

                score += 2
                reasons.append(
                    f"حجم بالا: {vol_ratio:.2f}x"
                )

            if (
                close > float(prev["close"])
                and float(last["close"]) > float(last["open"])
            ):

                score += 1
                reasons.append(
                    "کندل صعودی"
                )

            if close > ema20 * 0.998:

                score += 1
                reasons.append(
                    "قیمت نزدیک EMA20"
                )

            if score >= MIN_SCORE:
                direction = "LONG"

        # ====================================================
        # SHORT
        # ====================================================

        elif close < ema20 < ema50:

            score += 3
            reasons.append(
                "روند نزولی: EMA20 پایین‌تر از EMA50"
            )

            if 38 <= rsi <= 60:

                score += 2
                reasons.append(
                    f"RSI مناسب: {rsi:.1f}"
                )

            if vol_ratio >= 1.25:

                score += 2
                reasons.append(
                    f"حجم بالا: {vol_ratio:.2f}x"
                )

            if (
                close < float(prev["close"])
                and float(last["close"]) < float(last["open"])
            ):

                score += 1
                reasons.append(
                    "کندل نزولی"
                )

            if close < ema20 * 1.002:

                score += 1
                reasons.append(
                    "قیمت نزدیک EMA20"
                )

            if score >= MIN_SCORE:
                direction = "SHORT"

        # ====================================================

        if direction is None:
            return None

        if score < MIN_SCORE:
            return None

        # ====================================================
        # Entry / SL / TP
        # ====================================================

        decimals = 2 if "BTC" in symbol else 3

        entry = round(close, decimals)

        atr_mult = 1.6

        if direction == "LONG":

            sl = round(
                entry - atr * atr_mult,
                decimals
            )

            risk = entry - sl

            if risk <= 0:
                return None

            tp1 = round(
                entry + risk * 1.5,
                decimals
            )

            tp2 = round(
                entry + risk * 2.8,
                decimals
            )

        else:

            sl = round(
                entry + atr * atr_mult,
                decimals
            )

            risk = sl - entry

            if risk <= 0:
                return None

            tp1 = round(
                entry - risk * 1.5,
                decimals
            )

            tp2 = round(
                entry - risk * 2.8,
                decimals
            )

        rr = 1.5

        leverage = 7 if score >= 8 else 5

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

    except Exception as e:

        logger.error(
            f"خطا در تحلیل {symbol}: {e}"
        )

        return None


# ============================================================
# ساخت پیام سیگنال
# ============================================================

def format_signal(sig: dict):

    if sig["direction"] == "LONG":

        emoji = "🟢"
        direction_fa = "لانگ"

    else:

        emoji = "🔴"
        direction_fa = "شورت"

    text = f"""
{emoji} **سیگنال اسکلپ ۵ دقیقه‌ای** {emoji}

📌 **{sig["symbol"]}**
📈 جهت: **{direction_fa}**

⏰ زمان کندل: `{sig["time"]} UTC`

🎯 ورود: `{sig["entry"]}`

🛑 حد ضرر: `{sig["sl"]}`

✅ حد سود ۱: `{sig["tp1"]}`

✅ حد سود ۲: `{sig["tp2"]}`

💪 قدرت سیگنال: **{sig["score"]}/10**

📊 RSI: `{sig["rsi"]}`

📊 حجم: `{sig["vol_ratio"]}x`

🔥 لوریج پیشنهادی: **{sig["leverage"]}x**

📝 دلایل:
"""

    for reason in sig["reasons"]:

        text += f"• {reason}\n"

    text += """
⚠️ مدیریت سرمایه را جدی بگیر.
"""

    return text.strip()


# ============================================================
# ارسال سیگنال‌ها
# ============================================================

async def check_and_send():

    global last_signal_time

    now = datetime.now(timezone.utc)

    for symbol in SYMBOLS:

        try:

            df = await fetch_data(symbol)

            signal = analyze(
                df,
                symbol
            )

            if not signal:
                continue

            key = f"{symbol}_{signal['direction']}"

            previous_time = last_signal_time.get(key)

            if previous_time:

                elapsed = (
                    now - previous_time
                ).total_seconds()

                if elapsed < COOLDOWN_MINUTES * 60:
                    continue

            message = format_signal(signal)

            try:

                await bot.send_message(
                    chat_id=CHAT_ID,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )

                last_signal_time[key] = now

                logger.info(
                    f"سیگنال ارسال شد: "
                    f"{symbol} "
                    f"{signal['direction']} "
                    f"score={signal['score']}"
                )

            except TelegramError as e:

                logger.error(
                    f"خطای تلگرام هنگام ارسال: {e}"
                )

        except Exception as e:

            logger.error(
                f"خطا در بررسی {symbol}: {e}"
            )


# ============================================================
# اجرای اصلی
# ============================================================

async def main():

    logger.info(
        "ربات سیگنال اسکلپ ۵ دقیقه‌ای شروع به کار کرد..."
    )

    # تست اتصال به Telegram
    try:

        me = await bot.get_me()

        logger.info(
            f"Telegram Bot connected: @{me.username}"
        )

    except Exception as e:

        logger.error(
            f"اتصال به Telegram ناموفق بود: {e}"
        )

        return

    # پیام شروع
    try:

        await bot.send_message(
            chat_id=CHAT_ID,
            text=(
                "✅ ربات سیگنال BTC و ETH "
                "(۵ دقیقه) فعال شد.\n\n"
                "منتظر سیگنال‌های باکیفیت باش."
            )
        )

        logger.info(
            "پیام فعال شدن ربات با موفقیت ارسال شد."
        )

    except TelegramError as e:

        logger.error(
            f"نتوانستم پیام شروع را ارسال کنم: {e}"
        )

        # اینجا برنامه را نمی‌کشیم
        # تا Railway بی‌دلیل Crash Loop نشود.

    # حلقه اصلی
    while True:

        try:

            await check_and_send()

        except Exception as e:

            logger.error(
                f"خطا در حلقه اصلی: {e}"
            )

        await asyncio.sleep(45)


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        logger.info(
            "ربات متوقف شد."
        )

    finally:

        try:
            asyncio.run(exchange.close())
        except Exception:
            pass
