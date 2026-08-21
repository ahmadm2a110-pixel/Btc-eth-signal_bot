import asyncio
import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = " 8739764593:AAGVmuSxvyb4USsTCjqtUNFTgB9oS7TpXw4"
CHAT_ID = 8739764593

SYMBOLS = ["BTC/USDT", "ETH/USDT"]

# Entry timeframe
ENTRY_TIMEFRAME = "5m"

# Higher timeframe confirmation
HTF_15M = "15m"
HTF_1H = "1h"
HTF_4H = "4h"

LIMIT_5M = 200
LIMIT_15M = 200
LIMIT_1H = 200
LIMIT_4H = 200

# Minimum score
MIN_SCORE = 8

# Signal cooldown
COOLDOWN_MINUTES = 15

# Bot check interval
CHECK_INTERVAL_SECONDS = 30

# ATR stop multiplier
ATR_SL_MULTIPLIER = 1.5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# CONNECTIONS
# ============================================================

bot = Bot(token=BOT_TOKEN)

exchange = ccxt.binance({
    "enableRateLimit": True,
})


last_signal_time = {}


# ============================================================
# FETCH OHLCV
# ============================================================

async def fetch_ohlcv(symbol: str, timeframe: str, limit: int):

    try:

        data = await exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not data:
            return None

        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )

        return df

    except Exception as e:

        logger.error(
            f"OHLCV error [{symbol} {timeframe}]: {e}"
        )

        return None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df: pd.DataFrame):

    df = df.copy()

    df["ema20"] = ta.ema(
        df["close"],
        length=20,
    )

    df["ema50"] = ta.ema(
        df["close"],
        length=50,
    )

    df["ema200"] = ta.ema(
        df["close"],
        length=200,
    )

    df["rsi"] = ta.rsi(
        df["close"],
        length=14,
    )

    df["atr"] = ta.atr(
        df["high"],
        df["low"],
        df["close"],
        length=14,
    )

    df["volume_sma"] = ta.sma(
        df["volume"],
        length=20,
    )

    # ADX
    adx = ta.adx(
        df["high"],
        df["low"],
        df["close"],
        length=14,
    )

    if adx is not None:

        df["adx"] = adx["ADX_14"]
        df["dmp"] = adx["DMP_14"]
        df["dmn"] = adx["DMN_14"]

    else:

        df["adx"] = None
        df["dmp"] = None
        df["dmn"] = None

    # VWAP
    try:

        vwap = ta.vwap(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
        )

        if vwap is not None:
            df["vwap"] = vwap

    except Exception:

        df["vwap"] = None

    return df


# ============================================================
# TREND ANALYSIS
# ============================================================

def get_trend(df: pd.DataFrame):

    if df is None or len(df) < 200:
        return "UNKNOWN"

    df = add_indicators(df)

    # Last COMPLETED candle
    last = df.iloc[-2]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])

    if close > ema20 > ema50 > ema200:
        return "LONG"

    if close < ema20 < ema50 < ema200:
        return "SHORT"

    return "NEUTRAL"


# ============================================================
# MARKET STRUCTURE
# ============================================================

def get_structure(df: pd.DataFrame):

    if df is None or len(df) < 20:
        return "NEUTRAL"

    last = df.iloc[-2]

    recent = df.iloc[-12:-2]

    recent_high = recent["high"].max()
    recent_low = recent["low"].min()

    close = float(last["close"])

    if close > recent_high:
        return "BULL_BREAKOUT"

    if close < recent_low:
        return "BEAR_BREAKDOWN"

    # Simple momentum structure
    highs = recent["high"].tail(5).values
    lows = recent["low"].tail(5).values

    if len(highs) >= 5:

        higher_highs = (
            highs[-1] > highs[-3]
            and highs[-3] > highs[0]
        )

        higher_lows = (
            lows[-1] > lows[-3]
            and lows[-3] > lows[0]
        )

        if higher_highs and higher_lows:
            return "BULLISH"

        lower_highs = (
            highs[-1] < highs[-3]
            and highs[-3] < highs[0]
        )

        lower_lows = (
            lows[-1] < lows[-3]
            and lows[-3] < lows[0]
        )

        if lower_highs and lower_lows:
            return "BEARISH"

    return "NEUTRAL"


# ============================================================
# 5M SIGNAL
# ============================================================

def analyze_5m(
    df: pd.DataFrame,
    trend_15m: str,
    trend_1h: str,
    trend_4h: str,
):

    if df is None or len(df) < 100:
        return None

    df = add_indicators(df)

    # Completed candle
    last = df.iloc[-2]
    previous = df.iloc[-3]

    close = float(last["close"])
    open_price = float(last["open"])

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])

    rsi = float(last["rsi"])
    atr = float(last["atr"])

    volume = float(last["volume"])
    volume_sma = float(last["volume_sma"])

    adx = float(last["adx"])

    if any(
        pd.isna(x)
        for x in [
            ema20,
            ema50,
            rsi,
            atr,
            volume_sma,
            adx,
        ]
    ):
        return None

    if volume_sma <= 0 or atr <= 0:
        return None

    volume_ratio = volume / volume_sma

    structure = get_structure(df)

    score_long = 0
    score_short = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # HIGHER TIMEFRAME TREND
    # ========================================================

    if trend_4h == "LONG":

        score_long += 2
        long_reasons.append("روند 4H صعودی")

    elif trend_4h == "SHORT":

        score_short += 2
        short_reasons.append("روند 4H نزولی")

    if trend_1h == "LONG":

        score_long += 2
        long_reasons.append("روند 1H صعودی")

    elif trend_1h == "SHORT":

        score_short += 2
        short_reasons.append("روند 1H نزولی")

    if trend_15m == "LONG":

        score_long += 1
        long_reasons.append("روند 15M صعودی")

    elif trend_15m == "SHORT":

        score_short += 1
        short_reasons.append("روند 15M نزولی")

    # ========================================================
    # 5M TREND
    # ========================================================

    if close > ema20 > ema50:

        score_long += 2
        long_reasons.append(
            "EMA20 بالاتر از EMA50"
        )

    elif close < ema20 < ema50:

        score_short += 2
        short_reasons.append(
            "EMA20 پایین‌تر از EMA50"
        )

    # ========================================================
    # ADX
    # ========================================================

    if adx >= 20:

        if score_long > score_short:

            score_long += 1
            long_reasons.append(
                f"ADX مناسب ({adx:.1f})"
            )

        elif score_short > score_long:

            score_short += 1
            short_reasons.append(
                f"ADX مناسب ({adx:.1f})"
            )

    # ========================================================
    # VOLUME
    # ========================================================

    if volume_ratio >= 1.25:

        if score_long > score_short:

            score_long += 1
            long_reasons.append(
                f"حجم تأییدکننده ({volume_ratio:.2f}x)"
            )

        elif score_short > score_long:

            score_short += 1
            short_reasons.append(
                f"حجم تأییدکننده ({volume_ratio:.2f}x)"
            )

    # ========================================================
    # RSI
    # ========================================================

    if 45 <= rsi <= 65:

        if score_long > score_short:

            score_long += 1
            long_reasons.append(
                f"RSI مناسب ({rsi:.1f})"
            )

    if 35 <= rsi <= 55:

        if score_short > score_long:

            score_short += 1
            short_reasons.append(
                f"RSI مناسب ({rsi:.1f})"
            )

    # ========================================================
    # STRUCTURE
    # ========================================================

    if structure in (
        "BULLISH",
        "BULL_BREAKOUT",
    ):

        score_long += 1

        long_reasons.append(
            f"ساختار بازار: {structure}"
        )

    elif structure in (
        "BEARISH",
        "BEAR_BREAKDOWN",
    ):

        score_short += 1

        short_reasons.append(
            f"ساختار بازار: {structure}"
        )

    # ========================================================
    # CANDLE CONFIRMATION
    # ========================================================

    if (
        close > open_price
        and close > float(previous["close"])
    ):

        score_long += 1
        long_reasons.append(
            "کندل تأیید صعودی"
        )

    if (
        close < open_price
        and close < float(previous["close"])
    ):

        score_short += 1
        short_reasons.append(
            "کندل تأیید نزولی"
        )

    # ========================================================
    # DETERMINE DIRECTION
    # ========================================================

    if score_long >= MIN_SCORE and score_long > score_short:

        direction = "LONG"
        score = score_long
        reasons = long_reasons

    elif score_short >= MIN_SCORE and score_short > score_long:

        direction = "SHORT"
        score = score_short
        reasons = short_reasons

    else:

        return None

    # ========================================================
    # AVOID EXTREME ENTRIES
    # ========================================================

    distance_from_ema = abs(
        close - ema20
    ) / ema20

    if distance_from_ema > 0.012:

        logger.info(
            f"{direction} rejected: price too far from EMA20"
        )

        return None

    # ========================================================
    # PRICE LEVELS
    # ========================================================

    decimals = 2 if "BTC" in df.name if False else 2

    # Symbol-independent precision handled below
    entry = close

    if direction == "LONG":

        sl = entry - (
            atr * ATR_SL_MULTIPLIER
        )

        risk = entry - sl

        tp1 = entry + (
            risk * 1.5
        )

        tp2 = entry + (
            risk * 2.5
        )

    else:

        sl = entry + (
            atr * ATR_SL_MULTIPLIER
        )

        risk = sl - entry

        tp1 = entry - (
            risk * 1.5
        )

        tp2 = entry - (
            risk * 2.5
        )

    if risk <= 0:
        return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": score,
        "reasons": reasons,
        "rsi": rsi,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "structure": structure,
        "candle_time": last["timestamp"],
    }


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(
    symbol: str,
    signal: dict,
):

    direction = signal["direction"]

    if direction == "LONG":

        emoji = "🟢"
        direction_fa = "لانگ"

    else:

        emoji = "🔴"
        direction_fa = "شورت"

    decimals = 2 if "BTC" in symbol else 3

    entry = round(signal["entry"], decimals)
    sl = round(signal["sl"], decimals)
    tp1 = round(signal["tp1"], decimals)
    tp2 = round(signal["tp2"], decimals)

    candle_time = signal[
        "candle_time"
    ].strftime("%H:%M")

    text = (
        f"{emoji} *سیگنال اسکلپ ۵ دقیقه‌ای* {emoji}\n\n"

        f"📌 *{symbol.replace('/', '')}*\n"
        f"📈 جهت: *{direction_fa}*\n"
        f"⭐ امتیاز: *{signal['score']}/10*\n\n"

        f"⏰ کندل: `{candle_time} UTC`\n\n"

        f"🎯 ورود: `{entry}`\n"
        f"🛑 حد ضرر: `{sl}`\n"
        f"✅ TP1: `{tp1}`\n"
        f"✅ TP2: `{tp2}`\n\n"

        f"📊 RSI: `{signal['rsi']:.1f}`\n"
        f"📊 ADX: `{signal['adx']:.1f}`\n"
        f"📊 Volume: `{signal['volume_ratio']:.2f}x`\n"
        f"🏗 Structure: `{signal['structure']}`\n\n"

        f"📝 *دلایل:*\n"
    )

    for reason in signal["reasons"]:
        text += f"• {reason}\n"

    text += (
        "\n⚠️ این سیگنال تضمین سود نیست. "
        "مدیریت ریسک ضروری است."
    )

    return text


# ============================================================
# PROCESS SYMBOL
# ============================================================

async def process_symbol(symbol: str):

    try:

        df_5m = await fetch_ohlcv(
            symbol,
            "5m",
            LIMIT_5M,
        )

        df_15m = await fetch_ohlcv(
            symbol,
            "15m",
            LIMIT_15M,
        )

        df_1h = await fetch_ohlcv(
            symbol,
            "1h",
            LIMIT_1H,
        )

        df_4h = await fetch_ohlcv(
            symbol,
            "4h",
            LIMIT_4H,
        )

        if any(
            df is None
            for df in [
                df_5m,
                df_15m,
                df_1h,
                df_4h,
            ]
        ):
            return

        trend_15m = get_trend(df_15m)
        trend_1h = get_trend(df_1h)
        trend_4h = get_trend(df_4h)

        # Don't trade when higher timeframes strongly conflict
        if (
            trend_1h != "NEUTRAL"
            and trend_4h != "NEUTRAL"
            and trend_1h != trend_4h
        ):

            logger.info(
                f"{symbol}: HTF conflict -> NO TRADE"
            )

            return

        signal = analyze_5m(
            df_5m,
            trend_15m,
            trend_1h,
            trend_4h,
        )

        if signal is None:

            return

        key = (
            f"{symbol}_"
            f"{signal['direction']}_"
            f"{signal['candle_time']}"
        )

        # Same candle can only generate one signal
        if key in last_signal_time:
            return

        message = format_signal(
            symbol,
            signal,
        )

        try:

            await bot.send_message(
                chat_id=CHAT_ID,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
            )

            last_signal_time[key] = datetime.now(
                timezone.utc
            )

            logger.info(
                f"SIGNAL SENT: "
                f"{symbol} "
                f"{signal['direction']} "
                f"score={signal['score']}"
            )

        except TelegramError as e:

            logger.error(
                f"Telegram error [{symbol}]: {e}"
            )

    except Exception as e:

        logger.exception(
            f"Processing error [{symbol}]: {e}"
        )


# ============================================================
# TELEGRAM TEST
# ============================================================

async def telegram_test():

    try:

        me = await bot.get_me()

        logger.info(
            f"Telegram connected: @{me.username}"
        )

        await bot.send_message(
            chat_id=CHAT_ID,
            text=(
                "✅ ربات BTC/ETH اسکلپ ۵ دقیقه‌ای "
                "با موفقیت فعال شد.\n\n"
                "📊 فیلترهای 4H / 1H / 15M / 5M فعال هستند."
            ),
        )

        logger.info(
            "Startup message sent."
        )

        return True

    except TelegramError as e:

        logger.error(
            f"Telegram startup error: {e}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "======================================"
    )

    logger.info(
        "BTC/ETH SCALPING BOT STARTING"
    )

    logger.info(
        "======================================"
    )

    if not await telegram_test():

        logger.error(
            "Telegram initialization failed."
        )

        return

    while True:

        start = datetime.now(
            timezone.utc
        )

        logger.info(
            "Market scan started."
        )

        for symbol in SYMBOLS:

            await process_symbol(symbol)

        elapsed = (
            datetime.now(timezone.utc)
            - start
        ).total_seconds()

        sleep_time = max(
            5,
            CHECK_INTERVAL_SECONDS - int(elapsed),
        )

        await asyncio.sleep(
            sleep_time
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        logger.info(
            "Bot stopped."
        )

    except Exception as e:

        logger.exception(
            f"Fatal error: {e}"
        )

    finally:

        try:

            asyncio.run(
                exchange.close()
            )

        except Exception:
            pass
