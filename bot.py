import asyncio
import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import httpx


# ============================================================
# SETTINGS
# ============================================================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]

TIMEFRAME_5M = "5m"
TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"

LIMIT = 250

# Lower than the old 8-point system.
MIN_SCORE_A = 6
MIN_SCORE_A_PLUS = 8

CHECK_INTERVAL = 30

ATR_SL_MULTIPLIER = 1.5
TP1_RR = 1.5
TP2_RR = 2.5

# Price should not be extremely extended from EMA20.
MAX_ATR_DISTANCE_FROM_EMA = 2.5


# ============================================================
# NTFY SETTINGS
# ============================================================

NTFY_SERVER = "https://ntfy.sh"
NTFY_TOPIC = "btc_ah7K9xQ2_signal"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# BINANCE
# ============================================================

exchange = ccxt.binance({
    "enableRateLimit": True,
})


# ============================================================
# SIGNAL MEMORY
# ============================================================

sent_signals = set()


# ============================================================
# NTFY
# ============================================================

async def send_ntfy(
    message,
    title="Crypto Scalping Signal",
    priority="high",
    tags="chart_with_upwards_trend",
):

    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"

    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": tags,
    }

    try:

        async with httpx.AsyncClient(
            timeout=15
        ) as client:

            response = await client.post(
                url,
                content=message.encode("utf-8"),
                headers=headers,
            )

            response.raise_for_status()

        logger.info("ntfy notification sent successfully.")

        return True

    except Exception as e:

        logger.error(f"ntfy send error: {e}")

        return False


# ============================================================
# NTFY TEST
# ============================================================

async def ntfy_test():

    test_message = (
        "✅ ربات کریپتو با موفقیت فعال شد.\n\n"
        "📊 ارزها: BTC / ETH / SOL / BNB / XRP / LINK / ADA / AVAX\n\n"
        "⏱ تایم‌فریم اصلی: 5M\n"
        "🧭 روند: 15M / 1H / 4H\n\n"
        "🎯 سیستم آماده اسکن بازار است."
    )

    return await send_ntfy(
        test_message,
        title="Crypto Bot Started",
        priority="high",
        tags="rocket",
    )


# ============================================================
# FETCH MARKET DATA
# ============================================================

async def fetch_data(
    symbol,
    timeframe,
    limit=LIMIT,
):

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
            f"Market data error "
            f"{symbol} {timeframe}: {e}"
        )

        return None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

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

    adx = ta.adx(
        df["high"],
        df["low"],
        df["close"],
        length=14,
    )

    if adx is not None:
        df["adx"] = adx["ADX_14"]
    else:
        df["adx"] = float("nan")

    return df


# ============================================================
# HIGHER TIMEFRAME TREND
# ============================================================

def get_trend(df):

    if df is None:
        return "UNKNOWN"

    if len(df) < 205:
        return "UNKNOWN"

    df = add_indicators(df)

    # Last CLOSED candle
    last = df.iloc[-2]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])

    if any(
        pd.isna(x)
        for x in [
            close,
            ema20,
            ema50,
            ema200,
        ]
    ):
        return "UNKNOWN"

    # Strong trend
    if close > ema20 > ema50 > ema200:
        return "LONG"

    if close < ema20 < ema50 < ema200:
        return "SHORT"

    # Moderate trend
    if close > ema20 and ema20 > ema50:
        return "LONG"

    if close < ema20 and ema20 < ema50:
        return "SHORT"

    return "NEUTRAL"


# ============================================================
# MARKET STRUCTURE
# ============================================================

def get_market_structure(df):

    if df is None or len(df) < 30:
        return "NEUTRAL"

    last = df.iloc[-2]

    previous = df.iloc[-12:-2]

    close = float(last["close"])

    recent_high = float(
        previous["high"].max()
    )

    recent_low = float(
        previous["low"].min()
    )

    if close > recent_high:
        return "BULL_BREAKOUT"

    if close < recent_low:
        return "BEAR_BREAKDOWN"

    highs = previous["high"].tail(5).values
    lows = previous["low"].tail(5).values

    if len(highs) >= 5:

        bullish = (
            highs[-1] > highs[-3]
            and highs[-3] > highs[0]
            and lows[-1] > lows[-3]
            and lows[-3] > lows[0]
        )

        if bullish:
            return "BULLISH"

        bearish = (
            highs[-1] < highs[-3]
            and highs[-3] < highs[0]
            and lows[-1] < lows[-3]
            and lows[-3] < lows[0]
        )

        if bearish:
            return "BEARISH"

    return "NEUTRAL"


# ============================================================
# ANALYZE 5M
# ============================================================

def analyze_5m(
    df,
    trend_15m,
    trend_1h,
    trend_4h,
    symbol,
):

    if df is None:
        return None

    if len(df) < 100:
        return None

    df = add_indicators(df)

    # Last CLOSED 5M candle
    last = df.iloc[-2]
    previous = df.iloc[-3]

    close = float(last["close"])
    open_price = float(last["open"])

    high = float(last["high"])
    low = float(last["low"])

    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])

    rsi = float(last["rsi"])
    atr = float(last["atr"])

    volume = float(last["volume"])
    volume_sma = float(last["volume_sma"])

    adx = float(last["adx"])

    previous_close = float(previous["close"])

    if any(
        pd.isna(x)
        for x in [
            close,
            open_price,
            high,
            low,
            ema20,
            ema50,
            rsi,
            atr,
            volume_sma,
            adx,
            previous_close,
        ]
    ):
        return None

    if atr <= 0 or volume_sma <= 0:
        return None

    volume_ratio = volume / volume_sma

    structure = get_market_structure(df)

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # 4H DIRECTION
    # ========================================================

    if trend_4h == "LONG":

        long_score += 2

        long_reasons.append(
            "روند 4H صعودی"
        )

    elif trend_4h == "SHORT":

        short_score += 2

        short_reasons.append(
            "روند 4H نزولی"
        )

    # ========================================================
    # 1H DIRECTION
    # ========================================================

    if trend_1h == "LONG":

        long_score += 2

        long_reasons.append(
            "روند 1H صعودی"
        )

    elif trend_1h == "SHORT":

        short_score += 2

        short_reasons.append(
            "روند 1H نزولی"
        )

    # ========================================================
    # 15M DIRECTION
    # ========================================================

    if trend_15m == "LONG":

        long_score += 1

        long_reasons.append(
            "روند 15M صعودی"
        )

    elif trend_15m == "SHORT":

        short_score += 1

        short_reasons.append(
            "روند 15M نزولی"
        )

    # ========================================================
    # 5M EMA
    # ========================================================

    if close > ema20:

        long_score += 1

        long_reasons.append(
            "قیمت بالای EMA20"
        )

    elif close < ema20:

        short_score += 1

        short_reasons.append(
            "قیمت پایین EMA20"
        )

    # EMA20 / EMA50 alignment
    if ema20 > ema50:

        long_score += 1

        long_reasons.append(
            "EMA20 بالاتر از EMA50"
        )

    elif ema20 < ema50:

        short_score += 1

        short_reasons.append(
            "EMA20 پایین‌تر از EMA50"
        )

    # ========================================================
    # ADX
    # ========================================================

    if adx >= 18:

        if long_score > short_score:

            long_score += 1

            long_reasons.append(
                f"قدرت روند مناسب ADX={adx:.1f}"
            )

        elif short_score > long_score:

            short_score += 1

            short_reasons.append(
                f"قدرت روند مناسب ADX={adx:.1f}"
            )

    # ========================================================
    # VOLUME
    # ========================================================

    if volume_ratio >= 1.10:

        if long_score > short_score:

            long_score += 1

            long_reasons.append(
                f"حجم مناسب {volume_ratio:.2f}x"
            )

        elif short_score > long_score:

            short_score += 1

            short_reasons.append(
                f"حجم مناسب {volume_ratio:.2f}x"
            )

    # ========================================================
    # RSI MOMENTUM
    # ========================================================

    # LONG
    if long_score > short_score:

        if 48 <= rsi <= 68:

            long_score += 1

            long_reasons.append(
                f"مومنتوم RSI مناسب {rsi:.1f}"
            )

        elif 40 <= rsi < 48:

            # Weak but usable pullback
            long_score += 0.5

            long_reasons.append(
                f"RSI پولبک {rsi:.1f}"
            )

    # SHORT
    elif short_score > long_score:

        if 32 <= rsi <= 52:

            short_score += 1

            short_reasons.append(
                f"مومنتوم RSI مناسب {rsi:.1f}"
            )

        elif 52 < rsi <= 60:

            short_score += 0.5

            short_reasons.append(
                f"RSI پولبک {rsi:.1f}"
            )

    # ========================================================
    # MARKET STRUCTURE
    # ========================================================

    if structure in [
        "BULLISH",
        "BULL_BREAKOUT",
    ]:

        long_score += 1

        long_reasons.append(
            f"ساختار بازار: {structure}"
        )

    elif structure in [
        "BEARISH",
        "BEAR_BREAKDOWN",
    ]:

        short_score += 1

        short_reasons.append(
            f"ساختار بازار: {structure}"
        )

    # ========================================================
    # CANDLE MOMENTUM
    # ========================================================

    candle_range = high - low

    if candle_range > 0:

        body = abs(close - open_price)

        body_ratio = body / candle_range

        # Bullish candle
        if (
            close > open_price
            and close > previous_close
            and body_ratio >= 0.45
        ):

            long_score += 1

            long_reasons.append(
                "مومنتوم کندل صعودی"
            )

        # Bearish candle
        elif (
            close < open_price
            and close < previous_close
            and body_ratio >= 0.45
        ):

            short_score += 1

            short_reasons.append(
                "مومنتوم کندل نزولی"
            )

    # ========================================================
    # SELECT DIRECTION
    # ========================================================

    if (
        long_score >= MIN_SCORE_A
        and long_score > short_score
    ):

        direction = "LONG"
        score = long_score
        reasons = long_reasons

    elif (
        short_score >= MIN_SCORE_A
        and short_score > long_score
    ):

        direction = "SHORT"
        score = short_score
        reasons = short_reasons

    else:

        return None

    # ========================================================
    # HIGHER TIMEFRAME HARD CONFLICT
    # ========================================================

    # If both 1H and 4H strongly disagree with the signal,
    # reject it.
    if (
        direction == "LONG"
        and trend_1h == "SHORT"
        and trend_4h == "SHORT"
    ):

        return None

    if (
        direction == "SHORT"
        and trend_1h == "LONG"
        and trend_4h == "LONG"
    ):

        return None

    # ========================================================
    # DON'T CHASE PRICE
    # ========================================================

    distance_from_ema = (
        abs(close - ema20)
        / atr
    )

    if distance_from_ema > MAX_ATR_DISTANCE_FROM_EMA:

        logger.info(
            f"{symbol}: price too far from EMA20 "
            f"({distance_from_ema:.2f} ATR)"
        )

        return None

    # ========================================================
    # SIGNAL QUALITY
    # ========================================================

    if score >= MIN_SCORE_A_PLUS:

        signal_grade = "A+"

    else:

        signal_grade = "A"

    # ========================================================
    # ENTRY / SL / TP
    # ========================================================

    entry = close

    if direction == "LONG":

        sl = (
            entry
            - atr * ATR_SL_MULTIPLIER
        )

        risk = entry - sl

        tp1 = (
            entry
            + risk * TP1_RR
        )

        tp2 = (
            entry
            + risk * TP2_RR
        )

    else:

        sl = (
            entry
            + atr * ATR_SL_MULTIPLIER
        )

        risk = sl - entry

        tp1 = (
            entry
            - risk * TP1_RR
        )

        tp2 = (
            entry
            - risk * TP2_RR
        )

    if risk <= 0:
        return None

    return {
        "symbol": symbol,
        "direction": direction,
        "grade": signal_grade,
        "score": score,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": rsi,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "structure": structure,
        "reasons": reasons,
        "candle_time": last["timestamp"],
    }


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(signal):

    symbol = signal["symbol"]

    if signal["direction"] == "LONG":

        emoji = "🟢"
        direction_fa = "لانگ"

    else:

        emoji = "🔴"
        direction_fa = "شورت"

    grade = signal["grade"]

    if symbol.startswith("BTC"):

        decimals = 2

    elif symbol.startswith(
        ("ETH", "BNB", "SOL", "AVAX")
    ):

        decimals = 3

    else:

        decimals = 4

    entry = round(
        signal["entry"],
        decimals,
    )

    sl = round(
        signal["sl"],
        decimals,
    )

    tp1 = round(
        signal["tp1"],
        decimals,
    )

    tp2 = round(
        signal["tp2"],
        decimals,
    )

    candle_time = signal[
        "candle_time"
    ].strftime("%H:%M")

    text = (
        f"{emoji} سیگنال اسکلپ ۵ دقیقه‌ای {emoji}\n\n"

        f"📌 {symbol.replace('/', '')}\n"
        f"📈 جهت: {direction_fa}\n"
        f"🏆 کیفیت: {grade}\n"
        f"⭐ امتیاز: {signal['score']:.1f}\n\n"

        f"⏰ کندل بسته‌شده: "
        f"{candle_time} UTC\n\n"

        f"🎯 ورود: {entry}\n"
        f"🛑 حد ضرر: {sl}\n"
        f"✅ TP1: {tp1}\n"
        f"✅ TP2: {tp2}\n\n"

        f"📊 RSI: {signal['rsi']:.1f}\n"
        f"📊 ADX: {signal['adx']:.1f}\n"
        f"📊 Volume: "
        f"{signal['volume_ratio']:.2f}x\n"
        f"🏗 ساختار: "
        f"{signal['structure']}\n\n"

        f"📝 دلایل سیگنال:\n"
    )

    for reason in signal["reasons"]:

        text += (
            f"• {reason}\n"
        )

    text += (
        "\n⚠️ سیگنال تضمین سود نیست. "
        "ریسک هر معامله را محدود نگه دار."
    )

    return text


# ============================================================
# PROCESS SYMBOL
# ============================================================

async def process_symbol(symbol):

    try:

        logger.info(
            f"{symbol}: fetching market data..."
        )

        df_5m = await fetch_data(
            symbol,
            TIMEFRAME_5M,
        )

        df_15m = await fetch_data(
            symbol,
            TIMEFRAME_15M,
        )

        df_1h = await fetch_data(
            symbol,
            TIMEFRAME_1H,
        )

        df_4h = await fetch_data(
            symbol,
            TIMEFRAME_4H,
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

            logger.warning(
                f"{symbol}: missing market data"
            )

            return

        trend_15m = get_trend(
            df_15m
        )

        trend_1h = get_trend(
            df_1h
        )

        trend_4h = get_trend(
            df_4h
        )

        logger.info(
            f"{symbol} | "
            f"4H={trend_4h} | "
            f"1H={trend_1h} | "
            f"15M={trend_15m}"
        )

        # ====================================================
        # ANALYSIS
        # ====================================================

        signal = analyze_5m(
            df_5m,
            trend_15m,
            trend_1h,
            trend_4h,
            symbol,
        )

        if signal is None:

            logger.info(
                f"{symbol}: NO TRADE"
            )

            return

        # ====================================================
        # SAME CANDLE PROTECTION
        # ====================================================

        candle_id = (
            f"{symbol}_"
            f"{signal['direction']}_"
            f"{signal['candle_time']}"
        )

        if candle_id in sent_signals:

            return

        # ====================================================
        # SEND SIGNAL
        # ====================================================

        message = format_signal(
            signal
        )

        sent = await send_ntfy(
            message,
            title=(
                f"{symbol.replace('/', '')} "
                f"{signal['direction']} "
                f"{signal['grade']}"
            ),
            priority="high",
            tags=(
                "chart_with_upwards_trend"
                if signal["direction"] == "LONG"
                else "chart_with_downwards_trend"
            ),
        )

        if sent:

            sent_signals.add(
                candle_id
            )

            logger.info(
                f"SIGNAL SENT | "
                f"{symbol} | "
                f"{signal['direction']} | "
                f"{signal['grade']} | "
                f"SCORE={signal['score']:.1f}"
            )

    except Exception as e:

        logger.exception(
            f"Processing error "
            f"{symbol}: {e}"
        )


# ============================================================
# MAIN LOOP
# ============================================================

async def main():

    logger.info(
        "========================================"
    )

    logger.info(
        "CRYPTO SCALPING BOT STARTING"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Symbols: "
        + ", ".join(SYMBOLS)
    )

    # ========================================================
    # NTFY TEST
    # ========================================================

    ntfy_ok = await ntfy_test()

    if not ntfy_ok:

        logger.error(
            "ntfy test failed."
        )

        return

    logger.info(
        "ntfy connected successfully."
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        try:

            start_time = datetime.now(
                timezone.utc
            )

            logger.info(
                "========================================"
            )

            logger.info(
                "Market scan started..."
            )

            for symbol in SYMBOLS:

                await process_symbol(
                    symbol
                )

            elapsed = (
                datetime.now(
                    timezone.utc
                )
                - start_time
            ).total_seconds()

            sleep_time = max(
                5,
                CHECK_INTERVAL
                - int(elapsed),
            )

            logger.info(
                f"Scan complete. "
                f"Elapsed={elapsed:.1f}s | "
                f"Next scan in {sleep_time}s."
            )

            await asyncio.sleep(
                sleep_time
            )

        except Exception as e:

            logger.exception(
                f"Main loop error: {e}"
            )

            await asyncio.sleep(10)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "Bot stopped manually."
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
