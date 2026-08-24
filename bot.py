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


# ============================================================
# TIMEFRAMES
# ============================================================

# IMPORTANT:
# 15M is now the MAIN SIGNAL TIMEFRAME.
# 1H and 4H are used only for higher-timeframe trend.

TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"
TIMEFRAME_4H = "4h"


# ============================================================
# MARKET DATA
# ============================================================

LIMIT = 250


# ============================================================
# SIGNAL SETTINGS
# ============================================================

MIN_SCORE = 6

A_PLUS_SCORE = 8

# Bot scans frequently, but signal can only change
# when a new 15M candle closes.
CHECK_INTERVAL = 30


# ============================================================
# RISK SETTINGS
# ============================================================

ATR_SL_MULTIPLIER = 1.5

TP1_RR = 1.5
TP2_RR = 2.5

# Maximum distance from EMA20 in ATR units.
# Higher = more signals.
MAX_ATR_DISTANCE_FROM_EMA = 2.8


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
# NTFY SEND
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

    # --------------------------------------------------------
    # Retry 3 times
    # --------------------------------------------------------

    for attempt in range(1, 4):

        try:

            logger.info(
                f"ntfy sending attempt "
                f"{attempt}/3 -> {url}"
            )

            timeout = httpx.Timeout(
                connect=10.0,
                read=20.0,
                write=20.0,
                pool=10.0,
            )

            async with httpx.AsyncClient(
                timeout=timeout
            ) as client:

                response = await client.post(
                    url,
                    content=message.encode("utf-8"),
                    headers=headers,
                )

                logger.info(
                    f"ntfy HTTP status: "
                    f"{response.status_code}"
                )

                response.raise_for_status()

            logger.info(
                "ntfy notification sent successfully."
            )

            return True

        except httpx.ConnectTimeout as e:

            logger.error(
                f"ntfy CONNECT TIMEOUT "
                f"(attempt {attempt}/3): "
                f"{repr(e)}"
            )

        except httpx.ReadTimeout as e:

            logger.error(
                f"ntfy READ TIMEOUT "
                f"(attempt {attempt}/3): "
                f"{repr(e)}"
            )

        except httpx.ConnectError as e:

            logger.error(
                f"ntfy CONNECT ERROR "
                f"(attempt {attempt}/3): "
                f"{repr(e)}"
            )

        except httpx.HTTPStatusError as e:

            status = e.response.status_code
            body = e.response.text[:500]

            logger.error(
                f"ntfy HTTP ERROR "
                f"(attempt {attempt}/3) | "
                f"status={status} | "
                f"body={body}"
            )

        except httpx.RequestError as e:

            logger.error(
                f"ntfy REQUEST ERROR "
                f"(attempt {attempt}/3): "
                f"{repr(e)}"
            )

        except Exception as e:

            logger.exception(
                f"ntfy UNKNOWN ERROR "
                f"(attempt {attempt}/3): "
                f"{repr(e)}"
            )

        if attempt < 3:

            await asyncio.sleep(2)

    logger.error(
        "ntfy failed after 3 attempts."
    )

    return False


# ============================================================
# NTFY TEST
# ============================================================

async def ntfy_test():

    test_message = (
        "✅ ربات کریپتو با موفقیت فعال شد.\n\n"

        "📊 ارزها:\n"
        "BTC / ETH / SOL / BNB\n"
        "XRP / LINK / ADA / AVAX\n\n"

        "⏱ سیستم اصلی:\n"
        "15M Signal\n"
        "1H Trend\n"
        "4H Trend\n\n"

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
            f"{symbol} {timeframe}: "
            f"{repr(e)}"
        )

        return None


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
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

    # RSI
    df["rsi"] = ta.rsi(
        df["close"],
        length=14,
    )

    # ATR
    df["atr"] = ta.atr(
        df["high"],
        df["low"],
        df["close"],
        length=14,
    )

    # Volume
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

    # Last CLOSED 15M candle
    last = df.iloc[-2]

    # Previous closed candles
    previous = df.iloc[-12:-2]

    close = float(last["close"])

    recent_high = float(
        previous["high"].max()
    )

    recent_low = float(
        previous["low"].min()
    )

    # Breakout
    if close > recent_high:

        return "BULL_BREAKOUT"

    if close < recent_low:

        return "BEAR_BREAKDOWN"

    highs = previous["high"].tail(5).values
    lows = previous["low"].tail(5).values

    if len(highs) >= 5:

        bullish = (
            highs[-1] > highs[-3]
            and lows[-1] > lows[-3]
        )

        bearish = (
            highs[-1] < highs[-3]
            and lows[-1] < lows[-3]
        )

        if bullish:

            return "BULLISH"

        if bearish:

            return "BEARISH"

    return "NEUTRAL"


# ============================================================
# 15M ANALYSIS
# ============================================================

def analyze_15m(
    df,
    trend_15m,
    trend_1h,
    trend_4h,
    symbol,
):

    if df is None:
        return None

    if len(df) < 205:
        return None

    df = add_indicators(df)

    # ========================================================
    # IMPORTANT:
    # ALL SIGNAL CALCULATIONS USE CLOSED 15M CANDLE
    # ========================================================

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

    previous_close = float(
        previous["close"]
    )

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

    if atr <= 0:
        return None

    if volume_sma <= 0:
        return None

    volume_ratio = (
        volume / volume_sma
    )

    structure = get_market_structure(df)

    long_score = 0.0
    short_score = 0.0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # 4H TREND
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
    # 1H TREND
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
    # 15M TREND
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
    # 15M PRICE VS EMA20
    # ========================================================

    if close > ema20:

        long_score += 1

        long_reasons.append(
            "قیمت بالای EMA20 در 15M"
        )

    elif close < ema20:

        short_score += 1

        short_reasons.append(
            "قیمت پایین EMA20 در 15M"
        )

    # ========================================================
    # 15M EMA20 / EMA50
    # ========================================================

    if ema20 > ema50:

        long_score += 1

        long_reasons.append(
            "EMA20 بالاتر از EMA50 در 15M"
        )

    elif ema20 < ema50:

        short_score += 1

        short_reasons.append(
            "EMA20 پایین‌تر از EMA50 در 15M"
        )

    # ========================================================
    # ADX 15M
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
    # VOLUME 15M
    # ========================================================

    if volume_ratio >= 1.10:

        if long_score > short_score:

            long_score += 1

            long_reasons.append(
                f"حجم تأییدکننده {volume_ratio:.2f}x"
            )

        elif short_score > long_score:

            short_score += 1

            short_reasons.append(
                f"حجم تأییدکننده {volume_ratio:.2f}x"
            )

    # ========================================================
    # RSI 15M
    # ========================================================

    if long_score > short_score:

        if 47 <= rsi <= 68:

            long_score += 1

            long_reasons.append(
                f"RSI مناسب {rsi:.1f}"
            )

        elif 40 <= rsi < 47:

            long_score += 0.5

            long_reasons.append(
                f"RSI پولبک {rsi:.1f}"
            )

        elif 68 < rsi <= 72:

            long_score += 0.5

            long_reasons.append(
                f"RSI قوی {rsi:.1f}"
            )

    elif short_score > long_score:

        if 32 <= rsi <= 53:

            short_score += 1

            short_reasons.append(
                f"RSI مناسب {rsi:.1f}"
            )

        elif 53 < rsi <= 60:

            short_score += 0.5

            short_reasons.append(
                f"RSI پولبک {rsi:.1f}"
            )

        elif 28 <= rsi < 32:

            short_score += 0.5

            short_reasons.append(
                f"RSI قوی {rsi:.1f}"
            )

    # ========================================================
    # 15M MARKET STRUCTURE
    # ========================================================

    if structure == "BULL_BREAKOUT":

        long_score += 1.5

        long_reasons.append(
            "شکست صعودی ساختار 15M"
        )

    elif structure == "BULLISH":

        long_score += 1

        long_reasons.append(
            "ساختار بازار 15M صعودی"
        )

    elif structure == "BEAR_BREAKDOWN":

        short_score += 1.5

        short_reasons.append(
            "شکست نزولی ساختار 15M"
        )

    elif structure == "BEARISH":

        short_score += 1

        short_reasons.append(
            "ساختار بازار 15M نزولی"
        )

    # ========================================================
    # 15M CANDLE MOMENTUM
    # ========================================================

    candle_range = high - low

    if candle_range > 0:

        body = abs(
            close - open_price
        )

        body_ratio = (
            body / candle_range
        )

        # Strong bullish candle
        if (
            close > open_price
            and close > previous_close
            and body_ratio >= 0.40
        ):

            long_score += 1

            long_reasons.append(
                "مومنتوم کندل 15M صعودی"
            )

        # Strong bearish candle
        elif (
            close < open_price
            and close < previous_close
            and body_ratio >= 0.40
        ):

            short_score += 1

            short_reasons.append(
                "مومنتوم کندل 15M نزولی"
            )

        # Moderate bullish candle
        elif (
            close > open_price
            and close > previous_close
        ):

            long_score += 0.5

            long_reasons.append(
                "حرکت صعودی کندل 15M"
            )

        # Moderate bearish candle
        elif (
            close < open_price
            and close < previous_close
        ):

            short_score += 0.5

            short_reasons.append(
                "حرکت نزولی کندل 15M"
            )

    # ========================================================
    # HIGHER TIMEFRAME CONFLICT
    # ========================================================

    if (
        trend_1h in ["LONG", "SHORT"]
        and trend_4h in ["LONG", "SHORT"]
        and trend_1h != trend_4h
    ):

        if long_score > short_score:

            long_score -= 1

            long_reasons.append(
                "⚠️ اختلاف روند 1H و 4H"
            )

        elif short_score > long_score:

            short_score -= 1

            short_reasons.append(
                "⚠️ اختلاف روند 1H و 4H"
            )

    # ========================================================
    # SELECT SIGNAL
    # ========================================================

    if (
        long_score >= MIN_SCORE
        and long_score > short_score
    ):

        direction = "LONG"

        score = long_score

        reasons = long_reasons

    elif (
        short_score >= MIN_SCORE
        and short_score > long_score
    ):

        direction = "SHORT"

        score = short_score

        reasons = short_reasons

    else:

        return None

    # ========================================================
    # DON'T CHASE PRICE
    # ========================================================

    distance_from_ema = (
        abs(close - ema20)
        / atr
    )

    if (
        distance_from_ema
        > MAX_ATR_DISTANCE_FROM_EMA
    ):

        logger.info(
            f"{symbol}: price extended "
            f"{distance_from_ema:.2f} ATR "
            f"from 15M EMA20"
        )

        return None

    # ========================================================
    # EXTREME RSI PROTECTION
    # ========================================================

    if (
        direction == "LONG"
        and rsi > 75
    ):

        logger.info(
            f"{symbol}: LONG rejected "
            f"RSI={rsi:.1f}"
        )

        return None

    if (
        direction == "SHORT"
        and rsi < 25
    ):

        logger.info(
            f"{symbol}: SHORT rejected "
            f"RSI={rsi:.1f}"
        )

        return None

    # ========================================================
    # GRADE
    # ========================================================

    if score >= A_PLUS_SCORE:

        grade = "A+"

    else:

        grade = "A"

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
        "grade": grade,
        "score": score,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rsi": rsi,
        "adx": adx,
        "volume_ratio": volume_ratio,
        "structure": structure,
        "distance_from_ema": distance_from_ema,
        "trend_15m": trend_15m,
        "trend_1h": trend_1h,
        "trend_4h": trend_4h,
        "reasons": reasons,
        "candle_time": last["timestamp"],
    }


# ============================================================
# DECIMAL PRECISION
# ============================================================

def get_decimals(symbol):

    if symbol.startswith("BTC"):

        return 2

    if symbol.startswith(
        ("ETH", "BNB", "SOL", "AVAX")
    ):

        return 3

    return 4


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

    decimals = get_decimals(symbol)

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
        f"{emoji} {signal['grade']} "
        f"سیگنال اسکلپ ۱۵ دقیقه‌ای {emoji}\n\n"

        f"📌 {symbol.replace('/', '')}\n"
        f"📈 جهت: {direction_fa}\n"
        f"⭐ امتیاز: {signal['score']:.1f}\n\n"

        f"⏰ کندل بسته‌شده: "
        f"{candle_time} UTC\n\n"

        f"🎯 ورود: {entry}\n"
        f"🛑 حد ضرر: {sl}\n"
        f"✅ TP1: {tp1}\n"
        f"✅ TP2: {tp2}\n\n"

        f"🧭 4H: {signal['trend_4h']}\n"
        f"🧭 1H: {signal['trend_1h']}\n"
        f"🧭 15M: {signal['trend_15m']}\n\n"

        f"📊 RSI 15M: "
        f"{signal['rsi']:.1f}\n"

        f"📊 ADX 15M: "
        f"{signal['adx']:.1f}\n"

        f"📊 Volume 15M: "
        f"{signal['volume_ratio']:.2f}x\n"

        f"🏗 ساختار 15M: "
        f"{signal['structure']}\n"

        f"📏 فاصله EMA20: "
        f"{signal['distance_from_ema']:.2f} ATR\n\n"

        f"📝 دلایل:\n"
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

        # ====================================================
        # ONLY 15M + 1H + 4H
        # ====================================================

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
                df_15m,
                df_1h,
                df_4h,
            ]
        ):

            logger.warning(
                f"{symbol}: missing market data"
            )

            return

        # ====================================================
        # HIGHER TIMEFRAME TRENDS
        # ====================================================

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
        # ANALYZE CLOSED 15M CANDLE
        # ====================================================

        signal = analyze_15m(
            df_15m,
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

        logger.info(
            f"{symbol}: SIGNAL CANDIDATE | "
            f"{signal['direction']} | "
            f"score={signal['score']:.1f} | "
            f"grade={signal['grade']}"
        )

        # ====================================================
        # SAME 15M CANDLE PROTECTION
        # ====================================================

        candle_id = (
            f"{symbol}_"
            f"{signal['direction']}_"
            f"{signal['candle_time']}"
        )

        if candle_id in sent_signals:

            logger.info(
                f"{symbol}: "
                f"signal already sent for "
                f"this 15M candle"
            )

            return

        # ====================================================
        # FORMAT
        # ====================================================

        message = format_signal(
            signal
        )

        # ====================================================
        # SEND NTFY
        # ====================================================

        sent = await send_ntfy(
            message,
            title=(
                f"{symbol.replace('/', '')} "
                f"{signal['direction']} "
                f"{signal['grade']} "
                f"15M"
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
                f"SCORE={signal['score']:.1f} | "
                f"15M={signal['candle_time']}"
            )

    except Exception as e:

        logger.exception(
            f"Processing error "
            f"{symbol}: {repr(e)}"
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "========================================"
    )

    logger.info(
        "CRYPTO 15M SCALPING BOT STARTING"
    )

    logger.info(
        "========================================"
    )

    logger.info(
        "Symbols: "
        + ", ".join(SYMBOLS)
    )

    logger.info(
        f"Main signal timeframe: "
        f"{TIMEFRAME_15M}"
    )

    logger.info(
        "Higher timeframes: 1H / 4H"
    )

    logger.info(
        f"Minimum score: {MIN_SCORE}"
    )

    logger.info(
        f"A+ score: {A_PLUS_SCORE}+"
    )

    logger.info(
        f"Check interval: "
        f"{CHECK_INTERVAL}s"
    )

    # ========================================================
    # NTFY TEST
    # ========================================================

    ntfy_ok = await ntfy_test()

    if not ntfy_ok:

        logger.error(
            "ntfy test failed. "
            "Bot will stop before market scanning."
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
                "15M MARKET SCAN STARTED..."
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
                f"Scan complete | "
                f"Elapsed={elapsed:.1f}s | "
                f"Next scan in {sleep_time}s"
            )

            await asyncio.sleep(
                sleep_time
            )

        except Exception as e:

            logger.exception(
                f"Main loop error: {repr(e)}"
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
            f"Fatal error: {repr(e)}"
        )

    finally:

        try:

            asyncio.run(
                exchange.close()
            )

        except Exception:

            pass
