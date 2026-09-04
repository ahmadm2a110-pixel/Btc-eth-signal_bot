import asyncio
import logging
import os
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import httpx
import pandas as pd


# ============================================================
# SCALP HUNTER ENGINE
#
# H1 BIAS
#     ->
# M15 LIQUIDITY SWEEP
#     ->
# DISPLACEMENT
#     ->
# BREAK OF STRUCTURE
#     ->
# RETEST
#     ->
# MOMENTUM CONFIRMATION
#     ->
# SCORE
#     ->
# A+ / A / NO TRADE
#
# CLOSED CANDLES ONLY
# AUTO TRADING DISABLED
# NTFY ENABLED
# ============================================================


# ============================================================
# CONFIG
# ============================================================

EXCHANGE_ID = "binance"

MAIN_TIMEFRAME = "15m"
HIGHER_TIMEFRAME = "1h"

SCAN_INTERVAL = 30

CANDLE_LIMIT_M15 = 300
CANDLE_LIMIT_H1 = 250

MIN_SCORE = 78
A_PLUS_SCORE = 88

COOLDOWN_MINUTES = 45

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
VOLUME_PERIOD = 20

MAX_ENTRY_DISTANCE_ATR = 0.90
MIN_RR = 1.8

SWEEP_LOOKBACK = 12
STRUCTURE_LOOKBACK = 8

DISPLACEMENT_ATR = 1.15
DISPLACEMENT_BODY_RATIO = 0.62
DISPLACEMENT_VOLUME_MULTIPLIER = 1.25

RETEST_ATR_TOLERANCE = 0.35

MIN_ATR_PERCENT = 0.10
MAX_ATR_PERCENT = 4.50

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "LINK/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "DOT/USDT",
    "NEAR/USDT",
    "FIL/USDT",
    "CRV/USDT",
    "JASMY/USDT",
    "VET/USDT",
    "CELR/USDT",
]

NTFY_SERVER = "https://ntfy.sh"
NTFY_TOPIC = "btc_ah7K9xQ2_signal"

AUTO_TRADING = False


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("SCALP_HUNTER")


# ============================================================
# GLOBAL STATE
# ============================================================

last_signal_time = {}
last_signal_direction = {}
last_candle_seen = {}

active_setups = {}


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        0.0,
        index=df.index,
    )

    minus_dm = pd.Series(
        0.0,
        index=df.index,
    )

    plus_dm[
        (up_move > down_move)
        & (up_move > 0)
    ] = up_move[
        (up_move > down_move)
        & (up_move > 0)
    ]

    minus_dm[
        (down_move > up_move)
        & (down_move > 0)
    ] = down_move[
        (down_move > up_move)
        & (down_move > 0)
    ]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_value = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        ).mean()
        / atr_value
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        ).mean()
        / atr_value
    )

    denominator = (plus_di + minus_di).replace(0, float("nan"))

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator
    )

    return dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean().fillna(0)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataframe(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    df["ema20"] = ema(
        df["close"],
        EMA_FAST,
    )

    df["ema50"] = ema(
        df["close"],
        EMA_MID,
    )

    df["ema200"] = ema(
        df["close"],
        EMA_SLOW,
    )

    df["rsi"] = rsi(
        df["close"],
        RSI_PERIOD,
    )

    df["atr"] = atr(
        df,
        ATR_PERIOD,
    )

    df["adx"] = adx(
        df,
        ADX_PERIOD,
    )

    df["volume_sma"] = (
        df["volume"]
        .rolling(VOLUME_PERIOD)
        .mean()
    )

    df["body"] = (
        df["close"] - df["open"]
    ).abs()

    df["range"] = (
        df["high"] - df["low"]
    )

    df["body_ratio"] = (
        df["body"]
        / df["range"].replace(0, float("nan"))
    )

    df["atr_percent"] = (
        df["atr"]
        / df["close"]
        * 100
    )

    return df.dropna().reset_index(drop=True)


# ============================================================
# H1 MARKET BIAS
# ============================================================

def get_h1_bias(df):
    row = df.iloc[-2]

    if (
        row["ema20"] > row["ema50"]
        and row["ema50"] > row["ema200"]
        and row["close"] > row["ema20"]
    ):
        return "BULLISH"

    if (
        row["ema20"] < row["ema50"]
        and row["ema50"] < row["ema200"]
        and row["close"] < row["ema20"]
    ):
        return "BEARISH"

    if (
        row["ema20"] > row["ema50"]
        and row["close"] > row["ema50"]
    ):
        return "BULLISH_WEAK"

    if (
        row["ema20"] < row["ema50"]
        and row["close"] < row["ema50"]
    ):
        return "BEARISH_WEAK"

    return "NEUTRAL"


# ============================================================
# MARKET QUALITY
# ============================================================

def market_quality(df):
    row = df.iloc[-2]

    atr_pct = row["atr_percent"]

    if atr_pct < MIN_ATR_PERCENT:
        return False, "LOW_VOLATILITY"

    if atr_pct > MAX_ATR_PERCENT:
        return False, "EXTREME_VOLATILITY"

    if row["adx"] < 15:
        return False, "WEAK_TREND"

    return True, "OK"


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(df, direction):
    i = len(df) - 2

    if i < SWEEP_LOOKBACK + 2:
        return None

    current = df.iloc[i]

    previous = df.iloc[
        i - SWEEP_LOOKBACK:i
    ]

    previous_low = previous["low"].min()
    previous_high = previous["high"].max()

    if direction == "LONG":

        swept = current["low"] < previous_low
        reclaimed = current["close"] > previous_low

        if swept and reclaimed:
            return {
                "type": "BULLISH_SWEEP",
                "level": previous_low,
                "index": i,
            }

    if direction == "SHORT":

        swept = current["high"] > previous_high
        reclaimed = current["close"] < previous_high

        if swept and reclaimed:
            return {
                "type": "BEARISH_SWEEP",
                "level": previous_high,
                "index": i,
            }

    return None


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(df, direction):
    i = len(df) - 2

    row = df.iloc[i]

    if row["atr"] <= 0:
        return False

    body_ratio = row["body_ratio"]
    body_atr = row["body"] / row["atr"]

    volume_ok = (
        row["volume"]
        >= row["volume_sma"]
        * DISPLACEMENT_VOLUME_MULTIPLIER
    )

    strong_body = (
        body_ratio >= DISPLACEMENT_BODY_RATIO
    )

    strong_range = (
        body_atr >= DISPLACEMENT_ATR
    )

    if direction == "LONG":
        directional = row["close"] > row["open"]

    else:
        directional = row["close"] < row["open"]

    return (
        directional
        and strong_body
        and strong_range
        and volume_ok
    )


# ============================================================
# STRUCTURE BREAK
# ============================================================

def detect_structure_break(df, direction):
    i = len(df) - 2

    if i < STRUCTURE_LOOKBACK + 2:
        return None

    current = df.iloc[i]

    structure = df.iloc[
        i - STRUCTURE_LOOKBACK:i
    ]

    swing_high = structure["high"].max()
    swing_low = structure["low"].min()

    if direction == "LONG":

        if current["close"] > swing_high:
            return {
                "type": "BULLISH_BOS",
                "level": swing_high,
                "index": i,
            }

    if direction == "SHORT":

        if current["close"] < swing_low:
            return {
                "type": "BEARISH_BOS",
                "level": swing_low,
                "index": i,
            }

    return None


# ============================================================
# RETEST DETECTION
# ============================================================

def detect_retest(df, level, direction):
    i = len(df) - 2

    row = df.iloc[i]

    tolerance = row["atr"] * RETEST_ATR_TOLERANCE

    distance = abs(
        row["low"] - level
    ) if direction == "LONG" else abs(
        row["high"] - level
    )

    touched = distance <= tolerance

    if not touched:
        return False

    if direction == "LONG":
        rejection = row["close"] > level

    else:
        rejection = row["close"] < level

    return rejection


# ============================================================
# MOMENTUM CONFIRMATION
# ============================================================

def momentum_confirmation(df, direction):
    i = len(df) - 2

    row = df.iloc[i]
    previous = df.iloc[i - 1]

    if direction == "LONG":

        return (
            row["close"] > row["open"]
            and row["close"] > previous["high"]
            and row["rsi"] >= 52
            and row["adx"] >= 18
        )

    return (
        row["close"] < row["open"]
        and row["close"] < previous["low"]
        and row["rsi"] <= 48
        and row["adx"] >= 18
    )


# ============================================================
# EMA CHASE FILTER
# ============================================================

def chase_filter(df, direction):
    row = df.iloc[-2]

    if row["atr"] <= 0:
        return False

    distance = abs(
        row["close"] - row["ema20"]
    ) / row["atr"]

    if distance > MAX_ENTRY_DISTANCE_ATR:
        return False

    if direction == "LONG":
        return row["close"] >= row["ema20"]

    return row["close"] <= row["ema20"]


# ============================================================
# SCORE ENGINE
# ============================================================

def calculate_score(
    df,
    direction,
    bias,
    sweep,
    displacement,
    bos,
    retest,
    momentum,
):
    row = df.iloc[-2]

    score = 0
    confirmations = []

    # H1 bias
    if direction == "LONG":

        if bias == "BULLISH":
            score += 25
            confirmations.append(
                "H1 bullish alignment"
            )

        elif bias == "BULLISH_WEAK":
            score += 15
            confirmations.append(
                "H1 bullish weak alignment"
            )

    else:

        if bias == "BEARISH":
            score += 25
            confirmations.append(
                "H1 bearish alignment"
            )

        elif bias == "BEARISH_WEAK":
            score += 15
            confirmations.append(
                "H1 bearish weak alignment"
            )

    # Liquidity sweep
    if sweep:
        score += 15
        confirmations.append(
            "Liquidity sweep"
        )

    # Displacement
    if displacement:
        score += 15
        confirmations.append(
            "Displacement"
        )

    # BOS
    if bos:
        score += 15
        confirmations.append(
            "Market structure break"
        )

    # Retest
    if retest:
        score += 15
        confirmations.append(
            "BOS retest"
        )

    # Momentum
    if momentum:
        score += 10
        confirmations.append(
            "Momentum confirmation"
        )

    # RSI quality
    if direction == "LONG":

        if 52 <= row["rsi"] <= 68:
            score += 5
            confirmations.append(
                "Healthy bullish RSI"
            )

    else:

        if 32 <= row["rsi"] <= 48:
            score += 5
            confirmations.append(
                "Healthy bearish RSI"
            )

    return min(score, 100), confirmations


# ============================================================
# ENTRY / SL / TP
# ============================================================

def build_trade_levels(
    df,
    direction,
    structure_level,
    sweep_level,
):
    row = df.iloc[-2]

    entry = float(row["close"])
    atr_value = float(row["atr"])

    if direction == "LONG":

        structural_sl = min(
            structure_level,
            sweep_level,
        )

        sl = structural_sl - (
            atr_value * 0.15
        )

        risk = entry - sl

        if risk <= 0:
            return None

        tp1 = entry + risk * 1.8
        tp2 = entry + risk * 2.7
        tp3 = entry + risk * 4.0

    else:

        structural_sl = max(
            structure_level,
            sweep_level,
        )

        sl = structural_sl + (
            atr_value * 0.15
        )

        risk = sl - entry

        if risk <= 0:
            return None

        tp1 = entry - risk * 1.8
        tp2 = entry - risk * 2.7
        tp3 = entry - risk * 4.0

    rr = abs(tp2 - entry) / risk

    if rr < MIN_RR:
        return None

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk": risk,
        "rr": rr,
    }


# ============================================================
# SETUP ENGINE
# ============================================================

def analyze_symbol(symbol, m15, h1):
    if len(m15) < 100 or len(h1) < 100:
        return None

    bias = get_h1_bias(h1)

    if bias == "NEUTRAL":
        return None

    quality_ok, quality_reason = market_quality(m15)

    if not quality_ok:
        return None

    candidates = []

    if bias in ("BULLISH", "BULLISH_WEAK"):
        candidates.append("LONG")

    if bias in ("BEARISH", "BEARISH_WEAK"):
        candidates.append("SHORT")

    for direction in candidates:

        sweep = detect_liquidity_sweep(
            m15,
            direction,
        )

        if not sweep:
            continue

        displacement = detect_displacement(
            m15,
            direction,
        )

        if not displacement:
            continue

        bos = detect_structure_break(
            m15,
            direction,
        )

        if not bos:
            continue

        retest = detect_retest(
            m15,
            bos["level"],
            direction,
        )

        momentum = momentum_confirmation(
            m15,
            direction,
        )

        if not momentum:
            continue

        if not chase_filter(
            m15,
            direction,
        ):
            continue

        score, confirmations = calculate_score(
            m15,
            direction,
            bias,
            sweep,
            displacement,
            bos,
            retest,
            momentum,
        )

        if score < MIN_SCORE:
            continue

        levels = build_trade_levels(
            m15,
            direction,
            bos["level"],
            sweep["level"],
        )

        if not levels:
            continue

        grade = (
            "A+"
            if score >= A_PLUS_SCORE
            else "A"
        )

        candle = m15.iloc[-2]

        return {
            "symbol": symbol,
            "direction": direction,
            "grade": grade,
            "score": score,
            "bias": bias,
            "entry": levels["entry"],
            "sl": levels["sl"],
            "tp1": levels["tp1"],
            "tp2": levels["tp2"],
            "tp3": levels["tp3"],
            "rr": levels["rr"],
            "rsi": candle["rsi"],
            "adx": candle["adx"],
            "atr_percent": candle["atr_percent"],
            "timestamp": candle["timestamp"],
            "confirmations": confirmations,
        }

    return None


# ============================================================
# SIGNAL DEDUPLICATION
# ============================================================

def is_duplicate_signal(signal):
    symbol = signal["symbol"]
    direction = signal["direction"]

    now = datetime.now(timezone.utc)

    previous_time = last_signal_time.get(symbol)
    previous_direction = last_signal_direction.get(symbol)

    if previous_time:

        elapsed = (
            now - previous_time
        ).total_seconds() / 60

        if elapsed < COOLDOWN_MINUTES:
            return True

    if (
        previous_direction
        and previous_direction != direction
    ):
        if previous_time:

            elapsed = (
                now - previous_time
            ).total_seconds() / 60

            if elapsed < COOLDOWN_MINUTES:
                return True

    return False


def register_signal(signal):
    symbol = signal["symbol"]

    last_signal_time[symbol] = (
        datetime.now(timezone.utc)
    )

    last_signal_direction[symbol] = (
        signal["direction"]
    )


# ============================================================
# NTFY
# ============================================================

async def send_ntfy(message, priority="default"):
    url = (
        f"{NTFY_SERVER}/"
        f"{NTFY_TOPIC}"
    )

    headers = {
        "Title": "SCALP HUNTER SIGNAL",
        "Priority": priority,
        "Tags": "chart_with_upwards_trend",
    }

    try:

        async with httpx.AsyncClient(
            timeout=10
        ) as client:

            response = await client.post(
                url,
                content=message.encode("utf-8"),
                headers=headers,
            )

            if response.status_code >= 300:
                logger.error(
                    "NTFY error: %s",
                    response.text,
                )

    except Exception as exc:

        logger.error(
            "NTFY exception: %s",
            exc,
        )


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def format_signal(signal):
    direction = signal["direction"]

    emoji = (
        "🟢"
        if direction == "LONG"
        else "🔴"
    )

    confirmation_text = "\n".join(
        f"✓ {x}"
        for x in signal["confirmations"]
    )

    return f"""
{emoji} SCALP HUNTER — {signal["grade"]}

{signal["symbol"]} {direction}

Score: {signal["score"]}/100
H1 Bias: {signal["bias"]}

ENTRY
{signal["entry"]:.8f}

STOP
{signal["sl"]:.8f}

TP1
{signal["tp1"]:.8f}

TP2
{signal["tp2"]:.8f}

TP3
{signal["tp3"]:.8f}

RR: 1:{signal["rr"]:.2f}

RSI: {signal["rsi"]:.1f}
ADX: {signal["adx"]:.1f}
ATR: {signal["atr_percent"]:.2f}%

CONFIRMATIONS
{confirmation_text}

STATUS
WAIT FOR ENTRY CONFIRMATION

AUTO TRADING: DISABLED
""".strip()


# ============================================================
# SETUP INVALIDATION
# ============================================================

def check_active_setup(symbol, df):
    setup = active_setups.get(symbol)

    if not setup:
        return None

    row = df.iloc[-2]

    direction = setup["direction"]

    if direction == "LONG":

        if row["close"] <= setup["sl"]:
            return "STOP_INVALIDATED"

        if row["close"] >= setup["tp1"]:
            return "TP1_REACHED"

    else:

        if row["close"] >= setup["sl"]:
            return "STOP_INVALIDATED"

        if row["close"] <= setup["tp1"]:
            return "TP1_REACHED"

    return None


# ============================================================
# REGISTER ACTIVE SETUP
# ============================================================

def register_active_setup(signal):
    active_setups[
        signal["symbol"]
    ] = {
        "direction": signal["direction"],
        "entry": signal["entry"],
        "sl": signal["sl"],
        "tp1": signal["tp1"],
        "tp2": signal["tp2"],
        "tp3": signal["tp3"],
        "created": datetime.now(timezone.utc),
    }


# ============================================================
# EXCHANGE
# ============================================================

def create_exchange():
    exchange_class = getattr(
        ccxt,
        EXCHANGE_ID,
    )

    return exchange_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        },
    })


async def fetch_dataframe(
    exchange,
    symbol,
    timeframe,
    limit,
):
    try:

        rows = await exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if not rows:
            return None

        return prepare_dataframe(rows)

    except Exception as exc:

        logger.warning(
            "%s %s data error: %s",
            symbol,
            timeframe,
            exc,
        )

        return None


# ============================================================
# PROCESS SYMBOL
# ============================================================

async def process_symbol(exchange, symbol):
    try:

        m15 = await fetch_dataframe(
            exchange,
            symbol,
            MAIN_TIMEFRAME,
            CANDLE_LIMIT_M15,
        )

        h1 = await fetch_dataframe(
            exchange,
            symbol,
            HIGHER_TIMEFRAME,
            CANDLE_LIMIT_H1,
        )

        if m15 is None or h1 is None:
            return

        if len(m15) < 100 or len(h1) < 100:
            return

        closed_candle = m15.iloc[-2]

        candle_time = closed_candle[
            "timestamp"
        ]

        previous_candle = last_candle_seen.get(
            symbol
        )

        if previous_candle is not None:

            if candle_time <= previous_candle:
                return

        last_candle_seen[symbol] = candle_time

        invalidation = check_active_setup(
            symbol,
            m15,
        )

        if invalidation:

            logger.info(
                "%s active setup: %s",
                symbol,
                invalidation,
            )

            if invalidation == "STOP_INVALIDATED":

                await send_ntfy(
                    f"""
⚠️ SETUP INVALIDATED

{symbol}

Previous setup has been invalidated.

No new entry.

AUTO TRADING: DISABLED
""".strip(),
                    priority="high",
                )

                active_setups.pop(
                    symbol,
                    None,
                )

            elif invalidation == "TP1_REACHED":

                await send_ntfy(
                    f"""
✅ TP1 REACHED

{symbol}

TP1 has been reached.

Manage remaining position according to your plan.

AUTO TRADING: DISABLED
""".strip(),
                )

                active_setups.pop(
                    symbol,
                    None,
                )

            return

        signal = analyze_symbol(
            symbol,
            m15,
            h1,
        )

        if not signal:
            return

        if is_duplicate_signal(signal):
            logger.info(
                "%s signal blocked by cooldown",
                symbol,
            )
            return

        register_signal(signal)
        register_active_setup(signal)

        message = format_signal(signal)

        logger.info(
            "\n%s",
            message,
        )

        await send_ntfy(
            message,
            priority=(
                "high"
                if signal["grade"] == "A+"
                else "default"
            ),
        )

    except Exception as exc:

        logger.exception(
            "%s processing error: %s",
            symbol,
            exc,
        )


# ============================================================
# MAIN SCANNER
# ============================================================

async def scanner():
    exchange = create_exchange()

    try:

        await exchange.load_markets()

        logger.info(
            "SCALP HUNTER ENGINE STARTED"
        )

        logger.info(
            "Main timeframe: %s",
            MAIN_TIMEFRAME,
        )

        logger.info(
            "Higher timeframe: %s",
            HIGHER_TIMEFRAME,
        )

        logger.info(
            "Symbols: %d",
            len(SYMBOLS),
        )

        logger.info(
            "Auto trading: %s",
            "ENABLED"
            if AUTO_TRADING
            else "DISABLED",
        )

        logger.info(
            "NTFY: ENABLED"
        )

        while True:

            logger.info(
                "Scanning market..."
            )

            tasks = []

            for symbol in SYMBOLS:

                if symbol not in exchange.markets:
                    continue

                tasks.append(
                    process_symbol(
                        exchange,
                        symbol,
                    )
                )

            if tasks:

                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

            await asyncio.sleep(
                SCAN_INTERVAL
            )

    except asyncio.CancelledError:

        logger.info(
            "Scanner cancelled."
        )

    except Exception as exc:

        logger.exception(
            "Fatal scanner error: %s",
            exc,
        )

    finally:

        await exchange.close()

        logger.info(
            "Exchange connection closed."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            scanner()
        )

    except KeyboardInterrupt:

        logger.info(
            "Bot stopped by user."
        )
