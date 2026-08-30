import asyncio
import logging
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import httpx
import pandas as pd


# ============================================================
# AH1 TRADING ASSISTANT
# H4 CONTEXT -> H1 STRUCTURE -> SETUP -> SCORE -> ALERT
# LONG + SHORT
# NO AUTO TRADING
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "LINK/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "VET/USDT",
    "CELR/USDT",
    "DYDX/USDT",
    "JASMY/USDT",
    "CRV/USDT",
    "DOT/USDT",
    "FIL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "RUNE/USDT",
    "NEAR/USDT",
]

MAIN_TIMEFRAME = "1h"
HIGHER_TIMEFRAME = "4h"

H1_LIMIT = 250
H4_LIMIT = 250

SCAN_INTERVAL = 60

MIN_SCORE = 6
A_SCORE = 7
A_PLUS_SCORE = 8

ATR_PERIOD = 14
RSI_PERIOD = 14
ADX_PERIOD = 14

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

VOLUME_PERIOD = 20

MAX_ENTRY_DISTANCE_ATR = 1.5
RETEST_TOLERANCE_ATR = 0.35

SETUP_EXPIRY_HOURS = 6

NTFY_TOPIC = "btc_ah7K9xQ2_signal"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("AH1_ASSISTANT")


# ============================================================
# STATE
# ============================================================

setup_state = {}
last_alert = {}


# ============================================================
# EXCHANGE
# ============================================================

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "spot",
    },
})


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_price(value):
    value = safe_float(value)

    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 100:
        return f"{value:,.3f}"
    if value >= 1:
        return f"{value:,.4f}"
    if value >= 0.01:
        return f"{value:,.5f}"

    return f"{value:,.8f}"


def now_utc():
    return datetime.now(timezone.utc)


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        float("nan"),
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi.fillna(50)


def calculate_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_adx(df, period=14):
    high = df["high"]
    low = df["low"]

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

    plus_mask = (
        (up_move > down_move)
        & (up_move > 0)
    )

    minus_mask = (
        (down_move > up_move)
        & (down_move > 0)
    )

    plus_dm.loc[plus_mask] = up_move.loc[plus_mask]
    minus_dm.loc[minus_mask] = down_move.loc[minus_mask]

    previous_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr.replace(0, float("nan"))
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr.replace(0, float("nan"))
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            float("nan"),
        )
    )

    adx = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return (
        adx.fillna(0),
        plus_di.fillna(0),
        minus_di.fillna(0),
    )


def add_indicators(df):
    df = df.copy()

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

    df["rsi"] = calculate_rsi(
        df["close"],
        RSI_PERIOD,
    )

    df["atr"] = calculate_atr(
        df,
        ATR_PERIOD,
    )

    adx, plus_di, minus_di = calculate_adx(
        df,
        ADX_PERIOD,
    )

    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

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

    df["upper_wick"] = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    return df


# ============================================================
# DATA
# ============================================================

async def fetch_ohlcv(symbol, timeframe, limit):
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

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna().reset_index(drop=True)

        return df

    except Exception as exc:
        logger.error(
            "Data error %s %s: %s",
            symbol,
            timeframe,
            exc,
        )

        return None


def remove_open_candle(df):
    if df is None or len(df) < 3:
        return df

    return df.iloc[:-1].copy().reset_index(drop=True)


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_structure(df):
    if len(df) < 30:
        return "NEUTRAL"

    recent = df.iloc[-20:]

    previous_high = recent["high"].iloc[:-3].max()
    previous_low = recent["low"].iloc[:-3].min()

    last_close = safe_float(
        recent["close"].iloc[-1]
    )

    if last_close > previous_high:
        return "BULL_BREAKOUT"

    if last_close < previous_low:
        return "BEAR_BREAKDOWN"

    ema20 = safe_float(
        recent["ema20"].iloc[-1]
    )

    ema50 = safe_float(
        recent["ema50"].iloc[-1]
    )

    if ema20 > ema50:
        return "BULLISH"

    if ema20 < ema50:
        return "BEARISH"

    return "NEUTRAL"


def detect_support_resistance(df):
    lookback = min(
        50,
        len(df) - 1,
    )

    recent = df.iloc[-lookback:]

    resistance = recent["high"].iloc[:-2].max()
    support = recent["low"].iloc[:-2].min()

    return (
        safe_float(support),
        safe_float(resistance),
    )


def detect_trendline_break(df, direction):
    if len(df) < 40:
        return False

    recent = df.iloc[-20:]

    closes = recent["close"].values

    slope = (
        closes[-1] - closes[0]
    ) / max(
        1,
        len(closes) - 1,
    )

    if direction == "LONG":
        return slope > 0

    return slope < 0


# ============================================================
# CANDLE ANALYSIS
# ============================================================

def candle_confirmation(row, direction):
    open_price = safe_float(row["open"])
    close = safe_float(row["close"])
    high = safe_float(row["high"])
    low = safe_float(row["low"])

    body = abs(
        close - open_price
    )

    total_range = max(
        high - low,
        1e-12,
    )

    body_ratio = (
        body / total_range
    )

    if direction == "LONG":
        bullish = close > open_price

        lower_wick = (
            min(open_price, close)
            - low
        )

        pinbar = (
            lower_wick > body * 1.5
            and lower_wick / total_range > 0.35
        )

        momentum = (
            bullish
            and body_ratio >= 0.55
        )

        return pinbar or momentum

    bearish = close < open_price

    upper_wick = (
        high
        - max(open_price, close)
    )

    pinbar = (
        upper_wick > body * 1.5
        and upper_wick / total_range > 0.35
    )

    momentum = (
        bearish
        and body_ratio >= 0.55
    )

    return pinbar or momentum


# ============================================================
# H4 CONTEXT
# ============================================================

def analyze_h4(df):
    row = df.iloc[-1]

    close = safe_float(row["close"])
    ema20 = safe_float(row["ema20"])
    ema50 = safe_float(row["ema50"])
    ema200 = safe_float(row["ema200"])

    if (
        close > ema20
        and ema20 > ema50
        and ema50 > ema200
    ):
        return "BULLISH"

    if (
        close < ema20
        and ema20 < ema50
        and ema50 < ema200
    ):
        return "BEARISH"

    if close > ema50:
        return "BULLISH_WEAK"

    if close < ema50:
        return "BEARISH_WEAK"

    return "NEUTRAL"


# ============================================================
# DIRECTION ANALYSIS
# ============================================================

def analyze_direction(h1, h4, direction):
    row = h1.iloc[-1]

    price = safe_float(row["close"])

    ema20 = safe_float(row["ema20"])
    ema50 = safe_float(row["ema50"])
    ema200 = safe_float(row["ema200"])

    rsi = safe_float(row["rsi"])
    atr = safe_float(row["atr"])
    adx = safe_float(row["adx"])

    volume = safe_float(row["volume"])
    volume_sma = safe_float(row["volume_sma"])

    support, resistance = detect_support_resistance(h1)

    structure = detect_structure(h1)

    score = 0

    reasons = []
    warnings = []

    # --------------------------------------------------------
    # H4 CONTEXT
    # --------------------------------------------------------

    if direction == "LONG":

        if h4 == "BULLISH":
            score += 2
            reasons.append(
                "H4 trend aligned"
            )

        elif h4 == "BULLISH_WEAK":
            score += 1
            reasons.append(
                "H4 trend mildly bullish"
            )

        else:
            warnings.append(
                "H4 trend not aligned"
            )

    else:

        if h4 == "BEARISH":
            score += 2
            reasons.append(
                "H4 trend aligned"
            )

        elif h4 == "BEARISH_WEAK":
            score += 1
            reasons.append(
                "H4 trend mildly bearish"
            )

        else:
            warnings.append(
                "H4 trend not aligned"
            )

    # --------------------------------------------------------
    # EMA STRUCTURE
    # --------------------------------------------------------

    if direction == "LONG":

        if ema20 > ema50:
            score += 1
            reasons.append(
                "H1 EMA20 above EMA50"
            )

        if ema50 > ema200:
            score += 1
            reasons.append(
                "H1 EMA50 above EMA200"
            )

    else:

        if ema20 < ema50:
            score += 1
            reasons.append(
                "H1 EMA20 below EMA50"
            )

        if ema50 < ema200:
            score += 1
            reasons.append(
                "H1 EMA50 below EMA200"
            )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

    if direction == "LONG":

        if structure == "BULL_BREAKOUT":
            score += 2
            reasons.append(
                "Bullish structure breakout"
            )

        elif structure == "BULLISH":
            score += 1
            reasons.append(
                "Bullish market structure"
            )

        elif structure == "BEAR_BREAKDOWN":
            score -= 2
            warnings.append(
                "Bearish breakdown"
            )

    else:

        if structure == "BEAR_BREAKDOWN":
            score += 2
            reasons.append(
                "Bearish structure breakdown"
            )

        elif structure == "BEARISH":
            score += 1
            reasons.append(
                "Bearish market structure"
            )

        elif structure == "BULL_BREAKOUT":
            score -= 2
            warnings.append(
                "Bullish breakout"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if direction == "LONG":

        if 52 <= rsi <= 68:
            score += 1
            reasons.append(
                "Healthy bullish RSI"
            )

        elif rsi > 75:
            warnings.append(
                "RSI overextended"
            )

        elif rsi < 45:
            score -= 1

    else:

        if 32 <= rsi <= 48:
            score += 1
            reasons.append(
                "Healthy bearish RSI"
            )

        elif rsi < 25:
            warnings.append(
                "RSI overextended"
            )

        elif rsi > 55:
            score -= 1

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if adx >= 25:
        score += 1
        reasons.append(
            "ADX trend strength confirmed"
        )

    elif adx < 18:
        warnings.append(
            "Weak trend strength"
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if volume_sma > 0:

        volume_ratio = (
            volume / volume_sma
        )

        if volume_ratio >= 1.20:
            score += 1
            reasons.append(
                "Volume confirmation"
            )

        elif volume_ratio < 0.75:
            warnings.append(
                "Low volume"
            )

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    if candle_confirmation(
        row,
        direction,
    ):
        score += 1
        reasons.append(
            "Candle confirmation"
        )

    # --------------------------------------------------------
    # PRICE ACTION
    # --------------------------------------------------------

    if detect_trendline_break(
        h1,
        direction,
    ):
        score += 1
        reasons.append(
            "Price action supports direction"
        )

    # --------------------------------------------------------
    # ENTRY DISTANCE
    # --------------------------------------------------------

    distance_atr = 0.0

    if atr > 0:
        distance_atr = (
            abs(price - ema20)
            / atr
        )

    if distance_atr <= MAX_ENTRY_DISTANCE_ATR:
        score += 1
        reasons.append(
            "Entry distance acceptable"
        )

    else:
        warnings.append(
            f"Price extended "
            f"{distance_atr:.2f} ATR "
            f"from EMA20"
        )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    if direction == "LONG":

        if resistance > 0:

            if price > resistance:
                reasons.append(
                    "Above recent resistance"
                )

            elif (
                abs(price - resistance)
                <= atr * RETEST_TOLERANCE_ATR
            ):
                reasons.append(
                    "Near resistance retest"
                )

    else:

        if support > 0:

            if price < support:
                reasons.append(
                    "Below recent support"
                )

            elif (
                abs(price - support)
                <= atr * RETEST_TOLERANCE_ATR
            ):
                reasons.append(
                    "Near support retest"
                )

    # --------------------------------------------------------
    # GRADE
    # --------------------------------------------------------

    if score >= A_PLUS_SCORE:
        grade = "A+"

    elif score >= A_SCORE:
        grade = "A"

    elif score >= MIN_SCORE:
        grade = "WATCH"

    else:
        grade = "NO_TRADE"

    # --------------------------------------------------------
    # HARD INVALIDATION
    # --------------------------------------------------------

    if direction == "LONG":

        if (
            h4 == "BEARISH"
            and structure == "BEAR_BREAKDOWN"
        ):
            grade = "NO_TRADE"

    else:

        if (
            h4 == "BULLISH"
            and structure == "BULL_BREAKOUT"
        ):
            grade = "NO_TRADE"

    return {
        "direction": direction,
        "score": score,
        "grade": grade,
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi,
        "atr": atr,
        "adx": adx,
        "support": support,
        "resistance": resistance,
        "structure": structure,
        "distance_atr": distance_atr,
        "reasons": reasons,
        "warnings": warnings,
    }


# ============================================================
# TRADE MAP
# ============================================================

def build_trade_map(result):
    price = result["price"]
    atr = result["atr"]

    if atr <= 0:
        atr = price * 0.01

    direction = result["direction"]

    if direction == "LONG":

        entry_low = (
            price - atr * 0.25
        )

        entry_high = (
            price + atr * 0.10
        )

        invalidation = (
            price - atr * 1.25
        )

        tp1 = price + atr * 1.50
        tp2 = price + atr * 2.50
        tp3 = price + atr * 4.00

    else:

        entry_low = (
            price - atr * 0.10
        )

        entry_high = (
            price + atr * 0.25
        )

        invalidation = (
            price + atr * 1.25
        )

        tp1 = price - atr * 1.50
        tp2 = price - atr * 2.50
        tp3 = price - atr * 4.00

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "invalidation": invalidation,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
    }


# ============================================================
# NTFY
# ============================================================

async def send_ntfy(
    title,
    message,
    priority="default",
    tags=None,
):
    headers = {
        "Title": title,
        "Priority": priority,
    }

    if tags:
        headers["Tags"] = tags

    try:

        async with httpx.AsyncClient(
            timeout=15
        ) as client:

            response = await client.post(
                NTFY_URL,
                content=message,
                headers=headers,
            )

            response.raise_for_status()

            logger.info(
                "NTFY sent: %s",
                title,
            )

    except Exception as exc:

        logger.error(
            "NTFY error: %s",
            exc,
        )


# ============================================================
# ALERT FORMAT
# ============================================================

def format_setup(
    symbol,
    h4_context,
    result,
    trade_map,
):
    direction = result["direction"]

    icon = (
        "🟢"
        if direction == "LONG"
        else "🔴"
    )

    reasons = "\n".join(
        f"• {item}"
        for item in result["reasons"][:8]
    )

    warnings = "\n".join(
        f"• {item}"
        for item in result["warnings"][:5]
    )

    if not warnings:
        warnings = "• None"

    return (
        f"{icon} {symbol} — "
        f"{direction} SETUP\n\n"

        f"Grade: {result['grade']}\n"
        f"Score: {result['score']}/10+\n"
        f"H4: {h4_context}\n"
        f"H1 Structure: "
        f"{result['structure']}\n\n"

        f"Price: "
        f"{fmt_price(result['price'])}\n"

        f"RSI: "
        f"{result['rsi']:.1f}\n"

        f"ADX: "
        f"{result['adx']:.1f}\n"

        f"ATR: "
        f"{fmt_price(result['atr'])}\n"

        f"EMA20 distance: "
        f"{result['distance_atr']:.2f} ATR\n\n"

        f"CONFIRMATIONS\n"
        f"{reasons}\n\n"

        f"WARNINGS\n"
        f"{warnings}\n\n"

        f"ENTRY ZONE\n"
        f"{fmt_price(trade_map['entry_low'])}"
        f" - "
        f"{fmt_price(trade_map['entry_high'])}\n\n"

        f"INVALIDATION\n"
        f"{fmt_price(trade_map['invalidation'])}\n\n"

        f"TARGETS\n"
        f"TP1: {fmt_price(trade_map['tp1'])}\n"
        f"TP2: {fmt_price(trade_map['tp2'])}\n"
        f"TP3: {fmt_price(trade_map['tp3'])}\n\n"

        f"Action: WAIT FOR "
        f"CONFIRMATION / RETEST"
    )


# ============================================================
# ALERT CONTROL
# ============================================================

def get_setup_id(
    symbol,
    result,
):
    return (
        f"{symbol}:"
        f"{result['direction']}:"
        f"{result['structure']}"
    )


def should_alert(
    setup_id,
    result,
):
    current_time = now_utc()

    previous = last_alert.get(
        setup_id
    )

    if previous is None:
        return True

    previous_score = previous["score"]
    previous_grade = previous["grade"]
    previous_time = previous["time"]

    hours = (
        current_time - previous_time
    ).total_seconds() / 3600

    if hours >= SETUP_EXPIRY_HOURS:
        return True

    if result["grade"] != previous_grade:
        return True

    if result["score"] >= previous_score + 2:
        return True

    return False


def save_alert(
    setup_id,
    result,
):
    last_alert[setup_id] = {
        "score": result["score"],
        "grade": result["grade"],
        "time": now_utc(),
    }


# ============================================================
# SETUP MONITOR
# ============================================================

def monitor_previous_setup(
    symbol,
    h1,
    h4_context,
):
    state = setup_state.get(
        symbol
    )

    if not state:
        return None

    direction = state["direction"]

    result = analyze_direction(
        h1,
        h4_context,
        direction,
    )

    previous_score = state["score"]

    if result["grade"] == "NO_TRADE":

        return {
            "type": "INVALIDATED",
            "result": result,
        }

    if result["score"] <= previous_score - 2:

        return {
            "type": "WEAKENED",
            "result": result,
        }

    if result["score"] >= previous_score + 2:

        return {
            "type": "STRENGTHENED",
            "result": result,
        }

    return {
        "type": "ACTIVE",
        "result": result,
    }


def update_setup_state(
    symbol,
    result,
):
    setup_state[symbol] = {
        "direction": result["direction"],
        "score": result["score"],
        "grade": result["grade"],
        "time": now_utc(),
    }


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

async def analyze_symbol(symbol):

    h1 = await fetch_ohlcv(
        symbol,
        MAIN_TIMEFRAME,
        H1_LIMIT,
    )

    h4 = await fetch_ohlcv(
        symbol,
        HIGHER_TIMEFRAME,
        H4_LIMIT,
    )

    if h1 is None or h4 is None:

        logger.warning(
            "No data: %s",
            symbol,
        )

        return

    h1 = remove_open_candle(h1)
    h4 = remove_open_candle(h4)

    # --------------------------------------------------------
    # FIXED DATA VALIDATION
    # --------------------------------------------------------

    if len(h1) < 210 or len(h4) < 210:

        logger.warning(
            "Not enough data: %s | H1=%d H4=%d",
            symbol,
            len(h1),
            len(h4),
        )

        return

    h1 = add_indicators(h1)
    h4 = add_indicators(h4)

    h4_context = analyze_h4(h4)

    # --------------------------------------------------------
    # MONITOR EXISTING SETUP
    # --------------------------------------------------------

    previous_state = setup_state.get(
        symbol
    )

    if previous_state:

        monitor = monitor_previous_setup(
            symbol,
            h1,
            h4_context,
        )

        if monitor:

            event = monitor["type"]
            result = monitor["result"]

            if event == "INVALIDATED":

                await send_ntfy(
                    f"❌ {symbol} SETUP INVALIDATED",
                    (
                        f"{symbol} "
                        f"{previous_state['direction']} "
                        f"setup has been invalidated.\n\n"
                        f"H4: {h4_context}\n"
                        f"H1: {result['structure']}\n"
                        f"Score: {result['score']}"
                    ),
                    priority="high",
                    tags=(
                        "warning,"
                        "chart_with_downwards_trend"
                    ),
                )

                setup_state.pop(
                    symbol,
                    None,
                )

            elif event == "STRENGTHENED":

                trade_map = build_trade_map(
                    result
                )

                await send_ntfy(
                    f"⚡ {symbol} "
                    f"SETUP STRENGTHENED",
                    format_setup(
                        symbol,
                        h4_context,
                        result,
                        trade_map,
                    ),
                    priority="high",
                    tags="rocket",
                )

                update_setup_state(
                    symbol,
                    result,
                )

            elif event == "WEAKENED":

                await send_ntfy(
                    f"⚠️ {symbol} "
                    f"SETUP WEAKENED",
                    (
                        f"{symbol} "
                        f"{result['direction']}\n\n"
                        f"H4: {h4_context}\n"
                        f"H1: {result['structure']}\n"
                        f"Score: {result['score']}\n"
                        f"Previous score: "
                        f"{previous_state['score']}\n\n"
                        f"Action: WAIT"
                    ),
                    priority="default",
                    tags="warning",
                )

                update_setup_state(
                    symbol,
                    result,
                )

    # --------------------------------------------------------
    # LONG ANALYSIS
    # --------------------------------------------------------

    long_result = analyze_direction(
        h1,
        h4_context,
        "LONG",
    )

    # --------------------------------------------------------
    # SHORT ANALYSIS
    # --------------------------------------------------------

    short_result = analyze_direction(
        h1,
        h4_context,
        "SHORT",
    )

    candidates = [
        long_result,
        short_result,
    ]

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0]

    # --------------------------------------------------------
    # A+ / A
    # --------------------------------------------------------

    if best["grade"] in {
        "A+",
        "A",
    }:

        trade_map = build_trade_map(
            best
        )

        setup_id = get_setup_id(
            symbol,
            best,
        )

        if should_alert(
            setup_id,
            best,
        ):

            title = (
                f"🔥 {symbol} "
                f"{best['direction']} "
                f"{best['grade']}"
            )

            message = format_setup(
                symbol,
                h4_context,
                best,
                trade_map,
            )

            tags = (
                "chart_with_upwards_trend"
                if best["direction"] == "LONG"
                else "chart_with_downwards_trend"
            )

            await send_ntfy(
                title,
                message,
                priority="high",
                tags=tags,
            )

            save_alert(
                setup_id,
                best,
            )

            update_setup_state(
                symbol,
                best,
            )

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    elif best["grade"] == "WATCH":

        setup_id = get_setup_id(
            symbol,
            best,
        )

        if should_alert(
            setup_id,
            best,
        ):

            await send_ntfy(
                f"👀 {symbol} WATCH",
                (
                    f"{symbol} "
                    f"{best['direction']}\n\n"
                    f"Score: "
                    f"{best['score']}\n"
                    f"H4: {h4_context}\n"
                    f"H1: "
                    f"{best['structure']}\n"
                    f"RSI: "
                    f"{best['rsi']:.1f}\n"
                    f"ADX: "
                    f"{best['adx']:.1f}\n\n"
                    f"Action: WAIT FOR "
                    f"CONFIRMATION"
                ),
                priority="default",
                tags="eyes",
            )

            save_alert(
                setup_id,
                best,
            )


# ============================================================
# MARKET SCANNER
# ============================================================

async def scan_market():

    logger.info(
        "Starting market scan..."
    )

    for symbol in SYMBOLS:

        try:

            await analyze_symbol(
                symbol
            )

        except Exception as exc:

            logger.exception(
                "Analysis error %s: %s",
                symbol,
                exc,
            )

        await asyncio.sleep(
            0.25
        )

    logger.info(
        "Market scan completed"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "=========================================="
    )

    logger.info(
        "AH1 TRADING ASSISTANT STARTING"
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
        "Auto trading: DISABLED"
    )

    logger.info(
        "NTFY: ENABLED"
    )

    logger.info(
        "=========================================="
    )

    while True:

        try:

            await scan_market()

        except Exception as exc:

            logger.exception(
                "Scanner error: %s",
                exc,
            )

        await asyncio.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "Bot stopped."
        )

    finally:

        try:

            asyncio.run(
                exchange.close()
            )

        except Exception:
            pass
