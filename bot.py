import time
import logging
import warnings
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# CRYPTO INTRADAY MA LIVE BOT
# Multi-Symbol
# Binance Market Data
# 15m Live Scanner
# ntfy Notifications
# ============================================================

CONFIG = {
    "symbols": [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "DOT/USDT",
        "CELR/USDT",
        "LINK/USDT",
        "CRV/USDT",
        "NEAR/USDT",
        "ZEC/USDT",
        "ADA/USDT",
        "XRP/USDT",
        "FIL/USDT",
        "JASMY/USDT",
    ],

    "timeframe": "15m",
    "limit": 200,

    # Moving averages
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,

    # Pullback
    "pullback_tolerance": 0.004,

    # ATR / Risk
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "rr_ratio": 1.8,

    # ntfy
    "ntfy_enabled": True,
    "ntfy_server": "https://ntfy.sh",
    "ntfy_topic": "btc_ah7K9xQ2_signal",

    # Scanner
    "scan_interval_seconds": 30,
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MA-LIVE-BOT")


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
# STATE
# ============================================================

last_processed_candle = {}
last_signal_candle = {}


# ============================================================
# NTfy
# ============================================================

def send_ntfy(title: str, message: str, priority: int = 4):
    if not CONFIG["ntfy_enabled"]:
        return

    url = f"{CONFIG['ntfy_server']}/{CONFIG['ntfy_topic']}"

    try:
        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": str(priority),
                "Tags": "chart_with_upwards_trend,moneybag",
            },
            timeout=10,
        )

        if response.status_code >= 200 and response.status_code < 300:
            logger.info("ntfy notification sent: %s", title)
        else:
            logger.error(
                "ntfy error: HTTP %s | %s",
                response.status_code,
                response.text,
            )

    except Exception as e:
        logger.error("ntfy request failed: %s", e)


# ============================================================
# INDICATORS
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift(1)
    ).abs()

    low_close = (
        df["low"] - df["close"].shift(1)
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_9"] = (
        df["close"]
        .ewm(
            span=CONFIG["ema_fast"],
            adjust=False,
        )
        .mean()
    )

    df["ema_21"] = (
        df["close"]
        .ewm(
            span=CONFIG["ema_slow"],
            adjust=False,
        )
        .mean()
    )

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=CONFIG["ema_trend"],
            adjust=False,
        )
        .mean()
    )

    df["atr"] = calculate_atr(
        df,
        CONFIG["atr_period"],
    )

    df["ema_9_prev"] = df["ema_9"].shift(1)
    df["ema_21_prev"] = df["ema_21"].shift(1)

    return df


# ============================================================
# SIGNAL ENGINE
# ============================================================

def get_signal(df: pd.DataFrame):
    if len(df) < CONFIG["ema_trend"] + 10:
        return 0, None

    row = df.iloc[-2]

    ema9 = row["ema_9"]
    ema21 = row["ema_21"]
    ema50 = row["ema_50"]

    ema9_prev = row["ema_9_prev"]
    ema21_prev = row["ema_21_prev"]

    close = row["close"]
    open_price = row["open"]

    if any(
        pd.isna(x)
        for x in [
            ema9,
            ema21,
            ema50,
            ema9_prev,
            ema21_prev,
        ]
    ):
        return 0, None

    uptrend = close > ema50
    downtrend = close < ema50

    bullish_cross = (
        ema9 > ema21
        and ema9_prev <= ema21_prev
    )

    bearish_cross = (
        ema9 < ema21
        and ema9_prev >= ema21_prev
    )

    distance = abs(close - ema21) / close

    near_ema = (
        distance <= CONFIG["pullback_tolerance"]
    )

    bullish_candle = close > open_price
    bearish_candle = close < open_price

    long_cross = (
        bullish_cross
        and uptrend
    )

    long_pullback = (
        uptrend
        and near_ema
        and bullish_candle
        and close > ema21
    )

    short_cross = (
        bearish_cross
        and downtrend
    )

    short_pullback = (
        downtrend
        and near_ema
        and bearish_candle
        and close < ema21
    )

    if long_cross or long_pullback:
        return 1, "LONG"

    if short_cross or short_pullback:
        return -1, "SHORT"

    return 0, None


# ============================================================
# MARKET DATA
# ============================================================

def fetch_data(symbol: str):
    try:
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=CONFIG["timeframe"],
            limit=CONFIG["limit"],
        )

        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv,
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

        df.set_index("timestamp", inplace=True)

        return df

    except Exception as e:
        logger.error(
            "%s data error: %s",
            symbol,
            e,
        )
        return None


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def build_signal(symbol: str, side: str, df: pd.DataFrame):
    signal_candle = df.iloc[-2]

    entry = float(df.iloc[-1]["open"])

    atr = float(signal_candle["atr"])

    if np.isnan(atr) or atr <= 0:
        return None

    sl_distance = atr * CONFIG["atr_multiplier"]

    if side == "LONG":
        stop_loss = entry - sl_distance
        take_profit = (
            entry
            + sl_distance * CONFIG["rr_ratio"]
        )

    else:
        stop_loss = entry + sl_distance
        take_profit = (
            entry
            - sl_distance * CONFIG["rr_ratio"]
        )

    candle_time = signal_candle.name

    return {
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,
        "candle_time": candle_time,
    }


# ============================================================
# SEND SIGNAL
# ============================================================

def send_signal(signal: dict):
    symbol = signal["symbol"]
    side = signal["side"]

    entry = signal["entry"]
    stop_loss = signal["stop_loss"]
    take_profit = signal["take_profit"]
    atr = signal["atr"]

    if side == "LONG":
        emoji = "🟢"
    else:
        emoji = "🔴"

    title = f"{emoji} {symbol} {side}"

    message = (
        f"Symbol: {symbol}\n"
        f"Side: {side}\n"
        f"Entry: {entry:.8f}\n"
        f"Stop Loss: {stop_loss:.8f}\n"
        f"Take Profit: {take_profit:.8f}\n"
        f"RR: 1:{CONFIG['rr_ratio']}\n"
        f"ATR: {atr:.8f}\n"
        f"Timeframe: {CONFIG['timeframe']}\n"
        f"Candle: {signal['candle_time']}\n"
        f"Strategy: EMA 9/21/50 + Pullback"
    )

    logger.info(
        "%s | %s | Entry %.8f | SL %.8f | TP %.8f",
        symbol,
        side,
        entry,
        stop_loss,
        take_profit,
    )

    send_ntfy(
        title=title,
        message=message,
        priority=4,
    )


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(symbol: str):
    df = fetch_data(symbol)

    if df is None:
        return

    if len(df) < CONFIG["ema_trend"] + 10:
        logger.warning(
            "%s insufficient data",
            symbol,
        )
        return

    df = add_indicators(df)

    closed_candle = df.index[-2]

    previous_candle = last_processed_candle.get(symbol)

    if previous_candle == closed_candle:
        return

    last_processed_candle[symbol] = closed_candle

    signal, side = get_signal(df)

    if signal == 0:
        logger.info(
            "%s | %s | No signal",
            symbol,
            closed_candle,
        )
        return

    previous_signal_candle = (
        last_signal_candle.get(symbol)
    )

    if previous_signal_candle == closed_candle:
        return

    generated_signal = build_signal(
        symbol,
        side,
        df,
    )

    if generated_signal is None:
        return

    last_signal_candle[symbol] = closed_candle

    send_signal(generated_signal)


# ============================================================
# SCANNER
# ============================================================

def scan_all_symbols():
    logger.info(
        "Scanning %d symbols...",
        len(CONFIG["symbols"]),
    )

    for symbol in CONFIG["symbols"]:
        try:
            process_symbol(symbol)
        except Exception as e:
            logger.error(
                "%s processing error: %s",
                symbol,
                e,
            )


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("CRYPTO INTRADAY MA LIVE BOT")
    logger.info("Timeframe: %s", CONFIG["timeframe"])
    logger.info("Symbols: %d", len(CONFIG["symbols"]))
    logger.info("Strategy: EMA 9/21/50 + Pullback")
    logger.info("ntfy: %s", CONFIG["ntfy_enabled"])
    logger.info("=" * 60)

    send_ntfy(
        title="MA LIVE BOT ONLINE",
        message=(
            "Crypto Intraday MA Live Bot is online.\n"
            f"Timeframe: {CONFIG['timeframe']}\n"
            f"Symbols: {len(CONFIG['symbols'])}\n"
            "Scanner: ACTIVE"
        ),
        priority=3,
    )

    while True:
        try:
            scan_all_symbols()

        except Exception as e:
            logger.error(
                "Scanner error: %s",
                e,
            )

        time.sleep(
            CONFIG["scan_interval_seconds"]
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
