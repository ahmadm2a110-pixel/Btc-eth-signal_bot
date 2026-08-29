"""
Crypto Intraday MA Strategy + Backtest + Risk Management + ntfy
Multi-Symbol Version
---------------------------------------------------------------
ارزها: BTC, ETH, SOL, BNB, DOT, CELR, LINK, CRV, NEAR, ZEC, ADA, XRP, FIL, JASMY
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import warnings
warnings.filterwarnings("ignore")

# ====================== تنظیمات ======================

CONFIG = {
    # لیست ارزها
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
    "limit": 500,

    # مدیریت سرمایه و ریسک
    "initial_capital": 10000,
    "risk_per_trade": 0.01,        # ۱٪
    "rr_ratio": 1.8,
    "max_trades_per_day": 4,

    # پارامترهای استراتژی
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,
    "pullback_tolerance": 0.004,

    # استاپ‌لاس
    "stop_method": "atr",          # "atr" یا "percent"
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "stop_percent": 0.008,

    # ntfy
    "ntfy_enabled": True,
    "ntfy_topic": "btc_ah7K9xQ2_signal",
    "ntfy_server": "https://ntfy.sh",
}


# ====================== توابع کمکی ======================

def send_ntfy(title: str, message: str, priority: int = 3):
    if not CONFIG["ntfy_enabled"]:
        return
    url = f"{CONFIG['ntfy_server']}/{CONFIG['ntfy_topic']}"
    try:
        requests.post(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": str(priority),
                "Tags": "chart_with_upwards_trend,moneybag"
            },
            timeout=10
        )
        print(f"[ntfy] ارسال شد → {title}")
    except Exception as e:
        print(f"[ntfy] خطا: {e}")


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_9"] = calculate_ema(df["close"], CONFIG["ema_fast"])
    df["ema_21"] = calculate_ema(df["close"], CONFIG["ema_slow"])
    df["ema_50"] = calculate_ema(df["close"], CONFIG["ema_trend"])
    df["atr"] = calculate_atr(df, CONFIG["atr_period"])
    df["ema_9_prev"] = df["ema_9"].shift(1)
    df["ema_21_prev"] = df["ema_21"].shift(1)
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    uptrend = df["close"] > df["ema_50"]
    downtrend = df["close"] < df["ema_50"]

    bullish_cross = (df["ema_9"] > df["ema_21"]) & (df["ema_9_prev"] <= df["ema_21_prev"])
    bearish_cross = (df["ema_9"] < df["ema_21"]) & (df["ema_9_prev"] >= df["ema_21_prev"])

    distance = abs(df["close"] - df["ema_21"]) / df["close"]
    near_ema = distance <= CONFIG["pullback_tolerance"]
    bullish_candle = df["close"] > df["open"]
    bearish_candle = df["close"] < df["open"]

    df["signal"] = 0
    buy_cross = bullish_cross & uptrend
    buy_pullback = uptrend & near_ema & bullish_candle & (df["close"] > df["ema_21"])
    df.loc[buy_cross | buy_pullback, "signal"] = 1

    sell_cross = bearish_cross & downtrend
    sell_pullback = downtrend & near_ema & bearish_candle & (df["close"] < df["ema_21"])
    df.loc[sell_cross | sell_pullback, "signal"] = -1

    return df


# ====================== بک‌تست ======================

def run_backtest(df: pd.DataFrame, symbol: str) -> dict:
    capital = CONFIG["initial_capital"]
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    entry_time = None

    trades = []
    daily_trades = {}

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        current_time = row.name
        date_key = current_time.date() if hasattr(current_time, "date") else str(current_time)[:10]

        if date_key not in daily_trades:
            daily_trades[date_key] = 0

        # مدیریت پوزیشن باز
        if position != 0:
            hit_sl = hit_tp = False
            if position == 1:
                if row["low"] <= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                elif row["high"] >= take_profit:
                    hit_tp = True
                    exit_price = take_profit
            else:
                if row["high"] >= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                elif row["low"] <= take_profit:
                    hit_tp = True
                    exit_price = take_profit

            if hit_sl or hit_tp:
                pnl = (exit_price - entry_price) * position_size * position
                capital += pnl
                trades.append({
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "result": "TP" if hit_tp else "SL"
                })
                position = 0

        # ورود جدید
        if position == 0 and daily_trades[date_key] < CONFIG["max_trades_per_day"]:
            signal = prev["signal"]
            if signal != 0:
                entry_price = row["open"]
                atr_value = row["atr"] if not np.isnan(row["atr"]) else entry_price * 0.01

                if CONFIG["stop_method"] == "atr":
                    sl_distance = atr_value * CONFIG["atr_multiplier"]
                else:
                    sl_distance = entry_price * CONFIG["stop_percent"]

                if signal == 1:
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + (sl_distance * CONFIG["rr_ratio"])
                    position = 1
                else:
                    stop_loss = entry_price + sl_distance
                    take_profit = entry_price - (sl_distance * CONFIG["rr_ratio"])
                    position = -1

                risk_amount = capital * CONFIG["risk_per_trade"]
                position_size = risk_amount / sl_distance
                entry_time = current_time
                daily_trades[date_key] += 1

    # بستن پوزیشن باز
    if position != 0:
        exit_price = df.iloc[-1]["close"]
        pnl = (exit_price - entry_price) * position_size * position
        capital += pnl
        trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "side": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "result": "EOD"
        })

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return {"symbol": symbol, "error": "هیچ معامله‌ای انجام نشد"}

    total_return = (capital - CONFIG["initial_capital"]) / CONFIG["initial_capital"] * 100
    win_trades = trades_df[trades_df["pnl"] > 0]
    winrate = len(win_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    return {
        "symbol": symbol,
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(trades_df),
        "winrate_pct": round(winrate, 2),
        "trades": trades_df
    }


def print_backtest_summary(all_results: list):
    print("\n" + "="*70)
    print("خلاصه بک‌تست همه ارزها")
    print("="*70)
    print(f"{'نماد':<12} {'بازده %':<10} {'معاملات':<10} {'وین‌ریت %':<10} {'سرمایه نهایی'}")
    print("-"*70)
    for r in all_results:
        if "error" in r:
            print(f"{r['symbol']:<12} {'---':<10} {'---':<10} {'---':<10} {r['error']}")
        else:
            print(f"{r['symbol']:<12} {r['total_return_pct']:<10} {r['total_trades']:<10} {r['winrate_pct']:<10} {r['final_capital']}")
    print("="*70)


# ====================== سیگنال زنده ======================

def check_live_signal(df: pd.DataFrame, symbol: str):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    signal = prev["signal"]

    if signal == 0:
        return

    side = "خرید (LONG)" if signal == 1 else "فروش (SHORT)"
    price = last["close"]

    atr_value = last["atr"] if not np.isnan(last["atr"]) else price * 0.01
    if CONFIG["stop_method"] == "atr":
        sl_dist = atr_value * CONFIG["atr_multiplier"]
    else:
        sl_dist = price * CONFIG["stop_percent"]

    if signal == 1:
        sl = price - sl_dist
        tp = price + sl_dist * CONFIG["rr_ratio"]
    else:
        sl = price + sl_dist
        tp = price - sl_dist * CONFIG["rr_ratio"]

    title = f"سیگنال {side} | {symbol}"
    message = (
        f"نماد: {symbol}\n"
        f"قیمت: {price:.4f}\n"
        f"EMA9: {last['ema_9']:.4f} | EMA21: {last['ema_21']:.4f}\n"
        f"استاپ: {sl:.4f}\n"
        f"تارگت: {tp:.4f}\n"
        f"RR: 1:{CONFIG['rr_ratio']}"
    )

    print(f"\n{title}")
    print(message)
    send_ntfy(title, message, priority=4)


# ====================== دریافت دیتا ======================

def fetch_data(symbol: str):
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=CONFIG["timeframe"], limit=CONFIG["limit"])
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"[{symbol}] خطا در دریافت دیتا: {e}")
        return None


# ====================== اجرای اصلی ======================

if __name__ == "__main__":
    all_results = []

    print(f"شروع بررسی {len(CONFIG['symbols'])} ارز...\n")

    for symbol in CONFIG["symbols"]:
        print(f"---------- {symbol} ----------")
        df = fetch_data(symbol)

        if df is None or len(df) < 50:
            print(f"دیتای کافی برای {symbol} وجود ندارد.\n")
            continue

        df = add_indicators(df)
        df = generate_signals(df)

        # بک‌تست
        result = run_backtest(df, symbol)
        all_results.append(result)

        if "error" not in result:
            print(f"بازده: {result['total_return_pct']}% | معاملات: {result['total_trades']} | وین‌ریت: {result['winrate_pct']}%")

        # سیگنال زنده
        check_live_signal(df, symbol)
        print()

    # خلاصه نهایی
    print_backtest_summary(all_results)

    # ذخیره همه معاملات
    all_trades = []
    for r in all_results:
        if "trades" in r:
            all_trades.append(r["trades"])
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined.to_csv("all_backtest_trades.csv", index=False)
        print("\nفایل all_backtest_trades.csv ذخیره شد.")
