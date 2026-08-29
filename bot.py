"""
Crypto Intraday MA Strategy + Backtest + Risk Management + ntfy
---------------------------------------------------------------
استراتژی: EMA 9 / 21 / 50
- کراس
- پولبک
بک‌تست ساده + مدیریت ریسک + نوتیفیکیشن ntfy

نیازمندی‌ها:
    pip install pandas numpy ccxt requests
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings("ignore")

# ====================== تنظیمات ======================

CONFIG = {
    # نماد و تایم‌فریم
    "symbol": "BTC/USDT",
    "timeframe": "15m",
    "limit": 500,                  # تعداد کندل برای بک‌تست

    # مدیریت سرمایه و ریسک
    "initial_capital": 10000,      # سرمایه اولیه (USDT)
    "risk_per_trade": 0.01,        # ریسک هر معامله (۱٪)
    "rr_ratio": 1.8,               # نسبت ریوارد به ریسک
    "max_trades_per_day": 4,       # حداکثر معامله در روز

    # پارامترهای استراتژی
    "ema_fast": 9,
    "ema_slow": 21,
    "ema_trend": 50,
    "pullback_tolerance": 0.004,   # ۰.۴٪

    # استاپ‌لاس (می‌تونی یکی از این دو روش رو انتخاب کنی)
    "stop_method": "atr",          # "atr" یا "percent"
    "atr_period": 14,
    "atr_multiplier": 1.5,         # استاپ = ATR * این عدد
    "stop_percent": 0.008,         # اگر روش percent باشه (۰.۸٪)

    # ntfy
    "ntfy_enabled": True,
    "ntfy_topic": "btc_ah7K9xQ2_signal",   # ← تاپیک شما
    "ntfy_server": "https://ntfy.sh",
}


# ====================== توابع کمکی ======================

def send_ntfy(title: str, message: str, priority: int = 3):
    """ارسال نوتیفیکیشن به ntfy"""
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
        print(f"[ntfy] خطا در ارسال: {e}")


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

    # کراس
    bullish_cross = (df["ema_9"] > df["ema_21"]) & (df["ema_9_prev"] <= df["ema_21_prev"])
    bearish_cross = (df["ema_9"] < df["ema_21"]) & (df["ema_9_prev"] >= df["ema_21_prev"])

    # پولبک
    distance = abs(df["close"] - df["ema_21"]) / df["close"]
    near_ema = distance <= CONFIG["pullback_tolerance"]
    bullish_candle = df["close"] > df["open"]
    bearish_candle = df["close"] < df["open"]

    df["signal"] = 0

    # سیگنال خرید
    buy_cross = bullish_cross & uptrend
    buy_pullback = uptrend & near_ema & bullish_candle & (df["close"] > df["ema_21"])
    df.loc[buy_cross | buy_pullback, "signal"] = 1

    # سیگنال فروش
    sell_cross = bearish_cross & downtrend
    sell_pullback = downtrend & near_ema & bearish_candle & (df["close"] < df["ema_21"])
    df.loc[sell_cross | sell_pullback, "signal"] = -1

    return df


# ====================== بک‌تست ======================

def run_backtest(df: pd.DataFrame) -> dict:
    capital = CONFIG["initial_capital"]
    position = 0          # 1 = long, -1 = short, 0 = flat
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    position_size = 0.0
    entry_time = None

    trades = []
    equity_curve = []
    daily_trades = {}

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        current_time = row.name
        date_key = current_time.date() if hasattr(current_time, "date") else str(current_time)[:10]

        if date_key not in daily_trades:
            daily_trades[date_key] = 0

        # ===== مدیریت پوزیشن باز =====
        if position != 0:
            hit_sl = False
            hit_tp = False

            if position == 1:  # Long
                if row["low"] <= stop_loss:
                    hit_sl = True
                    exit_price = stop_loss
                elif row["high"] >= take_profit:
                    hit_tp = True
                    exit_price = take_profit
            else:  # Short
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
                    "entry_time": entry_time,
                    "exit_time": current_time,
                    "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": position_size,
                    "pnl": pnl,
                    "return_pct": (pnl / (entry_price * position_size)) * 100,
                    "result": "TP" if hit_tp else "SL"
                })

                position = 0
                position_size = 0.0

        # ===== ورود به معامله جدید =====
        if position == 0 and daily_trades[date_key] < CONFIG["max_trades_per_day"]:
            signal = prev["signal"]

            if signal != 0:
                entry_price = row["open"]
                atr_value = row["atr"] if not np.isnan(row["atr"]) else entry_price * 0.01

                if CONFIG["stop_method"] == "atr":
                    sl_distance = atr_value * CONFIG["atr_multiplier"]
                else:
                    sl_distance = entry_price * CONFIG["stop_percent"]

                if signal == 1:  # Long
                    stop_loss = entry_price - sl_distance
                    take_profit = entry_price + (sl_distance * CONFIG["rr_ratio"])
                    position = 1
                else:  # Short
                    stop_loss = entry_price + sl_distance
                    take_profit = entry_price - (sl_distance * CONFIG["rr_ratio"])
                    position = -1

                risk_amount = capital * CONFIG["risk_per_trade"]
                position_size = risk_amount / sl_distance
                entry_time = current_time
                daily_trades[date_key] += 1

        # ثبت equity
        unrealized = 0.0
        if position != 0:
            unrealized = (row["close"] - entry_price) * position_size * position
        equity_curve.append({
            "time": current_time,
            "equity": capital + unrealized
        })

    # بستن پوزیشن باز در انتها
    if position != 0:
        exit_price = df.iloc[-1]["close"]
        pnl = (exit_price - entry_price) * position_size * position
        capital += pnl
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "side": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": position_size,
            "pnl": pnl,
            "return_pct": (pnl / (entry_price * position_size)) * 100,
            "result": "EOD"
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index("time")

    if len(trades_df) == 0:
        return {"error": "هیچ معامله‌ای انجام نشد"}

    total_return = (capital - CONFIG["initial_capital"]) / CONFIG["initial_capital"] * 100
    win_trades = trades_df[trades_df["pnl"] > 0]
    winrate = len(win_trades) / len(trades_df) * 100
    avg_win = win_trades["pnl"].mean() if len(win_trades) > 0 else 0
    avg_loss = trades_df[trades_df["pnl"] <= 0]["pnl"].mean() if len(trades_df[trades_df["pnl"] <= 0]) > 0 else 0
    profit_factor = abs(win_trades["pnl"].sum() / trades_df[trades_df["pnl"] <= 0]["pnl"].sum()) if len(trades_df[trades_df["pnl"] <= 0]) > 0 else np.inf

    results = {
        "initial_capital": CONFIG["initial_capital"],
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return, 2),
        "total_trades": len(trades_df),
        "winrate_pct": round(winrate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "trades": trades_df,
        "equity_curve": equity_df
    }
    return results


def print_backtest_results(results: dict):
    if "error" in results:
        print(results["error"])
        return

    print("\n" + "="*55)
    print("نتایج بک‌تست")
    print("="*55)
    print(f"سرمایه اولیه     : {results['initial_capital']:,.0f} USDT")
    print(f"سرمایه نهایی     : {results['final_capital']:,.0f} USDT")
    print(f"بازده کل         : {results['total_return_pct']}%")
    print(f"تعداد معاملات    : {results['total_trades']}")
    print(f"وین‌ریت          : {results['winrate_pct']}%")
    print(f"میانگین سود      : {results['avg_win']:.2f}")
    print(f"میانگین ضرر      : {results['avg_loss']:.2f}")
    print(f"Profit Factor    : {results['profit_factor']}")
    print("="*55)

    print("\nآخرین ۵ معامله:")
    print(results["trades"][["side", "entry_price", "exit_price", "pnl", "result"]].tail().to_string())
    print()


# ====================== سیگنال زنده ======================

def check_live_signal(df: pd.DataFrame):
    """بررسی آخرین سیگنال و ارسال به ntfy"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = prev["signal"]
    if signal == 0:
        print("سیگنال جدیدی وجود ندارد.")
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

    title = f"سیگنال {side} | {CONFIG['symbol']}"
    message = (
        f"نماد: {CONFIG['symbol']}\n"
        f"قیمت: {price:.2f}\n"
        f"EMA9: {last['ema_9']:.2f} | EMA21: {last['ema_21']:.2f}\n"
        f"استاپ: {sl:.2f}\n"
        f"تارگت: {tp:.2f}\n"
        f"RR: 1:{CONFIG['rr_ratio']}"
    )

    print(title)
    print(message)
    send_ntfy(title, message, priority=4)


# ====================== دریافت دیتا ======================

def fetch_data():
    """گرفتن دیتا از بایننس"""
    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(
            CONFIG["symbol"],
            timeframe=CONFIG["timeframe"],
            limit=CONFIG["limit"]
        )
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        print(f"خطا در دریافت دیتا: {e}")
        print("از داده‌های نمونه استفاده می‌شود...")
        return generate_sample_data()


def generate_sample_data():
    """داده‌ی نمونه برای تست بدون اینترنت"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=CONFIG["limit"], freq="15min")
    price = 65000 + np.cumsum(np.random.randn(CONFIG["limit"]) * 80)
    df = pd.DataFrame({
        "open": price + np.random.randn(CONFIG["limit"]) * 30,
        "high": price + abs(np.random.randn(CONFIG["limit"]) * 60),
        "low": price - abs(np.random.randn(CONFIG["limit"]) * 60),
        "close": price,
        "volume": np.random.randint(100, 2000, CONFIG["limit"])
    }, index=dates)
    return df


if __name__ == "__main__":
    print("در حال دریافت داده‌ها...")
    df = fetch_data()

    print("محاسبه اندیکاتورها و سیگنال‌ها...")
    df = add_indicators(df)
    df = generate_signals(df)

    # ----- بک‌تست -----
    print("\nدر حال اجرای بک‌تست...")
    results = run_backtest(df)
    print_backtest_results(results)

    # ----- سیگنال زنده -----
    print("\nبررسی سیگنال زنده...")
    check_live_signal(df)

    # ذخیره نتایج
    if "trades" in results:
        results["trades"].to_csv("backtest_trades.csv", index=False)
        results["equity_curve"].to_csv("equity_curve.csv")
        print("فایل‌های backtest_trades.csv و equity_curve.csv ذخیره شدند.")
