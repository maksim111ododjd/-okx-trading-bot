#!/usr/bin/env python3
# main.py — OKX trading bot (paper mode default) with Telegram control + Flask health endpoint
import os, time, json, hmac, hashlib, base64, csv, math, traceback, threading
from datetime import datetime, timezone
from typing import Optional
import requests
import pandas as pd
import numpy as np
import ta
from flask import Flask, jsonify
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# -------------------- CONFIG from env --------------------
OKX_API_KEY        = os.getenv("OKX_API_KEY", "").strip()
OKX_API_SECRET     = os.getenv("OKX_API_SECRET", "").strip()
OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "").strip()

TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = int(os.getenv("TELEGRAM_CHAT_ID", "0"))

PAPER_MODE         = os.getenv("PAPER_MODE", "true").lower() in ("1","true","yes")
BASE_URL           = os.getenv("BASE_URL", "https://www.okx.com")

SYMBOLS            = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,SOL-USDT").split(",")
TIMEFRAME          = os.getenv("TIMEFRAME", "5m")
POLL_INTERVAL      = int(os.getenv("POLL_INTERVAL", "30"))

RISK_PER_TRADE     = float(os.getenv("RISK_PER_TRADE", "1.0"))   # % of balance to risk per trade
MIN_USDT_TO_TRADE  = float(os.getenv("MIN_USDT_TO_TRADE", "5"))

EMA_FAST = int(os.getenv("EMA_FAST","5"))
EMA_SLOW = int(os.getenv("EMA_SLOW","10"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD","14"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD","14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER","1.5"))
TP_MULT = float(os.getenv("TP_MULT","3.0"))
TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT","1.0"))

LOG_TRADES_FILE = os.getenv("LOG_TRADES_FILE","trades_log.csv")
PNL_FILE = os.getenv("PNL_FILE","pnl.csv")

X_SIM_HEADER = {"x-simulated-trading": "1"} if PAPER_MODE else {}

# -------------------- Utilities --------------------
def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec='seconds')

def safe_print(*a, **k):
    print(f"{now_iso()} |", *a, **k)

# -------------------- Telegram helper (simple send) --------------------
def send_telegram_text(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        safe_print("Telegram not configured")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML"}
        requests.post(url, json=payload, timeout=8)
    except Exception as e:
        safe_print("Telegram send error:", e)

# -------------------- OKX sign/time --------------------
def get_okx_server_iso():
    url = BASE_URL + "/api/v5/public/time"
    r = requests.get(url, timeout=10)
    j = r.json()
    ts_ms = int(j["data"][0]["ts"])
    iso = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat(timespec='milliseconds').replace("+00:00","Z")
    return iso

def sign_okx(method, path, body, timestamp):
    body_str = json.dumps(body, separators=(",",":")) if body else ""
    message = f"{timestamp}{method.upper()}{path}{body_str}"
    mac = hmac.new(OKX_API_SECRET.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()

def okx_request(method, path, body=None, params=None):
    try:
        ts = get_okx_server_iso()
    except Exception as e:
        raise RuntimeError(f"Time fetch error: {e}")
    sign = sign_okx(method, path, body, ts)
    headers = {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": OKX_API_PASSPHRASE,
        "Content-Type": "application/json",
        **X_SIM_HEADER
    }
    url = BASE_URL + path
    if method.upper() == "GET":
        r = requests.get(url, headers=headers, params=params, timeout=15)
    else:
        r = requests.post(url, headers=headers, json=body, timeout=15)
    try:
        return r.json()
    except:
        return {"err": f"non-json response: {r.status_code} {r.text}"}

# -------------------- Market data --------------------
def fetch_klines_okx(symbol, bar=TIMEFRAME, limit=200):
    path = "/api/v5/market/candles"
    params = {"instId": symbol, "bar": bar, "limit": str(limit)}
    r = requests.get(BASE_URL + path, params=params, timeout=15)
    j = r.json()
    if "data" not in j or not j["data"]:
        return pd.DataFrame()
    df = pd.DataFrame(j["data"])
    df = df.iloc[:, :6]
    df.columns = ["ts","open","high","low","close","volume"]
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df.iloc[::-1].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df

def add_indicators(df):
    if df.empty or len(df) < max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 2:
        return df
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=ATR_PERIOD)
    df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_sig'] = macd.macd_signal()
    return df

# -------------------- Account & sizing --------------------
def get_usdt_balance():
    try:
        res = okx_request("GET", "/api/v5/account/balance")
        if not isinstance(res, dict) or "data" not in res:
            return 0.0
        for d in res["data"][0].get("details", []):
            if d.get("ccy") == "USDT":
                try:
                    return float(d.get("availBal", 0.0))
                except:
                    return 0.0
        return 0.0
    except Exception as e:
        safe_print("Balance fetch error:", e)
        return 0.0

def calc_qty_by_risk(price, usdt_balance, risk_percent, stop_distance_usd):
    risk_usd = usdt_balance * (risk_percent/100.0)
    if stop_distance_usd <= 1e-9:
        return 0.0
    pos_usd = risk_usd / stop_distance_usd
    qty = pos_usd / price
    return max(0.0, round(qty, 6))

# -------------------- Execution simulation/real --------------------
open_positions = {}  # symbol -> pos dict

def simulate_open(symbol, side, qty, entry, sl, tp):
    pos = {"symbol":symbol,"side":side,"qty":qty,"entry":entry,"sl":sl,"tp":tp,"trail":None,"opened_at":now_iso()}
    open_positions[symbol] = pos
    return {"sim":True,"result":"ok","filled_price":entry}

def simulate_close(symbol, price):
    pos = open_positions.get(symbol)
    if not pos:
        return None
    side = pos["side"]
    qty = pos["qty"]
    pnl = (price - pos["entry"]) * qty if side=="buy" else (pos["entry"] - price) * qty
    pos["closed_at"] = now_iso()
    pos["closed_price"] = price
    pos["pnl"] = pnl
    archive_trade(pos)
    del open_positions[symbol]
    return {"sim":True,"pnl":pnl}

def real_market_order(symbol, side, qty):
    path = "/api/v5/trade/order"
    body = {"instId": symbol, "tdMode":"cash", "side": side.lower(), "ordType":"market", "sz": str(qty)}
    return okx_request("POST", path, body)

def place_order(symbol, side, qty, entry, sl, tp):
    if PAPER_MODE:
        return simulate_open(symbol, side, qty, entry, sl, tp)
    else:
        return real_market_order(symbol, side, qty)

def close_position(symbol, price):
    if PAPER_MODE:
        return simulate_close(symbol, price)
    else:
        pos = open_positions.get(symbol)
        if not pos:
            return None
        opp = "sell" if pos["side"]=="buy" else "buy"
        resp = real_market_order(symbol, opp, pos["qty"])
        archive_trade({"symbol":symbol,"qty":pos["qty"],"entry":pos["entry"],"closed_price":price,"pnl":0.0})
        del open_positions[symbol]
        return resp

# -------------------- Logging --------------------
def ensure_file(fn, headers):
    if not os.path.exists(fn):
        with open(fn, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

ensure_file(LOG_TRADES_FILE, ["time","symbol","action","side","qty","price","sl","tp","pnl","note"])
ensure_file(PNL_FILE, ["time","symbol","qty","entry","exit","pnl","balance"])

def log_trade_open(symbol, side, qty, price, sl, tp):
    row = {"time": now_iso(), "symbol":symbol, "action":"OPEN", "side":side, "qty":qty, "price":price, "sl":sl, "tp":tp, "pnl":0.0, "note":""}
    with open(LOG_TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

def log_trade_close(pos):
    row = {"time": now_iso(), "symbol":pos["symbol"], "qty":pos["qty"], "entry":pos["entry"], "exit":pos.get("closed_price",""), "pnl":pos.get("pnl",0.0), "balance": get_usdt_balance()}
    with open(PNL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)
    row2 = {"time": now_iso(), "symbol":pos["symbol"], "action":"CLOSE", "side":pos["side"], "qty":pos["qty"], "price":pos.get("closed_price",""), "sl":pos["sl"], "tp":pos["tp"], "pnl":pos.get("pnl",0.0), "note":""}
    with open(LOG_TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row2.keys()))
        writer.writerow(row2)

def archive_trade(pos):
    row = {"time": now_iso(), "symbol":pos["symbol"], "qty":pos["qty"], "entry":pos["entry"], "exit": pos.get("closed_price",""), "pnl": pos.get("pnl",0.0), "balance": get_usdt_balance()}
    with open(PNL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

# -------------------- Strategy --------------------
def compute_signal_and_params(df):
    df = add_indicators(df)
    if df.empty:
        return None
    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"]) if not np.isnan(last["atr"]) else max(1.0, price*0.01)
    rsi = float(last["rsi"]) if not np.isnan(last["rsi"]) else 50.0
    ema_buy = last["ema_fast"] > last["ema_slow"]
    ema_sell = last["ema_fast"] < last["ema_slow"]
    macd_ok = float(last.get("macd", 0.0)) > float(last.get("macd_sig", 0.0))
    if rsi < 42 and ema_buy and macd_ok:
        side = "buy"
    elif rsi > 58 and ema_sell and not macd_ok:
        side = "sell"
    else:
        return None
    stop_dist = ATR_MULTIPLIER * atr
    tp_dist = TP_MULT * stop_dist
    if side == "buy":
        sl_price = price - stop_dist
        tp_price = price + tp_dist
    else:
        sl_price = price + stop_dist
        tp_price = price - tp_dist
    return {"side":side, "price":price, "sl":sl_price, "tp":tp_price, "stop_dist":stop_dist}

# -------------------- Trading control --------------------
trading_active = False
trading_lock = threading.Lock()

def start_trading():
    global trading_active
    with trading_lock:
        if not trading_active:
            trading_active = True
            send_telegram_text("▶️ Торговля включена (paper mode)")
            safe_print("Trading activated")

def stop_trading():
    global trading_active
    with trading_lock:
        if trading_active:
            trading_active = False
            send_telegram_text("⏸ Торговля приостановлена")
            safe_print("Trading stopped")

# -------------------- Main trading loop --------------------
def trading_loop():
    safe_print("Trading loop thread started. PAPER_MODE=", PAPER_MODE)
    send_telegram_text(f"🚀 Bot started (paper={PAPER_MODE})")
    last_report = time.time()
    while True:
        try:
            if not trading_active:
                time.sleep(2)
                continue
            usdt = get_usdt_balance()
            safe_print(f"USDT: {usdt:.2f} | Open positions: {len(open_positions)}")
            if usdt < MIN_USDT_TO_TRADE:
                safe_print("Баланс USDT мал (< min). Ждём...")
                time.sleep(POLL_INTERVAL)
                continue
            for sym in SYMBOLS:
                try:
                    df = fetch_klines_okx(sym, TIMEFRAME, limit=200)
                    if df.empty:
                        safe_print(sym, "no data")
                        continue
                    params = compute_signal_and_params(df)
                    price = float(df["close"].iloc[-1])
                    if sym in open_positions:
                        pos = open_positions[sym]
                        atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else 0
                        if pos["side"] == "buy":
                            new_trail = price - TRAILING_ATR_MULT * atr
                            if pos.get("trail") is None or new_trail > pos["trail"]:
                                pos["trail"] = new_trail
                                pos["sl"] = max(pos["sl"], pos["trail"])
                        else:
                            new_trail = price + TRAILING_ATR_MULT * atr
                            if pos.get("trail") is None or new_trail < pos["trail"]:
                                pos["trail"] = new_trail
                                pos["sl"] = min(pos["sl"], pos["trail"])
                        if (pos["side"]=="buy" and (price <= pos["sl"] or price >= pos["tp"])) or (pos["side"]=="sell" and (price >= pos["sl"] or price <= pos["tp"])):
                            res = close_position(sym, price)
                            pnl = res.get("pnl", 0) if isinstance(res, dict) else 0
                            safe_print(f"CLOSED {sym} {pos['side']} qty={pos['qty']} entry={pos['entry']:.4f} exit={price:.4f} pnl={pnl:.4f}")
                            log_trade_close({"symbol":sym,"qty":pos['qty'],"entry":pos['entry'],"closed_price":price,"pnl":pnl,"side":pos['side'],"sl":pos['sl'],"tp":pos['tp']})
                            send_telegram_text(f"⚡ CLOSED {sym} {pos['side'].upper()} qty={pos['qty']} entry={pos['entry']:.4f} exit={price:.4f} pnl={pnl:.4f}")
                            continue
                    if sym not in open_positions and params and usdt >= MIN_USDT_TO_TRADE:
                        stop_usd = params["stop_dist"]
                        qty = calc_qty_by_risk(params["price"], usdt, RISK_PER_TRADE, stop_usd)
                        if qty <= 0:
                            safe_print(sym, "qty 0 skip")
                            continue
                        qty = round(qty, 6)
                        resp = place_order(sym, params["side"], qty, params["price"], params["sl"], params["tp"])
                        log_trade_open(sym, params["side"], qty, params["price"], params["sl"], params["tp"])
                        send_telegram_text(f"🟢 OPEN {sym} {params['side'].upper()} qty={qty} entry={params['price']:.4f} SL={params['sl']:.4f} TP={params['tp']:.4f}")
                        safe_print(f"OPEN {sym} {params['side']} qty={qty} entry={params['price']:.4f}")
                except Exception as e_sym:
                    safe_print(f"Error {sym}:", e_sym)
                    safe_print(traceback.format_exc())
            if time.time() - last_report > 60*10:
                total_pnl = 0.0
                if os.path.exists(PNL_FILE):
                    try:
                        dfp = pd.read_csv(PNL_FILE)
                        if "pnl" in dfp.columns:
                            total_pnl = dfp["pnl"].sum()
                    except:
                        total_pnl = 0.0
                send_telegram_text(f"📊 Report: Balance {get_usdt_balance():.2f} USDT | Closed PnL: {total_pnl:.2f} USDT | Open positions: {len(open_positions)}")
                last_report = time.time()
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            safe_print("Trading loop error:", e)
            safe_print(traceback.format_exc())
            send_telegram_text(f"⚠ Trading loop error: {e}")
            time.sleep(5)

# -------------------- Telegram handlers --------------------
async def cmd_startbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != TELEGRAM_CHAT_ID:
        return
    start_trading()
    await update.message.reply_text("🚀 Торговля включена (paper mode).")

async def cmd_stopbot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != TELEGRAM_CHAT_ID:
        return
    stop_trading()
    await update.message.reply_text("⏸ Торговля приостановлена.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != TELEGRAM_CHAT_ID:
        return
    usdt = get_usdt_balance()
    msg = f"🛰 Status\nPaper: {PAPER_MODE}\nBalance USDT: {usdt:.2f}\nOpen positions: {len(open_positions)}"
    await update.message.reply_text(msg)

async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != TELEGRAM_CHAT_ID:
        return
    bal = get_usdt_balance()
    await update.message.reply_text(f"💰 Balance: {bal:.4f} USDT")

async def cmd_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != TELEGRAM_CHAT_ID:
        return
    if not open_positions:
        await update.message.reply_text("Нет открытых позиций.")
        return
    out = []
    for s,p in open_positions.items():
        out.append(f"{s} | {p['side'].upper()} qty={p['qty']} entry={p['entry']:.4f} SL={p['sl']:.4f} TP={p['tp']:.4f}")
    await update.message.reply_text("\n".join(out))

async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != TELEGRAM_CHAT_ID:
        return
    total_pnl = 0.0
    if os.path.exists(PNL_FILE):
        try:
            dfp = pd.read_csv(PNL_FILE)
            if "pnl" in dfp.columns:
                total_pnl = dfp["pnl"].sum()
        except:
            total_pnl = 0.0
    await update.message.reply_text(f"📊 Closed PnL total: {total_pnl:.4f} USDT")

# -------------------- Flask health endpoints --------------------
app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"msg":"OKX trading bot service","ok":True})

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "open_positions": len(open_positions)})

# -------------------- Startup --------------------
def start_background_workers():
    # 1) trading thread
    t = threading.Thread(target=trading_loop, name="trading-loop", daemon=True)
    t.start()
    safe_print("Trading thread started (daemon).")

    # 2) telegram bot (async) - run in separate thread
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        # -------------------- Startup --------------------
import asyncio  # добавь этот импорт прямо сюда (рядом с def)

def start_background_workers():
    # 1) торговый поток
    t = threading.Thread(target=trading_loop, name="trading-loop", daemon=True)
    t.start()
    safe_print("Trading thread started (daemon).")

    # 2) Telegram бот — теперь корректно через asyncio.run()
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        def run_telegram():
            try:
                app_tg = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
                app_tg.add_handler(CommandHandler("startbot", cmd_startbot))
                app_tg.add_handler(CommandHandler("stopbot", cmd_stopbot))
                app_tg.add_handler(CommandHandler("status", cmd_status))
                app_tg.add_handler(CommandHandler("balance", cmd_balance))
                app_tg.add_handler(CommandHandler("positions", cmd_positions))
                app_tg.add_handler(CommandHandler("report", cmd_report))
                safe_print("Telegram bot starting (polling)...")

                # ВАЖНО: теперь запускаем через asyncio.run
                asyncio.run(app_tg.run_polling(poll_interval=3.0))
            except Exception as e:
                safe_print("Telegram thread error:", e)
                send_telegram_text(f"⚠ Telegram thread error: {e}")

        th = threading.Thread(target=run_telegram, name="telegram-poll", daemon=True)
        th.start()
        safe_print("Telegram polling thread started.")
    else:
        safe_print("Telegram not configured (TELEGRAM_TOKEN/CHAT_ID missing).")

# -------------------- Run server --------------------
if __name__ == "__main__":
    # start workers
    start_background_workers()
    # start flask webserver (Render uses PORT env)
    port = int(os.getenv("PORT", "8000"))
    safe_print("Starting Flask on port", port)
    from flask import Flask as _F  # already imported but ensure
    # Run flask (this will be main thread)
    app.run(host="0.0.0.0", port=port)
