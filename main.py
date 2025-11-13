#!/usr/bin/env python3
"""
main.py — OKX multi-symbol trading bot (paper mode by default)
Features:
- OKX REST v5 signing
- indicators: EMA, RSI, ATR, MACD
- position sizing by % risk of balance
- SL = ATR * ATR_MULTIPLIER, TP = SL * TP_MULT
- trailing stop using ATR
- Telegram notifications
- CSV logging: trades_log.csv, pnl.csv
- runs continuously, robust error handling
CONFIG via environment variables (no secrets in repo)
"""
import os, time, json, hmac, hashlib, base64, csv, math, traceback
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import ta

# -------------------- CONFIG from env --------------------
OKX_API_KEY        = os.getenv("OKX_API_KEY", "")
OKX_API_SECRET     = os.getenv("OKX_API_SECRET", "")
OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE", "")

TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

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

# header for simulated trading in OKX (paper)
X_SIM_HEADER = {"x-simulated-trading": "1"} if PAPER_MODE else {}

# -------------------- Utilities --------------------
def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec='seconds')

def safe_print(*a, **k):
    print(f"{now_iso()} |", *a, **k)

# -------------------- Telegram --------------------
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
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
    # keep first 6 columns: ts, open, high, low, close, vol
    df = df.iloc[:, :6]
    df.columns = ["ts","open","high","low","close","volume"]
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df = df.iloc[::-1].reset_index(drop=True)  # okx returns newest first
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
    # additional close record in trades log
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
    # dynamic thresholds (can be tuned)
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

# -------------------- Main loop --------------------
def run_multi_bot():
    send_telegram(f"🚀 Bot started (paper={PAPER_MODE})")
    safe_print("Starting multi-bot. Press CTRL+C to stop.")
    last_report = time.time()
    try:
        while True:
            try:
                usdt = get_usdt_balance()
            except Exception as e:
                safe_print("Balance fetch error:", e)
                usdt = 0.0
            safe_print(f"USDT: {usdt:.2f} | Open positions: {len(open_positions)}")
            if usdt < MIN_USDT_TO_TRADE:
                safe_print("Баланс USDT мал (< min). Ждём...")
            for sym in SYMBOLS:
                try:
                    df = fetch_klines_okx(sym, TIMEFRAME, limit=200)
                    if df.empty:
                        safe_print(sym, "no data")
                        continue
                    params = compute_signal_and_params(df)
                    price = float(df["close"].iloc[-1])
                    # monitor existing position
                    if sym in open_positions:
                        pos = open_positions[sym]
                        # update trailing
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
                        # check sl / tp
                        if (pos["side"]=="buy" and (price <= pos["sl"] or price >= pos["tp"])) or (pos["side"]=="sell" and (price >= pos["sl"] or price <= pos["tp"])):
                            res = close_position(sym, price)
                            pnl = res.get("pnl", 0) if isinstance(res, dict) else 0
                            safe_print(f"CLOSED {sym} {pos['side']} qty={pos['qty']} entry={pos['entry']:.4f} exit={price:.4f} pnl={pnl:.4f}")
                            log_trade_close({"symbol":sym,"qty":pos['qty'],"entry":pos['entry'],"closed_price":price,"pnl":pnl,"side":pos['side'],"sl":pos['sl'],"tp":pos['tp']})
                            send_telegram(f"⚡ CLOSED {sym} {pos['side'].upper()} qty={pos['qty']} entry={pos['entry']:.4f} exit={price:.4f} pnl={pnl:.4f}")
                            continue
                    # open new position
                    if sym not in open_positions and params and usdt >= MIN_USDT_TO_TRADE:
                        stop_usd = params["stop_dist"]
                        qty = calc_qty_by_risk(params["price"], usdt, RISK_PER_TRADE, stop_usd)
                        if qty <= 0:
                            safe_print(sym, "qty 0 skip")
                            continue
                        qty = round(qty, 6)
                        resp = place_order(sym, params["side"], qty, params["price"], params["sl"], params["tp"])
                        log_trade_open(sym, params["side"], qty, params["price"], params["sl"], params["tp"])
                        send_telegram(f"🟢 OPEN {sym} {params['side'].upper()} qty={qty} entry={params['price']:.4f} SL={params['sl']:.4f} TP={params['tp']:.4f}")
                        safe_print(f"OPEN {sym} {params['side']} qty={qty} entry={params['price']:.4f}")
                except Exception as e_sym:
                    safe_print(f"Error {sym}:", e_sym)
                    safe_print(traceback.format_exc())
            # periodic report
            if time.time() - last_report > 60*10:
                total_pnl = 0.0
                if os.path.exists(PNL_FILE):
                    try:
                        dfp = pd.read_csv(PNL_FILE)
                        if "pnl" in dfp.columns:
                            total_pnl = dfp["pnl"].sum()
                    except:
                        total_pnl = 0.0
                send_telegram(f"📊 Report: Balance {get_usdt_balance():.2f} USDT | Closed PnL: {total_pnl:.2f} USDT | Open positions: {len(open_positions)}")
                last_report = time.time()
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        safe_print("User stopped bot.")
        send_telegram("⛔ Bot stopped by user")
    except Exception as e:
        safe_print("Main loop crashed:", e)
        safe_print(traceback.format_exc())
        send_telegram(f"⚠ Bot error: {e}")
        raise

# -------------------- Quick test --------------------
# -------------------- Web server (for Render / health checks) --------------------
from flask import Flask, jsonify
import threading

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"ok": True, "msg": "OKX trading bot service"}), 200

@app.route("/healthz")
def healthz():
    # можно дополнительно проверять доступность файлов или состояние (например len(open_positions))
    return jsonify({"ok": True, "open_positions": len(open_positions)}), 200

def start_flask():
    # Render предоставляет порт в переменной PORT
    port = int(os.getenv("PORT", "8000"))
    # host 0.0.0.0 чтобы принимать внешние запросы
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Запускаем бота в фоновом демоне
    t = threading.Thread(target=run_multi_bot, daemon=True, name="bot-thread")
    t.start()
    safe_print("Background bot thread started.")
    # Запускаем Flask в главном потоке (он не блокирует фон. поток)
    try:
        start_flask()
    except Exception as e:
        safe_print("Flask server error:", e)
        # если Flask упал — даём информацию в телеграм (если настроен)
        send_telegram(f"⚠ Flask crashed: {e}")
        raise
