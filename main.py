# BOT TRADING REAL – Bybit + Qwen3-VL-32B-Instruct (Basado en deepsel buenisimo)
# ==============================================================================
import os, time, requests, json, numpy as np, pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import json_repair
import base64
from openai import OpenAI
import hashlib
import hmac

# =================== CONFIGURACIÓN DE APIS ===================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("Falta SILICONFLOW_API_KEY")

SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
MODELO_VISION = "Qwen/Qwen3-VL-32B-Instruct"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("Faltan BYBIT_API_KEY o BYBIT_API_SECRET")

# =================== FUNCIONES BYBIT (REALES) ===================
def bybit_request(endpoint, method="GET", params=None, body=None):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    query_string = ""
    if params:
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    if body:
        body_str = json.dumps(body)
        payload = timestamp + BYBIT_API_KEY + recv_window + body_str
    else:
        payload = timestamp + BYBIT_API_KEY + recv_window + query_string
    signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}{endpoint}"
    if method == "GET":
        resp = requests.get(url, headers=headers, params=params)
    else:
        resp = requests.post(url, headers=headers, json=body)
    return resp.json()

def set_leverage():
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": "10", "sellLeverage": "10"}
        result = bybit_request("/v5/position/set-leverage", method="POST", body=body)
        ret_code = result.get('retCode')
        if ret_code == 0 or ret_code == 110043:
            print("✅ Apalancamiento 10x configurado")
        else:
            print(f"⚠️ Error configurando apalancamiento: {result}")
    except Exception as e:
        print(f"❌ Excepción configurando apalancamiento: {e}")

def get_real_balance():
    try:
        params = {"accountType": "UNIFIED", "coin": "USDT"}
        result = bybit_request("/v5/account/wallet-balance", method="GET", params=params)
        return float(result['result']['list'][0]['coin'][0]['walletBalance'])
    except Exception as e:
        print(f"❌ Error obteniendo saldo: {e}")
        return None

def get_free_margin():
    try:
        params = {"accountType": "UNIFIED"}
        result = bybit_request("/v5/account/wallet-balance", method="GET", params=params)
        if result.get('retCode') == 0:
            for coin in result['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    wallet = float(coin['walletBalance'])
                    used = float(coin.get('usedMargin', 0))
                    return wallet - used
    except Exception as e:
        print(f"❌ Error obteniendo margen libre: {e}")
    return 0.0

def get_real_position_size():
    try:
        params = {"category": "linear", "symbol": "BTCUSDT"}
        result = bybit_request("/v5/position/list", method="GET", params=params)
        if result.get('retCode') == 0:
            for pos in result['result']['list']:
                if pos['symbol'] == "BTCUSDT":
                    return abs(float(pos['size']))
        return 0.0
    except Exception as e:
        print(f"❌ Error get_real_position_size: {e}")
        return 0.0

def place_market_order(side, qty):
    try:
        body = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC"
        }
        result = bybit_request("/v5/order/create", method="POST", body=body)
        if result.get('retCode') == 0:
            return result['result']['orderId']
        else:
            print(f"❌ Error orden market: {result}")
            return None
    except Exception as e:
        print(f"❌ Excepción place_market_order: {e}")
        return None

def close_position_qty(qty, side_to_close):
    try:
        real_size = get_real_position_size()
        if real_size <= 0.0:
            print("⚠️ No hay posición real. Se omite cierre.")
            return "already_closed"
        qty_to_close = min(qty, real_size)
        if qty_to_close <= 0.0 or qty_to_close < 0.001:
            print(f"⚠️ Cantidad a cerrar ({qty_to_close}) menor al mínimo (0.001 BTC).")
            return "already_closed"
        close_side = "Sell" if side_to_close == "Buy" else "Buy"
        body = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "side": close_side,
            "orderType": "Market",
            "qty": str(round(qty_to_close, 3)),
            "timeInForce": "GTC",
            "reduceOnly": True
        }
        result = bybit_request("/v5/order/create", method="POST", body=body)
        if result.get('retCode') == 0:
            return result['result']['orderId']
        else:
            print(f"❌ Error close_position_qty: {result}")
            return None
    except Exception as e:
        print(f"❌ Excepción close_position_qty: {e}")
        return None

def sync_positions_with_bybit():
    global REAL_ACTIVE_TRADES
    try:
        real_size = get_real_position_size()
        if real_size == 0.0 and REAL_ACTIVE_TRADES:
            print("🧹 Sincronización: limpiando trades fantasmas.")
            REAL_ACTIVE_TRADES.clear()
            guardar_memoria()
        elif real_size > 0.0 and not REAL_ACTIVE_TRADES:
            print("⚠️ Hay posición real pero el bot no la registra.")
        else:
            print("✅ Sincronización OK")
    except Exception as e:
        print(f"❌ Error sync: {e}")

def force_sync_active_trades():
    global REAL_ACTIVE_TRADES
    real_size = get_real_position_size()
    if real_size == 0.0 and REAL_ACTIVE_TRADES:
        print("🧹 Sync forzada: limpiando fantasmas.")
        REAL_ACTIVE_TRADES.clear()
        guardar_memoria()

# ====== MEMORIA PERSISTENTE ======
MEMORY_FILE = "memoria_bot_real.json"

def convertir_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convertir_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convertir_serializable(item) for item in obj]
    return obj

def guardar_memoria():
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS
    active_trades_meta = {}
    for tid, t in REAL_ACTIVE_TRADES.items():
        active_trades_meta[tid] = {
            "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
            "razon": t.get("razon", ""), "tp1_ejecutado": t["tp1_ejecutado"],
            "tp2_ejecutado": t.get("tp2_ejecutado", False),
            "sl_actual": t.get("sl_actual"), "trailing_logic": t.get("trailing_logic", "EMA20"),
            "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante")
        }
    data = {
        "TRADE_HISTORY": TRADE_HISTORY,
        "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
        "REAL_BALANCE": REAL_BALANCE,
        "WIN_COUNT": WIN_COUNT,
        "LOSS_COUNT": LOSS_COUNT,
        "TOTAL_TRADES": TOTAL_TRADES,
        "ULTIMO_APRENDIZAJE": ULTIMO_APRENDIZAJE,
        "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS,
        "ACTIVE_TRADES_META": active_trades_meta,
        "ULTIMO_PROFIT_FACTOR": ULTIMO_PROFIT_FACTOR
    }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        print("💾 Memoria guardada")
    except Exception as e: print(f"Error guardando memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS, REAL_BALANCE, WIN_COUNT, LOSS_COUNT
    global TOTAL_TRADES, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, REAL_ACTIVE_TRADES
    if not os.path.exists(MEMORY_FILE): return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        TRADE_HISTORY = data.get("TRADE_HISTORY", [])
        REGLAS_APRENDIDAS = data.get("REGLAS_APRENDIDAS", REGLAS_APRENDIDAS)
        REAL_BALANCE = data.get("REAL_BALANCE", None)
        WIN_COUNT = data.get("WIN_COUNT", 0)
        LOSS_COUNT = data.get("LOSS_COUNT", 0)
        TOTAL_TRADES = data.get("TOTAL_TRADES", 0)
        ULTIMO_APRENDIZAJE = data.get("ULTIMO_APRENDIZAJE", 0)
        TOKENS_ACUMULADOS = data.get("TOKENS_ACUMULADOS", 0)
        ULTIMO_PROFIT_FACTOR = data.get("ULTIMO_PROFIT_FACTOR", 1.0)
        active_meta = data.get("ACTIVE_TRADES_META", {})
        for tid, meta in active_meta.items():
            REAL_ACTIVE_TRADES[int(tid)] = meta
        print(f"🧠 Memoria cargada. Trades: {TOTAL_TRADES}")
    except Exception as e: print(f"Error cargando memoria: {e}")

def parse_json_seguro(raw):
    if not raw or raw.strip() == "": return None
    try:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except: return None

# =================== CONFIGURACIÓN DEL BOT ===================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3

PCT_TP1, PCT_TP2 = 0.50, 0.30   # 50% TP1, 30% TP2, 20% restante a trailing

REAL_BALANCE = None
REAL_ACTIVE_TRADES = {}
TRADE_COUNTER = 0
WIN_COUNT = 0
LOSS_COUNT = 0
TOTAL_TRADES = 0
TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
DAILY_START_BALANCE = None
STOPPED_TODAY = False
CURRENT_DAY = None

ULTIMO_APRENDIZAJE = 0
ULTIMO_PROFIT_FACTOR = 1.0
REGLAS_APRENDIDAS = "Aún no hay lecciones. Busca confluencia."
TOKENS_ACUMULADOS = 0

# =================== COMUNICACIÓN TELEGRAM ===================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e: print(f"Error Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta_imagen, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
    except Exception as e: print(f"Error imagen Telegram: {e}")

def reporte_estado():
    pnl_global = REAL_BALANCE - (DAILY_START_BALANCE or REAL_BALANCE)
    winrate = (WIN_COUNT / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0
    mensaje = (
        f"📊 **ESTADO REAL BTC**\n"
        f"💰 Balance: {REAL_BALANCE:.2f} USDT\n"
        f"📈 PnL día: {pnl_global:+.2f} USDT\n"
        f"🎯 Winrate: {winrate:.1f}%\n"
        f"⚡ Activos: {len(REAL_ACTIVE_TRADES)}/{MAX_CONCURRENT_TRADES}\n"
        f"📐 PF (10t): {ULTIMO_PROFIT_FACTOR:.2f}"
    )
    telegram_mensaje(mensaje)

# =================== INDICADORES Y ZONAS (IDÉNTICO AL PAPER) ===================
def obtener_velas(limit=150):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=20)
        data = r.json()
        if data.get("retCode") != 0: return pd.DataFrame()
        lista = data.get("result")["list"][::-1]
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']: df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except: return pd.DataFrame()

def calcular_indicadores(df):
    if df.empty: return df
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    tr = pd.concat([(df['high']-df['low']), (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss)))
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2):
    if df.empty or len(df) < 40: return 0,0,0,0,"LATERAL","LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-120:]
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    tend = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    micro = 'SUBIENDO' if micro_slope > 0.2 else 'CAYENDO' if micro_slope < -0.2 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tend, micro

def analizar_anatomia_vela(v):
    rango = v['high'] - v['low']
    if rango == 0: return "Doji Plano"
    c_pct = (abs(v['close'] - v['open']) / rango) * 100
    s_sup = ((v['high'] - max(v['close'], v['open'])) / rango) * 100
    s_inf = ((min(v['close'], v['open']) - v['low']) / rango) * 100
    color = "VERDE" if v['close'] > v['open'] else "ROJA"
    return f"{color} (C:{c_pct:.0f}%|MS:{s_sup:.0f}%|MI:{s_inf:.0f}%)"

def analizar_patrones_conjuntos(df, idx):
    if idx < 3: return "Consolidación"
    v3, v2, v1 = df.iloc[idx], df.iloc[idx-1], df.iloc[idx-2]
    verde3, verde2, verde1 = v3['close'] > v3['open'], v2['close'] > v2['open'], v1['close'] > v1['open']
    patrones = []
    if not verde1 and verde3 and v3['close'] > (v1['open']+v1['close'])/2: patrones.append("ESTRELLA MAÑANA")
    if verde1 and not verde3 and v3['close'] < (v1['open']+v1['close'])/2: patrones.append("ESTRELLA ATARDECER")
    if verde1 and verde2 and verde3 and v3['close'] > v2['close']: patrones.append("3 SOLDADOS")
    if not verde2 and verde3 and v3['close'] > v2['open']: patrones.append("ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open']: patrones.append("ENVOLVENTE BAJISTA")
    return " | ".join(patrones) if patrones else "Sin patrón claro"

def generar_descripcion_nison(df, idx=-2):
    v = df.iloc[idx]
    sop, res, slope, inter, tend, micro = detectar_zonas_mercado(df, idx)
    desc = f"""
PRECIO: {v['close']:.2f} | EMA20: {v['ema20']:.2f} | ATR: {v['atr']:.2f}
SOP: {sop:.2f} | RES: {res:.2f} | TEND: {tend} | MICRO: {micro}
ANATOMÍA: {analizar_anatomia_vela(v)}
PATRONES: {analizar_patrones_conjuntos(df, idx)}
"""
    return desc, v['atr']

def generar_grafico_para_vision(df, soporte, resistencia, slope, intercept, precio):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    fig, ax = plt.subplots(figsize=(16,8))
    x = np.arange(len(df_plot))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def analizar_con_qwen(descripcion_texto, atr, reglas, imagen):
    global TOKENS_ACUMULADOS
    try:
        img_b64 = pil_to_base64(imagen)
        prompt = f"""
Eres Trader Senior. Mira el gráfico (Cian=Sop, Magenta=Res, Amarillo=EMA20).
Define niveles VISUALES de salida. 
1. sl_price: Invalidación real.
2. tp1_price: Obstáculo cercano.
3. tp2_price: Nivel de liquidez lejano.
4. trailing_logic: "EMA20" o "LOW_CANDLE".

JSON (una línea):
{{"decision":"Buy/Sell/Hold","razon":"texto","sl_price":0.0,"tp1_price":0.0,"tp2_price":0.0,"trailing_logic":"EMA20/LOW_CANDLE"}}

DATOS: {descripcion_texto}
MEMORIA: {reglas}
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role":"user", "content":[{"type":"text","text":prompt}, {"type":"image_url","image_url":{"url":img_b64}}]}],
            temperature=0.1
        )
        TOKENS_ACUMULADOS += response.usage.total_tokens
        datos = parse_json_seguro(response.choices[0].message.content)
        if not datos: return "Hold", "", 0, 0, 0, "EMA20"
        return datos.get("decision","Hold"), datos.get("razon",""), datos.get("sl_price"), datos.get("tp1_price"), datos.get("tp2_price"), datos.get("trailing_logic","EMA20")
    except Exception as e:
        print(f"❌ Error en IA: {e}")
        return "Hold", "Error API", 0, 0, 0, "EMA20"

# =================== GESTIÓN REAL (APERTURA Y CIERRE) ===================
def real_abrir_posicion(decision, precio, atr, razon, sl_ia, tp1_ia, tp2_ia, logic_ia, df, sop, res, slo, inter):
    global REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES, TOTAL_TRADES
    if len(REAL_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES:
        print(f"⚠️ Máximo {MAX_CONCURRENT_TRADES} trades alcanzado.")
        return

    if REAL_BALANCE is None:
        REAL_BALANCE = get_real_balance()
        if REAL_BALANCE is None:
            return

    # Calcular stop loss final
    sl_final = float(sl_ia) if sl_ia else (precio - atr*1.5 if decision=="Buy" else precio + atr*1.5)
    if decision == "Buy" and sl_final >= precio:
        sl_final = precio - atr
    if decision == "Sell" and sl_final <= precio:
        sl_final = precio + atr

    distancia = abs(precio - sl_final)
    risk_amount = REAL_BALANCE * RISK_PER_TRADE
    qty_btc = risk_amount / distancia
    max_qty = (REAL_BALANCE * LEVERAGE) / precio
    qty_btc = min(qty_btc, max_qty)

    if qty_btc <= 0:
        print("⚠️ Cantidad a operar demasiado pequeña.")
        return

    qty_btc = round(qty_btc, 3)

    # Verificar mínimos de exchange
    if qty_btc * precio < 100.0:
        qty_btc = round(100.0 / precio, 3)
        print(f"⚠️ Ajustando a nocional mínimo: {qty_btc} BTC")
    if qty_btc < 0.001:
        print("⚠️ Mínimo 0.001 BTC no alcanzado.")
        return

    # Verificar margen disponible
    margen_necesario = (qty_btc * precio) / LEVERAGE
    free_margin = get_free_margin()
    if free_margin < margen_necesario:
        print(f"⚠️ Margen insuficiente. Libre: {free_margin:.2f} | Necesario: {margen_necesario:.2f}")
        return

    # Ejecutar orden real
    order_id = place_market_order(decision, qty_btc)
    if not order_id:
        print("❌ No se pudo abrir la orden.")
        return

    TRADE_COUNTER += 1
    t = {
        "id": TRADE_COUNTER, "decision": decision, "entrada": precio,
        "sl_inicial": sl_final, "sl_actual": sl_final,
        "tp1": tp1_ia, "tp2": tp2_ia, "trailing_logic": logic_ia,
        "qty_original": qty_btc, "qty_restante": qty_btc,
        "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0,
        "razon": razon, "order_id": order_id
    }
    REAL_ACTIVE_TRADES[TRADE_COUNTER] = t
    msg = f"🚀 [#{TRADE_COUNTER}] {decision} REAL en {precio:.2f} | Qty {qty_btc} BTC\nRazon: {razon}"
    print(msg)
    telegram_mensaje(msg)

    # Gráfico simple para Telegram
    fig, ax = plt.subplots()
    ax.plot(df.tail(20)['close'].values)
    plt.savefig("/tmp/in.png")
    plt.close()
    telegram_enviar_imagen("/tmp/in.png", msg)

def real_revisar_sl_tp(df):
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES
    if not REAL_ACTIVE_TRADES:
        return

    force_sync_active_trades()  # Limpia fantasmas antes de revisar
    if not REAL_ACTIVE_TRADES:
        return

    c = df['close'].iloc[-1]
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    ema = df['ema20'].iloc[-1]
    atr = df['atr'].iloc[-1]
    h_prev = df['high'].iloc[-2]
    l_prev = df['low'].iloc[-2]

    cerrar_ids = []
    for tid, t in list(REAL_ACTIVE_TRADES.items()):
        # TP1 (50%)
        if not t['tp1_ejecutado'] and t['tp1'] is not None and t['tp1'] > 0:
            if (t['decision']=="Buy" and h >= t['tp1']) or (t['decision']=="Sell" and l <= t['tp1']):
                qty_cerrar = round(t['qty_original'] * PCT_TP1, 3)
                if qty_cerrar < 0.001:
                    qty_cerrar = 0.001
                if qty_cerrar > 0 and qty_cerrar <= t['qty_restante']:
                    result = close_position_qty(qty_cerrar, t['decision'])
                    if result == "already_closed":
                        t['qty_restante'] = 0
                        t['tp1_ejecutado'] = True
                        cerrar_ids.append(tid)
                        continue
                    elif result:
                        ganancia = abs(t['tp1'] - t['entrada']) * qty_cerrar
                        t['pnl_parcial'] += ganancia
                        REAL_BALANCE = get_real_balance()
                        t['qty_restante'] -= qty_cerrar
                        t['tp1_ejecutado'] = True
                        t['sl_actual'] = t['entrada']  # breakeven
                        print(f"🎯 TP1 real #{tid} +{ganancia:.2f} USDT")
                        telegram_mensaje(f"🎯 TP1 #{tid} hit. SL a breakeven.")
                    else:
                        print(f"❌ Falló cierre TP1 #{tid}")

        # TP2 (30%)
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2'] is not None and t['tp2'] > 0:
            if (t['decision']=="Buy" and h >= t['tp2']) or (t['decision']=="Sell" and l <= t['tp2']):
                qty_cerrar = round(t['qty_original'] * PCT_TP2, 3)
                if qty_cerrar < 0.001:
                    qty_cerrar = 0.001
                if qty_cerrar > 0 and qty_cerrar <= t['qty_restante']:
                    result = close_position_qty(qty_cerrar, t['decision'])
                    if result == "already_closed":
                        t['qty_restante'] = 0
                        t['tp2_ejecutado'] = True
                        cerrar_ids.append(tid)
                        continue
                    elif result:
                        ganancia = abs(t['tp2'] - t['entrada']) * qty_cerrar
                        t['pnl_parcial'] += ganancia
                        REAL_BALANCE = get_real_balance()
                        t['qty_restante'] -= qty_cerrar
                        t['tp2_ejecutado'] = True
                        print(f"🎯 TP2 real #{tid} +{ganancia:.2f} USDT")
                        telegram_mensaje(f"🎯 TP2 #{tid} hit.")
                    else:
                        print(f"❌ Falló cierre TP2 #{tid}")

        # Trailing / SL
        cerrar = False
        motivo = ""
        if t['tp1_ejecutado']:
            # Actualizar trailing
            if t['trailing_logic'] == "EMA20":
                nuevo_sl = ema - (atr * 0.2) if t['decision'] == "Buy" else ema + (atr * 0.2)
            else:  # LOW_CANDLE
                nuevo_sl = l_prev if t['decision'] == "Buy" else h_prev
            if t['decision'] == "Buy":
                if nuevo_sl > t['sl_actual']:
                    t['sl_actual'] = nuevo_sl
                if l <= t['sl_actual']:
                    cerrar, motivo = True, "Trailing"
            else:
                if nuevo_sl < t['sl_actual']:
                    t['sl_actual'] = nuevo_sl
                if h >= t['sl_actual']:
                    cerrar, motivo = True, "Trailing"
        else:
            if (t['decision'] == "Buy" and l <= t['sl_inicial']) or (t['decision'] == "Sell" and h >= t['sl_inicial']):
                cerrar, motivo = True, "Stop Loss"

        if cerrar and t['qty_restante'] > 0:
            real_size = get_real_position_size()
            if real_size <= 0.0:
                cerrar_ids.append(tid)
                continue
            qty_to_close = min(t['qty_restante'], real_size)
            if qty_to_close < 0.001:
                cerrar_ids.append(tid)
                continue
            result = close_position_qty(qty_to_close, t['decision'])
            if result == "already_closed" or result:
                close_price = df['close'].iloc[-1]
                pnl_resto = (close_price - t['entrada']) * qty_to_close if t['decision'] == "Buy" else (t['entrada'] - close_price) * qty_to_close
                pnl_total = t['pnl_parcial'] + pnl_resto
                REAL_BALANCE = get_real_balance()
                TOTAL_TRADES += 1
                if pnl_total > 0:
                    WIN_COUNT += 1
                else:
                    LOSS_COUNT += 1
                TRADE_HISTORY.append(convertir_serializable({
                    "pnl": pnl_total, "resultado_win": pnl_total > 0, "decision": t['decision'], "razon": t['razon']
                }))
                cerrar_ids.append(tid)
                print(f"📤 CERRADO #{tid} ({motivo}) | PnL: {pnl_total:.2f} USDT")
                telegram_mensaje(f"📤 CERRADO #{tid} ({motivo}). PnL: {pnl_total:.2f} USDT")
                reporte_estado()
            else:
                print(f"❌ Falló cierre final #{tid}")

    for tid in cerrar_ids:
        del REAL_ACTIVE_TRADES[tid]

    if len(TRADE_HISTORY) > 0 and len(TRADE_HISTORY) % 10 == 0:
        aprender_de_trades()

# =================== AUTOAPRENDIZAJE (IDÉNTICO AL PAPER) ===================
def aprender_de_trades():
    global REGLAS_APRENDIDAS, ULTIMO_APRENDIZAJE, ULTIMO_PROFIT_FACTOR
    try:
        ult = TRADE_HISTORY[-10:]
        gan = sum(t['pnl'] for t in ult if t['pnl']>0)
        per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
        ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
        ult_serial = convertir_serializable(ult)
        prompt = f"Analiza estos 10 trades reales y dame una lección corta: {json.dumps(ult_serial)}"
        resp = client.chat.completions.create(model=MODELO_VISION, messages=[{"role":"user","content":prompt}])
        REGLAS_APRENDIDAS = resp.choices[0].message.content
        print(f"🧠 APRENDIZAJE: {REGLAS_APRENDIDAS}")
        telegram_mensaje(f"🧠 APRENDIZAJE: {REGLAS_APRENDIDAS}")
        ULTIMO_APRENDIZAJE = TOTAL_TRADES
        guardar_memoria()
    except Exception as e:
        print(f"❌ Error en aprendizaje: {e}")

# =================== RISK MANAGEMENT DIARIO ===================
def risk_management_check():
    global DAILY_START_BALANCE, STOPPED_TODAY, CURRENT_DAY, REAL_BALANCE
    hoy = datetime.now(timezone.utc).date()
    if CURRENT_DAY != hoy:
        CURRENT_DAY = hoy
        if REAL_BALANCE is None:
            REAL_BALANCE = get_real_balance()
        DAILY_START_BALANCE = REAL_BALANCE
        STOPPED_TODAY = False
        print(f"📅 Nuevo día: {hoy}. Balance inicial: {DAILY_START_BALANCE:.2f}")
    if REAL_BALANCE is not None and DAILY_START_BALANCE is not None:
        drawdown = (REAL_BALANCE - DAILY_START_BALANCE) / DAILY_START_BALANCE
        if drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
            STOPPED_TODAY = True
            print(f"🚨 Drawdown diario superado ({MAX_DAILY_DRAWDOWN_PCT*100}%). Operaciones detenidas hasta mañana.")
    return not STOPPED_TODAY

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global REAL_BALANCE, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, TRADE_HISTORY, REAL_ACTIVE_TRADES
    cargar_memoria()
    sync_positions_with_bybit()   # Limpia fantasmas al inicio
    set_leverage()
    REAL_BALANCE = get_real_balance()
    if REAL_BALANCE is None:
        print("❌ No se pudo obtener saldo real. Abortando.")
        return
    print(f"🤖 BOT REAL V99.43 INICIADO - Balance: {REAL_BALANCE:.2f} USDT - Max trades: {MAX_CONCURRENT_TRADES}")
    telegram_mensaje(f"🤖 Bot Real Online - Balance: {REAL_BALANCE:.2f} USDT")

    ultima_vela = None
    iteracion = 0
    while True:
        try:
            iteracion += 1
            df_raw = obtener_velas()
            if df_raw.empty:
                time.sleep(SLEEP_SECONDS)
                continue
            df = calcular_indicadores(df_raw)
            if df.empty:
                time.sleep(SLEEP_SECONDS)
                continue

            precio_actual = df['close'].iloc[-1]
            REAL_BALANCE = get_real_balance()
            vela_c = df.index[-2]
            if ultima_vela is None:
                ultima_vela = vela_c

            if len(REAL_ACTIVE_TRADES) < MAX_CONCURRENT_TRADES and ultima_vela != vela_c:
                if risk_management_check():
                    sop, res, slo, inter, t, m = detectar_zonas_mercado(df)
                    desc, atr = generar_descripcion_nison(df)
                    img = generar_grafico_para_vision(df, sop, res, slo, inter, precio_actual)
                    dec, raz, sl, tp1, tp2, log = analizar_con_qwen(desc, atr, REGLAS_APRENDIDAS, img)
                    print(f"🤖 Decisión IA: {dec}")
                    if dec in ["Buy","Sell"]:
                        real_abrir_posicion(dec, precio_actual, atr, raz, sl, tp1, tp2, log, df, sop, res, slo, inter)
                    else:
                        print(f"⏸️ IA decidió HOLD. Motivo: {raz[:100]}")
                ultima_vela = vela_c
            else:
                print(f"⏳ Misma vela o límite de trades alcanzado. Activos: {len(REAL_ACTIVE_TRADES)}/{MAX_CONCURRENT_TRADES}")

            if REAL_ACTIVE_TRADES:
                print("🔎 Revisando trades activos...")
                real_revisar_sl_tp(df)

            if iteracion % 10 == 0:
                winrate = (WIN_COUNT / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0
                print(f"📈 RESUMEN: Balance={REAL_BALANCE:.2f} | Trades={TOTAL_TRADES} | Winrate={winrate:.1f}% | PF={ULTIMO_PROFIT_FACTOR:.2f}")

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
