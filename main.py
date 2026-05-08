# BOT TRADING REAL – Bybit + Qwen3-VL-32B-Instruct
# Versión con riesgo ultrapequeño (0.01-0.5 USDT) ajustado automáticamente al margen libre.
# Apalancamiento fijo 34x. Sin filtros de calidad de entrada (solo la IA decide).
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

# =================== FUNCIONES BYBIT ===================
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
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": "34", "sellLeverage": "34"}
        result = bybit_request("/v5/position/set-leverage", method="POST", body=body)
        ret_code = result.get('retCode')
        if ret_code == 0 or ret_code == 110043:
            print("✅ Apalancamiento 34x configurado")
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
            return "already_closed"
        qty_to_close = min(qty, real_size)
        if qty_to_close <= 0.0 or qty_to_close < 0.001:
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
            print(f"✅ Orden de cierre enviada: {qty_to_close} BTC")
            return result['result']['orderId']
        else:
            print(f"❌ Error close_position_qty: {result}")
            return None
    except Exception as e:
        print(f"❌ Excepción close_position_qty: {e}")
        return None

def close_position_qty_confirm(qty, side_to_close, max_wait=5):
    size_before = get_real_position_size()
    if size_before <= 0:
        return "already_closed"
    qty_to_close = min(qty, size_before)
    if qty_to_close < 0.001:
        return "already_closed"
    order_id = close_position_qty(qty_to_close, side_to_close)
    if not order_id or order_id == "already_closed":
        return None
    for _ in range(max_wait * 2):
        time.sleep(0.5)
        size_after = get_real_position_size()
        if size_before - size_after >= qty_to_close * 0.99:
            print(f"✅ Confirmada reducción: {size_before:.4f} -> {size_after:.4f}")
            return order_id
    print(f"❌ No se confirmó reducción tras {max_wait}s. antes={size_before:.4f}, después={size_after:.4f}")
    return None

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
            "sl_actual": t.get("sl_actual"), "trailing_logic": t.get("trailing_logic", "BREAKEVEN"),
            "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
            "breakeven_activado": t.get("breakeven_activado", False)
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
RISK_PER_TRADE_MAX = 3.0      # Riesgo máximo deseado (se reducirá si no cabe)
RISK_PER_TRADE_MIN = 0.01     # Riesgo mínimo aceptable (0.01 USDT)
LEVERAGE = 34
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 1
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5
MIN_RISK_REWARD_TP1 = 1.5
MIN_RISK_REWARD_TP2 = 2.5

# Límites de distancia al stop loss (como fracción del precio)
MIN_SL_DIST_PCT = 0.0015      # 0.15% mínimo
MAX_SL_DIST_PCT = 0.005       # 0.5% máximo

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

# =================== TELEGRAM ===================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        if len(texto) > 4000:
            texto = texto[:4000]
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
    if REAL_BALANCE is None:
        return
    pnl_global = REAL_BALANCE - (DAILY_START_BALANCE or REAL_BALANCE)
    winrate = (WIN_COUNT / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0
    max_din = get_dynamic_max_trades()
    mensaje = (
        f"📊 **ESTADO REAL BTC**\n"
        f"💰 Balance: {REAL_BALANCE:.2f} USDT\n"
        f"📈 PnL día: {pnl_global:+.2f} USDT\n"
        f"🎯 Winrate: {winrate:.1f}%\n"
        f"⚡ Activos: {len(REAL_ACTIVE_TRADES)}/{max_din}\n"
        f"📐 PF (10t): {ULTIMO_PROFIT_FACTOR:.2f}"
    )
    telegram_mensaje(mensaje)

# =================== INDICADORES Y ZONAS ===================
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
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
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
    if slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', linewidth=2, label='Tendencia', alpha=0.7)
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

def analizar_con_qwen(imagen):
    global TOKENS_ACUMULADOS
    try:
        img_b64 = pil_to_base64(imagen)
        prompt = """
Eres un trader experto que analiza gráficos de velas japonesas de BTCUSDT (timeframe 5 minutos).
Observa el gráfico que se te muestra. Contiene:
- Velas japonesas (verdes alcistas, rojas bajistas) con cuerpos y mechas.
- Línea cyan horizontal: SOPORTE.
- Línea magenta horizontal: RESISTENCIA.
- Línea amarilla: EMA20.
- Línea blanca discontinua: TENDENCIA LINEAL (regresión).

Analiza visualmente:
- El tamaño de los cuerpos y las mechas (rechazos).
- La distancia del precio actual al soporte, resistencia y EMA20.
- Si la EMA20 actúa como soporte o resistencia dinámica.
- La pendiente de la tendencia (alcista, bajista, lateral).

NO utilices patrones de velas (como envolventes, estrellas, etc.). Solo analiza la acción del precio en bruto.

Decide si es momento de COMPRAR (Buy), VENDER (Sell) o NO HACER NADA (Hold).
Define niveles concretos de salida basados en el gráfico:
- sl_price: Precio donde la operación se invalida (justo debajo del soporte si es Buy, o encima de la resistencia si es Sell). Debe ser ajustado (distancia entre 0.15% y 0.5% del precio).
- tp1_price: Primer objetivo (un nivel visible donde el precio pueda encontrar resistencia si es Buy, o soporte si es Sell). Debe estar a una distancia mínima de 1.5 veces la distancia del stop loss.
- tp2_price: Segundo objetivo (más lejano, al menos 2.5 veces la distancia del stop loss).
- trailing_logic: Siempre "BREAKEVEN".

Responde ÚNICAMENTE con un JSON en una línea, sin texto adicional, con esta estructura:
{"decision":"Buy/Sell/Hold","razon":"texto muy corto","sl_price":0.0,"tp1_price":0.0,"tp2_price":0.0,"trailing_logic":"BREAKEVEN"}
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role":"user", "content":[{"type":"text","text":prompt}, {"type":"image_url","image_url":{"url":img_b64}}]}],
            temperature=0.1
        )
        TOKENS_ACUMULADOS += response.usage.total_tokens
        datos = parse_json_seguro(response.choices[0].message.content)
        if not datos:
            return "Hold", "Error parsing", 0, 0, 0, "BREAKEVEN"
        return datos.get("decision","Hold"), datos.get("razon",""), datos.get("sl_price"), datos.get("tp1_price"), datos.get("tp2_price"), "BREAKEVEN"
    except Exception as e:
        print(f"❌ Error en IA: {e}")
        return "Hold", "Error API", 0, 0, 0, "BREAKEVEN"

# =================== FUNCIONES AUXILIARES ===================
def sync_active_trades_with_bybit():
    global REAL_ACTIVE_TRADES
    real_size = get_real_position_size()
    if real_size == 0.0 and REAL_ACTIVE_TRADES:
        print("🧹 Sincronización: No hay posición real. Limpiando trades fantasmas.")
        REAL_ACTIVE_TRADES.clear()
        guardar_memoria()
    elif real_size > 0.0 and not REAL_ACTIVE_TRADES:
        print("⚠️ Hay posición real pero el bot no la registra. Se recomienda cerrar manualmente.")
    else:
        mem_size = sum(t['qty_restante'] for t in REAL_ACTIVE_TRADES.values())
        if abs(mem_size - real_size) > 0.002:
            print(f"⚠️ Discrepancia de tamaño: memoria {mem_size:.3f} BTC, real {real_size:.3f} BTC. Reconstruyendo...")
            if REAL_ACTIVE_TRADES:
                tid = list(REAL_ACTIVE_TRADES.keys())[0]
                REAL_ACTIVE_TRADES[tid]['qty_restante'] = real_size
                for other in list(REAL_ACTIVE_TRADES.keys())[1:]:
                    del REAL_ACTIVE_TRADES[other]
                guardar_memoria()

def get_dynamic_max_trades():
    if REAL_BALANCE is None:
        return 1
    max_by_balance = int(REAL_BALANCE // MIN_MARGIN_PER_TRADE)
    if max_by_balance < 1:
        max_by_balance = 1
    return min(MAX_CONCURRENT_TRADES, max_by_balance)

# =================== APERTURA DE OPERACIONES CON RIESGO ULTRAPEQUEÑO ===================
def real_abrir_posicion(decision, precio, razon, sl_ia, tp1_ia, tp2_ia, logic_ia, df, sop, res, slo, inter):
    global REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES
    max_trades = get_dynamic_max_trades()
    if len(REAL_ACTIVE_TRADES) >= max_trades:
        print(f"⚠️ Máximo dinámico de trades ({max_trades}) alcanzado.")
        return

    if REAL_BALANCE is None:
        REAL_BALANCE = get_real_balance()
        if REAL_BALANCE is None:
            return

    free_margin = get_free_margin()
    if free_margin <= 0:
        print("❌ Margen libre insuficiente o nulo.")
        return

    # 1. Obtener distancia al stop loss (priorizar IA, luego límites)
    atr_valor = df['atr'].iloc[-1] if not df.empty and 'atr' in df else precio * 0.003
    min_dist_usd = max(precio * MIN_SL_DIST_PCT, atr_valor * 0.5)
    max_dist_usd = min(precio * MAX_SL_DIST_PCT, atr_valor * 1.5)
    
    if decision == "Buy":
        distancia_prop = (precio - sl_ia) if sl_ia and sl_ia > 0 else min_dist_usd
    else:
        distancia_prop = (sl_ia - precio) if sl_ia and sl_ia > 0 else min_dist_usd
    distancia_final = max(min_dist_usd, min(distancia_prop, max_dist_usd))
    
    # 2. Calcular el riesgo máximo que permite el margen libre para esta distancia
    riesgo_maximo_posible = (free_margin * distancia_final * LEVERAGE) / precio
    # Tomar el mínimo entre el riesgo máximo deseado y el que cabe
    riesgo_usar = min(RISK_PER_TRADE_MAX, riesgo_maximo_posible)
    # Asegurar que no sea menor que el mínimo permitido
    if riesgo_usar < RISK_PER_TRADE_MIN:
        print(f"❌ Riesgo calculado {riesgo_usar:.4f} USDT es menor al mínimo {RISK_PER_TRADE_MIN}. No se abre.")
        return
    
    # 3. Calcular cantidad teórica
    qty_btc = riesgo_usar / distancia_final
    
    # 4. Ajustar por nocional mínimo de Bybit (100 USDT)
    nocional = qty_btc * precio
    if nocional < 100.0:
        qty_btc_necesaria = 100.0 / precio
        # El riesgo real aumentará porque la cantidad es mayor
        riesgo_real = qty_btc_necesaria * distancia_final
        if riesgo_real > RISK_PER_TRADE_MAX:
            print(f"⚠️ Para cumplir nocional mínimo, el riesgo sería {riesgo_real:.2f} USDT (> máx {RISK_PER_TRADE_MAX}). Abortando.")
            return
        # Verificar que el margen para esta nueva cantidad no supere el libre
        margen_necesario = (qty_btc_necesaria * precio) / LEVERAGE
        if margen_necesario > free_margin + 0.01:
            print(f"❌ Nocional mínimo requiere margen {margen_necesario:.2f} > libre {free_margin:.2f}. Cancelado.")
            return
        qty_btc = qty_btc_necesaria
        riesgo_usar = riesgo_real
        print(f"⚠️ Ajustado por nocional mínimo: qty={qty_btc:.4f} BTC, riesgo real={riesgo_usar:.4f} USDT")
    
    # 5. Validar cantidad mínima (0.001 BTC)
    if qty_btc < 0.001:
        print(f"⚠️ Cantidad {qty_btc:.4f} BTC es menor a 0.001. No se abre.")
        return
    
    qty_btc = round(qty_btc, 3)
    
    # 6. Calcular SL final
    if decision == "Buy":
        sl_ajustado = precio - distancia_final
    else:
        sl_ajustado = precio + distancia_final
        
    if (decision == "Buy" and sl_ajustado >= precio) or (decision == "Sell" and sl_ajustado <= precio):
        print("⚠️ Stop loss inválido. No se abre.")
        return
    
    # 7. Recalcular TP1 y TP2 con ratios mínimos
    tp1_min_dist = distancia_final * MIN_RISK_REWARD_TP1
    tp2_min_dist = distancia_final * MIN_RISK_REWARD_TP2
    
    if decision == "Buy":
        tp1_original = tp1_ia if tp1_ia and tp1_ia > precio else precio + tp1_min_dist
        tp2_original = tp2_ia if tp2_ia and tp2_ia > tp1_original else precio + tp2_min_dist
        tp1_ajustado = max(tp1_original, precio + tp1_min_dist)
        tp2_ajustado = max(tp2_original, precio + tp2_min_dist)
        if tp2_ajustado <= tp1_ajustado:
            tp2_ajustado = tp1_ajustado + distancia_final * 0.5
    else:
        tp1_original = tp1_ia if tp1_ia and tp1_ia < precio else precio - tp1_min_dist
        tp2_original = tp2_ia if tp2_ia and tp2_ia < tp1_original else precio - tp2_min_dist
        tp1_ajustado = min(tp1_original, precio - tp1_min_dist)
        tp2_ajustado = min(tp2_original, precio - tp2_min_dist)
        if tp2_ajustado >= tp1_ajustado:
            tp2_ajustado = tp1_ajustado - distancia_final * 0.5
    
    # 8. Abrir orden
    order_id = place_market_order(decision, qty_btc)
    if not order_id:
        print("❌ No se pudo abrir la orden.")
        return
    
    TRADE_COUNTER += 1
    qty_tp1 = round(qty_btc * TP1_PERCENT, 3)
    qty_restante = round(qty_btc - qty_tp1, 3)
    t = {
        "id": TRADE_COUNTER, "decision": decision, "entrada": precio,
        "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
        "tp1": tp1_ajustado, "tp2": tp2_ajustado, "trailing_logic": "BREAKEVEN",
        "qty_original": qty_btc, "qty_restante": qty_restante,
        "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0,
        "razon": razon, "order_id": order_id, "breakeven_activado": False
    }
    REAL_ACTIVE_TRADES[TRADE_COUNTER] = t
    
    msg = (f"🚀 [#{TRADE_COUNTER}] {decision} en {precio:.2f}\n"
           f"💰 Riesgo efectivo: {riesgo_usar:.3f} USDT | SL dist: {distancia_final:.1f} USD\n"
           f"📊 Qty: {qty_btc} BTC | Margen: {(qty_btc*precio)/LEVERAGE:.2f} USDT\n"
           f"🎯 SL: {sl_ajustado:.2f} | TP1: {tp1_ajustado:.2f} | TP2: {tp2_ajustado:.2f}\n"
           f"📝 {razon}")
    print(msg)
    telegram_mensaje(msg)
    
    img_completa = generar_grafico_para_vision(df, sop, res, slo, inter, precio)
    img_completa.save("/tmp/in_completo.png")
    telegram_enviar_imagen("/tmp/in_completo.png", msg)

# =================== GESTIÓN DE TRADES ACTIVOS (SL/TP) ===================
def real_revisar_sl_tp(df):
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES
    sync_active_trades_with_bybit()
    if not REAL_ACTIVE_TRADES:
        return

    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]

    cerrar_ids = []
    for tid, t in list(REAL_ACTIVE_TRADES.items()):
        # TP1 (50% de la posición)
        if not t['tp1_ejecutado'] and t['tp1'] is not None and t['tp1'] > 0:
            if (t['decision']=="Buy" and h >= t['tp1']) or (t['decision']=="Sell" and l <= t['tp1']):
                qty_tp1 = round(t['qty_original'] * TP1_PERCENT, 3)
                if qty_tp1 >= 0.001 and t['qty_restante'] > 0:
                    result = close_position_qty_confirm(qty_tp1, t['decision'])
                    if result and result != "already_closed":
                        pnl_parcial = (t['tp1'] - t['entrada']) * qty_tp1 if t['decision']=="Buy" else (t['entrada'] - t['tp1']) * qty_tp1
                        t['pnl_parcial'] += pnl_parcial
                        t['qty_restante'] = round(t['qty_original'] - qty_tp1, 3)
                        t['tp1_ejecutado'] = True
                        t['breakeven_activado'] = True
                        offset = 2.0
                        t['sl_actual'] = t['entrada'] - offset if t['decision']=="Buy" else t['entrada'] + offset
                        print(f"🎯 TP1 #{tid}: +{pnl_parcial:.2f} USDT. Restante {t['qty_restante']} BTC, SL breakeven en {t['sl_actual']:.2f}")
                        telegram_mensaje(f"🎯 TP1 #{tid}: +{pnl_parcial:.2f} USDT (cerrado {qty_tp1} BTC)")
                        if t['qty_restante'] <= 0.0001:
                            cerrar_ids.append(tid)
                    else:
                        print(f"⚠️ TP1 no confirmado #{tid}")
                else:
                    cerrar_ids.append(tid)

        # TP2
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2'] is not None and t['tp2'] > 0 and t['qty_restante'] > 0:
            if (t['decision']=="Buy" and h >= t['tp2']) or (t['decision']=="Sell" and l <= t['tp2']):
                qty_restante = t['qty_restante']
                if qty_restante >= 0.001:
                    result = close_position_qty_confirm(qty_restante, t['decision'])
                    if result and result != "already_closed":
                        pnl_resto = (t['tp2'] - t['entrada']) * qty_restante if t['decision']=="Buy" else (t['entrada'] - t['tp2']) * qty_restante
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
                        msg = f"✅ CIERRE COMPLETO #{tid} por TP2\nPnL: {pnl_total:+.2f} USDT\nBalance: {REAL_BALANCE:.2f}"
                        print(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                    else:
                        print(f"❌ Falló cierre TP2 #{tid}")
                else:
                    cerrar_ids.append(tid)

        # Stop Loss
        if t['qty_restante'] > 0.001:
            condicion_stop = False
            if t['decision'] == "Buy" and l <= t['sl_actual']:
                condicion_stop = True
            elif t['decision'] == "Sell" and h >= t['sl_actual']:
                condicion_stop = True

            if condicion_stop:
                qty_restante = t['qty_restante']
                if qty_restante >= 0.001:
                    result = close_position_qty_confirm(qty_restante, t['decision'])
                    if result and result != "already_closed":
                        pnl_resto = (t['sl_actual'] - t['entrada']) * qty_restante if t['decision']=="Buy" else (t['entrada'] - t['sl_actual']) * qty_restante
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
                        motivo = "SL inicial" if not t.get('breakeven_activado') else "Breakeven"
                        msg = f"❌ CIERRE #{tid} por {motivo}\nPnL: {pnl_total:+.2f} USDT"
                        print(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                    else:
                        print(f"❌ Falló cierre stop #{tid}")
                else:
                    cerrar_ids.append(tid)

    for tid in cerrar_ids:
        del REAL_ACTIVE_TRADES[tid]

    if TOTAL_TRADES > 0 and TOTAL_TRADES % 10 == 0 and TOTAL_TRADES != ULTIMO_APRENDIZAJE:
        aprender_de_trades()

# =================== AUTOAPRENDIZAJE ===================
def aprender_de_trades():
    global REGLAS_APRENDIDAS, ULTIMO_APRENDIZAJE, ULTIMO_PROFIT_FACTOR
    try:
        ult = TRADE_HISTORY[-10:]
        gan = sum(t['pnl'] for t in ult if t['pnl']>0)
        per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
        ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
        winrate = (WIN_COUNT / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0
        resumen = f"📊 **APRENDIZAJE #{TOTAL_TRADES}**\nWinrate: {winrate:.1f}%\nProfit Factor: {ULTIMO_PROFIT_FACTOR:.2f}"
        telegram_mensaje(resumen)
        try:
            ult_serial = convertir_serializable(ult)
            prompt = f"Analiza estos 10 trades reales y dame una lección corta (máximo 200 caracteres): {json.dumps(ult_serial)}"
            resp = client.chat.completions.create(model=MODELO_VISION, messages=[{"role":"user","content":prompt}], timeout=10)
            REGLAS_APRENDIDAS = resp.choices[0].message.content
            telegram_mensaje(f"🧠 Lección IA: {REGLAS_APRENDIDAS}")
        except:
            pass
    except Exception as e:
        print(f"Error aprendizaje: {e}")
    finally:
        ULTIMO_APRENDIZAJE = TOTAL_TRADES
        guardar_memoria()

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
            print(f"🚨 Drawdown diario superado. Operaciones detenidas.")
    return not STOPPED_TODAY

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global REAL_BALANCE, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, TRADE_HISTORY, REAL_ACTIVE_TRADES
    cargar_memoria()
    set_leverage()
    REAL_BALANCE = get_real_balance()
    if REAL_BALANCE is None:
        print("❌ No se pudo obtener saldo real. Abortando.")
        return
    max_dinamico = get_dynamic_max_trades()
    print(f"🤖 BOT RIESGO ULTRAPEQUEÑO - Balance: {REAL_BALANCE:.2f} USDT - Max trades: {max_dinamico}")
    telegram_mensaje(f"🤖 Bot Online - Riesgo dinámico entre 0.01 y {RISK_PER_TRADE_MAX} USDT")

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
            max_trades_actual = get_dynamic_max_trades()
            vela_c = df.index[-2]
            if ultima_vela is None:
                ultima_vela = vela_c

            sync_active_trades_with_bybit()

            if len(REAL_ACTIVE_TRADES) < max_trades_actual and ultima_vela != vela_c:
                if risk_management_check():
                    sop, res, slo, inter, t, m = detectar_zonas_mercado(df)
                    img = generar_grafico_para_vision(df, sop, res, slo, inter, precio_actual)
                    dec, raz, sl, tp1, tp2, log = analizar_con_qwen(img)
                    print(f"🤖 IA: {dec} - {raz}")
                    if dec in ["Buy","Sell"]:
                        real_abrir_posicion(dec, precio_actual, raz, sl, tp1, tp2, log, df, sop, res, slo, inter)
                    else:
                        print(f"⏸️ IA dice HOLD: {raz[:100]}")
                ultima_vela = vela_c
            else:
                if ultima_vela == vela_c:
                    print("⏳ Misma vela, no repite análisis.")
                else:
                    print(f"⏸️ Límite de trades alcanzado ({len(REAL_ACTIVE_TRADES)}/{max_trades_actual})")

            if REAL_ACTIVE_TRADES:
                real_revisar_sl_tp(df)

            if iteracion % 10 == 0:
                reporte_estado()
                winrate = (WIN_COUNT / TOTAL_TRADES * 100) if TOTAL_TRADES > 0 else 0
                print(f"📈 Balance={REAL_BALANCE:.2f} | Trades={TOTAL_TRADES} | Winrate={winrate:.1f}% | PF={ULTIMO_PROFIT_FACTOR:.2f}")

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
