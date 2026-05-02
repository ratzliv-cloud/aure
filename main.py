# BOT TRADING V99.43 – QWEN3-VL-32B-Instruct (EDICIÓN ROBUSTA RESTAURADA)
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

# =================== CONFIGURACIÓN DE APIS ===================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
MODELO_VISION = "Qwen/Qwen3-VL-32B-Instruct"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

# ====== MEMORIA CON CONVERSIÓN DE TIPOS ======
MEMORY_FILE = "memoria_bot.json"

def convertir_serializable(obj):
    """Filtro infalible para NumPy y tipos no nativos."""
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, dict): return {str(k): convertir_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [convertir_serializable(item) for item in obj]
    return obj

def guardar_memoria():
    data = {
        "TRADE_HISTORY": TRADE_HISTORY,
        "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
        "PAPER_BALANCE": float(PAPER_BALANCE),
        "PAPER_WIN": int(PAPER_WIN),
        "PAPER_LOSS": int(PAPER_LOSS),
        "PAPER_TRADES_TOTALES": int(PAPER_TRADES_TOTALES),
        "ULTIMO_APRENDIZAJE": int(ULTIMO_APRENDIZAJE),
        "TOKENS_ACUMULADOS": int(TOKENS_ACUMULADOS),
        "ULTIMO_PROFIT_FACTOR": float(ULTIMO_PROFIT_FACTOR)
    }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
    except Exception as e: print(f"Error memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS, PAPER_BALANCE, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_TOTALES, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR
    if not os.path.exists(MEMORY_FILE): return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        TRADE_HISTORY = data.get("TRADE_HISTORY", [])
        REGLAS_APRENDIDAS = data.get("REGLAS_APRENDIDAS", REGLAS_APRENDIDAS)
        PAPER_BALANCE = data.get("PAPER_BALANCE", 100.0)
        PAPER_WIN = data.get("PAPER_WIN", 0)
        PAPER_LOSS = data.get("PAPER_LOSS", 0)
        PAPER_TRADES_TOTALES = data.get("PAPER_TRADES_TOTALES", 0)
        ULTIMO_APRENDIZAJE = data.get("ULTIMO_APRENDIZAJE", 0)
        TOKENS_ACUMULADOS = data.get("TOKENS_ACUMULADOS", 0)
        ULTIMO_PROFIT_FACTOR = data.get("ULTIMO_PROFIT_FACTOR", 1.0)
    except: pass

# =================== CONFIGURACIÓN TRADING ===================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
MAX_CONCURRENT_TRADES = 3
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120

PCT_TP1, PCT_TP2 = 0.50, 0.30

PAPER_BALANCE = 100.0
PAPER_ACTIVE_TRADES = {}
TRADE_COUNTER, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES = 0, 0, 0, 0
TRADE_HISTORY = []
MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None
ULTIMO_APRENDIZAJE, ULTIMO_PROFIT_FACTOR = 0, 1.0
REGLAS_APRENDIDAS = "Sin lecciones."
TOKENS_ACUMULADOS = 0

# =================== ANALÍTICA DE VELAS (RESTAURADA) ===================
def analizar_anatomia_vela(v):
    rango = max(v['high'] - v['low'], 0.001)
    cuerpo = abs(v['close'] - v['open'])
    c_pct = (cuerpo / rango) * 100
    s_sup = ((v['high'] - max(v['close'], v['open'])) / rango) * 100
    s_inf = ((min(v['close'], v['open']) - v['low']) / rango) * 100
    color = "VERDE" if v['close'] > v['open'] else "ROJA"
    return f"{color} (Cuerpo:{c_pct:.1f}% | M.Sup:{s_sup:.1f}% | M.Inf:{s_inf:.1f}%)"

def analizar_patrones_conjuntos(df, idx):
    if idx < 3: return "Datos insuficientes"
    v3, v2, v1 = df.iloc[idx], df.iloc[idx-1], df.iloc[idx-2]
    # Lógica de patrones restaurada
    p = []
    verde3 = v3['close'] > v3['open']
    verde2 = v2['close'] > v2['open']
    verde1 = v1['close'] > v1['open']
    
    # Envolventes
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']: p.append("🐂 ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']: p.append("🐻 ENVOLVENTE BAJISTA")
    # Estrellas
    if not verde1 and verde3 and v3['close'] > (v1['open']+v1['close'])/2 and abs(v2['close']-v2['open']) < (v1['high']-v1['low'])*0.2: p.append("🌟 ESTRELLA MAÑANA")
    # Fuerza
    if verde1 and verde2 and verde3 and v3['close'] > v2['close']: p.append("🚀 TRES SOLDADOS")
    
    # Anatomía individual para la actual
    r3 = max(v3['high'] - v3['low'], 0.001)
    inf3 = ((min(v3['close'], v3['open']) - v3['low']) / r3) * 100
    sup3 = ((v3['high'] - max(v3['close'], v3['open'])) / r3) * 100
    if inf3 > 60: p.append("🔨 MARTILLO")
    if sup3 > 60: p.append("🌠 ESTRELLA FUGAZ")
    
    return " | ".join(p) if p else "Consolidación"

def detectar_zonas_mercado(df, idx=-2):
    df_eval = df.iloc[:idx+1]
    sop = df_eval['low'].rolling(40).min().iloc[-1]
    res = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-100:]
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    tend = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return sop, res, slope, intercept, tend

def generar_descripcion_nison(df, idx=-2):
    v = df.iloc[idx]
    sop, res, slope, inter, tend = detectar_zonas_mercado(df, idx)
    desc = f"""
=== ANALISIS TECNICO ===
Precio: {v['close']:.2f} | EMA20: {v['ema20']:.2f}
Soporte: {sop:.2f} | Resistencia: {res:.2f} | Tendencia: {tend}
Vela Actual: {analizar_anatomia_vela(v)}
Vela -1: {analizar_anatomia_vela(df.iloc[idx-1])}
Patrones: {analizar_patrones_conjuntos(df, idx)}
RSI: {v['rsi']:.1f} | ATR: {v['atr']:.2f}
"""
    return desc, v['atr']

# =================== COMUNICACION Y GRAFICOS ===================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN: return
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": texto})

def telegram_enviar_imagen(ruta, caption=""):
    if not TELEGRAM_TOKEN: return
    with open(ruta, 'rb') as f:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": f})

def generar_grafico_vision(df, sop, res, precio):
    df_plot = df.tail(120)
    fig, ax = plt.subplots(figsize=(16,8))
    x = np.arange(len(df_plot))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = 'lime' if c >= o else 'red'
        ax.vlines(x[i], l, h, color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((x[i]-0.3, min(o,c)), 0.6, max(abs(c-o), 0.1), color=color))
    ax.axhline(sop, color='cyan', ls='--', alpha=0.6)
    ax.axhline(res, color='magenta', ls='--', alpha=0.6)
    ax.plot(x, df_plot['ema20'], color='yellow', lw=1.5)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# =================== IA Y EJECUCION ===================
def analizar_con_qwen(desc, imagen, reglas):
    global TOKENS_ACUMULADOS
    buffered = io.BytesIO()
    imagen.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    prompt = f"""
Analiza como Trader Senior. Gráfico: Amarillo=EMA20, Cian=Sop, Magenta=Res.
1. sl_price: Invalidación visual detrás de mechas.
2. tp1_price: Obstáculo visual cercano.
3. tp2_price: Nivel de liquidez lejano.
4. trailing_logic: "EMA20" o "LOW_CANDLE".

IMPORTANTE: Solo opera si hay confluencia clara.
JSON: {{"decision":"Buy/Sell/Hold","razon":"...","sl_price":0.0,"tp1_price":0.0,"tp2_price":0.0,"trailing_logic":"EMA20"}}
DATOS: {desc}
REGLAS: {reglas}
"""
    try:
        resp = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role":"user", "content":[{"type":"text","text":prompt}, {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64}"}}]}],
            temperature=0.1
        )
        TOKENS_ACUMULADOS += resp.usage.total_tokens
        return json.loads(json_repair.repair_json(resp.choices[0].message.content))
    except: return {"decision":"Hold"}

def paper_abrir_posicion(dec, precio, atr, data_ia, df):
    global TRADE_COUNTER
    if len(PAPER_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES: return
    
    sl = float(data_ia.get("sl_price", 0))
    if dec == "Buy" and (sl >= precio or sl == 0): sl = precio - (atr * 1.5)
    if dec == "Sell" and (sl <= precio or sl == 0): sl = precio + (atr * 1.5)
    
    dist = abs(precio - sl)
    size = (PAPER_BALANCE * RISK_PER_TRADE) / dist
    size = min(size, (PAPER_BALANCE * LEVERAGE) / precio)
    
    TRADE_COUNTER += 1
    PAPER_ACTIVE_TRADES[TRADE_COUNTER] = {
        "id": TRADE_COUNTER, "decision": dec, "entrada": precio, "sl_inicial": sl, "sl_actual": sl,
        "tp1": data_ia.get("tp1_price"), "tp2": data_ia.get("tp2_price"), "trailing_logic": data_ia.get("trailing_logic", "EMA20"),
        "size_btc": size, "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0, "razon": data_ia.get("razon")
    }
    telegram_mensaje(f"🚀 ENTRADA #{TRADE_COUNTER} {dec} en {precio:.2f}\nRazon: {data_ia.get('razon')}")
    guardar_memoria()

def paper_revisar_sl_tp(df):
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY
    if not PAPER_ACTIVE_TRADES: return
    c, h, l = df['close'].iloc[-1], df['high'].iloc[-1], df['low'].iloc[-1]
    ema, atr = df['ema20'].iloc[-1], df['atr'].iloc[-1]
    h_prev, l_prev = df['high'].iloc[-2], df['low'].iloc[-2]
    
    cerrar_ids = []
    for tid, t in PAPER_ACTIVE_TRADES.items():
        # TP1 (50%)
        if not t['tp1_ejecutado'] and t['tp1']:
            if (t['decision']=="Buy" and h>=t['tp1']) or (t['decision']=="Sell" and l<=t['tp1']):
                ganancia = abs(t['tp1'] - t['entrada']) * (t['size_btc'] * PCT_TP1)
                t['pnl_parcial'] += ganancia
                PAPER_BALANCE += ganancia
                t['tp1_ejecutado'] = True
                t['sl_actual'] = t['entrada']
                telegram_mensaje(f"🎯 TP1 #{tid} hit. SL a Breakeven.")

        # Trailing / SL
        cerrar, motivo = False, ""
        if t['tp1_ejecutado']:
            if t['decision']=="Buy":
                t['sl_actual'] = max(t['sl_actual'], ema-(atr*0.2) if t['trailing_logic']=="EMA20" else l_prev)
                if l <= t['sl_actual']: cerrar, motivo = True, "Trailing Stop"
            else:
                t['sl_actual'] = min(t['sl_actual'], ema+(atr*0.2) if t['trailing_logic']=="EMA20" else h_prev)
                if h >= t['sl_actual']: cerrar, motivo = True, "Trailing Stop"
        else:
            if (t['decision']=="Buy" and l <= t['sl_inicial']) or (t['decision']=="Sell" and h >= t['sl_inicial']):
                cerrar, motivo = True, "Stop Loss"

        if cerrar:
            pct = 0.50 if t['tp1_ejecutado'] else 1.0
            pnl_final = (t['sl_actual'] - t['entrada']) * t['size_btc'] * pct if t['decision']=="Buy" else (t['entrada']-t['sl_actual']) * t['size_btc'] * pct
            pnl_total = float(t['pnl_parcial'] + pnl_final)
            PAPER_BALANCE += float(pnl_final)
            PAPER_TRADES_TOTALES += 1
            if pnl_total > 0: PAPER_WIN += 1
            else: PAPER_LOSS += 1
            TRADE_HISTORY.append({"pnl": pnl_total, "resultado_win": bool(pnl_total > 0), "decision": t['decision']})
            cerrar_ids.append(tid)
            telegram_mensaje(f"📤 CERRADO #{tid} ({motivo}). PnL: {pnl_total:.2f} USDT")

    for tid in cerrar_ids: del PAPER_ACTIVE_TRADES[tid]
    if cerrar_ids: guardar_memoria()

# =================== LOOP ===================
def run_bot():
    cargar_memoria()
    ultima_vela = None
    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            if df.empty: continue
            vela_c = df.index[-2]
            if len(PAPER_ACTIVE_TRADES) < MAX_CONCURRENT_TRADES and ultima_vela != vela_c:
                desc, atr = generar_descripcion_nison(df)
                sop, res, _, _, _ = detectar_zonas_mercado(df)
                img = generar_grafico_vision(df, sop, res, df['close'].iloc[-1])
                data_ia = analizar_con_qwen(desc, img, REGLAS_APRENDIDAS)
                dec = data_ia.get("decision", "Hold")
                if dec in ["Buy", "Sell"]:
                    paper_abrir_posicion(dec, df['close'].iloc[-1], atr, data_ia, df)
                ultima_vela = vela_c
            paper_revisar_sl_tp(df)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"Error: {e}"); time.sleep(30)

if __name__ == '__main__':
    run_bot()
