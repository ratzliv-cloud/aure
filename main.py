# BOT TRADING V99.43 – QWEN3-VL-32B-Instruct (EDICIÓN COMPLETA CORREGIDA + LOGS + FIX JSON)
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
if not SILICONFLOW_API_KEY:
    raise ValueError("Falta SILICONFLOW_API_KEY. Obtén una en https://cloud.siliconflow.com")

SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
MODELO_VISION = "Qwen/Qwen3-VL-32B-Instruct"

# --- CORRECCIÓN: Definición de variables de Telegram ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

# ====== MEMORIA PERSISTENTE ======
MEMORY_FILE = "memoria_bot.json"

def convertir_serializable(obj):
    # Maneja cualquier escalar de NumPy (int, float, bool) convirtiéndolo a tipo nativo Python
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
    for tid, t in PAPER_ACTIVE_TRADES.items():
        active_trades_meta[tid] = {
            "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
            "razon": t.get("razon", ""), "tp1_ejecutado": t["tp1_ejecutado"],
            "tp2_ejecutado": t.get("tp2_ejecutado", False),
            "sl_actual": t.get("sl_actual"), "trailing_logic": t.get("trailing_logic", "EMA20")
        }
    data = {
        "TRADE_HISTORY": TRADE_HISTORY,
        "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
        "PAPER_BALANCE": PAPER_BALANCE,
        "PAPER_WIN": PAPER_WIN,
        "PAPER_LOSS": PAPER_LOSS,
        "PAPER_TRADES_TOTALES": PAPER_TRADES_TOTALES,
        "ULTIMO_APRENDIZAJE": ULTIMO_APRENDIZAJE,
        "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS,
        "PAPER_ACTIVE_META": active_trades_meta,
        "ULTIMO_PROFIT_FACTOR": ULTIMO_PROFIT_FACTOR
    }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        print("💾 Memoria guardada")
    except Exception as e: print(f"Error guardando memoria: {e}")

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
        print(f"🧠 Memoria cargada. Trades: {PAPER_TRADES_TOTALES}")
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

PCT_TP1, PCT_TP2 = 0.50, 0.30  # El 20% restante va a trailing

PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_ACTIVE_TRADES = {}
TRADE_COUNTER, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES = 0, 0, 0, 0
TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

ULTIMO_APRENDIZAJE, ULTIMO_PROFIT_FACTOR = 0, 1.0
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
    pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
    winrate = (PAPER_WIN / PAPER_TRADES_TOTALES * 100) if PAPER_TRADES_TOTALES > 0 else 0
    mensaje = (
        f"📊 **ESTADO DEL BOT**\n"
        f"💰 Balance: {PAPER_BALANCE:.2f} USDT\n"
        f"📈 PnL: {pnl_global:+.2f} USDT\n"
        f"🎯 Winrate: {winrate:.1f}%\n"
        f"⚡ Activos: {len(PAPER_ACTIVE_TRADES)}\n"
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

# =================== ANATOMÍA Y PATRONES (NISON) ===================
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

# =================== VISIÓN IA Y GRÁFICOS ===================OS ===================
def
def generar_ generar_graficografico_para_vision_para_vision(df,(df, soporte soporte, resistencia, slope, intercept, precio, resistencia, slope, intercept, precio):
   ):
    df_plot = df.tail df_plot = df.tail(GRAFICO_V(GRAFICO_VELASELAS_LIMIT)._LIMIT).copycopy()
    fig()
    fig, ax = plt.subplots(fig, ax = plt.subplots(figsize=(16,8size=(16,8))
    x = np))
    x = np.arange(len(df_.arange(len(df_plot))
    for i inplot))
    for i in range(len range(len(df_plot)):
        o(df_plot)):
        o, h, l, h, l, c, c = df_plot = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low['low'].iloc[i], df'].iloc[i], df_plot_plot['close'].iloc[i['close'].iloc[i]
        color = '#00ff00' if]
        color = '#00ff00' if c >= c >= o else '#ff o else '#ff0000'
        ax.v0000'
        ax.vlines(x[i],lines(x[i], l, l, h, color= h, color=color,color, linewidth=1 linewidth.5)
        ax.add=1.5)
        ax.add_patch_patch(plt.Rect(plt.Rectangle((x[i]-0angle((x[i]-0.35, min(o,c.35, min(o,c)), 0.7,)), 0.7, max(abs(c-o),  max(abs(c-o), 0.1), color=0.1), color=color,color, alpha=0. alpha=0.9))
    ax.ax9))
    ax.axhline(soporte, colorhline(soporte, color='cyan', ls='cyan', ls='--='--', l', lw=2,w=2, label=' label='SoporteSoporte')
    ax.ax')
    ax.axhline(reshline(resistencia, coloristencia, color='mag='magenta', ls='enta', ls='--',--', lw=2 lw=2, label='Res, label='Resistenciaistencia')
   ')
    ax.plot(x, df_ ax.plot(x, df_plot['plot['ema20ema20'], 'yellow','], 'yellow', lw=2, label='EMA lw=2, label='EMA2020')
    ax.set_')
    ax.set_facecolorfacecolor('#121('#121212')
    fig212')
    fig.patch.set_.patch.set_facecolorfacecolor('#121212')
    plt.tight('#121212')
    plt_layout()
    buf.tight_layout()
    buf = io = io.BytesIO.BytesIO()
    plt.savefig(buf,()
    plt.savefig(buf, format='png', dpi format='png', dpi=100)
   =100)
    buf.se buf.seek(0ek(0)
    img)
    img = Image.open(buf)
    = Image.open(buf plt.close)
    plt.close()
   ()
    return img return img

def pil_to_base64

def pil_to_base64(img):
    buff(img):
    buffered = io.BytesIO()
   ered = io.B img.save(buffered,ytesIO()
    img.save(buffered, format=" format="PNGPNG")
   ")
    return f"data return f"data:image/png;base:image/png;base64,64,{base64.b{base64.b64encode(buffered.get64encode(buffered.getvalue()).decode()}value()).decode()}"

def"

def analizar_con_q analizar_con_qwen(wen(descripcion_texto,descripcion_texto, atr, reglas, atr, reglas, imagen imagen):
    global TOKENS_):
    global TOKENS_ACUMACUMULADOSULADOS
    try:
        img_b64 =
    try:
        img_b64 = pil_to_base64 pil_to_base64(im(imagen)
        promptagen)
        prompt = f = f"""
Eres Tra"""
Eres Trader Seniorder Senior. Mira el gráfico (Cian=Sop. Mira el gráfico (Cian=Sop, Mag, Magenta=Res,enta=Res, Amarillo Amarillo=EMA20=EMA20).
Define niveles VISUAL).
Define niveles VISUALES deES de salida salida.. 
1. 
1. sl_price sl_price: Invalid: Invalidación real.
2. tpación real.
2. tp1_price: Obst1_price: Obstáculoáculo cercano.
3 cercano.
3. tp. tp2_price2_price: N: Nivel deivel de liquidez liquidez lejano.
4. trailing_logic lejano.
4. trailing_logic: ": "EMA20EMA20" o "LOW_C" o "LANDLEOW_CANDLE".

JSON".

JSON (una línea (una):
{{"decision":" línea):
{{"Buy/Sell/Hdecision":"Buy/Sell/Hold","razonold","razon":"text":"texto","o","sl_pricesl_price":0.0":0.0,"tp1_price,"tp1_price":0.0,"tp2_price":0.0,"tp2_price":0":0.0,"trailing_logic.0,"trailing_logic":"EMA20/LOW_CANDLE":"EMA20/LOW_CANDLE""}}

DATOS}}

DATOS: {descrip: {descripcion_textcion_texto}
MEMORo}
MEMORIA:IA: {reg {reglas}
"""
        response =las}
"""
        response = client.ch client.chat.completionsat.com.createpletions.create(
            model(
            model=MODELO_VISION=MODELO_VISION,
            messages=[{"role,
            messages=[{"role":"user", "":"user", "contentcontent":[{"type":"text","text":prompt},":[{"type":"text","text":prompt}, {"type {"type":"image_url","image_url":"image_url","image_url":{"url":img_b64":{"url":img_b64}}}}]}],
            temperature=]}],
            temperature=0.1
       0.1
        )
        TOKENS_AC )
        TOKENS_ACUMULADOS += responseUMULADOS += response.usage.usage.total_tokens.total_tokens
        datos
        datos = parse_json_se = parse_json_seguro(response.guro(response.choiceschoices[0].[0].message.content)
       message.content)
        if not if not datos: return " datosHold", "", 0, : return "Hold", "", 0, 0, 0, 0, "EMA200, "EMA20"
        return datos.get("decision"
        return datos.get("decision","Hold"), datos.get("","Hold"), datos.get("razon",""),razon",""), datos.get("sl_price"), datos.get("sl_price"), datos.get datos.get("tp("tp1_price1_price"), datos.get(""), datos.get("tp2tp2_price"), datos.get_price"), datos.get("trailing_logic("trailing_logic","EMA","EMA2020")
    except")
    except Exception as Exception as e:
        print e:
        print(f"❌ Error(f"❌ Error en en llamada a IA: { llamada a IA: {e}")
        returne}")
        return "Hold", "Error API",  "Hold", "Error API", 0, 00, 0, , 0, "EMA200, "EMA20"

# ==================="

# =================== GEST GESTIÓN EJECIÓN EJECUCIUCIÓN ===================ÓN ===================
def
def paper_abrir_posicion paper_abrir_posicion(decision, precio, atr, razon, sl(decision, precio, atr, razon, sl_ia, tp1__ia, tp1_ia, tp2_iaia, tp2_ia, logic_ia, logic, df, sop_ia, df, sop, res, slo, inter, res, slo, inter):
    global):
    global PAPER PAPER_BAL_BALANCE,ANCE, TRADE_COUN TRADE_COUNTERTER
    if
    if len(P len(PAPERAPER_ACT_ACTIVE_TRADESIVE_TR) >=ADES) >= MAX_CON MAX_CONCURRENTCURRENT_TRAD_TRADES: returnES: return
    
    sl
    
    sl_final_final = float(sl_ia = float(sl_ia) if sl_) if sl_ia elseia else (pre (precio - atr*1.5 if decision=="Buy" else precio + atr*1.5)
    if decision == "Buy" and sl_final >= precio: sl_final = preciocio - atr*1.5 if decision=="Buy" else precio + atr*1.5)
    if decision == "Buy" and sl_final >= precio: sl_final = precio - at - atrr
    if
    if decision == decision == "Sell" "S and sl_finalell" and sl_final <= precio: sl <= precio: sl_final = precio_final = precio + at + atr

    distanciar

    distancia = abs(precio - sl = abs(precio - sl_final)
    size__final)
    size_btc = (PAPbtc = (PAPER_BALANCEER_BALANCE * RIS * RISK_PER_TRADE) /K_PER_TRADE) / distancia
    size_bt distancia
    size_btc = min(size_btc = min(size_btc,c, (PAPER (PAPER_BALANCE * LEVER_BALANCE *AGE) / precio)

    LEVERAGE) / precio)

    TRADE_COUN TRADE_COUNTER +=TER += 1
    t = 1
    t = {
        "id {
        "id": TR": TRADE_COUNTER, "ADE_COUNTER, "decision": decision, "entdecision": decision,rada": precio, "entrada": precio, "sl "sl_inicial": sl_final_inicial": sl, "sl_actual": sl_f_final, "sl_actual": sl_finalinal,
        "tp1,
        "tp1": tp": tp1_ia, "tp1_ia, "tp2": tp22": tp2_ia_ia, "trailing_logic":, "trailing_logic": logic_ia, "size logic_ia, "size_btc": size__btc": size_btcbtc,
        "tp,
        "tp1_e1_ejecjecutadoutado": False, "tp2": False, "tp2_ejecut_ejecutado":ado": False, False, "pnl_ "pnl_parcialparcial": 0.": 0.0,0, "razon": razon "razon": razon
    }
    PAPER
    }
    PAPER_ACTIVE_TRADES_ACTIVE_TRADES[TRADE_COUNTER[TRADE_COUNTER] = t
    msg = f] = t
    msg = f""🚀 [#{TR🚀 [#{TRADE_COUNTERADE_CO}] {UNTER}] {decision}decision} en {precio:.2 en {precio:.2f}\nRf}\nRazon:azon: {razon {razon}"
    print(f"📈 {msg}"
    print(f"📈 {msg}")
    telegram_mensaje(msg)
    # Generar gráfico simple para Telegram
    fig, ax = plt.subplots(); ax.plot(df.tail(20)['close}")
    telegram_mensaje(msg)
    # Generar gráfico simple para Telegram
    fig, ax = plt.subplots(); ax.plot(df.tail(20)'].values); plt['close'].values); plt.savefig("/tmp.savefig("/tmp/in.png/in.png"); plt.close"); plt()
    telegram.close()
    telegram_enviar__enviar_imagenimagen("/tmp/in.png", msg)

def("/tmp/in.png", msg paper_re)

def paper_revisarvisar_sl_sl_tp_tp(df(df):
    global):
    global PAPER_BALANCE, PAPER_BALANCE, PAPER PAPER_WIN_WIN, PAP, PAPER_LOSS,ER_LOSS, PAPER PAPER_TRADES_T_TRADES_TOTALESOTALES, TRADE_H, TRADE_HISTORYISTORY
    if not
    if not PAPER_ACTIVE_TR PAPER_ACTIVE_TRADESADES: return: return
    c,
    c, h, h, l = l = df[' df['close'].ilocclose'].iloc[-1[-1], df['high], df['high'].iloc[-'].iloc[-1],1], df['low']. df['low'].ilociloc[-1]
    ema[-1]
    ema, atr = df[', atr = df['ema20'].iloc[-1], df['atrema20'].iloc[-1], df['atr'].iloc[-1]
    h_prev'].iloc[-1]
    h_prev, l, l_prev_prev = df = df['high'].iloc[-['high'].iloc[-2],2], df[' df['low'].low'].iloc[-2iloc[-2]
]
    
    cerrar_ids    
    cerrar_ids = = []
    for []
    for tid, tid, t in t in PAPER_ACT PAPER_ACTIVE_TRADES.itemsIVE_TRADES.items():
        #():
        # TP1 (50 TP1 (50%)
       %)
        if not t['tp1 if not t['tp1_ejecut_ejecutado']ado'] and t['tp and t['tp1']:
            if (1']:
            if (t['decision']=="Buy" and h>=t['t['decision']=="Buy" and h>=t['tp1']) or (t['decision']=="Selltp1']) or (t['decision']=="Sell" and l<=t['tp1']):
                gan" and l<=t['tp1']):
                ganancia = abs(t['tpancia = abs(t['tp1'] - t1'] - t['ent['entrada']) * (rada']) * (t['t['size_btc'] *size_btc'] * PCT_ PCT_TP1)
                t['pnlTP1)
                t['pnl_parcial']_parcial'] += gan += ganancia
                PAPER_Bancia
                PAPER_BALANCE += gananciaALANCE += ganancia
                t['tp
                t['tp1_ejecutado1_ejecutado'] ='] = True
                t True
                t['sl['sl_actual'] =_actual'] = t[' t['entrada']
               entrada']
                print(f"🎯 TP print(f"🎯 TP1 alcan1 alcanzado en tradezado en trade #{tid #{tid}")
                telegram_m}")
                telegram_mensajeensaje(f"🎯 TP1 #{tid(f"🎯 TP1 #{tid} hit} hit. SL a Bre. SL a Breakevenakeven.")
        
        #.")
 TP2 (30%)
               
        # TP2 (30%)
        if t['tp1_ejec if t['tp1_ejecutado'] andutado'] and not t not t['tp2_e['tp2_ejecutadojecutado'] and t['tp2'] and t['tp2']']:
            if (t:
            if (t['decision['decision']=="Buy"']=="Buy" and h>=t['tp2']) and h>=t['tp2']) or (t[' or (t['decision']=="Sell"decision']=="Sell" and l and l<=t<=t['tp2']['tp):
                gan2']):
                gananciaancia = abs(t[' = abs(t['tp2'] -tp2'] - t['entrada']) * (t['size_btc'] * PCT_TP2)
                t['pn t['entrada']) * (t['size_btc'] * PCT_TP2)
                t['pnl_parciall_parcial'] += ganancia
                PAPER_B'] += ganancia
                PAPERALANCE +=_BALANCE += ganancia
                t[' ganancia
                t['tp2tp2_ejecutado']_ejecutado'] = True
                print(f"🎯 TP = True
                print(f"🎯 TP2 alcan2 alcanzado en trade #{tidzado en trade #{tid}")
               }")
                telegram_m telegram_mensajeensaje(f"🎯 TP2(f"🎯 TP2 #{tid} hit #{tid} hit.")

        # Trailing / SL.")

        # Trailing / SL
        cerrar, motivo =
        cerrar, motivo = False, ""
        if t False, ""
        if t['tp1_ejecutado']['tp1_ejecutado']:
            if:
            if t[' t['decision']=="Buy":
               decision']=="Buy":
                t['sl_actual'] t['sl_actual'] = max(t['sl_actual'], ema = max(t['sl_actual'], ema-(at-(atr*0.r*0.2) if t['tra2) if t['trailing_logic']=="iling_logic']=="EMA20EMA20" else l_" else l_prevprev)
                if l <= t[')
                if l <= t['sl_actual']sl_actual']: cerrar,: cerr motivo = True, "Traar, motivo = True, "Trailing"
            else:
               iling"
            else:
                t['sl_actual'] t['sl_actual'] = min(t['sl_ = min(t['sl_actual'], ema+(atactual'], ema+(atr*0.2)r*0.2) if t['trailing_logic if t['trailing_logic']=="EMA20" else']=="EMA20" else h_prev h_prev)
                if)
                if h >= t[' h >= t['sl_actual']: cerrsl_actual']: cerrar, motivo = True,ar, motivo = True, "Tra "Trailing"
        elseiling"
        else:
            if (:
            if (t['decision']=="Buyt['decision']=="Buy" and l <= t['" andsl_in l <= t['sl_inicial']) or (t['icial']) or (t['decision']=="Sdecision']=="Sell"ell" and h >= t['sl and h >= t['sl_inicial']_inicial']):
                cerrar, motivo = True, "Stop):
                cerrar, motivo = True, "Stop Loss"

        if cerrar:
            Loss"

        if cerrar:
            pct =  pct = 0.0.20 if t['tp220 if t['tp2_ejecutado']_ejecutado'] else (0.50 if t[' else (0.50 if t['tp1tp1_ej_ejecutado'] else 1.ecutado'] else 1.0)
            pnl_f0)
            pnl_f = ( = (t['sl_t['sl_actual']actual']-t['ent-t['entrada']rada'])*t['size_bt)*t['size_btc']*pc']*pct ifct if t[' t['decision']=="Buy" else (t['entrada']decision']=="Buy" else (t['entrada']-t['sl_actual-t['sl_actual'])*'])*t['size_btct['size_btc']*pct']*pct
           
            pnl_t = pnl_t = t['pnl_par t['pnl_parcial']cial'] + pnl_f + pnl_f
            PAPER_BAL
            PAPER_BALANCE +=ANCE += pnl_f
            PAPER_TRADES pnl_f
            PAPER_TRADES_TOTALES +=_TOTALES += 1
            if pnl_t 1
            if pnl_t >  > 0: PAPER_WIN += 0: PAPER_WIN += 11
            else
            else: PAPER_L: PAPER_LOSS +=OSS += 1
            
            1
            
            # --- # --- CORRECCIÓN: Convertir a tipos serializ CORRECCIÓN: Convertir a tipos serializables antes de guardar en historial ---
            trade_record = convertirables antes de guardar en historial ---
            trade_record = convertir_serializable({
                "pnl_serializable({
                "pnl": p": pnl_t,
               nl_t,
                "result "resultado_win":ado_win": pnl pnl_t > 0,
               _t > 0,
                "decision "decision": t": t['decision'],
               ['decision'],
                "razon": "razon": t['razon']
            t['razon']
            })
            TRADE })
            TRADE_HIST_HISTORY.append(tradeORY.append(trade_record_record)
            
            cerrar_ids.append)
            
            cerrar_ids.append(tid(tid)
           )
            print(f"📤 CERRADO #{tid} ({motivo}) | print(f"📤 CERRADO #{tid} ({motivo}) | PnL: {pnl_t:.2f} USDT | Balance PnL: {pnl_t:.2f} USDT | Balance: {PAPER_B: {PAPER_BALANCE:.2f}")
            telegramALANCE:.2f}")
            telegram_mens_mensaje(faje(f""📤 CERRADO📤 CERRADO #{tid} ({ #{tidmotivo} ({motivo}). P}). PnLnL: {pnl: {_t:.2fpnl_t:.2f} USDT")
            report} USDT")
            reporte_estadoe_estado()

   ()

    for tid in cerr for tid in cerrar_idsar_ids: del: del PAPER PAPER_ACTIVE_TRADES_ACTIVE_TRADES[tid[tid]
   ]
    if len(TR if len(TRADE_HADE_HISTORYISTORY) >) > 0 and len 0 and len(TR(TRADE_HADE_HISTORY) %ISTORY) % 10 10 == 0 == 0: aprender_de: aprender_de_trades()

#_trades()

# =================== AUTOAPRENDIZ =================== AUTOAPRENDIZAJE Y LOAJE Y LOOP ===================
defOP = aprender_de==================
def aprender_de_trades():
   _trades():
    global REGLAS_APR global REGLAS_APRENDIDAS, ULTENDIDAS, ULTIMO_APRENDIZIMO_APRENDIZAJE, ULTIMOAJE, ULTIMO_PROF_PROFIT_FACTORIT_FACTOR
    try:
        ult
    try:
        ult = TR = TRADE_HISTORYADE_HISTORY[-10[-10:]
       :]
        gan = gan = sum(t sum(t['pnl']['pn for t in ult if tl'] for t in ult if t['pnl']['pnl']>0)
        per = abs(sum(t['>0)
        per = abs(sum(t['pnlpnl'] for t in ult if t[''] for t in ult if t['pnlpnl']']<0))
        ULTIMO<0))
        ULTIMO_PROFIT_FACTOR_PROFIT_FACTOR = gan/per if per> = gan/per if per>0 else 10 else 1.0.0
        
        # --- CORREC
        
        # --- CORRECCIÓN: Convertir laCIÓN: Convertir la lista a tipos serializables lista a tipos serial antes de json.dizables antes de json.dumps ---
        ult_umps ---
        ult_serializableserializable = convertir_serial = convertir_serializable(ult)
        prompt = fizable(ult)
        prompt = f"Analiza estos"Analiza estos 10 trades y dame 10 trades y dame una lección corta: una lección corta: {json.dumps {json.dumps(ult(ult_serial_serializable)}izable)}"
        
        resp"
        
        resp = client = client.chat.chat.completions.create(model=MODEL.completions.create(model=MODELO_VO_VISION,ISION, messages=[{"role":"user messages=[{"role":"user","content":prom","content":prompt}pt}])
       ])
        REGLAS_APRENDIDAS = resp REGLAS_APRENDIDAS = resp.choices.choices[0[0].message].message.content
        print.content
        print(f"(f"🧠 APREND🧠 APRENDIZAIZAJE: {REJE: {REGLAS_APGLAS_APRENDRENDIDASIDAS}")
       }")
        telegram_m telegram_mensaje(f"ensaje(f"🧠🧠 APRENDIZA APRENDIZAJE:JE: {REGLAS {REGLAS_AP_APRENDIDASRENDIDAS}")
       }")
        ULTIMO_ ULTIMO_APRAPRENDIZENDIZAJEAJE = PAP = PAPER_TRADESER_TRADES_TOTAL_TOTALES
        guardES
        guardar_memoria()
    except Exception as ear_memoria()
    except Exception as e:
       :
        print(f"❌ Error en aprendizaje print(f"❌ Error (no afecta al en aprendizaje trading): {e} (no afecta al trading): {")

def risk_management_checke}")

def risk_():
    global PAPER_management_check():
    global PAPDAILYER_DAILY_START_BAL_STARTANCE,_BALANCE, PAPER PAPER_STOP_STOPPEDPED_TOD_TODAY, PAPERAY, PAPER_CURRENT_CURRENT_DAY
    hoy =_DAY
    hoy = datetime.now(timezone datetime.now(timezone.ut.utc).date()
    ifc).date()
    if PAPER_CURRENT_DAY PAPER_CURRENT_DAY != hoy != hoy:
        PAPER:
        PAPER_CURRENT_CURRENT_DAY_DAY, PAP, PAPER_DAILYER_DAILY_START_START_BALANCE,_BALANCE, PAPER PAPER_STOPPED_TODAY =_STOPPED_TODAY = hoy, PAPER_BALANCE, hoy, PAPER_BALANCE, False
        print(f" False
        print📅 Nuevo día:(f"📅 Nuevo {hoy}. Balance inicial día: {hoy}. Balance inicial: {: {PAPPAPER_ER_DAILYDAILY_START_BALANCE:.2f}")
   _START_BALANCE:.2f}")
    if ( if (PAPER_BPAPER_BALANCEALANCE - PAPER_ - PAPER_DAILYDAILY_START_START_BAL_BALANCE) / PAPER_DAILYANCE) / PAPER_DAILY_START_START_BAL_BALANCE <= -MAX_DAILY_DANCE <= -MAX_DAILY_DRAWDRAWDOWN_POWN_PCTCT:
        PAPER_ST:
        PAPER_STOPPED_TOPPED_TODAY = TrueODAY = True
       
        print(f" print(f"🚨 Draw🚨 Drawdown didown diario superado ({MAX_DAILYario superado ({MAX_DAILY_DRA_DRAWDOWNWDOWN_PCT_PCT*100}%).*100}%). Bot det Bot detenido hasta maenido hasta mañana.")
    returnñana not PAPER_STOPP.")
    return not PAPER_STOPPED_TODAY

def run_bot():
    cargar_memoria()
    print("🤖 BOT V99.43 INICIADO - Modo LOGS DETALLADOS + FIX JSON")
    telegram_mensaje("🤖 Bot V99.43 Online - Estructural Visual")
    ultima_vela = None
    iteracion = 0
    while True:
        try:
            iteracion += 1
            now_str = datetime.now().strftime("%Y-%m-%d %H:%MED_TODAY

def run_bot():
    cargar_memoria()
    print("🤖 BOT V99.43 INICIADO - Modo LOGS DETALLADOS + FIX JSON")
    telegram_mensaje("🤖 Bot V99.43 Online - Estructural Visual")
    ultima_vela = None
    iteracion = 0
    while True:
        try:
            iteracion += 1
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%S")
            print(f")
            print(f"\n🔄 [{now_str}] Ciclo #{iteracion} - In"\n🔄 [{now_str}] Ciclo #{iteracion} - Iniciandoiciando...")
            
            # Obtener datos
            df_raw = obtener_velas()
            if df_raw.empty...")
            
            # Obtener datos
            df_raw = obtener_velas()
            if df_raw.empty:
               :
                print("⚠️ print("⚠️ No se No se pudieron obtener velas. Reintentando... pudieron obtener velas. Reintentando...")
                time.sleep(SLEEP")
                time.sleep(S_SECONDS)
                continueLEEP_SECONDS)
               
            df continue
            df = calcular = calcular_indicadores(df_indicadores(df_raw_raw)
            if)
            if df.empty:
                df.empty:
                print(" print("⚠️ DataFrame vac⚠️ DataFrame vacío tras indicadoresío tras indicadores. Re. Reintentintentando...")
               ando...")
                time.sleep time.sleep(SLEEP_SEC(SLEEP_SECONDSONDS)
                continue
            
            precio_actual = df[')
                continue
            
            precio_actual = df['close'].close'].iloc[-1]
           iloc[-1]
            print(f print(f""📊 Pre📊 Precio actual: {precio_actual:.2f}cio actual: {precio_actual:.2f} | T | Trades activos:rades activos: {len(PAP {lenER_ACTIVE(PAPER_ACTIVE_TRADES)}/{MAX_CONCURRENT_TR_TRADES)}/{MAX_CONCURRENT_TRADES} | Balance:ADES} | Balance: {P {PAPERAPER_BALANCE:.2f_BALANCE:.2f}")
            
            # Verificar nueva v}")
            
            # Verificar nueva velaela
            v
            vela_c = dfela_c.index[-2]
            if = df.index[-2]
            if ultima ultima_vela is None_vela is:
                ult None:
                ultima_ima_vela = vvela = vela_cela_c
                print(f"🕯
                print(f️ Vela"🕯️ Vela base base establecida: {vela_c}")
            
            # Condiciones para nuevo análisis
            establecida: {vela_c}")
            
            # Condiciones para nuevo análisis
            if len if len(PAPER_(PAPER_ACTIVE_TRADACTIVE_TRADES)ES) < MAX_CONCURRENT_TRADES < MAX_CONCURRENT_TRADES and ultima_ and ultima_velavela != vela_c != vela_c:
                print(f":
                print(f"🕯️ Nueva🕯 vela️ Nueva vela detectada detectada: {vela: {vela_c}_c} (anterior: {ultima_ (anterior: {ultima_velavela})")
                print("🔍 Ej})")
                print("🔍 Ejecutecutando análisis de mercado...ando análisis de mercado...")
                sop")
                sop, res, res, sl, slo, inter,o, inter, t, m = detectar_zon t, m = detectar_zonas_mercado(dfas_mercado(df)
                desc, at)
                desc, atr =r = generar_descripcion generar_descripcion_nison_nison(df)
                print("🧠 En(df)
                print("🧠 Enviandoviando gráfico a IA para decis gráfico a IA para decisión...")
                img =ión...")
                generar_grafico_para img = generar_grafico_vision(df,_para_vision(df, sop, res, sop, res, slo slo, inter, df['close']., inter, df['close'].iloc[-iloc[-1])
                dec1, raz, sl, tp1,])
                dec, raz, sl, tp1, tp2, log tp2, log = analizar_con = analizar_con_qwen_qwen(desc(desc, atr, REGLAS_APREND, atr, REGLAS_APRENDIDASIDAS, img, img)
               )
                print(f print(f"🤖 Decisión IA: {"🤖 Decisión IA: {dec} | Razón:dec} | Razón: {raz {raz[:60]}...[:60]}...")
                
                if")
                
                if dec in dec in ["Buy","S ["Buy","Sell"]:
                   ell"]:
                    print(f"✅ IA sugiere print(f"✅ IA sugiere operación {dec}. Verificando operación {dec}. Verificando risk management risk management...")
                    if...")
                    if risk_management_check():
                        risk_management_check():
                        print(f" print(f"🚀 Abriendo🚀 Ab posición {dec}riendo posición {dec}...")
                        paper_ab...")
                        paperrir_posicion(dec,_abrir_posicion(dec, df[' df['close'].ilocclose'].iloc[-1],[-1], at atr, raz, sl,r, raz, sl, tp1, tp tp1, tp2,2, log, df, log, df, sop, sop, res, res, slo, inter)
                    slo, inter)
                    else:
                        print(" else:
                        print("⛔ Risk management⛔ Risk management bloqueó la operación (drawdown bloqueó la operación ( diario alcanzado).drawdown diario alcanzado).")
                else")
                else:
                    print(f":
                    print(f"⏸️ IA decid⏸️ IA decidió HOLD. Motivoió HOLD. Motivo: {: {raz[:100] if raz else 'raz[:100] if raz else 'No especificado'}")
               No especificado'}")
                ultima_vela = ultima_vela = vela_c
            elif vela_c
            elif ultima_vela == ultima_vela == vela vela_c:
                print_c:
                print("⏳ Misma vela("⏳ Misma vela, no se repite análisis, no se repite análisis. Esperando nueva. Esperando nueva vela vela...")
            else:
               ...")
            else:
                print(f"⏸️ Má print(f"⏸️ Máximo deximo de trades concurrent trades concurrentes alcanzado ({len(PAPes alcanzado ({len(PAPER_ACTIVEER_ACTIVE_TRAD_TRADES)}/{MAX_CONCURRENT_TRADES}). NoES)}/{MAX_CONCURRENT_TRADES}). No se anal se analiza nueva entradaiza nueva entrada.")
            
           .")
            
            # Revisar SL/TP de # Revisar SL/TP de trades activ trades activosos
            if
            if PAPER_ACT PAPER_ACTIVE_TRIVE_TRADES:
               ADES:
                print("🔎 Revisando condiciones de SL/TP print("🔎 Revisando condiciones de SL/TP en trades activos... en trades activos")
                paper_revisar_sl_tp(df)
           ... else:
                print")
                paper_revisar_sl_tp(df)
            else("💤 Sin:
                print("💤 Sin trades activ trades activos enos en este momento este momento.")
            
            # Mostrar.")
            
            # Mostrar resumen cada 10 cic resumen cada 10 cicloslos
            if iteracion % 10
            if iteracion % 10 == 0:
                winrate = ( == 0:
                winrate = (PAPER_WIN /PAPER_WIN / PAPER_TRADES_TOTALES PAPER_TRADES_TOTALES *  * 100) if PAP100) if PAPER_TRER_TRADESADES_TOTALES > 0_TOTALES > 0 else  else 00
                print
                print(f"📈(f"📈 RESUMEN RESUMEN [{now_str} [{now_str}]: Balance={PAPER_BALANCE:.2f} | Trades={PAPER_TRADES_TOTALES]: Balance={PAPER_BALANCE:.2f} | Trades={PAPER_TRADES_TOTALES} | Winrate} |={winrate:. Winrate={winrate:.1f}% | PF1f}% | PF={ULTIMO_PRO={ULTIMO_PROFITFIT_FACTOR:._FACTOR:.2f}")
            
            time.sleep(SLEEP2f}")
            
            time.sleep(SLEEP_SECONDS_SECONDS)
        except Exception)
        except Exception as e as e:
            print(f:
            print(f"❌ ERROR"❌ ERROR CRÍ CRÍTICO: {TICO: {ee}")
            import traceback}")
            import traceback
            traceback.print_exc
            traceback.print_exc()
            time.sleep(30)

if __()
            time.sleep(name__30)

if __ == '__name__ == '__main__':
    run_bot()
main__':
    run_bot()
