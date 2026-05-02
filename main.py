# BOT TRADING V99.43 – QWEN3-VL-32B-Instruct (EDICIÓN COMPLETA CORREGIDA + LOGS)
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

# =================== VISIÓN IA Y GRÁFICOS ===================
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
        print(f"❌ Error en llamada a IA: {e}")
        return "Hold", "Error API", 0, 0, 0, "EMA20"

# =================== GESTIÓN EJECUCIÓN ===================
def paper_abrir_posicion(decision, precio, atr, razon, sl_ia, tp1_ia, tp2_ia, logic_ia, df, sop, res, slo, inter):
    global PAPER_BALANCE, TRADE_COUNTER
    if len(PAPER_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES: return
    
    sl_final = float(sl_ia) if sl_ia else (precio - atr*1.5 if decision=="Buy" else precio + atr*1.5)
    if decision == "Buy" and sl_final >= precio: sl_final = precio - atr
    if decision == "Sell" and sl_final <= precio: sl_final = precio + atr

    distancia = abs(precio - sl_final)
    size_btc = (PAPER_BALANCE * RISK_PER_TRADE) / distancia
    size_btc = min(size_btc, (PAPER_BALANCE * LEVERAGE) / precio)

    TRADE_COUNTER += 1
    t = {
        "id": TRADE_COUNTER, "decision": decision, "entrada": precio, "sl_inicial": sl_final, "sl_actual": sl_final,
        "tp1": tp1_ia, "tp2": tp2_ia, "trailing_logic": logic_ia, "size_btc": size_btc,
        "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0, "razon": razon
    }
    PAPER_ACTIVE_TRADES[TRADE_COUNTER] = t
    msg = f"🚀 [#{TRADE_COUNTER}] {decision} en {precio:.2f}\nRazon: {razon}"
    print(f"📈 {msg}")
    telegram_mensaje(msg)
    # Generar gráfico simple para Telegram
    fig, ax = plt.subplots(); ax.plot(df.tail(20)['close'].values); plt.savefig("/tmp/in.png"); plt.close()
    telegram_enviar_imagen("/tmp/in.png", msg)

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
                print(f"🎯 TP1 alcanzado en trade #{tid}")
                telegram_mensaje(f"🎯 TP1 #{tid} hit. SL a Breakeven.")
        
        # TP2 (30%)
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2']:
            if (t['decision']=="Buy" and h>=t['tp2']) or (t['decision']=="Sell" and l<=t['tp2']):
                ganancia = abs(t['tp2'] - t['entrada']) * (t['size_btc'] * PCT_TP2)
                t['pnl_parcial'] += ganancia
                PAPER_BALANCE += ganancia
                t['tp2_ejecutado'] = True
                print(f"🎯 TP2 alcanzado en trade #{tid}")
                telegram_mensaje(f"🎯 TP2 #{tid} hit.")

        # Trailing / SL
        cerrar, motivo = False, ""
        if t['tp1_ejecutado']:
            if t['decision']=="Buy":
                t['sl_actual'] = max(t['sl_actual'], ema-(atr*0.2) if t['trailing_logic']=="EMA20" else l_prev)
                if l <= t['sl_actual']: cerrar, motivo = True, "Trailing"
            else:
                t['sl_actual'] = min(t['sl_actual'], ema+(atr*0.2) if t['trailing_logic']=="EMA20" else h_prev)
                if h >= t['sl_actual']: cerrar, motivo = True, "Trailing"
        else:
            if (t['decision']=="Buy" and l <= t['sl_inicial']) or (t['decision']=="Sell" and h >= t['sl_inicial']):
                cerrar, motivo = True, "Stop Loss"

        if cerrar:
            pct = 0.20 if t['tp2_ejecutado'] else (0.50 if t['tp1_ejecutado'] else 1.0)
            pnl_f = (t['sl_actual']-t['entrada'])*t['size_btc']*pct if t['decision']=="Buy" else (t['entrada']-t['sl_actual'])*t['size_btc']*pct
            pnl_t = t['pnl_parcial'] + pnl_f
            PAPER_BALANCE += pnl_f
            PAPER_TRADES_TOTALES += 1
            if pnl_t > 0: PAPER_WIN += 1
            else: PAPER_LOSS += 1
            TRADE_HISTORY.append({"pnl": pnl_t, "resultado_win": pnl_t > 0, "decision": t['decision'], "razon": t['razon']})
            cerrar_ids.append(tid)
            print(f"📤 CERRADO #{tid} ({motivo}) | PnL: {pnl_t:.2f} USDT | Balance: {PAPER_BALANCE:.2f}")
            telegram_mensaje(f"📤 CERRADO #{tid} ({motivo}). PnL: {pnl_t:.2f} USDT")
            reporte_estado()

    for tid in cerrar_ids: del PAPER_ACTIVE_TRADES[tid]
    if len(TRADE_HISTORY) > 0 and len(TRADE_HISTORY) % 10 == 0: aprender_de_trades()

# =================== AUTOAPRENDIZAJE Y LOOP ===================
def aprender_de_trades():
    global REGLAS_APRENDIDAS, ULTIMO_APRENDIZAJE, ULTIMO_PROFIT_FACTOR
    ult = TRADE_HISTORY[-10:]
    gan = sum(t['pnl'] for t in ult if t['pnl']>0)
    per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
    ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
    prompt = f"Analiza estos 10 trades y dame una lección corta: {json.dumps(ult)}"
    try:
        resp = client.chat.completions.create(model=MODELO_VISION, messages=[{"role":"user","content":prompt}])
        REGLAS_APRENDIDAS = resp.choices[0].message.content
        print(f"🧠 APRENDIZAJE: {REGLAS_APRENDIDAS}")
        telegram_mensaje(f"🧠 APRENDIZAJE: {REGLAS_APRENDIDAS}")
        ULTIMO_APRENDIZAJE = PAPER_TRADES_TOTALES
        guardar_memoria()
    except Exception as e:
        print(f"❌ Error en aprendizaje: {e}")

def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY, PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY = hoy, PAPER_BALANCE, False
        print(f"📅 Nuevo día: {hoy}. Balance inicial: {PAPER_DAILY_START_BALANCE:.2f}")
    if (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE <= -MAX_DAILY_DRAWDOWN_PCT:
        PAPER_STOPPED_TODAY = True
        print(f"🚨 Drawdown diario superado ({MAX_DAILY_DRAWDOWN_PCT*100}%). Bot detenido hasta mañana.")
    return not PAPER_STOPPED_TODAY

def run_bot():
    cargar_memoria()
    print("🤖 BOT V99.43 INICIADO - Modo LOGS DETALLADOS")
    telegram_mensaje("🤖 Bot V99.43 Online - Estructural Visual")
    ultima_vela = None
    iteracion = 0
    while True:
        try:
            iteracion += 1
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n🔄 [{now_str}] Ciclo #{iteracion} - Iniciando...")
            
            # Obtener datos
            df_raw = obtener_velas()
            if df_raw.empty:
                print("⚠️ No se pudieron obtener velas. Reintentando...")
                time.sleep(SLEEP_SECONDS)
                continue
            df = calcular_indicadores(df_raw)
            if df.empty:
                print("⚠️ DataFrame vacío tras indicadores. Reintentando...")
                time.sleep(SLEEP_SECONDS)
                continue
            
            precio_actual = df['close'].iloc[-1]
            print(f"📊 Precio actual: {precio_actual:.2f} | Trades activos: {len(PAPER_ACTIVE_TRADES)}/{MAX_CONCURRENT_TRADES} | Balance: {PAPER_BALANCE:.2f}")
            
            # Verificar nueva vela
            vela_c = df.index[-2]
            if ultima_vela is None:
                ultima_vela = vela_c
                print(f"🕯️ Vela base establecida: {vela_c}")
            
            # Condiciones para nuevo análisis
            if len(PAPER_ACTIVE_TRADES) < MAX_CONCURRENT_TRADES and ultima_vela != vela_c:
                print(f"🕯️ Nueva vela detectada: {vela_c} (anterior: {ultima_vela})")
                print("🔍 Ejecutando análisis de mercado...")
                sop, res, slo, inter, t, m = detectar_zonas_mercado(df)
                desc, atr = generar_descripcion_nison(df)
                print("🧠 Enviando gráfico a IA para decisión...")
                img = generar_grafico_para_vision(df, sop, res, slo, inter, df['close'].iloc[-1])
                dec, raz, sl, tp1, tp2, log = analizar_con_qwen(desc, atr, REGLAS_APRENDIDAS, img)
                print(f"🤖 Decisión IA: {dec} | Razón: {raz[:60]}...")
                
                if dec in ["Buy","Sell"]:
                    print(f"✅ IA sugiere operación {dec}. Verificando risk management...")
                    if risk_management_check():
                        print(f"🚀 Abriendo posición {dec}...")
                        paper_abrir_posicion(dec, df['close'].iloc[-1], atr, raz, sl, tp1, tp2, log, df, sop, res, slo, inter)
                    else:
                        print("⛔ Risk management bloqueó la operación (drawdown diario alcanzado).")
                else:
                    print(f"⏸️ IA decidió HOLD. Motivo: {raz[:100] if raz else 'No especificado'}")
                ultima_vela = vela_c
            elif ultima_vela == vela_c:
                print("⏳ Misma vela, no se repite análisis. Esperando nueva vela...")
            else:
                print(f"⏸️ Máximo de trades concurrentes alcanzado ({len(PAPER_ACTIVE_TRADES)}/{MAX_CONCURRENT_TRADES}). No se analiza nueva entrada.")
            
            # Revisar SL/TP de trades activos
            if PAPER_ACTIVE_TRADES:
                print("🔎 Revisando condiciones de SL/TP en trades activos...")
                paper_revisar_sl_tp(df)
            else:
                print("💤 Sin trades activos en este momento.")
            
            # Mostrar resumen cada 10 ciclos o si hay cambios significativos
            if iteracion % 10 == 0:
                winrate = (PAPER_WIN / PAPER_TRADES_TOTALES * 100) if PAPER_TRADES_TOTALES > 0 else 0
                print(f"📈 RESUMEN [{now_str}]: Balance={PAPER_BALANCE:.2f} | Trades={PAPER_TRADES_TOTALES} | Winrate={winrate:.1f}% | PF={ULTIMO_PROFIT_FACTOR:.2f}")
            
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
