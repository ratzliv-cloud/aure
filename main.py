c# BOT TRADING V99.40 – GEMINI VISION (MULTI-TRADES + BARRIDOS)
# MODELO MULTIMODAL CON RAZONAMIENTO HOLÍSTICO (TEXTO + IMAGEN)
# ==============================================================================
import os, time, requests, json, re, numpy as np, pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import google.generativeai as genai
import json_repair

# ====== CONFIGURACIÓN DE GOOGLE GEMINI ======
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Falta GEMINI_API_KEY en las variables de entorno.")

genai.configure(api_key=GEMINI_API_KEY)
# Puedes usar "gemini-2.0-flash" (rápido y excelente en visión) o "gemini-1.5-pro" (mayor razonamiento)
MODELO_IA = "gemini-2.0-flash" 

# ====== MEMORIA ======
MEMORY_FILE = "memoria_bot_gemini.json"

def convertir_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): return float(obj)
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convertir_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [convertir_serializable(item) for item in obj]
    return obj

def guardar_memoria():
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS
    data = {
        "TRADE_HISTORY": TRADE_HISTORY,
        "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
        "ADAPTIVE_SL_MULT": ADAPTIVE_SL_MULT,
        "ADAPTIVE_TP1_MULT": ADAPTIVE_TP1_MULT,
        "ADAPTIVE_TRAILING_MULT": ADAPTIVE_TRAILING_MULT,
        "PAPER_BALANCE": PAPER_BALANCE,
        "PAPER_WIN": PAPER_WIN,
        "PAPER_LOSS": PAPER_LOSS,
        "PAPER_TRADES_TOTALES": PAPER_TRADES_TOTALES,
        "ULTIMO_APRENDIZAJE": ULTIMO_APRENDIZAJE,
        "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS
    }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        print("💾 Memoria guardada")
    except Exception as e:
        print(f"Error guardando memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS

    if not os.path.exists(MEMORY_FILE):
        print("📁 Nueva memoria (primer inicio)")
        return
    try:
        with open(MEMORY_FILE, "r") as f: data = json.load(f)
        TRADE_HISTORY = data.get("TRADE_HISTORY", [])
        REGLAS_APRENDIDAS = data.get("REGLAS_APRENDIDAS", REGLAS_APRENDIDAS)
        ADAPTIVE_SL_MULT = data.get("ADAPTIVE_SL_MULT", ADAPTIVE_SL_MULT)
        ADAPTIVE_TP1_MULT = data.get("ADAPTIVE_TP1_MULT", ADAPTIVE_TP1_MULT)
        ADAPTIVE_TRAILING_MULT = data.get("ADAPTIVE_TRAILING_MULT", ADAPTIVE_TRAILING_MULT)
        PAPER_BALANCE = data.get("PAPER_BALANCE", PAPER_BALANCE)
        PAPER_WIN = data.get("PAPER_WIN", 0)
        PAPER_LOSS = data.get("PAPER_LOSS", 0)
        PAPER_TRADES_TOTALES = data.get("PAPER_TRADES_TOTALES", 0)
        ULTIMO_APRENDIZAJE = data.get("ULTIMO_APRENDIZAJE", 0)
        TOKENS_ACUMULADOS = data.get("TOKENS_ACUMULADOS", 0)
        print(f"🧠 Memoria cargada: {PAPER_TRADES_TOTALES} trades cerrados.")
    except Exception as e:
        print(f"Error cargando memoria: {e}")

def parse_json_seguro(raw):
    if not raw or raw.strip() == "": return None
    try:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except:
        return None

# =================== CONFIGURACIÓN ===================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
LEVERAGE = 10
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3

DEFAULT_SL_MULT = 1.2
DEFAULT_TP1_MULT = 1.5
DEFAULT_TRAILING_MULT = 1.8
PORCENTAJE_CIERRE_TP1 = 0.5

PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_ACTIVE_TRADES = {}
TRADE_COUNTER = 0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None

ADAPTIVE_SL_MULT = DEFAULT_SL_MULT
ADAPTIVE_TP1_MULT = DEFAULT_TP1_MULT
ADAPTIVE_TRAILING_MULT = DEFAULT_TRAILING_MULT
ULTIMO_APRENDIZAJE = 0
REGLAS_APRENDIDAS = "Aún no hay trades. Busca confluencia entre tendencia, patrones y barridos."

ULTIMA_DECISION = "Hold"
ULTIMA_MOTIVO = "Esperando señal"
TOKENS_ACUMULADOS = 0
CONTADOR_CICLOS = 0

# =================== COMUNICACIÓN TELEGRAM ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e: print(f"Error Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta_imagen, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
    except Exception as e: print(f"Error imagen: {e}")

def reporte_estado():
    pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
    winrate = (PAPER_WIN / PAPER_TRADES_TOTALES * 100) if PAPER_TRADES_TOTALES > 0 else 0
    mensaje = (
        f"📊 **REPORTE DE ESTADO**\n"
        f"💰 Balance: {PAPER_BALANCE:.2f} USDT\n"
        f"📈 PnL Global: {pnl_global:+.2f} USDT\n"
        f"🎯 Winrate: {winrate:.1f}% ({PAPER_WIN}W/{PAPER_LOSS}L)\n"
        f"⚡ Activos: {len(PAPER_ACTIVE_TRADES)}/{MAX_CONCURRENT_TRADES}\n"
        f"🧠 Modelo: {MODELO_IA}\n"
        f"🔢 Tokens consumidos: {TOKENS_ACUMULADOS}"
    )
    telegram_mensaje(mensaje)

# =================== DATOS E INDICADORES ===================
def obtener_velas(limit=150):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=20)
        data_json = r.json()
        if data_json.get("retCode") != 0: return pd.DataFrame()
        result = data_json.get("result")
        if not result or "list" not in result: return pd.DataFrame()
        df = pd.DataFrame(result["list"][::-1], columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']: df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"❌ Excepción obtener_velas: {e}")
        return pd.DataFrame()

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

def detectar_zonas_mercado(df, idx=-2, ventana_macro=120):
    if df.empty or len(df) < 40: return 0, 0, 0, 0, "LATERAL", "LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-ventana_macro:] if len(df_eval) >= ventana_macro else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    micro_tendencia = 'CAYENDO' if micro_slope < -0.2 else 'SUBIENDO' if micro_slope > 0.2 else 'LATERAL'
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tendencia, micro_tendencia

# =================== MOTOR HOLÍSTICO DE ANÁLISIS ===================
def analizar_anatomia_vela(v):
    rango = v['high'] - v['low']
    if rango == 0: return "Doji Plano (0%)"
    c_pct = (abs(v['close'] - v['open']) / rango) * 100
    s_sup = ((v['high'] - max(v['close'], v['open'])) / rango) * 100
    s_inf = ((min(v['close'], v['open']) - v['low']) / rango) * 100
    color = "VERDE" if v['close'] > v['open'] else "ROJA"
    return f"{color} (Cuerpo:{c_pct:.0f}% | M.Sup:{s_sup:.0f}% | M.Inf:{s_inf:.0f}%)"

def analizar_patrones_conjuntos(df, idx):
    if idx < 3 or df.empty: return "Datos insuficientes"
    v3, v2, v1 = df.iloc[idx], df.iloc[idx-1], df.iloc[idx-2]
    r3 = v3['high'] - v3['low']
    c3_pct = (abs(v3['close'] - v3['open']) / r3) * 100 if r3 > 0 else 0
    sup3 = ((v3['high'] - max(v3['close'], v3['open'])) / r3) * 100 if r3 > 0 else 0
    inf3 = ((min(v3['close'], v3['open']) - v3['low']) / r3) * 100 if r3 > 0 else 0
    verde3, verde2, verde1 = v3['close'] > v3['open'], v2['close'] > v2['open'], v1['close'] > v1['open']
    patrones = []
    
    if not verde1 and verde3 and v3['close'] > (v1['open']+v1['close'])/2: patrones.append("🌟 ESTRELLA DE LA MAÑANA")
    if verde1 and not verde3 and v3['close'] < (v1['open']+v1['close'])/2: patrones.append("🌟 ESTRELLA DEL ATARDECER")
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']: patrones.append("🐂 ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']: patrones.append("🐻 ENVOLVENTE BAJISTA")
    elif inf3 > 60 and c3_pct < 25 and sup3 < 15: patrones.append("🔨 MARTILLO")
    elif sup3 > 60 and c3_pct < 25 and inf3 < 15: patrones.append("🌠 ESTRELLA FUGAZ")
    return " | ".join(patrones) if patrones else "Consolidación normal"

def generar_descripcion_nison(df, idx=-2):
    if df.empty or len(df) < abs(idx)+1: return "Datos insuficientes", 0
    vela_actual = df.iloc[idx]
    precio, atr, ema20 = vela_actual['close'], df['atr'].iloc[idx], df['ema20'].iloc[idx]
    soporte, resistencia, _, _, tendencia, micro = detectar_zonas_mercado(df, idx)
    patrones_generales = analizar_patrones_conjuntos(df, idx)

    anat_v1 = analizar_anatomia_vela(df.iloc[idx-2]) if idx-2 >= 0 else "N/A"
    anat_v2 = analizar_anatomia_vela(df.iloc[idx-1]) if idx-1 >= 0 else "N/A"
    anat_v3 = analizar_anatomia_vela(df.iloc[idx]) if idx >= 0 else "N/A"

    margen_fakeout = atr * 0.4
    if precio > ema20:
        if vela_actual['low'] < (ema20 - margen_fakeout): rol_ema = "🔥 BARRIDO EN EMA20 (alcista): mecha perforando por debajo. Señal de compra."
        else: rol_ema = "Cabalgando SOBRE EMA20."
    else:
        if vela_actual['high'] > (ema20 + margen_fakeout): rol_ema = "🔥 BARRIDO EN EMA20 (bajista): mecha perforando por encima. Señal de venta."
        else: rol_ema = "Presionado BAJO EMA20."

    polaridad = f"Precio: {precio:.2f}. "
    if vela_actual['low'] < soporte and precio > soporte: polaridad += f"🔥 SPRING (barrido en soporte {soporte:.2f}) -> probable alza."
    elif vela_actual['high'] > resistencia and precio < resistencia: polaridad += f"🚨 UPTHRUST (barrido en resistencia {resistencia:.2f}) -> probable baja."

    descripcion = f"""
=== MATRIZ DE CONFLUENCIA Y TRAMPAS DE LIQUIDEZ (5M) ===
1. ESTRUCTURA: Tendencia Macro: {tendencia} | Impulso Micro: {micro}
2. S/R Y TRAMPAS: {polaridad}
3. EMA20: Accion: {rol_ema}
4. VELAS EXACTAS (V1/V2/Gatillo): {anat_v1} -> {anat_v2} -> {anat_v3}
5. PATRÓN COMPUESTO: {patrones_generales}
"""
    return descripcion, atr

# =================== GRÁFICOS PARA LA IA ===================
def generar_grafico_contexto(df, soporte, resistencia, slope, intercept):
    if df.empty: return None
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
        
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte 40-velas')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia 40-velas')
    ax.plot(x, intercept + slope * x, color='white', linestyle='-.', lw=1.5, alpha=0.6, label='Tendencia Global')
    
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
        
    ax.set_title("Radiografía del Mercado", color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15)
    ax.legend(loc='lower left', facecolor='black', labelcolor='white')
    
    plt.tight_layout()
    ruta = "/tmp/chart_contexto_ia.png"
    plt.savefig(ruta, dpi=100)
    plt.close()
    return ruta

# =================== IA GEMINI VISION (MULTIMODAL) ===================
def analizar_con_gemini_vision(descripcion, atr, reglas_aprendidas, ruta_imagen):
    global TOKENS_ACUMULADOS
    try:
        system_msg = f"""Eres un Maestro del Price Action y un experto leyendo gráficos de Velas Japonesas.
Tienes acceso a la "MATRIZ DE CONFLUENCIA" (texto) y a la IMAGEN del gráfico actual. 
La imagen tiene EMA20 (amarillo), Soportes (cyan) y Resistencias (magenta).

🔥 REGLA EVOLUTIVA DE TU MENTOR: "{reglas_aprendidas}"

LÓGICA OPERATIVA:
- Evalúa VISUALMENTE si el precio está rompiendo zonas o si está haciendo una mecha de trampa (fakeout).
- Busca barridos de liquidez: si ves que una vela dejó una mecha larga fuera de la EMA, soporte o resistencia y se devuelve, es la confirmación más fuerte.
- Si el contexto es dudoso, ruidoso, o las velas actuales anulan la tendencia (ej: tendencia alcista pero gran envolvente roja al final), escoge "Hold".

Responde ÚNICAMENTE con JSON válido."""

        user_msg = f"{descripcion}\n\nATR: {atr:.2f}. \nAnaliza el gráfico adjunto y el texto para tomar la mejor decisión (Buy, Sell o Hold)."

        # Configuración forzando JSON en Gemini
        gen_config = genai.types.GenerationConfig(
            temperature=0.0, 
            response_mime_type="application/json",
            response_schema={"type": "object", "properties": {
                "decision": {"type": "string", "enum": ["Buy", "Sell", "Hold"]},
                "patron": {"type": "string"},
                "razones": {"type": "array", "items": {"type": "string"}},
                "sl_mult": {"type": "number"},
                "tp1_mult": {"type": "number"},
                "trailing_mult": {"type": "number"}
            }, "required": ["decision", "patron", "razones", "sl_mult", "tp1_mult", "trailing_mult"]}
        )

        model = genai.GenerativeModel(model_name=MODELO_IA, system_instruction=system_msg, generation_config=gen_config)
        img = Image.open(ruta_imagen)
        
        response = model.generate_content([user_msg, img])
        
        # Conteo de tokens
        if response.usage_metadata:
            total_tokens = response.usage_metadata.total_token_count
            TOKENS_ACUMULADOS += total_tokens
            print(f"📊 Tokens usados: {total_tokens} | Acumulado={TOKENS_ACUMULADOS}")
        else: total_tokens = 0

        raw = response.text
        datos = parse_json_seguro(raw)
        
        if not datos: return "Hold", ["JSON inválido"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT), total_tokens

        decision = datos.get("decision", "Hold")
        sl_m = max(0.5, min(2.5, float(datos.get("sl_mult", 1.2))))
        tp_m = max(0.8, min(3.0, float(datos.get("tp1_mult", 1.5))))
        tr_m = max(1.0, min(3.0, float(datos.get("trailing_mult", 1.8))))

        return decision, datos.get("razones", []), datos.get("patron", ""), (sl_m, tp_m, tr_m), total_tokens

    except Exception as e:
        print(f"❌ Error IA Gemini: {e}")
        return "Hold", [f"Error: {e}"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT), 0

# =================== AUTOAPRENDIZAJE (GEMINI TEXTO) ===================
def aprender_de_trades():
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT, ULTIMO_APRENDIZAJE, REGLAS_APRENDIDAS, TOKENS_ACUMULADOS
    total = len(TRADE_HISTORY)
    if total < 10 or (total - ULTIMO_APRENDIZAJE) < 10: return
    print("🧠 Iniciando autoaprendizaje con Gemini...")
    ultimos = TRADE_HISTORY[-10:]
    wins = sum(1 for t in ultimos if t['resultado_win'])
    resumen = "\n".join([f"{i+1}. {t['decision']} {'WIN✅' if t['resultado_win'] else 'LOSS❌'} | PnL:{t['pnl']:.2f} | {t.get('patron','?')}" for i, t in enumerate(ultimos)])

    system_msg = "Eres el Mentor de Trading de la IA. Analiza 10 trades recientes y genera una instrucción (regla) de mejora. Responde sólo en JSON."
    user_msg = f"Winrate: {wins*10}%.\nHistorial:\n{resumen}\nGenera la nueva regla y ajusta multiplicadores."

    try:
        gen_config = genai.types.GenerationConfig(
            temperature=0.4, response_mime_type="application/json",
            response_schema={"type": "object", "properties": {
                "analisis": {"type": "string"}, "nueva_regla": {"type": "string"},
                "sl_mult_sugerido": {"type": "number"}, "tp1_mult_sugerido": {"type": "number"}, "trailing_mult_sugerido": {"type": "number"}
            }, "required": ["analisis", "nueva_regla", "sl_mult_sugerido", "tp1_mult_sugerido", "trailing_mult_sugerido"]}
        )
        model = genai.GenerativeModel(model_name=MODELO_IA, system_instruction=system_msg, generation_config=gen_config)
        resp = model.generate_content(user_msg)
        
        if resp.usage_metadata: TOKENS_ACUMULADOS += resp.usage_metadata.total_token_count
        datos = parse_json_seguro(resp.text)
        
        if datos:
            REGLAS_APRENDIDAS = datos.get("nueva_regla", REGLAS_APRENDIDAS)
            ADAPTIVE_SL_MULT = max(0.5, min(2.5, float(datos.get("sl_mult_sugerido", ADAPTIVE_SL_MULT))))
            ADAPTIVE_TP1_MULT = max(0.8, min(3.0, float(datos.get("tp1_mult_sugerido", ADAPTIVE_TP1_MULT))))
            ADAPTIVE_TRAILING_MULT = max(1.0, min(3.0, float(datos.get("trailing_mult_sugerido", ADAPTIVE_TRAILING_MULT))))
            msg = f"🧠 AUTOAPRENDIZAJE (10 trades)\n📊 Winrate: {wins*10}%\n📜 Regla: \"{REGLAS_APRENDIDAS}\"\n⚙️ SL:{ADAPTIVE_SL_MULT:.2f} TP1:{ADAPTIVE_TP1_MULT:.2f} TR:{ADAPTIVE_TRAILING_MULT:.2f}"
            telegram_mensaje(msg)
        ULTIMO_APRENDIZAJE = total
        guardar_memoria()
    except Exception as e: print(f"Error aprendizaje: {e}")

# =================== GRÁFICOS DE TELEGRAM ===================
def generar_grafico_operacion(df, trade_info, soporte, resistencia, slope, intercept, tipo="Entrada"):
    if df.empty: return None
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', lw=2)
    ax.axhline(resistencia, color='magenta', ls='--', lw=2)
    ax.plot(x, intercept + slope * x, color='white', linestyle='-.', lw=1.5, alpha=0.6)
    if 'ema20' in df_plot.columns: ax.plot(x, df_plot['ema20'], 'yellow', lw=2)
    
    if tipo == "Entrada":
        decision = trade_info['decision']
        p_act = df_plot['close'].iloc[-2]
        ax.scatter(len(df_plot)-2, p_act + (-30 if decision=='Buy' else 30), s=400, marker='^' if decision=='Buy' else 'v', c='lime' if decision=='Buy' else 'red', zorder=5)
        txt = f"[#{trade_info['id']}] {decision.upper()}\nSL:{trade_info['sl_inicial']:.2f} TP1:{trade_info['tp1']:.2f}"
    else:
        ax.axhline(trade_info['entrada'], color='blue', ls=':', lw=2, label='Entrada')
        ax.axhline(trade_info['sl_actual'], color='white', ls=':', lw=2, label='Salida')
        txt = f"[#{trade_info['id']}] {'WIN' if trade_info['resultado_win'] else 'LOSS'} | PnL: {trade_info['pnl']:.2f} USD"
    
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    ruta = f"/tmp/chart_{tipo.lower()}_{trade_info['id']}.png"
    plt.savefig(ruta, dpi=120)
    plt.close()
    return ruta

# =================== GESTIÓN MULTI-TRADE ===================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if drawdown <= -MAX_DAILY_DRAWDOWN_PCT and not PAPER_STOPPED_TODAY:
        telegram_mensaje("🛑 Drawdown diario máximo. Bot pausado por hoy.")
        PAPER_STOPPED_TODAY = True
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, atr, razones, patron, multis_ia, df, sop, res, slo, inter):
    global PAPER_BALANCE, TRADE_COUNTER
    if len(PAPER_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES: return False
    for t in PAPER_ACTIVE_TRADES.values():
        if t['decision'] == decision and abs(t['entrada'] - precio) < atr * 0.2: return False
        
    sl_m = (multis_ia[0] * 0.6) + (ADAPTIVE_SL_MULT * 0.4)
    tp_m = (multis_ia[1] * 0.6) + (ADAPTIVE_TP1_MULT * 0.4)
    tr_m = (multis_ia[2] * 0.6) + (ADAPTIVE_TRAILING_MULT * 0.4)
    
    sl_inicial = precio - (atr * sl_m) if decision == "Buy" else precio + (atr * sl_m)
    tp1 = precio + (atr * tp_m) if decision == "Buy" else precio - (atr * tp_m)
    distancia = abs(precio - sl_inicial)
    if distancia == 0: return False
    
    TRADE_COUNTER += 1
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    size_btc = min((riesgo_usd / distancia) * precio, PAPER_BALANCE * LEVERAGE) / precio
    
    trade = {
        "id": TRADE_COUNTER, "decision": decision, "entrada": precio,
        "sl_inicial": sl_inicial, "tp1": tp1, "trailing_mult": tr_m,
        "size_btc": size_btc, "size_restante": size_btc, "tp1_ejecutado": False,
        "pnl_parcial": 0.0, "sl_actual": sl_inicial, "max_precio": precio,
        "patron": patron, "atr_entrada": atr
    }
    PAPER_ACTIVE_TRADES[TRADE_COUNTER] = trade
    msg = f"📌 [#{TRADE_COUNTER}] {decision.upper()} a {precio:.2f}\nSL:{sl_inicial:.2f} TP1:{tp1:.2f}\n{patron}"
    print(msg)
    
    ruta_img = generar_grafico_operacion(df, trade, sop, res, slo, inter, "Entrada")
    if ruta_img: telegram_enviar_imagen(ruta_img, msg)
    return True

def paper_revisar_sl_tp(df, sop, res, slo, inter):
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY
    if df.empty: return
    h, l = df['high'].iloc[-1], df['low'].iloc[-1]
    trades_a_cerrar = []
    
    for t_id, t in PAPER_ACTIVE_TRADES.items():
        cerrar, motivo = False, ""
        t['max_precio'] = max(t['max_precio'], h) if t['decision'] == "Buy" else min(t['max_precio'], l)
        
        if not t['tp1_ejecutado']:
            if (t['decision'] == "Buy" and h >= t['tp1']) or (t['decision'] == "Sell" and l <= t['tp1']):
                beneficio = abs(t['tp1'] - t['entrada']) * (t['size_btc'] * PORCENTAJE_CIERRE_TP1)
                t['pnl_parcial'] = beneficio
                PAPER_BALANCE += beneficio
                t['size_restante'] *= (1 - PORCENTAJE_CIERRE_TP1)
                t['tp1_ejecutado'] = True
                t['sl_actual'] = t['entrada']
                telegram_mensaje(f"🎯 TP1 #{t_id}: +{beneficio:.2f} USD, SL a BE")
                
        if t['tp1_ejecutado']:
            n_sl = t['max_precio'] - (t['atr_entrada'] * t['trailing_mult']) if t['decision'] == "Buy" else t['max_precio'] + (t['atr_entrada'] * t['trailing_mult'])
            if (t['decision'] == "Buy" and n_sl > t['sl_actual']) or (t['decision'] == "Sell" and n_sl < t['sl_actual']):
                t['sl_actual'] = n_sl
            if (t['decision'] == "Buy" and l <= t['sl_actual']) or (t['decision'] == "Sell" and h >= t['sl_actual']):
                cerrar, motivo = True, "Trailing Stop"
        else:
            if (t['decision'] == "Buy" and l <= t['sl_inicial']) or (t['decision'] == "Sell" and h >= t['sl_inicial']):
                cerrar, motivo = True, "Stop Loss"
                t['sl_actual'] = t['sl_inicial']
                
        if cerrar:
            pnl_rest = (t['sl_actual'] - t['entrada']) * t['size_restante'] if t['decision'] == "Buy" else (t['entrada'] - t['sl_actual']) * t['size_restante']
            pnl_total = t['pnl_parcial'] + pnl_rest
            PAPER_BALANCE += pnl_rest
            PAPER_TRADES_TOTALES += 1
            win = pnl_total > 0
            if win: PAPER_WIN += 1 
            else: PAPER_LOSS += 1
            
            t['pnl'], t['resultado_win'] = pnl_total, win
            trades_a_cerrar.append(t_id)
            TRADE_HISTORY.append({"fecha": datetime.now(timezone.utc).isoformat(), "decision": t['decision'], "patron": t['patron'], "pnl": pnl_total, "resultado_win": win})
            guardar_memoria()
            
            msg = f"📤 [{motivo}] #{t_id} {t['decision'].upper()} cerrado | PnL: {pnl_total:.2f} USD"
            print(msg)
            ruta_img = generar_grafico_operacion(df, t, sop, res, slo, inter, "Salida")
            if ruta_img: telegram_enviar_imagen(ruta_img, msg)
            
    for t_id in trades_a_cerrar: del PAPER_ACTIVE_TRADES[t_id]
    if trades_a_cerrar and PAPER_TRADES_TOTALES % 10 == 0: aprender_de_trades()

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, CONTADOR_CICLOS, TOKENS_ACUMULADOS
    cargar_memoria()
    print(f"🤖 BOT V99.40 INICIADO - Modelo: {MODELO_IA} (Vision Multimodal)")
    telegram_mensaje(f"🤖 BOT V99.40 INICIADO - Modelo: {MODELO_IA} Vision\nEl bot ahora 'VE' los gráficos para confirmar barridos de liquidez.")
    ultima_vela = None
    
    while True:
        try:
            df_raw = obtener_velas()
            if df_raw.empty: time.sleep(30); continue
            df = calcular_indicadores(df_raw)
            if df.empty: time.sleep(30); continue
            
            vela_cerrada = df.index[-2]
            precio, atr = df['close'].iloc[-1], df['atr'].iloc[-1]
            sop, res, slo, inter, tend, micro = detectar_zonas_mercado(df)
            
            activos = len(PAPER_ACTIVE_TRADES)
            print(f"\n💓 Heartbeat | P:{precio:.2f} | Activos:{activos}/{MAX_CONCURRENT_TRADES} | Cerrados:{PAPER_TRADES_TOTALES} | PnL:{PAPER_BALANCE - PAPER_BALANCE_INICIAL:+.2f}")
            
            CONTADOR_CICLOS += 1
            if CONTADOR_CICLOS % 10 == 0: reporte_estado()
            
            if activos < MAX_CONCURRENT_TRADES and ultima_vela != vela_cerrada:
                desc, atr_val = generar_descripcion_nison(df)
                if not desc or len(desc) < 50:
                    time.sleep(SLEEP_SECONDS); continue
                    
                print(f"--- Generando y Evaluando Gráfico Vision {vela_cerrada.strftime('%H:%M')} ---")
                
                # 1. Generar la imagen limpia del contexto para la IA
                ruta_contexto = generar_grafico_contexto(df, sop, res, slo, inter)
                
                # 2. Enviar a Gemini (Texto + Imagen)
                if ruta_contexto:
                    decision, razones, patron, multis, tokens_usados = analizar_con_gemini_vision(desc, atr_val, REGLAS_APRENDIDAS, ruta_contexto)
                    ULTIMA_DECISION, ULTIMO_MOTIVO = decision, razones[0] if razones else ""
                    
                    if decision in ["Buy","Sell"] and risk_management_check():
                        paper_abrir_posicion(decision, precio, atr_val, razones, patron, multis, df, sop, res, slo, inter)
                    else:
                        print(f"⏸️ Hold o Riesgo pausado: {ULTIMO_MOTIVO[:80]}")
                
                ultima_vela = vela_cerrada
                
            if PAPER_ACTIVE_TRADES:
                sop, res, slo, inter, _, _ = detectar_zonas_mercado(df, -1)
                paper_revisar_sl_tp(df, sop, res, slo, inter)
                
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ Error loop: {e}")
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
