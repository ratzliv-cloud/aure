# BOT TRADING V99.40 – QWEN3-VL-32B-Instruct vía SiliconFlow (MULTI-TRADES + VISIÓN + MEMORIA DINÁMICA + PF ADAPTIVE)
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

# =================== CONFIGURACIÓN DE SILICONFLOW ===================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("Falta SILICONFLOW_API_KEY. Obtén una en https://cloud.siliconflow.com")

SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
MODELO_VISION = "Qwen/Qwen3-VL-32B-Instruct"

# ====== MEMORIA ======
MEMORY_FILE = "memoria_bot.json"

def convertir_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convertir_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convertir_serializable(item) for item in obj]
    else:
        return obj

def guardar_memoria():
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS
    # Creamos una copia de PAPER_ACTIVE_TRADES sin los datos grandes para memoria
    active_trades_meta = {}
    for tid, t in PAPER_ACTIVE_TRADES.items():
        active_trades_meta[tid] = {
            "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
            "patron": t.get("patron", ""), "tp1_ejecutado": t["tp1_ejecutado"]
        }
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
        "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS,
        "PAPER_ACTIVE_META": active_trades_meta,
        "ULTIMO_PROFIT_FACTOR": ULTIMO_PROFIT_FACTOR
    }
    data_serializable = convertir_serializable(data)
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data_serializable, f, indent=4)
        print("💾 Memoria guardada")
    except Exception as e:
        print(f"Error guardando memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR

    if not os.path.exists(MEMORY_FILE):
        print("📁 Nueva memoria (primer inicio)")
        return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
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
        ULTIMO_PROFIT_FACTOR = data.get("ULTIMO_PROFIT_FACTOR", 1.0)
        print(f"🧠 Memoria cargada: {PAPER_TRADES_TOTALES} trades, último aprendizaje #{ULTIMO_APRENDIZAJE}, PF={ULTIMO_PROFIT_FACTOR:.2f}")
    except Exception as e:
        print(f"Error cargando memoria: {e}")

def parse_json_seguro(raw):
    if not raw or raw.strip() == "":
        return None
    try:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except:
        try:
            stack = []
            start = raw.find('{')
            if start == -1: return None
            end = start
            for i, ch in enumerate(raw[start:], start):
                if ch == '{':
                    stack.append(ch)
                elif ch == '}':
                    if stack:
                        stack.pop()
                        if not stack:
                            end = i
                            break
            if not stack and end > start:
                json_str = raw[start:end+1]
                return json.loads(json_str)
        except:
            pass
        return None

# =================== CONFIGURACIÓN DEL BOT ===================
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
ULTIMO_PROFIT_FACTOR = 1.0
REGLAS_APRENDIDAS = "Aún no hay trades. Busca confluencia entre patrones de velas, tendencia y barridos de liquidez."

ULTIMA_DECISION = "Hold"
ULTIMO_MOTIVO = "Esperando señal"
TOKENS_ACUMULADOS = 0

# =================== COMUNICACIÓN TELEGRAM ===================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        print(f"Error Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta_imagen, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
    except Exception as e:
        print(f"Error imagen: {e}")

def reporte_estado():
    pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
    winrate = (PAPER_WIN / PAPER_TRADES_TOTALES * 100) if PAPER_TRADES_TOTALES > 0 else 0
    activos = len(PAPER_ACTIVE_TRADES)
    mensaje = (
        f"📊 **REPORTE DE ESTADO**\n"
        f"💰 Balance: {PAPER_BALANCE:.2f} USDT\n"
        f"📈 PnL Global: {pnl_global:+.2f} USDT\n"
        f"🎯 Winrate: {winrate:.1f}% ({PAPER_WIN}W/{PAPER_LOSS}L)\n"
        f"🔄 Trades cerrados: {PAPER_TRADES_TOTALES}\n"
        f"⚡ Activos: {activos}/{MAX_CONCURRENT_TRADES}\n"
        f"🧠 Modelo: {MODELO_VISION}\n"
        f"🔢 Tokens consumidos: {TOKENS_ACUMULADOS}\n"
        f"📐 Profit Factor (últimos 10): {ULTIMO_PROFIT_FACTOR:.2f}"
    )
    telegram_mensaje(mensaje)

# =================== DATOS E INDICADORES ===================
def obtener_velas(limit=150):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=20)
        data_json = r.json()
        if data_json.get("retCode") != 0:
            print(f"❌ Error API: {data_json.get('retMsg')}")
            return pd.DataFrame()
        result = data_json.get("result")
        if result is None or "list" not in result:
            return pd.DataFrame()
        lista_velas = result["list"][::-1]
        df = pd.DataFrame(lista_velas, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"❌ Excepción obtener_velas: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    if df.empty:
        return df
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
    if df.empty or len(df) < 40:
        return 0, 0, 0, 0, "LATERAL", "LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-ventana_macro:] if len(df_eval) >= ventana_macro else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    micro_tendencia = 'CAYENDO' if micro_slope < -0.2 else 'SUBIENDO' if micro_slope > 0.2 else 'LATERAL'
    tendencia = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tendencia, micro_tendencia

# =================== MOTOR HOLÍSTICO (VERSIÓN NEUTRAL MEJORADA) ===================
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
    verde3 = v3['close'] > v3['open']
    verde2 = v2['close'] > v2['open']
    verde1 = v1['close'] > v1['open']
    patrones = []
    if not verde1 and verde3 and v3['close'] > (v1['open']+v1['close'])/2: patrones.append("🌟 ESTRELLA DE LA MAÑANA")
    if verde1 and not verde3 and v3['close'] < (v1['open']+v1['close'])/2: patrones.append("🌟 ESTRELLA DEL ATARDECER")
    if verde1 and verde2 and verde3 and v3['close'] > v2['close'] and v2['close'] > v1['close']: patrones.append("🚀 TRES SOLDADOS BLANCOS")
    if not verde1 and not verde2 and not verde3 and v3['close'] < v2['close'] and v2['close'] < v1['close']: patrones.append("🩸 TRES CUERVOS NEGROS")
    if not verde2 and verde3 and v3['close'] > v2['open'] and v3['open'] < v2['close']: patrones.append("🐂 ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close'] < v2['open'] and v3['open'] > v2['close']: patrones.append("🐻 ENVOLVENTE BAJISTA")
    if verde3 and c3_pct > 70 and sup3 < 10: patrones.append("📈 VELA SÓLIDA EN MÁXIMOS")
    elif not verde3 and c3_pct > 70 and inf3 < 10: patrones.append("📉 VELA SÓLIDA EN MÍNIMOS")
    elif c3_pct < 15 and sup3 > 25 and inf3 > 25: patrones.append("⚖️ DOJI")
    elif inf3 > 60 and c3_pct < 25 and sup3 < 15: patrones.append("🔨 MARTILLO")
    elif sup3 > 60 and c3_pct < 25 and inf3 < 15: patrones.append("🌠 ESTRELLA FUGAZ")
    return " | ".join(patrones) if patrones else "Consolidación normal"

def generar_descripcion_nison(df, idx=-2):
    if df.empty or len(df) < abs(idx)+1:
        return "Datos insuficientes", 0

    vela_actual = df.iloc[idx]
    precio = vela_actual['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]

    soporte, resistencia, slope, intercept, tendencia, micro = detectar_zonas_mercado(df, idx)

    anat_v1 = analizar_anatomia_vela(df.iloc[idx-2]) if idx-2 >= 0 else "N/A"
    anat_v2 = analizar_anatomia_vela(df.iloc[idx-1]) if idx-1 >= 0 else "N/A"
    anat_v3 = analizar_anatomia_vela(df.iloc[idx]) if idx >= 0 else "N/A"

    patrones_generales = analizar_patrones_conjuntos(df, idx)

    df_mechas = df.iloc[max(0, idx-7):idx+1] if idx >= 7 else df.iloc[:idx+1]
    if len(df_mechas) >= 3:
        rangos = df_mechas['high'] - df_mechas['low']
        mechas_sup = (df_mechas['high'] - df_mechas[['close','open']].max(axis=1)) / rangos.replace(0, 0.001)
        mechas_inf = (df_mechas[['close','open']].min(axis=1) - df_mechas['low']) / rangos.replace(0, 0.001)
        cluster_txt = f"Mechas sup>55%: {sum(mechas_sup>0.55)} | Mechas inf>55%: {sum(mechas_inf>0.55)}"
    else:
        cluster_txt = "Datos insuficientes para clusters."

    ultimas_10 = df.iloc[max(0, idx-9):idx+1] if idx >= 9 else df.iloc[:idx+1]
    if len(ultimas_10) >= 5:
        sobre_ema = (ultimas_10['close'] > ultimas_10['ema20']).sum()
        pct_sobre = (sobre_ema / len(ultimas_10)) * 100
        lateralidad = f"{len(ultimas_10)} velas: {pct_sobre:.0f}% cierran sobre EMA20"
    else:
        lateralidad = "Datos insuficientes para evaluar EMA20."

    descripcion = f"""
=== DATOS DE MERCADO (sin interpretación) ===

Precio: {precio:.2f}
ATR: {atr:.2f}
EMA20: {ema20:.2f}
Soporte (40 velas): {soporte:.2f}
Resistencia (40 velas): {resistencia:.2f}
Tendencia macro (regresión 120v): {tendencia}
Micro tendencia (8 velas): {micro}

Anatomía de velas (últimas 3):
- Vela -2: {anat_v1}
- Vela -1: {anat_v2}
- Vela 0 (actual): {anat_v3}

Patrones reconocidos (3 velas consecutivas):
{patrones_generales}

Clusters de mechas (últimas 8 velas):
{cluster_txt}

Posición relativa respecto a EMA20:
{lateralidad}
"""
    return descripcion, atr

# =================== GRÁFICO PARA SILICONFLOW ===================
def generar_grafico_para_vision(df, soporte, resistencia, slope, intercept, ultimo_precio=None):
    if df.empty:
        return None
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    ax.plot(x, intercept + slope * x, color='white', linestyle='-.', lw=1.5, alpha=0.6, label='Tendencia')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    ax.legend(loc='lower right', facecolor='black', labelcolor='white')
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
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

# =================== IA QWEN3-VL-32B-INSTRUCT (VISIÓN + TEXTO) MEJORADA ===================
def analizar_con_qwen(descripcion_texto, atr, reglas_aprendidas, imagen):
    global TOKENS_ACUMULADOS
    try:
        imagen.thumbnail((1200, 800), Image.Resampling.LANCZOS)
        img_base64 = pil_to_base64(imagen)

        prompt = f"""
Eres un **Analista de Price Action** que sigue el método de Steven Nison y la escuela de "Market Structure".

Antes de dar tu decisión, DEBES seguir este razonamiento en orden:

1. **Tendencia global** (regresión de 120 velas) y micro tendencia (últimas 8 velas): ¿Es alcista, bajista o lateral?

2. **Posición respecto a EMA20**: ¿El precio está por encima o por debajo? ¿La EMA20 actúa como soporte o resistencia?

3. **Soportes y resistencias clave** (máximos/mínimos de 40 velas): ¿El precio ha roto alguno? Si rompió, ¿es un rompimiento verdadero (cierre fuera + vela grande) o falso (rechazo rápido)?

4. **Anatomía de las últimas 3 velas** (cuerpo, mecha superior, mecha inferior): ¿Hay figuras de reversión (Martillo, Estrella Fugaz, Doji)? ¿O de continuación (velas sólidas sin mechas)?

5. **Clusters de mechas** (últimas 8 velas): ¿Hay zonas donde el precio ha rebotado varias veces? Eso indica liquidez o barreras psicológicas.

6. **Patrones de múltiples velas** (Envolvente, Estrella Matutina, Tres Soldados, etc.) – solo como confirmación, no como señal única.

7. **Volatilidad (ATR)**: Indica si el mercado está expandido o contraído.

**Con toda esa información**, decide si tomar **Buy**, **Sell** o **Hold**.  
Si hay conflicto entre señales (ej. tendencia alcista pero patrón de reversión bajista), prioriza la tendencia macro y espera confirmación.

**IMPORTANTE**: No tomes decisiones solo por un patrón de vela. Explica brevemente en "razones" qué elementos visuales usaste (ej. "rechazo en resistencia + martillo + sobreventa en EMA").

**Lección de la experiencia reciente (debes considerarla, pero no seguirla ciegamente si el gráfico muestra lo contrario):**
"{reglas_aprendidas}"

**Formato JSON (una línea):**
{{"decision":"Buy/Sell/Hold","patron":"ninguno o nombre","razones":["razón1","razón2"],"sl_mult":1.2,"tp1_mult":1.5,"trailing_mult":1.8}}

Datos numéricos (solo referencia):
{descripcion_texto}
ATR: {atr:.2f}
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {"role": "system", "content": "Eres un trader profesional que analiza gráficos de velas con metodología de Steven Nison. Responde SOLO con JSON válido en una línea."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        raw = response.choices[0].message.content
        if hasattr(response, 'usage') and response.usage:
            TOKENS_ACUMULADOS += response.usage.total_tokens
        else:
            TOKENS_ACUMULADOS += len(raw.split()) + len(prompt.split()) + 1000
        print(f"📊 Tokens acumulados: {TOKENS_ACUMULADOS}")

        if not raw or raw.strip() == "":
            return "Hold", ["Respuesta vacía"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

        datos = parse_json_seguro(raw)
        if not datos:
            return "Hold", ["JSON inválido"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

        decision = datos.get("decision", "Hold")
        sl_m = max(0.5, min(2.5, float(datos.get("sl_mult", 1.2))))
        tp_m = max(0.8, min(3.0, float(datos.get("tp1_mult", 1.5))))
        tr_m = max(1.0, min(3.0, float(datos.get("trailing_mult", 1.8))))
        razones = datos.get("razones", [])
        
        # ===== FILTRO DE CALIDAD DE SEÑAL =====
        if decision != "Hold":
            elementos_analizados = set()
            for r in razones:
                r_low = r.lower()
                if "tendencia" in r_low or "trend" in r_low: elementos_analizados.add("trend")
                if "ema" in r_low: elementos_analizados.add("ema")
                if "soporte" in r_low or "resistencia" in r_low or "support" in r_low or "resistance" in r_low: elementos_analizados.add("s_r")
                if "mecha" in r_low or "wick" in r_low: elementos_analizados.add("wick")
                if "cluster" in r_low or "liquidez" in r_low: elementos_analizados.add("cluster")
                if "patrón" in r_low or "vela" in r_low or "candle" in r_low: elementos_analizados.add("pattern")
            if len(elementos_analizados) < 3:
                print(f"⚠️ Señal pobre: solo {len(elementos_analizados)} elementos analizados -> Hold")
                return "Hold", ["Análisis incompleto - faltan elementos visuales"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

        return decision, razones, datos.get("patron", ""), (sl_m, tp_m, tr_m)

    except Exception as e:
        print(f"❌ Error Qwen3-VL-32B-Instruct: {e} -> Hold")
        return "Hold", [f"Error: {e}"], "", (DEFAULT_SL_MULT, DEFAULT_TP1_MULT, DEFAULT_TRAILING_MULT)

# =================== AUTOAPRENDIZAJE (cada 10 trades NUEVOS) ===================
def aprender_de_trades():
    global ADAPTIVE_SL_MULT, ADAPTIVE_TP1_MULT, ADAPTIVE_TRAILING_MULT, ULTIMO_APRENDIZAJE, REGLAS_APRENDIDAS, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR
    total = PAPER_TRADES_TOTALES
    if total < 10 or (total - ULTIMO_APRENDIZAJE) < 10:
        return
    print("🧠 Iniciando autoaprendizaje...")
    ultimos = TRADE_HISTORY[-10:]
    wins = sum(1 for t in ultimos if t['resultado_win'])
    winrate = wins / 10.0
    ganancias = sum(t['pnl'] for t in ultimos if t['resultado_win'])
    perdidas = abs(sum(t['pnl'] for t in ultimos if not t['resultado_win']))
    profit_factor = ganancias / perdidas if perdidas > 0 else 1.0
    ULTIMO_PROFIT_FACTOR = profit_factor
    
    resumen = ""
    for i, t in enumerate(ultimos):
        estado = "WIN ✅" if t['resultado_win'] else "LOSS ❌"
        resumen += f"{i+1}. {t['decision']} {estado} | {t.get('patron','?')} | PnL:{t['pnl']:.2f}\n"

    # Ajuste adaptativo por Profit Factor
    if profit_factor < 1.2 and winrate < 0.5:
        ADAPTIVE_TP1_MULT = min(3.0, ADAPTIVE_TP1_MULT * 1.1)
        ADAPTIVE_SL_MULT = max(0.5, ADAPTIVE_SL_MULT * 0.95)
        telegram_mensaje(f"⚙️ Ajuste automático por PF bajo ({profit_factor:.2f}): TP1→{ADAPTIVE_TP1_MULT:.2f}, SL→{ADAPTIVE_SL_MULT:.2f}")

    system_msg = """
Eres el Mentor de Trading de una IA. Analiza los últimos 10 trades.
Extrae una lección concreta para mejorar: ¿qué funcionó? ¿qué falló?
Responde ÚNICAMENTE con un JSON en una línea:
{"analisis":"explicación breve","nueva_regla":"lección práctica para futuras decisiones","sl_mult_sugerido":1.5,"tp1_mult_sugerido":1.5,"trailing_mult_sugerido":1.8}
"""
    user_msg = f"Winrate: {winrate*100:.0f}% ({wins}W, {10-wins}L). Profit Factor: {profit_factor:.2f}\n\nHistorial:\n{resumen}\n\nDicta la nueva lección."
    try:
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        raw = response.choices[0].message.content
        if response.usage:
            TOKENS_ACUMULADOS += response.usage.total_tokens
        else:
            TOKENS_ACUMULADOS += len(raw.split()) + 500
        datos = parse_json_seguro(raw)
        if datos:
            REGLAS_APRENDIDAS = datos.get("nueva_regla", REGLAS_APRENDIDAS)
            # Solo aplicar sugerencias si el profit factor no es muy malo, si no preservamos el ajuste manual
            if profit_factor > 0.8:
                ADAPTIVE_SL_MULT = max(0.5, min(2.5, float(datos.get("sl_mult_sugerido", ADAPTIVE_SL_MULT))))
                ADAPTIVE_TP1_MULT = max(0.8, min(3.0, float(datos.get("tp1_mult_sugerido", ADAPTIVE_TP1_MULT))))
                ADAPTIVE_TRAILING_MULT = max(1.0, min(3.0, float(datos.get("trailing_mult_sugerido", ADAPTIVE_TRAILING_MULT))))
            msg = f"🧠 AUTOAPRENDIZAJE (10 trades nuevos)\n📊 Winrate: {winrate*100:.1f}% ({wins}W/{10-wins}L) | PF: {profit_factor:.2f}\n📜 Nueva lección: \"{REGLAS_APRENDIDAS}\"\n⚙️ SL:{ADAPTIVE_SL_MULT:.2f} TP1:{ADAPTIVE_TP1_MULT:.2f} Trail:{ADAPTIVE_TRAILING_MULT:.2f}"
            telegram_mensaje(msg)
            print(msg)
        ULTIMO_APRENDIZAJE = total
        guardar_memoria()
    except Exception as e:
        print(f"Error aprendizaje: {e}")

# =================== GRÁFICOS PARA TELEGRAM ===================
def generar_grafico(df, trade_info, soporte, resistencia, slope, intercept, tipo="Entrada"):
    if df.empty:
        return None
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    ax.plot(x, intercept + slope * x, color='white', linestyle='-.', lw=1.5, alpha=0.6, label='Tendencia')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    if tipo == "Entrada":
        decision = trade_info['decision']
        p_act = df_plot['close'].iloc[-2]
        ax.scatter(len(df_plot)-2, p_act + (-30 if decision=='Buy' else 30), s=400, marker='^' if decision=='Buy' else 'v', c='lime' if decision=='Buy' else 'red', zorder=5)
        txt = f"[#{trade_info['id']}] {decision.upper()}\n{trade_info['patron']}\nSL:{trade_info['sl_inicial']:.2f} TP1:{trade_info['tp1']:.2f}"
    else:
        win = trade_info['resultado_win']
        ax.axhline(trade_info['entrada'], color='blue', ls=':', lw=2, label='Entrada')
        ax.axhline(trade_info['sl_actual'], color='white', ls=':', lw=2, label='Salida')
        estado = "WIN" if win else "LOSS"
        txt = f"[#{trade_info['id']}] {estado} | PnL: {trade_info['pnl']:.2f} USD"
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.1)
    ax.legend(loc='lower right', facecolor='black', labelcolor='white')
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
    if len(PAPER_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES:
        return False
    for t in PAPER_ACTIVE_TRADES.values():
        if t['decision'] == decision and abs(t['entrada'] - precio) < atr * 0.2:
            return False
    sl_m = (multis_ia[0] * 0.6) + (ADAPTIVE_SL_MULT * 0.4)
    tp_m = (multis_ia[1] * 0.6) + (ADAPTIVE_TP1_MULT * 0.4)
    tr_m = (multis_ia[2] * 0.6) + (ADAPTIVE_TRAILING_MULT * 0.4)
    
    distancia_sl = atr * sl_m
    sl_inicial = precio - distancia_sl if decision == "Buy" else precio + distancia_sl
    tp1 = precio + (atr * tp_m) if decision == "Buy" else precio - (atr * tp_m)
    
    if distancia_sl == 0:
        return False
    
    # Cálculo correcto del tamaño de posición
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    size_btc = riesgo_usd / distancia_sl
    max_size_btc = PAPER_BALANCE / precio
    size_btc = min(size_btc, max_size_btc)
    
    if size_btc <= 0:
        return False
        
    TRADE_COUNTER += 1
    trade = {
        "id": TRADE_COUNTER, "decision": decision, "entrada": precio,
        "sl_inicial": sl_inicial, "tp1": tp1, "trailing_mult": tr_m,
        "size_btc": size_btc, "size_restante": size_btc, "tp1_ejecutado": False,
        "pnl_parcial": 0.0, "sl_actual": sl_inicial, "max_precio": precio,
        "patron": patron, "atr_entrada": atr, "trailing_activo": False,
        "avance_minimo_alcanzado": False
    }
    PAPER_ACTIVE_TRADES[TRADE_COUNTER] = trade
    msg = f"📌 [#{TRADE_COUNTER}] {decision.upper()} a {precio:.2f}\nSL:{sl_inicial:.2f} TP1:{tp1:.2f}\n{patron}\nRisk: {riesgo_usd:.2f} USDT | Size: {size_btc:.4f} BTC"
    print(msg)
    telegram_mensaje(msg)
    ruta_img = generar_grafico(df, trade, sop, res, slo, inter, "Entrada")
    if ruta_img:
        telegram_enviar_imagen(ruta_img, msg)
    return True

def paper_revisar_sl_tp(df, sop, res, slo, inter):
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY
    if df.empty:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    trades_a_cerrar = []
    for t_id, t in PAPER_ACTIVE_TRADES.items():
        cerrar = False
        motivo = ""
        # Actualizar máximo/mínimo para trailing
        if t['decision'] == "Buy":
            t['max_precio'] = max(t['max_precio'], h)
            # Verificar si se alcanzó avance mínimo para activar trailing (1.2 * ATR)
            if not t['avance_minimo_alcanzado'] and (t['max_precio'] - t['entrada']) > t['atr_entrada'] * 1.2:
                t['avance_minimo_alcanzado'] = True
        else:
            t['max_precio'] = min(t['max_precio'], l)
            if not t['avance_minimo_alcanzado'] and (t['entrada'] - t['max_precio']) > t['atr_entrada'] * 1.2:
                t['avance_minimo_alcanzado'] = True
        
        # Verificar TP1
        if not t['tp1_ejecutado']:
            if (t['decision'] == "Buy" and h >= t['tp1']) or (t['decision'] == "Sell" and l <= t['tp1']):
                beneficio = abs(t['tp1'] - t['entrada']) * (t['size_btc'] * PORCENTAJE_CIERRE_TP1)
                t['pnl_parcial'] = beneficio
                PAPER_BALANCE += beneficio
                t['size_restante'] *= (1 - PORCENTAJE_CIERRE_TP1)
                t['tp1_ejecutado'] = True
                # SL a entrada + 0.5 ATR (breakeven plus)
                if t['decision'] == "Buy":
                    t['sl_actual'] = t['entrada'] + t['atr_entrada'] * 0.5
                else:
                    t['sl_actual'] = t['entrada'] - t['atr_entrada'] * 0.5
                telegram_mensaje(f"🎯 TP1 #{t_id}: +{beneficio:.2f} USD, SL movido a {t['sl_actual']:.2f}")
        
        # Trailing stop después de TP1 solo si se ha alcanzado avance mínimo
        if t['tp1_ejecutado'] and t['avance_minimo_alcanzado']:
            nuevo_sl = t['max_precio'] - (t['atr_entrada'] * t['trailing_mult']) if t['decision'] == "Buy" else t['max_precio'] + (t['atr_entrada'] * t['trailing_mult'])
            if t['decision'] == "Buy":
                if nuevo_sl > t['sl_actual']:
                    old = t['sl_actual']
                    t['sl_actual'] = nuevo_sl
                    telegram_mensaje(f"🔄 [#{t_id}] Trailing Stop ajustado a: {t['sl_actual']:.2f} (antes {old:.2f})")
            else:
                if nuevo_sl < t['sl_actual']:
                    old = t['sl_actual']
                    t['sl_actual'] = nuevo_sl
                    telegram_mensaje(f"🔄 [#{t_id}] Trailing Stop ajustado a: {t['sl_actual']:.2f} (antes {old:.2f})")
            # Verificar si el SL fue tocado
            if (t['decision'] == "Buy" and l <= t['sl_actual']) or (t['decision'] == "Sell" and h >= t['sl_actual']):
                cerrar = True
                motivo = "Trailing Stop"
        else:
            # Stop loss inicial (si no se ha alcanzado TP1 aún)
            if not t['tp1_ejecutado']:
                if (t['decision'] == "Buy" and l <= t['sl_inicial']) or (t['decision'] == "Sell" and h >= t['sl_inicial']):
                    cerrar = True
                    motivo = "Stop Loss"
                    t['sl_actual'] = t['sl_inicial']
            else:
                # Si TP1 ejecutado pero no se ha alcanzado avance mínimo (raro), usar sl_actual como está
                if (t['decision'] == "Buy" and l <= t['sl_actual']) or (t['decision'] == "Sell" and h >= t['sl_actual']):
                    cerrar = True
                    motivo = "Stop Loss (post TP1)"
        
        if cerrar:
            pnl_rest = (t['sl_actual'] - t['entrada']) * t['size_restante'] if t['decision'] == "Buy" else (t['entrada'] - t['sl_actual']) * t['size_restante']
            pnl_total = t['pnl_parcial'] + pnl_rest
            PAPER_BALANCE += pnl_rest
            PAPER_TRADES_TOTALES += 1
            win = pnl_total > 0
            if win:
                PAPER_WIN += 1
            else:
                PAPER_LOSS += 1
            t['pnl'] = pnl_total
            t['resultado_win'] = win
            trades_a_cerrar.append(t_id)
            TRADE_HISTORY.append({
                "fecha": datetime.now(timezone.utc).isoformat(),
                "decision": t['decision'],
                "patron": t['patron'],
                "pnl": pnl_total,
                "resultado_win": win
            })
            guardar_memoria()
            msg = f"📤 [{motivo}] #{t_id} {t['decision'].upper()} -> {t['sl_actual']:.2f} | PnL: {pnl_total:.2f} USD"
            telegram_mensaje(msg)
            reporte_estado()
            ruta_img = generar_grafico(df, t, sop, res, slo, inter, "Salida")
            if ruta_img:
                telegram_enviar_imagen(ruta_img, msg)
    # Cerrar trades marcados
    for t_id in trades_a_cerrar:
        del PAPER_ACTIVE_TRADES[t_id]
    # Autoaprendizaje si hay al menos 10 trades nuevos desde la última vez
    if PAPER_TRADES_TOTALES - ULTIMO_APRENDIZAJE >= 10:
        aprender_de_trades()

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, TOKENS_ACUMULADOS
    cargar_memoria()
    print(f"🤖 BOT V99.40 INICIADO - Modelo {MODELO_VISION} con SiliconFlow (multimodal + memoria dinámica + PF adaptive)")
    telegram_mensaje(f"🤖 BOT V99.40 INICIADO - Modelo {MODELO_VISION}\nSistema Multi-Trades con análisis visual de gráficos y autoaprendizaje cada 10 trades.\nProfit Factor adaptativo activado.")
    ultima_vela = None
    while True:
        try:
            df_raw = obtener_velas()
            if df_raw.empty:
                time.sleep(60)
                continue
            df = calcular_indicadores(df_raw)
            if df.empty:
                time.sleep(60)
                continue
            vela_cerrada = df.index[-2]
            precio = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            sop, res, slo, inter, tend, micro = detectar_zonas_mercado(df)
            pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
            winrate = (PAPER_WIN/PAPER_TRADES_TOTALES*100) if PAPER_TRADES_TOTALES>0 else 0
            activos = len(PAPER_ACTIVE_TRADES)
            print(f"\n💓 Heartbeat | P:{precio:.2f} | ATR:{atr:.2f} | Activos:{activos}/{MAX_CONCURRENT_TRADES} | Cerrados:{PAPER_TRADES_TOTALES} | PnL:{pnl_global:+.2f} | WR:{winrate:.1f}% | PF(últ10):{ULTIMO_PROFIT_FACTOR:.2f} | Tokens: {TOKENS_ACUMULADOS}")
            
            if activos < MAX_CONCURRENT_TRADES and ultima_vela != vela_cerrada:
                desc, atr_val = generar_descripcion_nison(df)
                if not desc or len(desc) < 50:
                    print("⚠️ Descripción muy corta, se omite ciclo.")
                    ultima_vela = vela_cerrada
                    time.sleep(SLEEP_SECONDS)
                    continue
                print(f"--- Evaluando {vela_cerrada.strftime('%H:%M')} con Qwen3-VL-32B-Instruct (imagen + texto) ---")
                img = generar_grafico_para_vision(df, sop, res, slo, inter, precio)
                if img is None:
                    print("⚠️ No se pudo generar imagen, se omite ciclo.")
                    ultima_vela = vela_cerrada
                    time.sleep(SLEEP_SECONDS)
                    continue
                decision, razones, patron, multis = analizar_con_qwen(desc, atr_val, REGLAS_APRENDIDAS, img)
                ULTIMA_DECISION, ULTIMO_MOTIVO = decision, razones[0] if razones else ""
                if decision in ["Buy","Sell"] and risk_management_check():
                    paper_abrir_posicion(decision, precio, atr_val, razones, patron, multis, df, sop, res, slo, inter)
                else:
                    print(f"⏸️ Hold: {ULTIMO_MOTIVO[:80]}")
                ultima_vela = vela_cerrada
            if PAPER_ACTIVE_TRADES:
                sop, res, slo, inter, _, _ = detectar_zonas_mercado(df, -1)
                paper_revisar_sl_tp(df, sop, res, slo, inter)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ Error loop: {e}")
            import traceback; traceback.print_exc()
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
