# BOT TRADING V99.45 – IA controla SL/TP/TAILING 100% VISUAL
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

# =================== CONFIGURACIÓN ===================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("Falta SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
MODELO_VISION = "Qwen/Qwen3-VL-32B-Instruct"

MEMORY_FILE = "memoria_bot.json"
SYMBOL = "BTCUSDT"
INTERVAL = "5"
RISK_PER_TRADE = 0.02
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3
PCT_TP1 = 0.50   # 50%
PCT_TP2 = 0.30   # 30% (resto 20% para trailing)
PAPER_BALANCE_INICIAL = 100.0
MAX_DAILY_DRAWDOWN_PCT = 0.20

# Variables globales
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_ACTIVE_TRADES = {}
TRADE_COUNTER = 0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
TRADE_HISTORY = []
PAPER_DAILY_START_BALANCE = PAPER_BALANCE_INICIAL
PAPER_STOPPED_TODAY = False
PAPER_CURRENT_DAY = None
ULTIMO_APRENDIZAJE = 0
ULTIMO_PROFIT_FACTOR = 1.0
REGLAS_APRENDIDAS = "Aún no hay trades."
ULTIMA_DECISION = "Hold"
ULTIMO_MOTIVO = ""
TOKENS_ACUMULADOS = 0

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

# =================== FUNCIONES AUXILIARES ===================
def convertir_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): return float(obj)
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convertir_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [convertir_serializable(item) for item in obj]
    else: return obj

def guardar_memoria():
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS
    active_meta = {tid: {"id": t["id"], "decision": t["decision"], "entrada": t["entrada"], "razon": t.get("razon","")} for tid, t in PAPER_ACTIVE_TRADES.items()}
    data = {
        "TRADE_HISTORY": TRADE_HISTORY,
        "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
        "PAPER_BALANCE": PAPER_BALANCE,
        "PAPER_WIN": PAPER_WIN,
        "PAPER_LOSS": PAPER_LOSS,
        "PAPER_TRADES_TOTALES": PAPER_TRADES_TOTALES,
        "ULTIMO_APRENDIZAJE": ULTIMO_APRENDIZAJE,
        "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS,
        "PAPER_ACTIVE_META": active_meta,
        "ULTIMO_PROFIT_FACTOR": ULTIMO_PROFIT_FACTOR
    }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        print("💾 Memoria guardada")
    except Exception as e: print(f"Error guardando: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS, PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR
    if not os.path.exists(MEMORY_FILE): return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        TRADE_HISTORY = data.get("TRADE_HISTORY", [])
        REGLAS_APRENDIDAS = data.get("REGLAS_APRENDIDAS", REGLAS_APRENDIDAS)
        PAPER_BALANCE = data.get("PAPER_BALANCE", PAPER_BALANCE)
        PAPER_WIN = data.get("PAPER_WIN", 0)
        PAPER_LOSS = data.get("PAPER_LOSS", 0)
        PAPER_TRADES_TOTALES = data.get("PAPER_TRADES_TOTALES", 0)
        ULTIMO_APRENDIZAJE = data.get("ULTIMO_APRENDIZAJE", 0)
        TOKENS_ACUMULADOS = data.get("TOKENS_ACUMULADOS", 0)
        ULTIMO_PROFIT_FACTOR = data.get("ULTIMO_PROFIT_FACTOR", 1.0)
        print(f"🧠 Memoria cargada: {PAPER_TRADES_TOTALES} trades, PF={ULTIMO_PROFIT_FACTOR:.2f}")
    except Exception as e: print(f"Error cargando: {e}")

def parse_json_seguro(raw):
    if not raw: return None
    try:
        return json.loads(json_repair.repair_json(raw))
    except:
        try:
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                return json.loads(raw[start:end+1])
        except: pass
        return None

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except: pass

def telegram_enviar_imagen(ruta, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        with open(ruta, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
    except: pass

def reporte_estado():
    pnl_global = PAPER_BALANCE - PAPER_BALANCE_INICIAL
    winrate = (PAPER_WIN/PAPER_TRADES_TOTALES*100) if PAPER_TRADES_TOTALES>0 else 0
    activos = len(PAPER_ACTIVE_TRADES)
    mensaje = (f"📊 REPORTE\n💰 Balance: {PAPER_BALANCE:.2f}\n📈 PnL: {pnl_global:+.2f}\n🎯 WR: {winrate:.1f}% ({PAPER_WIN}W/{PAPER_LOSS}L)\n🔄 Cerrados: {PAPER_TRADES_TOTALES}\n⚡ Activos: {activos}\n📐 PF(10): {ULTIMO_PROFIT_FACTOR:.2f}")
    telegram_mensaje(mensaje)

# =================== DATOS ===================
def obtener_velas(limit=150):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}, timeout=20)
        data = r.json()
        if data.get("retCode") != 0: return pd.DataFrame()
        result = data.get("result")
        if not result or "list" not in result: return pd.DataFrame()
        lista = result["list"][::-1]
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"Error obtener_velas: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    if df.empty: return df
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    tr = pd.concat([(df['high']-df['low']), (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2):
    if df.empty or len(df)<40: return 0,0,0,0,"LATERAL","LATERAL"
    df_eval = df if idx==-1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-120:] if len(df_eval)>=120 else df_eval['close'].values
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    micro = 'CAYENDO' if micro_slope<-0.2 else 'SUBIENDO' if micro_slope>0.2 else 'LATERAL'
    tend = 'ALCISTA' if slope>0.01 else 'BAJISTA' if slope<-0.01 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tend, micro

def analizar_anatomia_vela(v):
    rango = v['high']-v['low']
    if rango==0: return "Doji Plano"
    c_pct = abs(v['close']-v['open'])/rango*100
    s_sup = (v['high']-max(v['close'],v['open']))/rango*100
    s_inf = (min(v['close'],v['open'])-v['low'])/rango*100
    color = "VERDE" if v['close']>v['open'] else "ROJA"
    return f"{color} (Cuerpo:{c_pct:.0f}% | M.Sup:{s_sup:.0f}% | M.Inf:{s_inf:.0f}%)"

def analizar_patrones_conjuntos(df, idx):
    if idx<3: return "Datos insuficientes"
    v3,v2,v1 = df.iloc[idx], df.iloc[idx-1], df.iloc[idx-2]
    r3 = v3['high']-v3['low']
    c3 = abs(v3['close']-v3['open'])/r3*100 if r3>0 else 0
    sup3 = (v3['high']-max(v3['close'],v3['open']))/r3*100 if r3>0 else 0
    inf3 = (min(v3['close'],v3['open'])-v3['low'])/r3*100 if r3>0 else 0
    verde3 = v3['close']>v3['open']
    verde2 = v2['close']>v2['open']
    verde1 = v1['close']>v1['open']
    patrones = []
    if not verde1 and verde3 and v3['close']>(v1['open']+v1['close'])/2: patrones.append("ESTRELLA MATUTINA")
    if verde1 and not verde3 and v3['close']<(v1['open']+v1['close'])/2: patrones.append("ESTRELLA VESPERTINA")
    if verde1 and verde2 and verde3 and v3['close']>v2['close']>v1['close']: patrones.append("TRES SOLDADOS")
    if not verde1 and not verde2 and not verde3 and v3['close']<v2['close']<v1['close']: patrones.append("TRES CUERVOS")
    if not verde2 and verde3 and v3['close']>v2['open'] and v3['open']<v2['close']: patrones.append("ENVOLVENTE ALCISTA")
    if verde2 and not verde3 and v3['close']<v2['open'] and v3['open']>v2['close']: patrones.append("ENVOLVENTE BAJISTA")
    if verde3 and c3>70 and sup3<10: patrones.append("VELA SÓLIDA MÁXIMOS")
    elif not verde3 and c3>70 and inf3<10: patrones.append("VELA SÓLIDA MÍNIMOS")
    elif c3<15 and sup3>25 and inf3>25: patrones.append("DOJI")
    elif inf3>60 and c3<25 and sup3<15: patrones.append("MARTILLO")
    elif sup3>60 and c3<25 and inf3<15: patrones.append("ESTRELLA FUGAZ")
    return " | ".join(patrones) if patrones else "Consolidación"

def generar_descripcion_nison(df, idx=-2):
    if df.empty or len(df)<abs(idx)+1: return "Datos insuficientes",0
    vela = df.iloc[idx]
    precio = vela['close']
    atr = df['atr'].iloc[idx]
    ema20 = df['ema20'].iloc[idx]
    sop, res, _, _, tend, micro = detectar_zonas_mercado(df, idx)
    anat1 = analizar_anatomia_vela(df.iloc[idx-2]) if idx-2>=0 else "N/A"
    anat2 = analizar_anatomia_vela(df.iloc[idx-1]) if idx-1>=0 else "N/A"
    anat3 = analizar_anatomia_vela(df.iloc[idx])
    patrones = analizar_patrones_conjuntos(df, idx)
    df_mechas = df.iloc[max(0,idx-7):idx+1]
    if len(df_mechas)>=3:
        rangos = df_mechas['high']-df_mechas['low']
        mech_sup = (df_mechas['high']-df_mechas[['close','open']].max(axis=1))/rangos.replace(0,0.001)
        mech_inf = (df_mechas[['close','open']].min(axis=1)-df_mechas['low'])/rangos.replace(0,0.001)
        cluster = f"Mechas sup>55%: {sum(mech_sup>0.55)} | inf>55%: {sum(mech_inf>0.55)}"
    else:
        cluster = "Insuficiente"
    ultimas10 = df.iloc[max(0,idx-9):idx+1]
    if len(ultimas10)>=5:
        sobre = (ultimas10['close']>ultimas10['ema20']).sum()
        lateralidad = f"{len(ultimas10)} velas: {sobre*100/len(ultimas10):.0f}% sobre EMA20"
    else:
        lateralidad = "Datos insuficientes"
    desc = f"""
=== DATOS (sin interpretación) ===
Precio: {precio:.2f}  ATR: {atr:.2f}  EMA20: {ema20:.2f}
Soporte40: {sop:.2f}  Resistencia40: {res:.2f}
Tendencia macro: {tend}  Micro: {micro}
Anatomía últimas 3: {anat1} | {anat2} | {anat3}
Patrones: {patrones}
Clusters mechas: {cluster}
Posición EMA20: {lateralidad}
"""
    return desc, atr

def generar_grafico_para_vision(df, sop, res, slope, intercept, precio=None):
    if df.empty: return None
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    x = np.arange(len(df_plot))
    fig, ax = plt.subplots(figsize=(16,8))
    for i in range(len(df_plot)):
        o,h,l,c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c>=o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, lw=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o),0.1), color=color, alpha=0.9))
    ax.axhline(sop, color='cyan', ls='--', lw=2, label='Soporte')
    ax.axhline(res, color='magenta', ls='--', lw=2, label='Resistencia')
    ax.plot(x, intercept+slope*x, 'white', ls='-.', lw=1.5, alpha=0.6, label='Tendencia')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    ax.grid(alpha=0.1)
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
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# =================== IA (entrada y actualización de stops) ===================
def analizar_con_qwen(descripcion_texto, reglas_aprendidas, imagen, contexto_extra=""):
    global TOKENS_ACUMULADOS
    try:
        imagen.thumbnail((1200,800), Image.Resampling.LANCZOS)
        img_b64 = pil_to_base64(imagen)
        prompt = f"""
Eres un **Analista de Price Action** experto en estructura de mercado y velas japonesas.

{contexto_extra}

Debes analizar el gráfico y decidir:
- **Decisión**: Buy, Sell o Hold (solo para primera entrada, si es actualización de stop, ignora o pon Hold).
- **Razón principal**: Frase corta que resuma el análisis visual completo.
- **Razones**: Lista detallada de elementos observados (tendencia, EMA, S/R, anatomía, clusters).
- **Niveles de precio** (si aplica):
  - sl_price (stop loss)
  - tp1_price (primer objetivo parcial, 50%)
  - tp2_price (segundo objetivo parcial, 30%, opcional)
  - nuevo_sl_price (para actualización de stop después de TP1)
- **Modo de trailing** (para el remanente): puede ser "ninguno", "minimos_consecutivos:N" (N velas), "ema20", "maximos_consecutivos:N", o un precio fijo "sl_fijo".

**Reglas**: No uses multiplicadores ATR. Basa todos los niveles en la estructura visual del gráfico (soportes, resistencias, EMAs, máximos/mínimos recientes, zonas de liquidez). Sé flexible y contextual.

Lección reciente: "{reglas_aprendidas}"

Datos numéricos (solo referencia visual):
{descripcion_texto}

Responde SOLO con JSON en una línea, ejemplo:
{{"decision":"Buy","razon_principal":"Rebote en soporte + martillo","razones":["tendencia alcista","EMA20 como soporte","vela martillo"],"sl_price":42000.00,"tp1_price":42500.00,"tp2_price":42800.00,"nuevo_sl_price":null,"modo_trailing":"minimos_consecutivos:3"}}
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {"role": "system", "content": "Eres un trader. Responde SOLO con JSON válido en una línea."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_b64}}]}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        raw = response.choices[0].message.content
        if hasattr(response, 'usage') and response.usage:
            TOKENS_ACUMULADOS += response.usage.total_tokens
        else:
            TOKENS_ACUMULADOS += len(raw.split()) + len(prompt.split()) + 1000
        datos = parse_json_seguro(raw)
        if not datos:
            return None
        return datos
    except Exception as e:
        print(f"Error IA: {e}")
        return None

# =================== ACTUALIZACIÓN DE STOP CON IA ===================
def actualizar_stop_con_ia(trade, df, sop, res, slope, intercept):
    """Llama a la IA para que decida el nuevo stop después de TP1"""
    desc, _ = generar_descripcion_nison(df, -1)
    img = generar_grafico_para_vision(df, sop, res, slope, intercept, trade['entrada'])
    if img is None:
        return None
    contexto = f"""
Estado del trade:
- Dirección: {trade['decision']}
- Precio entrada: {trade['entrada']:.2f}
- TP1 ya alcanzado (se cerró 50%)
- Tamaño restante: {trade['size_btc'] * (1 - PCT_TP1):.4f} BTC (50% original)
- SL actual: {trade['sl_actual']:.2f}
- Precio actual aprox: {df['close'].iloc[-1]:.2f}
- Máximo/mínimo desde entrada: {trade['max_precio']:.2f}

Necesito que observes el gráfico actual y me digas:
- nuevo_sl_price: nuevo nivel de stop loss (debe estar por encima del precio actual si es Buy, o por debajo si es Sell, basado en soportes/resistencias/EMAs).
- modo_trailing: para el remanente, qué regla visual quieres aplicar. Puede ser "minimos_consecutivos:3", "maximos_consecutivos:3", "ema20", "fijo", o "ninguno". Si usas "fijo", el stop no se moverá más.
Devuelve un JSON con esos dos campos.
"""
    datos = analizar_con_qwen(desc, REGLAS_APRENDIDAS, img, contexto)
    if datos:
        nuevo_sl = datos.get("nuevo_sl_price")
        modo = datos.get("modo_trailing")
        return nuevo_sl, modo
    return None, None

# =================== GESTIÓN TRADES ===================
def risk_management_check():
    global PAPER_DAILY_START_BALANCE, PAPER_STOPPED_TODAY, PAPER_CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if PAPER_CURRENT_DAY != hoy:
        PAPER_CURRENT_DAY = hoy
        PAPER_DAILY_START_BALANCE = PAPER_BALANCE
        PAPER_STOPPED_TODAY = False
    drawdown = (PAPER_BALANCE - PAPER_DAILY_START_BALANCE) / PAPER_DAILY_START_BALANCE
    if drawdown <= -MAX_DAILY_DRAWDOWN_PCT and not PAPER_STOPPED_TODAY:
        telegram_mensaje("🛑 Drawdown máximo diario. Bot pausado.")
        PAPER_STOPPED_TODAY = True
    return not PAPER_STOPPED_TODAY

def paper_abrir_posicion(decision, precio, razon_principal, razones, sl_price, tp1_price, tp2_price, df, sop, res, slope, intercept):
    global PAPER_BALANCE, TRADE_COUNTER
    if len(PAPER_ACTIVE_TRADES) >= MAX_CONCURRENT_TRADES:
        return False
    if sl_price is None:
        print("⚠️ IA no dio sl_price, no se abre.")
        return False
    distancia = abs(precio - sl_price)
    if distancia <= 0:
        return False
    riesgo = PAPER_BALANCE * RISK_PER_TRADE
    size = riesgo / distancia
    max_size = PAPER_BALANCE / precio
    size = min(size, max_size)
    if size <= 0:
        return False
    TRADE_COUNTER += 1
    trade = {
        "id": TRADE_COUNTER, "decision": decision, "entrada": precio,
        "sl_inicial": sl_price, "sl_actual": sl_price,
        "tp1": tp1_price, "tp2": tp2_price,
        "size_btc": size,
        "size_restante_tp1": size,   # después de TP1 será size*(1-PCT_TP1)
        "size_restante_tp2": size,   # después de TP2 será size*(1-PCT_TP1-PCT_TP2)
        "tp1_ejecutado": False, "tp2_ejecutado": False,
        "pnl_parcial_tp1": 0.0, "pnl_parcial_tp2": 0.0,
        "max_precio": precio,
        "razon": razon_principal,
        "modo_trailing": None,   # se definirá después de TP1
        "ultima_actualizacion_stop": 0
    }
    PAPER_ACTIVE_TRADES[TRADE_COUNTER] = trade
    msg = f"📌 #{TRADE_COUNTER} {decision.upper()} a {precio:.2f}\nSL:{sl_price:.2f}"
    if tp1_price: msg += f" TP1:{tp1_price:.2f}"
    if tp2_price: msg += f" TP2:{tp2_price:.2f}"
    msg += f"\n{razon_principal}\nRisk:{riesgo:.2f} Size:{size:.4f}"
    telegram_mensaje(msg)
    ruta = generar_grafico_para_vision(df, sop, res, slope, intercept, precio)
    if ruta:
        telegram_enviar_imagen(ruta, msg)
    return True

def aplicar_trailing_visual(trade, df):
    """Aplica la regla de trailing definida por la IA"""
    if trade['modo_trailing'] is None:
        return
    modo = trade['modo_trailing']
    if modo == "ninguno":
        return
    if modo.startswith("minimos_consecutivos:"):
        try:
            n = int(modo.split(":")[1])
            if len(df) >= n:
                ultimos_min = df['low'].iloc[-n:].min()
                if trade['decision'] == 'Buy' and ultimos_min > trade['sl_actual']:
                    trade['sl_actual'] = ultimos_min
                elif trade['decision'] == 'Sell' and ultimos_min < trade['sl_actual']:
                    trade['sl_actual'] = ultimos_min
        except: pass
    elif modo.startswith("maximos_consecutivos:"):
        try:
            n = int(modo.split(":")[1])
            if len(df) >= n:
                ultimos_max = df['high'].iloc[-n:].max()
                if trade['decision'] == 'Buy' and ultimos_max > trade['sl_actual']:
                    trade['sl_actual'] = ultimos_max
                elif trade['decision'] == 'Sell' and ultimos_max < trade['sl_actual']:
                    trade['sl_actual'] = ultimos_max
        except: pass
    elif modo == "ema20":
        if 'ema20' in df.columns and not df['ema20'].isna().iloc[-1]:
            ema = df['ema20'].iloc[-1]
            if trade['decision'] == 'Buy' and ema > trade['sl_actual']:
                trade['sl_actual'] = ema
            elif trade['decision'] == 'Sell' and ema < trade['sl_actual']:
                trade['sl_actual'] = ema
    # si es "fijo", no hacer nada

def paper_revisar_sl_tp(df, sop, res, slope, intercept):
    global PAPER_BALANCE, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES, TRADE_HISTORY
    if df.empty: return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    trades_a_cerrar = []
    for t_id, t in PAPER_ACTIVE_TRADES.items():
        # Actualizar max/min para referencia
        if t['decision'] == 'Buy':
            t['max_precio'] = max(t['max_precio'], h)
        else:
            t['max_precio'] = min(t['max_precio'], l)
        # Verificar TP1
        if not t['tp1_ejecutado'] and t['tp1'] is not None:
            if (t['decision'] == 'Buy' and h >= t['tp1']) or (t['decision'] == 'Sell' and l <= t['tp1']):
                beneficio = abs(t['tp1'] - t['entrada']) * (t['size_btc'] * PCT_TP1)
                t['pnl_parcial_tp1'] = beneficio
                PAPER_BALANCE += beneficio
                t['tp1_ejecutado'] = True
                # Reducir tamaño restante
                t['size_restante_tp1'] = t['size_btc'] * (1 - PCT_TP1)
                telegram_mensaje(f"🎯 TP1 #{t_id}: +{beneficio:.2f} USD. Consultando IA para nuevo stop...")
                # Llamar a IA para que defina nuevo SL y modo trailing
                nuevo_sl, modo = actualizar_stop_con_ia(t, df, sop, res, slope, intercept)
                if nuevo_sl is not None:
                    t['sl_actual'] = nuevo_sl
                    telegram_mensaje(f"🔄 #{t_id} Nuevo SL según IA: {nuevo_sl:.2f}")
                if modo:
                    t['modo_trailing'] = modo
                    telegram_mensaje(f"📐 #{t_id} Modo trailing: {modo}")
        # Verificar TP2
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2'] is not None:
            if (t['decision'] == 'Buy' and h >= t['tp2']) or (t['decision'] == 'Sell' and l <= t['tp2']):
                beneficio = abs(t['tp2'] - t['entrada']) * (t['size_btc'] * PCT_TP2)
                t['pnl_parcial_tp2'] = beneficio
                PAPER_BALANCE += beneficio
                t['tp2_ejecutado'] = True
                t['size_restante_tp2'] = t['size_btc'] * (1 - PCT_TP1 - PCT_TP2)
                telegram_mensaje(f"🎯 TP2 #{t_id}: +{beneficio:.2f} USD. Queda {t['size_restante_tp2']:.4f} BTC en trailing.")
                # Opcional: re-evaluar trailing después de TP2
                nuevo_sl, modo = actualizar_stop_con_ia(t, df, sop, res, slope, intercept)
                if nuevo_sl: t['sl_actual'] = nuevo_sl
                if modo: t['modo_trailing'] = modo
        # Aplicar trailing si está definido (después de TP1 o TP2)
        if t['tp1_ejecutado']:
            aplicar_trailing_visual(t, df)
        # Verificar si el stop actual se toca
        if (t['decision'] == 'Buy' and l <= t['sl_actual']) or (t['decision'] == 'Sell' and h >= t['sl_actual']):
            # Cerrar el resto
            if t['tp2_ejecutado']:
                remaining = t['size_restante_tp2']
            elif t['tp1_ejecutado']:
                remaining = t['size_restante_tp1']
            else:
                remaining = t['size_btc']
            pnl_rest = (t['sl_actual'] - t['entrada']) * remaining if t['decision'] == 'Buy' else (t['entrada'] - t['sl_actual']) * remaining
            pnl_total = t['pnl_parcial_tp1'] + t['pnl_parcial_tp2'] + pnl_rest
            PAPER_BALANCE += pnl_rest
            PAPER_TRADES_TOTALES += 1
            win = pnl_total > 0
            if win: PAPER_WIN += 1
            else: PAPER_LOSS += 1
            t['pnl'] = pnl_total
            t['resultado_win'] = win
            trades_a_cerrar.append(t_id)
            TRADE_HISTORY.append({
                "fecha": datetime.now(timezone.utc).isoformat(),
                "decision": t['decision'],
                "razon": t['razon'],
                "pnl": pnl_total,
                "resultado_win": win
            })
            guardar_memoria()
            msg = f"📤 Cierre #{t_id} {t['decision'].upper()} -> {t['sl_actual']:.2f} | PnL: {pnl_total:.2f} USD"
            telegram_mensaje(msg)
            reporte_estado()
            # Gráfico de cierre
            fig = generar_grafico_para_vision(df, sop, res, slope, intercept, t['entrada'])
            if fig:
                telegram_enviar_imagen(fig, msg)
    for tid in trades_a_cerrar:
        del PAPER_ACTIVE_TRADES[tid]
    if PAPER_TRADES_TOTALES - ULTIMO_APRENDIZAJE >= 10:
        aprender_de_trades()

# =================== AUTOAPRENDIZAJE ===================
def aprender_de_trades():
    global ULTIMO_APRENDIZAJE, REGLAS_APRENDIDAS, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR
    total = PAPER_TRADES_TOTALES
    if total < 10 or (total - ULTIMO_APRENDIZAJE) < 10: return
    ultimos = TRADE_HISTORY[-10:]
    wins = sum(1 for t in ultimos if t['resultado_win'])
    winrate = wins/10.0
    ganancias = sum(t['pnl'] for t in ultimos if t['resultado_win'])
    perdidas = abs(sum(t['pnl'] for t in ultimos if not t['resultado_win']))
    pf = ganancias/perdidas if perdidas>0 else 1.0
    ULTIMO_PROFIT_FACTOR = pf
    resumen = "\n".join([f"{i+1}. {t['decision']} {'WIN' if t['resultado_win'] else 'LOSS'} | {t.get('razon','?')} | {t['pnl']:.2f}" for i,t in enumerate(ultimos)])
    system_msg = "Eres mentor. Analiza últimos 10 trades, extrae lección para mejorar. Responde JSON: {\"analisis\":\"\",\"nueva_regla\":\"\"}"
    user_msg = f"Winrate: {winrate*100:.0f}% ({wins}W, {10-wins}L). PF: {pf:.2f}\nHistorial:\n{resumen}\nLección:"
    try:
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.3, max_tokens=1000
        )
        raw = response.choices[0].message.content
        datos = parse_json_seguro(raw)
        if datos:
            REGLAS_APRENDIDAS = datos.get("nueva_regla", REGLAS_APRENDIDAS)
            telegram_mensaje(f"🧠 AUTOAPRENDIZAJE\nWR: {winrate*100:.1f}% PF:{pf:.2f}\nLección: {REGLAS_APRENDIDAS}")
        ULTIMO_APRENDIZAJE = total
        guardar_memoria()
    except Exception as e: print(f"Error aprendizaje: {e}")

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global ULTIMA_DECISION, ULTIMO_MOTIVO, TOKENS_ACUMULADOS
    cargar_memoria()
    print("🤖 BOT V99.45 - IA gestiona stops y trailing 100% visual")
    telegram_mensaje("🤖 V99.45: SL/TP/TAILING definidos por IA visual en cada paso")
    ultima_vela = None
    while True:
        try:
            df_raw = obtener_velas()
            if df_raw.empty: time.sleep(60); continue
            df = calcular_indicadores(df_raw)
            if df.empty: time.sleep(60); continue
            vela_cerrada = df.index[-2]
            precio = df['close'].iloc[-1]
            sop, res, slope, intercept, tend, micro = detectar_zonas_mercado(df)
            activos = len(PAPER_ACTIVE_TRADES)
            print(f"💓 {vela_cerrada.strftime('%H:%M')} | P:{precio:.0f} | Activos:{activos} | Cerrados:{PAPER_TRADES_TOTALES} | Balance:{PAPER_BALANCE:.2f} | Tokens:{TOKENS_ACUMULADOS}")
            if activos < MAX_CONCURRENT_TRADES and ultima_vela != vela_cerrada:
                desc, _ = generar_descripcion_nison(df)
                if len(desc)>=50:
                    img = generar_grafico_para_vision(df, sop, res, slope, intercept, precio)
                    if img:
                        datos_ia = analizar_con_qwen(desc, REGLAS_APRENDIDAS, img, "")
                        if datos_ia and datos_ia.get("decision") in ["Buy","Sell"] and risk_management_check():
                            paper_abrir_posicion(
                                datos_ia["decision"], precio,
                                datos_ia.get("razon_principal",""),
                                datos_ia.get("razones",[]),
                                datos_ia.get("sl_price"), datos_ia.get("tp1_price"), datos_ia.get("tp2_price"),
                                df, sop, res, slope, intercept
                            )
                ultima_vela = vela_cerrada
            if PAPER_ACTIVE_TRADES:
                paper_revisar_sl_tp(df, sop, res, slope, intercept)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"Error loop: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
