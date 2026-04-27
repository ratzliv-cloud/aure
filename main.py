import os
import io
import base64
from openai import OpenAI
from PIL import Image

# --- Configuración ---
# Reemplace 'SU_API_KEY' con su clave API real de SiliconFlow (cloud.siliconflow.com)
API_KEY = "sk-hmrrxzwkjrcsgjrtygrvabzxmyzxsbkigyynuhseipsvqaaw"
# El endpoint de API correcto es .cn
BASE_URL = "https://api.siliconflow.cn/v1"
MODELO = "deepseek-ai/DeepSeek-V4-Flash"

# Inicializar el cliente con la configuración correcta
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- 1. Función para convertir imagen a base64 ---
def convertir_imagen_a_base64(imagen):
    """Convierte una imagen PIL a formato base64 para la API."""
    buffer = io.BytesIO()
    imagen.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

# --- 2. Prueba de Diagnóstico con modelo de solo texto ---
print("🧪 Probando conexión con modelo de texto...")
try:
    response_texto = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", # Modelo de solo texto para diagnóstico
        messages=[{"role": "user", "content": "Responde solo 'OK'"}],
        max_tokens=5
    )
    print("✅ ¡Conexión exitosa con modelo de texto!")
    print(f"Respuesta del modelo: {response_texto.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error en prueba de texto: {e}")
    # Si falla aquí, el problema es la clave API o la URL base
    exit()

# --- 3. Prueba Multimodal con DeepSeek-V4-Flash ---
print("🧠 Probando análisis multimodal con DeepSeek-V4-Flash...")
# Crear una imagen de prueba simple
imagen_prueba = Image.new('RGB', (100, 100), color='red')
imagen_base64 = convertir_imagen_a_base64(imagen_prueba)

try:
    response = client.chat.completions.create(
        model=MODELO,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe brevemente esta imagen."},
                    {"type": "image_url", "image_url": {"url": imagen_base64}}
                ]
            }
        ],
        max_tokens=100
    )
    print("✅ ¡Análisis multimodal exitoso!")
    print(f"Respuesta del modelo: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Error en análisis multimodal: {e}")
