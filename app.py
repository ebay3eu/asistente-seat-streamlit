import streamlit as st
import json
from openai import OpenAI
from pinecone import Pinecone

# --- Configuración de la Página y Título ---
st.set_page_config(
    page_title="Asistente Virtual SEAT",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Asistente Virtual SEAT")

# --- Conexión a los servicios ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Error: No se encontraron las claves de API. Asegúrate de configurar los 'Secrets' en Streamlit Cloud.")
    st.stop()

# --- Inicialización de Clientes ---
@st.cache_resource
def get_clients():
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("asistente-seat")
    return client_openai, pinecone_index

client_openai, pinecone_index = get_clients()

# --- Funciones de Lógica ---
# AÑADIDO: La función ahora acepta el historial del chat
def extraer_criterios_de_busqueda(pregunta_usuario, historial_chat):
    # AÑADIDO: Construimos un resumen de la conversación para darle contexto
    conversacion_para_contexto = "\n".join([f"{msg['role']}: {msg['content']}" for msg in historial_chat])

    prompt = f"""
    Analiza la última pregunta del usuario teniendo en cuenta el historial de la conversación para darle contexto.
    HISTORIAL DE LA CONVERSACIÓN:
    {conversacion_para_contexto}

    ÚLTIMA PREGUNTA DEL USUARIO: "{pregunta_usuario}"

    Extrae los criterios de la última pregunta en formato JSON. Si la última pregunta es un seguimiento de la anterior (ej: 'y por menos de 30.000€?'), combina los criterios.
    El JSON debe tener dos claves:
    1. "precio_max": un entero con el precio máximo si se menciona. Si no, 0.
    2. "descripcion": una cadena de texto que resuma TODAS las características que busca el usuario, combinando el historial si es necesario.

    Ejemplo con historial:
    - Historial: user: "busco un suv", assistant: "Tenemos Arona, Ateca y Tarraco"
    - Última pregunta: "y que sea híbrido"
    - JSON resultante: {{"precio_max": 0, "descripcion": "un suv que sea híbrido"}}

    Responde únicamente con el objeto JSON.
    """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        criterios = json.loads(response.choices[0].message.content)
        return criterios
    except Exception as e:
        st.error(f"Error al extraer criterios: {e}")
        return None

def busqueda_inteligente(criterios, top_k=5):
    # Esta función no necesita cambios, ya opera con los criterios extraídos
    filtro_metadata = {}
    if criterios.get("precio_max") and criterios["precio_max"] > 0:
        filtro_metadata["precio"] = {"$lte": criterios["precio_max"]}

    query_embedding = client_openai.embeddings.create(
        input=[criterios["descripcion"]], model="text-embedding-3-small"
    ).data[0].embedding
    
    res_busqueda = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filtro_metadata)

    if res_busqueda['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda['matches']]
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]

    st.info("La búsqueda inicial fue demasiado específica. Intentando una búsqueda más amplia...")
    pregunta_relajada = "dime todos los modelos de coche disponibles"
    query_embedding_relajado = client_openai.embeddings.create(input=[pregunta_relajada], model="text-embedding-3-small").data[0].embedding
    res_busqueda_relajada = pinecone_index.query(vector=query_embedding_relajado, top_k=top_k, include_metadata=True, filter=filtro_metadata)
    
    if res_busqueda_relajada['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda_relajada['matches']]
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]

    return None, None

# AÑADIDO: La función ahora acepta el historial del chat
def generar_respuesta_inteligente(pregunta_original, contexto, descripcion_buscada, historial_chat):
    prompt_sistema = f"""
    Eres "Asistente Virtual SEAT", un experto amable y servicial. Tu objetivo es ayudar al usuario a encontrar su coche ideal.
    La pregunta original del usuario fue: "{pregunta_original}"
    El usuario buscaba un coche con estas características: "{descripcion_buscada}"

    He realizado una búsqueda y he encontrado los siguientes modelos que podrían encajar parcialmente:
    CONTEXTO ENCONTRADO:
    {contexto}

    Tu tarea es responder al usuario de forma inteligente y conversacional.
    Usa el contexto encontrado para formular tu respuesta. Si el contexto no coincide con alguna parte específica de la descripción (ej: el usuario pidió "color verde"), explícalo amablemente.
    """
    
    # AÑADIDO: Construimos el payload de mensajes incluyendo el historial
    # Limitamos el historial a los últimos 4 mensajes para no exceder el límite de tokens
    mensajes_para_api = [{"role": "system", "content": prompt_sistema}] 
    mensajes_para_api.extend(historial_chat[-4:]) 
    mensajes_para_api.append({"role": "user", "content": pregunta_original})

    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=mensajes_para_api,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la respuesta con OpenAI: {e}")
        return "Hubo un problema al generar la respuesta."

# --- Interfaz de la Aplicación ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("¡Hola! Soy tu asistente virtual de SEAT. ¿En qué puedo ayudarte?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # AÑADIDO: Pasamos el historial a las funciones de lógica
            historial_relevante = st.session_state.messages[:-1] # Todo excepto la última pregunta
            criterios = extraer_criterios_de_busqueda(prompt, historial_relevante)
            
            if criterios:
                contexto, descripcion = busqueda_inteligente(criterios)
                if contexto:
                    # AÑADIDO: Pasamos el historial a la función de respuesta
                    respuesta = generar_respuesta_inteligente(prompt, contexto, descripcion, historial_relevante)
                    st.write(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                else:
                    st.write("Lo siento, no he encontrado ningún modelo que cumpla esos criterios.")
                    st.session_state.messages.append({"role": "assistant", "content": "Lo siento, no he encontrado ningún modelo que cumpla esos criterios."})
            else:
                st.write("No he podido entender tu petición. ¿Puedes reformularla?")
                st.session_state.messages.append({"role": "assistant", "content": "No he podido entender tu petición. ¿Puedes reformularla?"})