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
    """Inicializa y cachea los clientes de OpenAI y Pinecone."""
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("asistente-seat")
    return client_openai, pinecone_index

client_openai, pinecone_index = get_clients()

# --- Funciones de Lógica ---
def extraer_criterios_de_busqueda(pregunta_usuario, historial_chat):
    """Usa un LLM para convertir la pregunta en un JSON con criterios de búsqueda."""
    conversacion_para_contexto = "\n".join([f"{msg['role']}: {msg['content']}" for msg in historial_chat])
    prompt = f"""
    Analiza la última pregunta del usuario teniendo en cuenta el historial de la conversación para darle contexto.
    HISTORIAL:
    {conversacion_para_contexto}

    ÚLTIMA PREGUNTA: "{pregunta_usuario}"

    Extrae los criterios en un JSON con claves "precio_max" (int) y "descripcion" (string), combinando el historial si es necesario.
    Responde únicamente con el objeto JSON.
    """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None

def busqueda_inteligente(criterios, top_k=5):
    """Realiza una búsqueda en Pinecone, relajándola si es necesario."""
    # **LA CORRECCIÓN ESTÁ AQUÍ**
    # Si no hay descripción o está vacía, no podemos buscar.
    if not criterios.get("descripcion"):
        return None, None

    filtro_metadata = {}
    if criterios.get("precio_max") and criterios["precio_max"] > 0:
        filtro_metadata["precio"] = {"$lte": criterios["precio_max"]}

    query_embedding = client_openai.embeddings.create(input=[criterios["descripcion"]], model="text-embedding-3-small").data[0].embedding
    res_busqueda = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filtro_metadata)

    if res_busqueda['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda['matches']]
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]

    pregunta_relajada = "dime todos los modelos de coche disponibles"
    query_embedding_relajado = client_openai.embeddings.create(input=[pregunta_relajada], model="text-embedding-3-small").data[0].embedding
    res_busqueda_relajada = pinecone_index.query(vector=query_embedding_relajado, top_k=top_k, include_metadata=True, filter=filtro_metadata)
    
    if res_busqueda_relajada['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda_relajada['matches']]
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]

    return None, None

def generar_respuesta_inteligente(pregunta_original, contexto, descripcion_buscada, historial_chat):
    """Genera una respuesta en streaming explicando qué se encontró y qué no."""
    prompt_sistema = f"""
    Eres "Asistente Virtual SEAT", un experto amable y servicial. Tu objetivo es ayudar al usuario a encontrar su coche ideal.
    La pregunta original del usuario fue: "{pregunta_original}"
    El usuario buscaba un coche con estas características: "{descripcion_buscada}"

    He realizado una búsqueda y he encontrado los siguientes modelos que podrían encajar parcialmente:
    CONTEXTO ENCONTRADO:
    {contexto}

    Tu tarea es responder al usuario de forma inteligente y conversacional.
    Usa el contexto encontrado para formular tu respuesta. Si el contexto no coincide con alguna parte específica de la descripción, explícalo amablemente.
    """
    
    mensajes_para_api = [{"role": "system", "content": prompt_sistema}] 
    mensajes_para_api.extend(historial_chat[-4:]) 
    mensajes_para_api.append({"role": "user", "content": pregunta_original})

    try:
        stream = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=mensajes_para_api,
            temperature=0.5,
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
            
    except Exception as e:
        yield f"Error al generar la respuesta: {e}"

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
            historial_relevante = st.session_state.messages[:-1]
            criterios = extraer_criterios_de_busqueda(prompt, historial_relevante)
            
            if criterios:
                contexto, descripcion = busqueda_inteligente(criterios)
                if contexto:
                    respuesta_completa = st.write_stream(
                        generar_respuesta_inteligente(prompt, contexto, descripcion, historial_relevante)
                    )
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
                else:
                    respuesta_error = "Lo siento, no he encontrado ningún modelo que cumpla esos criterios."
                    st.write(respuesta_error)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_error})
            else:
                respuesta_error = "No he podido entender tu petición. ¿Puedes reformularla?"
                st.write(respuesta_error)
                st.session_state.messages.append({"role": "assistant", "content": respuesta_error})