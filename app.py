import streamlit as st
import json
import os
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
    """
    Usa un LLM para identificar la INTENCIÓN del usuario y extraer entidades.
    """
    conversacion_para_contexto = "\n".join([f"{msg['role']}: {msg['content']}" for msg in historial_chat])
    prompt = f"""
    Analiza la última pregunta del usuario y el historial para determinar su intención.
    HISTORIAL: {conversacion_para_contexto}
    ÚLTIMA PREGUNTA: "{pregunta_usuario}"

    Identifica una de las siguientes intenciones:
    1. "agendar_prueba": si el usuario quiere probar, conducir, ver en persona o reservar un test drive de un modelo.
    2. "enviar_ficha": si el usuario pide la ficha técnica, catálogo o documento de un modelo.
    3. "busqueda_general": para cualquier otra pregunta.

    Responde en formato JSON. El JSON debe tener:
    - "intent": la intención identificada.
    - "modelo": el nombre del modelo en minúsculas si la intención es "agendar_prueba" o "enviar_ficha". Si no, null.
    - "criterios": si la intención es "busqueda_general", un objeto con "precio_max" y "descripcion". Si no, null.

    Ejemplos:
    - Pregunta: "quiero probar el formentor" -> {{"intent": "agendar_prueba", "modelo": "formentor", "criterios": null}}
    - Pregunta: "ficha del ateca" -> {{"intent": "enviar_ficha", "modelo": "ateca", "criterios": null}}
    - Pregunta: "cuál es el más barato" -> {{"intent": "busqueda_general", "modelo": null, "criterios": {{"precio_max": 0, "descripcion": "el coche más barato"}}}}

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

# --- Las demás funciones de búsqueda y generación de respuesta no cambian ---
def busqueda_inteligente(criterios, top_k=10):
    if not criterios or not criterios.get("descripcion"): return None, None
    filtro_metadata = {}
    if criterios.get("precio_max", 0) > 0: filtro_metadata["precio"] = {"$lte": criterios["precio_max"]}
    terminos_genericos = ["coche", "coches", "vehículo", "vehículos", "un coche", "dime los modelos", "modelos disponibles", "dime que coches hay"]
    descripcion_normalizada = criterios.get("descripcion", "").lower().strip()
    if filtro_metadata and descripcion_normalizada in terminos_genericos:
        res_busqueda = pinecone_index.query(vector=[0.0] * 1536, top_k=top_k, include_metadata=True, filter=filtro_metadata)
    else:
        query_embedding = client_openai.embeddings.create(input=[criterios["descripcion"]], model="text-embedding-3-small").data[0].embedding
        res_busqueda = pinecone_index.query(vector=query_embedding, top_k=5, include_metadata=True, filter=filtro_metadata)
    if res_busqueda['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda['matches']]
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]
    return None, None

def generar_respuesta_inteligente(pregunta_original, contexto, descripcion_buscada, historial_chat):
    prompt_sistema = f"""
    Eres "Asistente Virtual SEAT". Tu objetivo es responder al usuario basándote exclusivamente en el contexto que te proporciono.
    La pregunta del usuario fue: "{pregunta_original}"
    Las características que buscaba eran: "{descripcion_buscada}"
    CONTEXTO ENCONTRADO: {contexto}
    Tu tarea es revisar los resultados del contexto y formular una respuesta clara y amable resumiendo los modelos encontrados.
    """
    mensajes_para_api = [{"role": "system", "content": prompt_sistema}] 
    mensajes_para_api.extend(historial_chat[-4:]) 
    mensajes_para_api.append({"role": "user", "content": pregunta_original})
    try:
        stream = client_openai.chat.completions.create(model="gpt-4o", messages=mensajes_para_api, temperature=0.5, stream=True)
        for chunk in stream: yield chunk.choices[0].delta.content or ""
    except Exception as e: yield f"Error al generar la respuesta: {e}"

# --- Interfaz de la Aplicación ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("¡Hola! Soy tu asistente virtual de SEAT. Puedo ayudarte a encontrar un modelo, enviarte una ficha técnica o ¡incluso agendar una prueba de conducción!")

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
            peticion = extraer_criterios_de_busqueda(prompt, historial_relevante)
            
            if peticion:
                intent = peticion.get("intent")
                modelo = peticion.get("modelo", "modelo de tu interés").title()

                if intent == "agendar_prueba":
                    st.write(f"¡Claro! Para agendar tu prueba de conducción para el **SEAT {modelo}**, por favor, completa el siguiente formulario:")
                    
                    with st.form(key="prueba_conduccion_form"):
                        nombre = st.text_input("Nombre completo")
                        email = st.text_input("Correo electrónico")
                        telefono = st.text_input("Teléfono de contacto")
                        submitted = st.form_submit_button("Enviar Solicitud")

                        if submitted:
                            # En una app real, aquí enviarías un email o guardarías en un CRM.
                            # Para la demo, solo mostramos un mensaje de éxito.
                            st.success(f"¡Gracias, {nombre}! Hemos recibido tu solicitud para probar el SEAT {modelo}. Un agente te contactará pronto en {email} o {telefono}. ¿Hay algo más que necesites?")
                            # Guardamos un mensaje de éxito en el historial
                            st.session_state.messages.append({"role": "assistant", "content": f"He procesado la solicitud de prueba de conducción para {nombre}."})
                
                elif intent == "enviar_ficha":
                    modelo_lower = peticion.get("modelo", "").lower()
                    file_path = os.path.join("fichas_tecnicas", f"{modelo_lower}.pdf")
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as pdf_file:
                            st.write(f"¡Por supuesto! Aquí tienes la ficha técnica del SEAT {modelo}. Haz clic para descargar:")
                            st.download_button(label=f"Descargar Ficha Técnica de {modelo}", data=pdf_file, file_name=f"ficha_tecnica_{modelo_lower}.pdf", mime="application/pdf")
                        st.session_state.messages.append({"role": "assistant", "content": f"He preparado la descarga de la ficha técnica del {modelo}."})
                    else:
                        st.warning(f"Lo siento, no he podido encontrar la ficha técnica para el SEAT {modelo}.")
                        st.session_state.messages.append({"role": "assistant", "content": f"No encontré la ficha para el {modelo}."})

                elif intent == "busqueda_general":
                    criterios = peticion.get("criterios")
                    contexto, descripcion = busqueda_inteligente(criterios)
                    if contexto:
                        respuesta_completa = st.write_stream(generar_respuesta_inteligente(prompt, contexto, descripcion, historial_relevante))
                        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
                    else:
                        st.warning("Lo siento, no he encontrado ningún modelo que cumpla con los criterios de tu búsqueda.")
                        st.session_state.messages.append({"role": "assistant", "content": "No encontré modelos con esos criterios."})
            else:
                st.error("No he podido entender tu petición. ¿Puedes reformularla?")
                st.session_state.messages.append({"role": "assistant", "content": "No entendí la petición."})