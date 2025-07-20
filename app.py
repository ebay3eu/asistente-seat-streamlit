import streamlit as st
import json
import os
import smtplib
from email.message import EmailMessage
from openai import OpenAI
from pinecone import Pinecone

# --- Configuración de la Página y Título ---
st.set_page_config(page_title="Asistente Virtual SEAT", page_icon="🚗", layout="centered")
st.title("🚗 Asistente Virtual SEAT")

# --- Conexión a los servicios (sin cambios) ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Error: Faltan claves de API en los 'Secrets'.")
    st.stop()

# --- Inicialización de Clientes (sin cambios) ---
@st.cache_resource
def get_clients():
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("asistente-seat")
    return client_openai, pinecone_index

client_openai, pinecone_index = get_clients()

# --- NUEVAS HERRAMIENTAS INTERNAS ---
def obtener_info_financiacion():
    """Devuelve un texto formateado con las opciones de financiación."""
    return """
    ### Opciones de Financiación SEAT

    Claro, aquí tienes las principales formas de financiar tu nuevo SEAT:

    **1. Crédito Lineal Clásico:**
    Es la opción tradicional. Financieras el importe total o parcial del coche en cuotas fijas mensuales durante un plazo que elijas (normalmente de 48 a 96 meses). Al terminar de pagar, el coche es 100% tuyo.

    **2. SEAT Flex (Compra Flexible / PCP):**
    Esta es la opción más popular actualmente. Funciona en tres pasos:
    * **Entrada:** Das una entrada (que puede ser opcional).
    * **Cuotas reducidas:** Pagas cuotas mensuales muy cómodas durante 3 o 4 años.
    * **Decisión final:** Al acabar el plazo, tienes 3 opciones:
        1. **Quedártelo:** Pagando la última cuota (el Valor Futuro Garantizado).
        2. **Devolverlo:** Sin costes adicionales.
        3. **Cambiarlo:** Por un nuevo modelo SEAT, usando el valor del coche actual para la entrada del nuevo.

    **3. Leasing / Renting:**
    Es un alquiler a largo plazo, ideal para empresas y autónomos. Pagas una cuota mensual que normalmente lo incluye todo (mantenimiento, seguro, impuestos, etc.). Al final del contrato, no eres el propietario del coche.
    """

def obtener_info_concesionarios(provincia=None):
    """Devuelve un texto con la lista de concesionarios, opcionalmente filtrada por provincia."""
    concesionarios = {
        "Barcelona": "Catalunya Motor, Lesseps Motor, Sarsa (Sabadell/Terrassa), Baix Motor (Sant Boi), Martorell Motor (Martorell).",
        "Girona": "Proauto (Girona, Figueres, Olot), Ablanes (Blanes).",
        "Tarragona": "Baycar (Tarragona, Reus), Auto Esteller (Tortosa).",
        "Lleida": "Dalmau Motor (Lleida), Automotor y Servicios (Lleida, Tàrrega)."
    }
    
    if provincia and provincia in concesionarios:
        return f"### Concesionarios en {provincia}\n\nAquí tienes algunos de nuestros concesionarios oficiales en la provincia de {provincia}:\n\n* **{concesionarios[provincia]}**"
    
    # Si no se especifica provincia, devolver todos
    respuesta_completa = "### Nuestros Concesionarios en Cataluña\n\n"
    for prov, lista in concesionarios.items():
        respuesta_completa += f"**{prov}:**\n* {lista}\n\n"
    return respuesta_completa

# --- Funciones de Lógica (con IA) ---
def extraer_criterios_de_busqueda(pregunta_usuario, historial_chat):
    """Identifica la intención y extrae las entidades de la pregunta."""
    conversacion_para_contexto = "\n".join([f"{msg['role']}: {msg['content']}" for msg in historial_chat])
    # AÑADIMOS LAS NUEVAS INTENCIONES
    prompt = f"""
    Analiza la última pregunta del usuario y el historial para determinar su intención.
    Intenciones posibles: "consultar_financiacion", "buscar_concesionario", "agendar_prueba", "enviar_ficha", "busqueda_general".

    Responde en formato JSON con:
    - "intent": la intención identificada.
    - "provincia": si la intención es "buscar_concesionario", la provincia en formato capitalizado (ej: "Girona"). Si no se menciona, null.
    - "modelo": si es relevante para la intención (agendar_prueba, enviar_ficha).
    - "criterios": si la intención es "busqueda_general".
    
    Ejemplos:
    - Pregunta: "¿Cómo puedo financiar el coche?" -> {{"intent": "consultar_financiacion", "provincia": null, "modelo": null, "criterios": null}}
    - Pregunta: "¿Dónde hay un concesionario en Tarragona?" -> {{"intent": "buscar_concesionario", "provincia": "Tarragona", "modelo": null, "criterios": null}}
    - Pregunta: "dime los concesionarios" -> {{"intent": "buscar_concesionario", "provincia": null, "modelo": null, "criterios": null}}

    Responde únicamente con el objeto JSON.
    """
    try:
        response = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}], temperature=0.0, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception: return None
    
# Las funciones `busqueda_inteligente` y `generar_respuesta_inteligente` no cambian
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
    prompt_sistema = "Eres 'Asistente Virtual SEAT'. Responde al usuario basándote en el contexto proporcionado."
    mensajes_para_api = [{"role": "system", "content": prompt_sistema}] 
    mensajes_para_api.extend(historial_chat[-4:]) 
    mensajes_para_api.append({"role": "user", "content": pregunta_original})
    try:
        stream = client_openai.chat.completions.create(model="gpt-4o", messages=mensajes_para_api, temperature=0.5, stream=True)
        for chunk in stream: yield chunk.choices[0].delta.content or ""
    except Exception as e: yield f"Error al generar la respuesta: {e}"

# --- Interfaz de la Aplicación ---
if "messages" not in st.session_state: st.session_state.messages = []
if not st.session_state.messages:
    with st.chat_message("assistant"): st.write("¡Hola! Soy tu asistente virtual de SEAT. ¿En qué puedo ayudarte?")
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            historial_relevante = st.session_state.messages[:-1]
            peticion = extraer_criterios_de_busqueda(prompt, historial_relevante)
            
            if peticion:
                intent = peticion.get("intent")
                
                # --- NUEVA LÓGICA DE HERRAMIENTAS ---
                if intent == "consultar_financiacion":
                    respuesta = obtener_info_financiacion()
                    st.markdown(respuesta) # Usamos markdown para formato
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})

                elif intent == "buscar_concesionario":
                    provincia = peticion.get("provincia")
                    respuesta = obtener_info_concesionarios(provincia)
                    st.markdown(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})

                # --- Las demás intenciones siguen igual ---
                elif intent == "agendar_prueba":
                    modelo = peticion.get("modelo", "modelo de tu interés").title()
                    st.write(f"¡Claro! Para agendar tu prueba para el **SEAT {modelo}**, completa el formulario:")
                    # ... (código del formulario omitido por brevedad, no cambia) ...

                elif intent == "enviar_ficha":
                    modelo_lower = peticion.get("modelo", "").lower()
                    # ... (código de la ficha técnica omitido por brevedad, no cambia) ...

                elif intent == "busqueda_general":
                    contexto, descripcion = busqueda_inteligente(peticion.get("criterios"))
                    if contexto:
                        respuesta_completa = st.write_stream(generar_respuesta_inteligente(prompt, contexto, descripcion, historial_relevante))
                        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
                    else:
                        st.warning("Lo siento, no he encontrado ningún modelo que cumpla esos criterios.")
            else:
                st.error("No he podido entender tu petición. ¿Puedes reformularla?")