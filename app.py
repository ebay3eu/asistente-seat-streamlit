import streamlit as st
import json
import os
import smtplib
from email.message import EmailMessage
from openai import OpenAI
from pinecone import Pinecone

# --- Configuración y Conexión (Sin cambios) ---
st.set_page_config(page_title="Asistente Virtual SEAT", page_icon="🚗", layout="centered")
st.title("🚗 Asistente Virtual SEAT")
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Error: Faltan claves de API en los 'Secrets'.")
    st.stop()

@st.cache_resource
def get_clients():
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("asistente-seat")
    return client_openai, pinecone_index
client_openai, pinecone_index = get_clients()

# --- Herramientas Internas (Sin cambios) ---
def obtener_info_financiacion():
    return """
    ### Opciones de Financiación SEAT
    Claro, aquí tienes las principales formas de financiar tu nuevo SEAT:
    **1. Crédito Lineal Clásico:** Financieras el importe total o parcial del coche en cuotas fijas mensuales. Al terminar, el coche es 100% tuyo.
    **2. SEAT Flex (Compra Flexible / PCP):** Es la opción más popular. Pagas una entrada opcional y cuotas mensuales reducidas durante 3-4 años. Al final decides si te lo quedas (pagando la última cuota), lo devuelves o lo cambias.
    **3. Leasing / Renting:** Es un alquiler a largo plazo, ideal para empresas y autónomos, con una cuota mensual que suele incluir mantenimiento, seguro, etc.
    """
def obtener_info_concesionarios(provincia=None):
    concesionarios = {
        "Barcelona": "Catalunya Motor, Lesseps Motor, Sarsa (Sabadell/Terrassa), Baix Motor (Sant Boi), Martorell Motor (Martorell).",
        "Girona": "Proauto (Girona, Figueres, Olot), Ablanes (Blanes).",
        "Tarragona": "Baycar (Tarragona, Reus), Auto Esteller (Tortosa).",
        "Lleida": "Dalmau Motor (Lleida), Automotor y Servicios (Lleida, Tàrrega)."
    }
    if provincia and provincia in concesionarios:
        return f"### Concesionarios en {provincia}\n\n* **{concesionarios[provincia]}**"
    respuesta_completa = "### Nuestros Concesionarios en Cataluña\n\n"
    for prov, lista in concesionarios.items():
        respuesta_completa += f"**{prov}:**\n* {lista}\n\n"
    return respuesta_completa

# --- Lógica de IA (Sin cambios) ---
def busqueda_inteligente(criterios, top_k=10):
    # ... (código sin cambios)
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
    # ... (código sin cambios)
    prompt_sistema = "Eres 'Asistente Virtual SEAT'. Responde al usuario basándote en el contexto proporcionado."
    mensajes_para_api = [{"role": "system", "content": prompt_sistema}] 
    mensajes_para_api.extend(historial_chat[-4:]) 
    mensajes_para_api.append({"role": "user", "content": pregunta_original})
    try:
        stream = client_openai.chat.completions.create(model="gpt-4o", messages=mensajes_para_api, temperature=0.5, stream=True)
        for chunk in stream: yield chunk.choices[0].delta.content or ""
    except Exception as e: yield f"Error al generar la respuesta: {e}"

# --- **NUEVA FUNCIÓN DE LÓGICA HÍBRIDA** ---
def procesar_pregunta(prompt, historial_chat):
    """
    Determina la intención del usuario, primero con palabras clave y luego con IA.
    """
    prompt_lower = prompt.lower()
    
    # 1. Detección por palabras clave (rápido y fiable)
    if any(keyword in prompt_lower for keyword in ["financ", "pagar", "cuotas"]):
        respuesta = obtener_info_financiacion()
        st.markdown(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        return

    if any(keyword in prompt_lower for keyword in ["concesionario", "tienda", "dónde estáis"]):
        # Podemos incluso extraer la provincia de forma simple
        provincias = ["Barcelona", "Girona", "Tarragona", "Lleida"]
        provincia_encontrada = None
        for p in provincias:
            if p.lower() in prompt_lower:
                provincia_encontrada = p
                break
        respuesta = obtener_info_concesionarios(provincia_encontrada)
        st.markdown(respuesta)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        return

    # 2. Si no hay palabras clave, usamos la IA para búsquedas complejas
    with st.spinner("Pensando..."):
        # La función de extracción ahora solo se usa para la búsqueda general
        prompt_extraccion = f"""
        Analiza la pregunta del usuario: "{prompt}" y el historial: {historial_chat}.
        Extrae en JSON los criterios "precio_max" y "descripcion".
        """
        try:
            response = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt_extraccion}], temperature=0.0, response_format={"type": "json_object"})
            criterios = json.loads(response.choices[0].message.content)
        except Exception:
            criterios = None

        if criterios:
            contexto, descripcion = busqueda_inteligente(criterios)
            if contexto:
                respuesta_completa = st.write_stream(generar_respuesta_inteligente(prompt, contexto, descripcion, historial_chat))
                st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
            else:
                st.warning("Lo siento, no he encontrado ningún modelo que cumpla esos criterios.")
        else:
            st.error("No he podido entender tu petición. ¿Puedes reformularla?")


# --- Interfaz de la Aplicación ---
if "messages" not in st.session_state: st.session_state.messages = []
if not st.session_state.messages:
    with st.chat_message("assistant"): st.write("¡Hola! Soy tu asistente virtual SEAT. ¿En qué puedo ayudarte?")
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # AHORA LLAMAMOS A NUESTRA NUEVA FUNCIÓN LÓGICA
        procesar_pregunta(prompt, st.session_state.messages[:-1])