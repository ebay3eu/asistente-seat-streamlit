import streamlit as st
import json
import os
from openai import OpenAI
from pinecone import Pinecone

# --- Configuración de la Página y Título ---
st.set_page_config(
    # TÍTULO QUE APARECE EN LA PESTAÑA DEL NAVEGADOR
    page_title="Asesor de Ventas SEAT",
    page_icon="🚗",
    layout="centered"
)

# --- NUEVO TÍTULO PRINCIPAL DE LA APLICACIÓN ---
st.title("🚗 Asesor de Ventas Digital SEAT")

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

# --- Herramientas Internas y Lógica de IA (sin cambios) ---
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
def determinar_intencion(prompt):
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in ["financ", "pagar", "cuotas"]): return {"intent": "consultar_financiacion"}
    if any(keyword in prompt_lower for keyword in ["concesionario", "tienda", "dónde estáis"]):
        provincias = ["Barcelona", "Girona", "Tarragona", "Lleida"]
        for p in provincias:
            if p.lower() in prompt_lower: return {"intent": "buscar_concesionario", "provincia": p}
        return {"intent": "buscar_concesionario"}
    if any(keyword in prompt_lower for keyword in ["ficha", "catálogo", "documento", "especificaciones"]): return {"intent": "enviar_ficha"}
    if any(keyword in prompt_lower for keyword in ["probar", "conducir", "test drive", "verlo"]): return {"intent": "agendar_prueba"}
    return {"intent": "busqueda_general"}
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
def extraer_criterios_ia(prompt, historial_chat):
    prompt_extraccion = f"""
    Analiza la pregunta del usuario: "{prompt}" y el historial: {historial_chat}.
    Extrae en formato JSON.
    - Si la pregunta parece una búsqueda general, extrae "precio_max" y "descripcion".
    - Si la pregunta pide una ficha o probar un modelo, extrae el "modelo".
    """
    try:
        response = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt_extraccion}], temperature=0.0, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception: return {}

# --- Interfaz de la Aplicación ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- NUEVO MENSAJE DE BIENVENIDA ---
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("""
        ¡Hola! Soy tu **Asesor de Ventas Digital de SEAT**. Estoy aquí para ayudarte a encontrar tu coche perfecto.

        **Puedes pedirme cosas como:**
        - **Buscar un modelo:** *"Busco un SUV familiar por menos de 40.000€"*
        - **Pedir un documento:** *"Envíame la ficha técnica del SEAT León"*
        - **Agendar una prueba:** *"Quiero probar el SEAT Arona"*
        - **Consultar opciones:** *"¿Qué opciones de financiación tenéis?"* o *"Dime los concesionarios en Barcelona"*

        ¿Cómo puedo ayudarte a empezar?
        """)

# Mostrar el historial de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obtener la nueva pregunta del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        historial_relevante = st.session_state.messages[:-1]
        peticion = determinar_intencion(prompt)
        intent = peticion.get("intent")
        
        if intent == "consultar_financiacion":
            respuesta = obtener_info_financiacion()
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

        elif intent == "buscar_concesionario":
            provincia = peticion.get("provincia")
            respuesta = obtener_info_concesionarios(provincia)
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

        elif intent == "enviar_ficha" or intent == "agendar_prueba":
            with st.spinner("Buscando modelo..."):
                entidades = extraer_criterios_ia(prompt, historial_relevante)
                modelo = entidades.get("modelo", "modelo de tu interés").title()
            
            if intent == "enviar_ficha":
                modelo_lower = modelo.lower()
                file_path = os.path.join("fichas_tecnicas", f"{modelo_lower}.pdf")
                if os.path.exists(file_path):
                    st.download_button(label=f"Descargar Ficha Técnica de {modelo}", data=open(file_path, "rb").read(), file_name=f"ficha_tecnica_{modelo_lower}.pdf", mime="application/pdf")
                    st.session_state.messages.append({"role": "assistant", "content": f"He preparado la descarga de la ficha técnica del {modelo}."})
                else: st.warning(f"Lo siento, no he podido encontrar la ficha técnica para el SEAT {modelo}.")

            elif intent == "agendar_prueba":
                st.write(f"¡Claro! Para agendar tu prueba para el **SEAT {modelo}**, completa el formulario:")
                with st.form(key="prueba_conduccion_form"):
                    nombre = st.text_input("Nombre completo")
                    email = st.text_input("Correo electrónico")
                    submitted = st.form_submit_button("Enviar Solicitud")
                    if submitted:
                        st.success(f"¡Gracias, {nombre}! Hemos recibido tu solicitud para probar el SEAT {modelo}. Un agente te contactará pronto.")
                        st.session_state.messages.append({"role": "assistant", "content": f"He procesado la solicitud de prueba de conducción para {nombre}."})
        
        elif intent == "busqueda_general":
            with st.spinner("Pensando..."):
                criterios = extraer_criterios_ia(prompt, historial_relevante)
                if criterios:
                    contexto, descripcion = busqueda_inteligente(criterios)
                    if contexto:
                        respuesta_completa = st.write_stream(generar_respuesta_inteligente(prompt, contexto, descripcion, historial_relevante))
                        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
                    else: st.warning("Lo siento, no he encontrado ningún modelo que cumpla esos criterios.")
                else: st.error("No he podido entender tu petición. ¿Puedes reformularla?")