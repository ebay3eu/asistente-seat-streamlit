import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

# --- Configuración de la Página y Título ---
st.set_page_config(
    page_title="Asistente Virtual SEAT",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Asistente Virtual SEAT")
st.caption("Tu experto en la gama de vehículos SEAT. Haz una pregunta para empezar.")

# --- Conexión a los servicios usando los "Secrets" de Streamlit ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Error: No se encontraron las claves de API. Asegúrate de configurar los 'Secrets' en Streamlit Community Cloud.")
    st.stop()

# --- Inicialización de Clientes (con caché para eficiencia) ---
@st.cache_resource
def get_clients():
    """Inicializa y cachea los clientes de OpenAI y Pinecone."""
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("asistente-seat")
    return client_openai, pinecone_index

client_openai, pinecone_index = get_clients()

# --- Funciones Principales ---
def encontrar_contexto_relevante(pregunta_usuario, top_k=5):
    """Encuentra los textos más relevantes en Pinecone para una pregunta."""
    try:
        # 1. Convierte la pregunta en un embedding
        res_embedding = client_openai.embeddings.create(
            input=[pregunta_usuario],
            model="text-embedding-3-small"
        )
        query_embedding = res_embedding.data[0].embedding

        # 2. Busca en Pinecone los vectores más similares
        res_busqueda = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # 3. Extrae los textos originales
        contextos = [item['metadata']['texto'] for item in res_busqueda['matches']]
        return "\n\n---\n\n".join(contextos)
    except Exception as e:
        st.error(f"Error al buscar en Pinecone: {e}")
        return ""

def generar_respuesta(pregunta_usuario, contexto):
    """Genera una respuesta con OpenAI basándose en el contexto."""
    prompt_sistema = f"""
    Eres "Asistente Virtual SEAT", un experto amable y servicial. Tu única fuente de conocimiento es el contexto que te proporciono.
    Responde a la pregunta del usuario de forma clara y concisa, basándote exclusivamente en la siguiente información.
    Si la respuesta no se encuentra en el contexto, di amablemente: "Lo siento, no tengo información sobre ese tema en mi base de datos."
    No inventes nada.

    CONTEXTO:
    {contexto}
    """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o", # Puedes usar "gpt-3.5-turbo" si prefieres
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pregunta_usuario}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la respuesta con OpenAI: {e}")
        return "Hubo un problema al generar la respuesta."

# --- Interfaz de la Aplicación ---
pregunta = st.text_input("Escribe aquí tu pregunta sobre la gama SEAT:", key="pregunta_usuario")

if st.button("Enviar Pregunta", type="primary"):
    if pregunta:
        with st.spinner("Buscando en la base de datos y generando una respuesta..."):
            # 1. Encontrar contexto relevante
            contexto_encontrado = encontrar_contexto_relevante(pregunta)
            
            # 2. Generar respuesta
            if contexto_encontrado:
                respuesta_final = generar_respuesta(pregunta, contexto_encontrado)
                st.markdown("### Respuesta:")
                st.write(respuesta_final)
            else:
                st.warning("No se pudo encontrar información relevante para tu pregunta.")
    else:
        st.warning("Por favor, escribe una pregunta.")