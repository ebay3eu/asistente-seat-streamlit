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
st.caption("Tu experto en la gama de vehículos SEAT. Usa los filtros y haz una pregunta.")

# --- Conexión a los servicios usando los "Secrets" de Streamlit ---
try:
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except FileNotFoundError:
    st.error("Error: No se encontraron las claves de API. Asegúrate de configurar los 'Secrets' en Streamlit Cloud.")
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
def encontrar_contexto_relevante(pregunta_usuario, filtro_metadata, top_k=5):
    """Encuentra los textos más relevantes en Pinecone usando un filtro."""
    try:
        res_embedding = client_openai.embeddings.create(
            input=[pregunta_usuario],
            model="text-embedding-3-small"
        )
        query_embedding = res_embedding.data[0].embedding

        # AÑADIMOS EL PARÁMETRO "filter" A LA BÚSQUEDA
        res_busqueda = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filtro_metadata
        )
        
        contextos = [item['metadata']['texto'] for item in res_busqueda['matches']]
        return "\n\n---\n\n".join(contextos)
    except Exception as e:
        st.error(f"Error al buscar en Pinecone: {e}")
        return ""

def generar_respuesta(pregunta_usuario, contexto):
    """Genera una respuesta con OpenAI basándose en el contexto."""
    prompt_sistema = f"""
    Eres "Asistente Virtual SEAT". Responde a la pregunta del usuario basándote exclusivamente en el siguiente contexto.
    Si el contexto está vacío o no contiene la respuesta, indica amablemente que no has encontrado modelos que cumplan los criterios.
    No inventes nada.

    CONTEXTO:
    {contexto}
    """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pregunta_usuario}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la respuesta con OpenAI: {e}")
        return "Hubo un problema al generar la respuesta."

# --- Interfaz de la Aplicación con Filtros ---
st.markdown("#### Filtra tu búsqueda")
precio_max = st.number_input(
    "Precio máximo (€)", 
    min_value=0, 
    max_value=100000, 
    value=0, # Por defecto sin filtro
    step=1000,
    help="Introduce un precio máximo para filtrar los resultados. Deja en 0 para no aplicar filtro."
)

st.markdown("---")

pregunta = st.text_input("Escribe aquí tu pregunta (ej: 'un coche con buen maletero', 'el más potente', etc.):", key="pregunta_usuario")

if st.button("Enviar Pregunta", type="primary"):
    if pregunta:
        # Construir el diccionario de filtro
        filtro = {}
        if precio_max > 0:
            filtro["precio"] = {"$lte": precio_max} # "$lte" = lower than or equal (menor o igual que)

        with st.spinner("Filtrando y buscando en la base de datos..."):
            contexto_encontrado = encontrar_contexto_relevante(pregunta, filtro)
            
            if contexto_encontrado:
                respuesta_final = generar_respuesta(pregunta, contexto_encontrado)
                st.markdown("### Resultados")
                st.write(respuesta_final)
            else:
                st.warning("No se encontraron modelos que cumplan con los criterios de búsqueda.")
    else:
        st.warning("Por favor, escribe una pregunta.")