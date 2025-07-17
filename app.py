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
st.caption("Tu experto en la gama de vehículos SEAT. Soy capaz de entender peticiones complejas y filtrar por precio.")

# --- Conexión a los servicios usando los "Secrets" de Streamlit ---
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

# --- MAGIA NUEVA (1): Función para extraer criterios de la pregunta ---
def extraer_criterios_de_busqueda(pregunta_usuario):
    """
    Usa un LLM para convertir la pregunta en lenguaje natural a un objeto JSON
    con los criterios de búsqueda que podemos usar.
    """
    prompt = f"""
    Analiza la siguiente pregunta de un usuario y extráela en un formato JSON.
    La pregunta es: "{pregunta_usuario}"

    El JSON debe tener dos claves:
    1. "precio_max": un entero con el precio máximo si se menciona. Si no, 0.
    2. "descripcion": una cadena de texto que resuma todas las demás características que busca el usuario (ej: 'coche grande con buen maletero', 'deportivo y potente', 'híbrido con techo panorámico').

    Ejemplos:
    - Pregunta: "un coche por menos de 30000 euros que sea bueno para viajar" -> JSON: {{"precio_max": 30000, "descripcion": "coche bueno para viajar"}}
    - Pregunta: "el más potente y deportivo" -> JSON: {{"precio_max": 0, "descripcion": "el más potente y deportivo"}}
    - Pregunta: "algo grande que pueda ser de color verde y con techo panoramico" -> JSON: {{"precio_max": 0, "descripcion": "algo grande de color verde y con techo panoramico"}}

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

# --- MAGIA NUEVA (2): Función de búsqueda que se relaja si no encuentra ---
def busqueda_inteligente(criterios, top_k=5):
    """
    Realiza una búsqueda en Pinecone. Si la descripción es muy específica y no
    devuelve resultados, la relaja a una búsqueda más general.
    """
    filtro_metadata = {}
    if criterios.get("precio_max") and criterios["precio_max"] > 0:
        filtro_metadata["precio"] = {"$lte": criterios["precio_max"]}

    # --- Intento 1: Búsqueda estricta con la descripción completa ---
    query_embedding = client_openai.embeddings.create(
        input=[criterios["descripcion"]], model="text-embedding-3-small"
    ).data[0].embedding
    
    res_busqueda = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filtro_metadata
    )

    # Si la búsqueda estricta funciona, devuelve los resultados
    if res_busqueda['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda['matches']]
        # Devolvemos el contexto y la descripción original que sí funcionó
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]

    # --- Intento 2: Búsqueda relajada (si la primera falló) ---
    # Si no hubo resultados, es probable que la descripción fuera demasiado específica
    # (ej: "color verde"). Ahora buscamos sin descripción, solo con el filtro de precio.
    # La pregunta que usaremos para el embedding será más genérica.
    st.info("La búsqueda inicial fue demasiado específica. Intentando una búsqueda más amplia...")
    
    pregunta_relajada = "dime todos los modelos de coche disponibles"
    query_embedding_relajado = client_openai.embeddings.create(
        input=[pregunta_relajada], model="text-embedding-3-small"
    ).data[0].embedding

    res_busqueda_relajada = pinecone_index.query(
        vector=query_embedding_relajado,
        top_k=top_k,
        include_metadata=True,
        filter=filtro_metadata
    )
    
    if res_busqueda_relajada['matches']:
        contexto = [item['metadata']['texto'] for item in res_busqueda_relajada['matches']]
        # Devolvemos el contexto y la descripción original que falló, para que el LLM sepa qué explicar
        return "\n\n---\n\n".join(contexto), criterios["descripcion"]

    return None, None


def generar_respuesta_inteligente(pregunta_original, contexto, descripcion_buscada):
    """
    Genera una respuesta que explica qué se encontró y qué no.
    """
    prompt_sistema = f"""
    Eres "Asistente Virtual SEAT", un experto amable y servicial. Tu objetivo es ayudar al usuario a encontrar su coche ideal.
    La pregunta original del usuario fue: "{pregunta_original}"
    El usuario buscaba un coche con estas características: "{descripcion_buscada}"

    He realizado una búsqueda y he encontrado los siguientes modelos que podrían encajar parcialmente:
    CONTEXTO ENCONTRADO:
    {contexto}

    Tu tarea es responder al usuario de forma inteligente:
    1. Si el contexto parece coincidir bien con la descripción buscada, simplemente resume los resultados y preséntalos.
    2. Si el contexto NO parece coincidir con alguna parte específica de la descripción (ej: el usuario pidió "color verde" pero en el contexto no se menciona), explícalo amablemente. Di qué es lo que NO encontraste, y presenta los resultados que SÍ encontraste como una alternativa.
    
    Ejemplo de respuesta inteligente:
    "He buscado un coche grande con techo panorámico y de color verde. Aunque en mi base de datos no tengo información específica sobre los colores, sí he encontrado estos modelos grandes que ofrecen el techo panorámico como extra opcional: [resume los resultados del contexto]."

    Responde de forma clara y útil.
    """
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pregunta_original}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error al generar la respuesta con OpenAI: {e}")
        return "Hubo un problema al generar la respuesta."

# --- Interfaz de la Aplicación ---
pregunta = st.text_input("Escribe aquí tu pregunta (ej: 'coche por menos de 30.000€', 'el más potente y deportivo', 'híbrido con techo panorámico')", key="pregunta_usuario")