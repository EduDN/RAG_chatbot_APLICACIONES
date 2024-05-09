import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
import base64

# Load environment variables
load_dotenv()

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
    (
        "user",
        """Eres un asistente de preguntas y respuestas que respondes en español. Para todas las demás consultas, tu objetivo principal es proporcionar respuestas lo más precisas posible, basadas en las instrucciones y el contexto que se te ha dado. Si una pregunta no coincide con el contexto proporcionado o está fuera del alcance del documento, por favor, aconseja al usuario que haga preguntas dentro del contexto del documento.
        Contexto:
        {context_str}
        Pregunta:
        {query_str}
        """
    )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Lo siento, no pude encontrar una respuesta."


# Inicialización de la aplicación de Streamlit
st.title("(PDF) Información e Inferencia🗞️")
st.markdown("Generación Aumentada por Recuperación") 
st.markdown("comienza a chatear ...🚀")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": '¡Hola! Sube un PDF y pregúntame cualquier cosa sobre su contenido.'}]

with st.sidebar:
    st.title("Menú:")
    uploaded_file = st.file_uploader("Sube tus archivos PDF y haz clic en el botón Enviar y Procesar")
    if st.button("Enviar y Procesar"):
        with st.spinner("Procesando..."):
            filepath = "data/saved_pdf.pdf"
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # displayPDF(filepath)  # Muestra el PDF cargado
            data_ingestion()  # Procesa el PDF cada vez que se carga un nuevo archivo
            st.success("Listo")

user_prompt = st.chat_input("Pregúntame cualquier cosa sobre el contenido del PDF:")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = handle_query(user_prompt)
    st.session_state.messages.append({'role': 'assistant', "content": response})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
