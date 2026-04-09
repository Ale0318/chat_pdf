import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ----------- ESTILOS PERSONALIZADOS -----------
st.markdown("""
    <style>
    /* Fondo completo */
    .stApp {
        background-color: #f5f5dc; /* blanco hueso */
    }

    /* Texto general */
    .block-container {
        color: #1e293b;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #1e293b;
        border-radius: 8px;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        color: #1e293b;
        border-radius: 8px;
    }

    /* Botones */
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }

    /* File uploader */
    .stFileUploader {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- TÍTULO -----------
st.title("💬 Chat PDF")
st.write("Versión de Python:", platform.python_version())

# ----------- IMAGEN -----------
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# ----------- SIDEBAR -----------
with st.sidebar:
    st.markdown("## 📄 Asistente de PDFs")
    st.markdown("Este agente te ayudará a analizar documentos usando IA 🚀")
    st.markdown("---")
    st.markdown("💡 Tip: Haz preguntas claras para mejores respuestas")

# ----------- API KEY -----------
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("⚠️ Ingresa tu API key para continuar")

# ----------- UPLOADER -----------
pdf = st.file_uploader("📂 Carga tu archivo PDF", type="pdf")

# ----------- PROCESAMIENTO -----------
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"📊 Texto extraído: {len(text)} caracteres")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"✅ Documento dividido en {len(chunks)} fragmentos")

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.subheader("❓ Haz tu pregunta")
        user_question = st.text_area("", placeholder="Ej: ¿Cuál es la idea principal del documento?")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=0, model_name="gpt-4o")

            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.subheader("🧠 Respuesta")
            st.success(response)

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Ingresa tu API key para continuar")
else:
    st.info("📌 Sube un PDF para empezar")
