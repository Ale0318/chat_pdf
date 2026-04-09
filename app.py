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
    .main {
        background-color: #f5f5dc; /* blanco hueso */
        color: #1e293b;
    }

    h1 {
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        text-align: center;
    }

    p {
        color: #475569;
        text-align: center;
    }

    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }

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

    .stFileUploader {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- TÍTULO -----------
st.markdown("""
    <h1 style='
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #1e293b;
        margin-bottom: 10px;
    '>
        💬 Chat PDF
    </h1>
""", unsafe_allow_html=True)

st.markdown(f"<p>Python version: {platform.python_version()}</p>", unsafe_allow_html=True)

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

        st.markdown("## ❓ Haz tu pregunta")
        user_question = st.text_area("", placeholder="Ej: ¿Cuál es la idea principal del documento?")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=0, model_name="gpt-4o")

            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("## 🧠 Respuesta")
            st.success(response)

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Ingresa tu API key para continuar")
else:
    st.info("📌 Sube un PDF para empezar")
