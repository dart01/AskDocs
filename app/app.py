import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_pipeline import (
    buscar_chunks_relevantes,
    crear_chunks,
    crear_indice_faiss,
    extraer_texto_pdf,
    generar_respuesta,
)

st.set_page_config(page_title="AskDocs", page_icon="📄", layout="wide")

# Estilos
st.markdown(
    """
<style>
    .main { background-color: #0f1117; }
    .stTextInput input { border-radius: 20px; }
    .mensaje-usuario {
        background-color: #1e3a5f;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: right;
    }
    .mensaje-asistente {
        background-color: #1a1a2e;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 3px solid #4a90d9;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("📄 AskDocs")
st.markdown("Sube un PDF y hazle preguntas en lenguaje natural.")
st.divider()

# Inicializar historial en session state
if "historial" not in st.session_state:
    st.session_state.historial = []
if "indice" not in st.session_state:
    st.session_state.indice = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "documento_cargado" not in st.session_state:
    st.session_state.documento_cargado = False

# Layout en dos columnas
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 Documento")
    archivo = st.file_uploader(
        "Sube tu PDF", type=["pdf"], label_visibility="collapsed"
    )

    if archivo and not st.session_state.documento_cargado:
        with st.spinner("Procesando documento..."):
            texto = extraer_texto_pdf(archivo)
            chunks = crear_chunks(texto)
            indice, _ = crear_indice_faiss(chunks)
            st.session_state.indice = indice
            st.session_state.chunks = chunks
            st.session_state.documento_cargado = True
            st.session_state.historial = []

    if st.session_state.documento_cargado:
        st.success(f"✓ {archivo.name}")
        st.info(f"📊 {len(st.session_state.chunks)} fragmentos indexados")

        if st.button("🗑️ Limpiar conversación"):
            st.session_state.historial = []
            st.rerun()

with col2:
    st.subheader("💬 Conversación")

    # Mostrar historial
    for mensaje in st.session_state.historial:
        if mensaje["rol"] == "usuario":
            st.markdown(
                f"""<div class='mensaje-usuario'>🙋 {mensaje["contenido"]}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class='mensaje-asistente'>🤖 {mensaje["contenido"]}</div>""",
                unsafe_allow_html=True,
            )
            if "fragmentos" in mensaje:
                with st.expander("Ver fragmentos del documento"):
                    for i, chunk in enumerate(mensaje["fragmentos"], 1):
                        st.markdown(f"**Fragmento {i}:**")
                        st.text(chunk[:300] + "...")
                        st.divider()

    # Input de pregunta
    if st.session_state.documento_cargado:
        pregunta = st.text_input("Escribe tu pregunta...", key="input_pregunta")

        if st.button("Preguntar 🚀") and pregunta:
            st.session_state.historial.append({"rol": "usuario", "contenido": pregunta})

            with st.spinner("Pensando..."):
                chunks_relevantes = buscar_chunks_relevantes(
                    pregunta, st.session_state.indice, st.session_state.chunks
                )
                respuesta = generar_respuesta(pregunta, chunks_relevantes)

            st.session_state.historial.append(
                {
                    "rol": "asistente",
                    "contenido": respuesta,
                    "fragmentos": chunks_relevantes,
                }
            )

            st.rerun()
    else:
        st.info("👈 Sube un PDF para comenzar")
