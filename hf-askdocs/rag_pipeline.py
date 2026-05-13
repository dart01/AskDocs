import os

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# Cargar modelo de embeddings
modelo_emb = SentenceTransformer("all-MiniLM-L6-v2")

# Cliente de Groq
cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extraer_texto_pdf(archivo):
    """Extrae todo el texto de un PDF página por página"""
    reader = PdfReader(archivo)
    texto_completo = ""
    for i, pagina in enumerate(reader.pages):
        texto = pagina.extract_text()
        if texto:
            texto_completo += f"\n[Página {i + 1}]\n{texto}"
    return texto_completo


def crear_chunks(texto, tamano=500, solapamiento=50):
    """Divide el texto en fragmentos con solapamiento"""
    palabras = texto.split()
    chunks = []
    i = 0
    while i < len(palabras):
        chunk = " ".join(palabras[i : i + tamano])
        chunks.append(chunk)
        i += tamano - solapamiento
    return chunks


def crear_indice_faiss(chunks):
    """Convierte chunks a embeddings y los guarda en FAISS"""
    embeddings = modelo_emb.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(embeddings)

    return indice, embeddings


def buscar_chunks_relevantes(pregunta, indice, chunks, k=4):
    """Busca los chunks más relevantes para la pregunta"""
    emb_pregunta = modelo_emb.encode([pregunta])
    emb_pregunta = np.array(emb_pregunta).astype("float32")

    distancias, indices = indice.search(emb_pregunta, k)
    chunks_relevantes = [chunks[i] for i in indices[0]]

    return chunks_relevantes


def generar_respuesta(pregunta, chunks_relevantes):
    """Genera una respuesta usando Groq con los chunks relevantes"""
    contexto = "\n\n---\n\n".join(chunks_relevantes)

    prompt = f"""Eres un asistente experto en analizar documentos. 
Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si la respuesta no está en el contexto, di "No encontré información sobre eso en el documento".
Cita las partes relevantes del documento en tu respuesta.

Contexto del documento:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    return respuesta.choices[0].message.content


def procesar_pdf_y_preguntar(archivo, pregunta):
    """Pipeline completo: PDF → respuesta"""
    print("Extrayendo texto...")
    texto = extraer_texto_pdf(archivo)

    print("Creando chunks...")
    chunks = crear_chunks(texto)
    print(f"Total chunks: {len(chunks)}")

    print("Creando índice FAISS...")
    indice, _ = crear_indice_faiss(chunks)

    print("Buscando chunks relevantes...")
    chunks_relevantes = buscar_chunks_relevantes(pregunta, indice, chunks)

    print("Generando respuesta...")
    respuesta = generar_respuesta(pregunta, chunks_relevantes)

    return respuesta, chunks_relevantes
