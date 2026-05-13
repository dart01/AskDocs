# 📄 AskDocs — Pregúntale a tus documentos

AskDocs nació de una necesidad muy concreta: a veces tienes un documento de 80 páginas y solo necesitas encontrar una respuesta específica. Leerlo completo toma horas. AskDocs lo hace en segundos.

Sube cualquier PDF — un contrato, un manual técnico, un paper académico — y hazle preguntas en lenguaje natural. El sistema encuentra las partes relevantes y responde con precisión citando el documento.

---

## Demo en vivo

👉 [Probar AskDocs](#) ← próximamente en Hugging Face Spaces

---

## Cómo funciona

AskDocs implementa un pipeline RAG (Retrieval-Augmented Generation) completo desde cero, sin frameworks que oculten la lógica interna.

### El pipeline de ingesta
Cuando subes un PDF, el sistema extrae todo el texto página por página, lo divide en fragmentos de 500 palabras con solapamiento para no perder contexto en los cortes, y convierte cada fragmento en un vector de 384 dimensiones usando sentence-transformers. Esos vectores se indexan en FAISS para búsquedas ultrarrápidas.

### El pipeline de consulta
Cuando haces una pregunta, el sistema la convierte en el mismo espacio vectorial y busca los 4 fragmentos más similares semánticamente. Luego le pasa esos fragmentos junto con tu pregunta al modelo Llama 3 corriendo en Groq, que genera una respuesta precisa basada únicamente en el contenido del documento.

El resultado es un sistema que nunca inventa información — si la respuesta no está en el documento, te lo dice.

---

## Stack técnico

| Capa | Herramienta |
|---|---|
| Extracción de PDF | pypdf |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Base vectorial | FAISS |
| LLM | Llama 3 via Groq API |
| Interfaz | Streamlit |

---

## Estructura del proyecto

```text
AskDocs/
├── app/
│   └── app.py              ← interfaz Streamlit
├── src/
│   └── rag_pipeline.py     ← pipeline RAG completo
├── data/                   ← PDFs de prueba (ignorado en git)
├── .env                    ← credenciales (ignorado en git)
├── requirements.txt
└── README.md
```

---

## Correr localmente

```bash
git clone https://github.com/dart01/askdocs.git
cd askdocs

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# Crear .env con tu API key de Groq
echo GROQ_API_KEY=tu_key > .env

streamlit run app/app.py
```

---

## Lo que aprendí

Implementar RAG desde cero sin LangChain me obligó a entender cada paso del pipeline en detalle. El chunking con solapamiento fue el aspecto más crítico — el tamaño y el solapamiento de los fragmentos afecta directamente la calidad de las respuestas. Con fragmentos muy pequeños el modelo pierde contexto, con fragmentos muy grandes introduce ruido innecesario.

---

## Autor

**Diego Riaño**
[LinkedIn](#) · [GitHub](https://github.com/dart01) · [Demo en vivo](#)