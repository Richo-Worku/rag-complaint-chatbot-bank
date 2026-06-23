import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import faiss
import pandas as pd
import streamlit as st
st.write("App is running")
# -----------------------------
# LOAD MODELS
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# LOAD FAISS + METADATA
# -----------------------------
index = faiss.read_index("vector_store/complaints_faiss.index")
metadata_df = pd.read_parquet("data/raw/complaint_embeddings.parquet")


# -----------------------------
# RETRIEVER
# -----------------------------
def retrieve_chunks(query, k=5):

    query_embedding = embedding_model.encode([query]).astype(np.float32)

    distances, indices = index.search(query_embedding, k)

    results = metadata_df.iloc[indices[0]].copy()
    results["distance"] = distances[0]

    return results


# -----------------------------
# GENERATOR
# -----------------------------
def generate_answer(query, k=5):

    results = retrieve_chunks(query, k)
    context = "\n\n".join(results["text"].tolist()[:3])

    prompt = f"""
Answer using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, results


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("💳 Credit Complaint RAG Assistant")

query = st.text_input("Enter your question:")

col1, col2 = st.columns(2)

with col1:
    ask = st.button("Ask")

with col2:
    clear = st.button("Clear")

if clear:
    st.rerun()


if ask and query:

    answer, sources = generate_answer(query)

    st.subheader("🧠 Answer")
    st.write(answer)

    st.subheader("📌 Source Chunks")

    for i, row in sources.iterrows():
        st.markdown(f"**Complaint ID:** {row['complaint_id']}")
        st.write(row["text"])
        st.markdown("---")