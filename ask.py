import os
import sys
import chromadb
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" o "anthropic"

col = chromadb.PersistentClient(path="db").get_collection("security_docs", embedding_function=ef)

# System prompt ingenuo A PROPÓSITO: lo endurecemos después del bloque de ataques
SYSTEM = "Responde la pregunta usando el contexto. Cita las fuentes como [n]."


def generate(system, user):
    if PROVIDER == "ollama":
        import ollama
        r = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r["message"]["content"]

    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def ask(question, k=4):
    res = col.query(query_texts=[question], n_results=k)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(
        f"[{i+1}] ({m['source']})\n{d}" for i, (d, m) in enumerate(zip(docs, metas))
    )
    return generate(SYSTEM, f"Contexto:\n{context}\n\nPregunta: {question}")


if __name__ == "__main__":
    print(ask(" ".join(sys.argv[1:])))