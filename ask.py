import sys
import anthropic
import chromadb

llm = anthropic.Anthropic()
col = chromadb.PersistentClient(path="db").get_collection("security_docs")

# System prompt ingenuo A PROPÓSITO: lo endurecemos después del bloque de ataques
SYSTEM = "Responde la pregunta usando el contexto. Cita las fuentes como [n]."

def ask(question, k=4):
    res = col.query(query_texts=[question], n_results=k)
    docs, metas = res["documents"][0], res["metadatas"][0]
    context = "\n\n".join(
        f"[{i+1}] ({m['source']})\n{d}" for i, (d, m) in enumerate(zip(docs, metas))
    )
    msg = llm.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"}],
    )
    return msg.content[0].text

if __name__ == "__main__":
    print(ask(" ".join(sys.argv[1:])))