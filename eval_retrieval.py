import json
import chromadb
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

col = chromadb.PersistentClient(path="db").get_collection("security_docs", embedding_function=ef)
evals = json.load(open("evals.json", encoding="utf-8"))

hit4 = top1 = 0
for e in evals:
    res = col.query(query_texts=[e["q"]], n_results=4)
    sources = [m["source"] for m in res["metadatas"][0]]
    ok = e["expected_source"] in sources
    first = sources[0] == e["expected_source"]
    hit4 += ok
    top1 += first
    print("OK  " if ok else "FAIL", "| top1" if first else "|     ", "|", e["q"])

n = len(evals)
print(f"\nhit@4: {hit4}/{n} ({hit4/n:.0%})  |  top1: {top1}/{n} ({top1/n:.0%})")