import sys
import chromadb
from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

col = chromadb.PersistentClient(path="db").get_collection("security_docs", embedding_function=ef)
res = col.query(query_texts=[" ".join(sys.argv[1:])], n_results=4)
for d, m in zip(res["documents"][0], res["metadatas"][0]):
    print(f"--- {m['source']} ---\n{d[:300]}\n")