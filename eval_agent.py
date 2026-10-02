from agent import run
from collections import Counter

TASK = "Revisa esta arquitectura: un chatbot que lee correos de clientes, consulta documentos internos con RAG y puede enviar correos mediante una API."
print(Counter(run(TASK, verbose=False)[0] for _ in range(10)))