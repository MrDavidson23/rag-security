from agent_guided import run
from collections import Counter

TASK = "Revisa esta arquitectura: un chatbot que lee correos de clientes, consulta documentos internos con RAG y puede enviar correos mediante una API."
runs = [run(TASK, verbose=False) for _ in range(10)]
print(Counter(r["status"] for r in runs), "| plan valido:", sum(r["plan_ok"] for r in runs), "/ 10")
print("intentos (solo ok):", Counter(r["attempts"] for r in runs if r["status"] == "ok"))
print("cubrió Prompt_Injection.txt:", sum("Prompt_Injection.txt" in r["sources"] for r in runs), "/ 10")
print("errores de los invalidos:", [r["error"] for r in runs if r["status"] == "invalido"])