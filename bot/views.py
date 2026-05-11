import os

from django.shortcuts import render, redirect
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chatbot_project",
    "chroma_db"
)


def chat_view(request):
    if "chat_history" not in request.session:
        request.session["chat_history"] = []

    if request.method == "POST":
        pregunta = request.POST.get("pregunta", "").strip()

        if pregunta:
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text")

                vectorstore = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=embeddings
                )

                documentos = vectorstore.similarity_search(pregunta, k=3)

                contexto = "\n\n".join([
                    doc.page_content for doc in documentos
                ])

                fuentes = list(set([
                    os.path.basename(doc.metadata.get("source", "Documento"))
                    for doc in documentos
                ]))

                modelo = OllamaLLM(model="llama3.2:3b")

                prompt = f"""
Eres un asistente virtual de Posgrado de la UNACH.
Tu tono debe ser amable, claro, institucional y cercano.

Responde únicamente con base en la información del CONTEXTO.
No inventes información.

Si la información no aparece claramente en el contexto, responde:
"Por el momento no tengo información suficiente en los documentos cargados para responder con seguridad. Te recomiendo consultar con la Dirección de Posgrado."

CONTEXTO:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

RESPUESTA:
"""

                respuesta = modelo.invoke(prompt)

            except Exception as e:
                respuesta = f"Error al consultar el chatbot: {e}"
                fuentes = []

            historial = request.session.get("chat_history", [])
            historial.append({
                "pregunta": pregunta,
                "respuesta": respuesta,
                "fuentes": fuentes,
            })

            request.session["chat_history"] = historial
            request.session.modified = True

        return redirect("bot:chat")

    return render(request, "bot/chat.html", {
        "chat_history": request.session.get("chat_history", [])
    })


def limpiar_chat(request):
    request.session["chat_history"] = []
    request.session.modified = True
    return redirect("bot:chat")