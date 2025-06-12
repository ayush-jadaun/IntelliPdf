import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def gemini_chat(messages, model="gemini-pro"):
    model = genai.GenerativeModel(model)
    response = model.generate_content(messages)
    return response.text.strip()

def gemini_summarize(text, model="gemini-pro"):
    model = genai.GenerativeModel(model)
    response = model.generate_content(f"Summarize this document:\n{text}")
    return response.text.strip()



def chat_with_documents(user_query, history, context_docs):
    # Build conversation as a list of messages
    messages = [
        {"role": "system", "parts": ["You are an expert research assistant."]},
        # Add conversation history if needed
        {"role": "user", "parts": [user_query]},
        # Optionally add context from documents
    ]
    return gemini_chat(messages)