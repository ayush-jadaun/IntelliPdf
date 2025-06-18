import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def gemini_chat(messages, model="gemini-1.5-flash-latest"):
    model = genai.GenerativeModel(model)
    response = model.generate_content(messages)
    return response.text.strip()

def gemini_summarize(text, model="gemini-1.5-flash-latest"):
    model = genai.GenerativeModel(model)
    response = model.generate_content(f"Summarize this document:\n{text}")
    return response.text.strip()

def generate_gemini_response(chat_message):
    """
    Wrapper to generate a Gemini response compatible with FastAPI chat endpoint.
    Accepts a ChatMessage (Pydantic model or dict) and returns the assistant's reply.
    """
    # Build Gemini messages list
    messages = []
    # Optionally add system prompt
    messages.append({"role": "system", "parts": ["You are an expert research assistant."]})

    # Add the user message
    if hasattr(chat_message, "text"):
        user_text = chat_message.text
    elif isinstance(chat_message, dict):
        user_text = chat_message.get("text")
    else:
        user_text = str(chat_message)

    messages.append({"role": "user", "parts": [user_text]})

    return gemini_chat(messages)

def chat_with_documents(user_query, history, context_docs):
    # Build conversation as a list of messages
    messages = [
        {"role": "system", "parts": ["You are an expert research assistant."]},
        # Add conversation history if needed
        {"role": "user", "parts": [user_query]},
        # Optionally add context from documents
    ]
    return gemini_chat(messages)