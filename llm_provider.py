import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider = "local"):
    if provider =="local":
        return ChatOpenAI(
            model="meta-llama-3.1-8b-instruct",
            temperature= 0.7,
            api_key=("LOCAL_API_KEY"),
            base_url="http://127.0.0.1:1234/v1"
        )
    elif provider == "groq":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            )