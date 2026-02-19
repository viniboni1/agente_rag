from docs_methods import ingest_pdf_documents, get_chunks
from vector_db_store import init_db
from factory_chains import create_chain
from langchain_core.messages import HumanMessage, AIMessage

from llm_provider import get_llm

def main():

    print("--- 🏛️ Iniciando Agente Filosófico ---")
    
    chat_history = []

    llm = get_llm("groq")

    document = ingest_pdf_documents(r"C:\Users\bonif\OneDrive\Documentos\agente_filosofico\livros")

    chunk = get_chunks (document)

    db = init_db(chunk)

    chain = create_chain(db, llm)

    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
        
        try:
            response = chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            print(f"\nAgente: {response}")
            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            
        except Exception as e:
            print(f"Erro: {e}")


if __name__ == "__main__":
    main()
