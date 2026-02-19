import os

from langchain_community.vectorstores import Chroma

from embedding_provider import get_embedding


def init_db(chunks= None, persist_directory= "./chroma_db"):
    model_embedding = get_embedding()
    
    if chunks:
        db = Chroma.from_documents(documents=chunks, embedding=model_embedding, persist_directory=persist_directory)
        print ("Chroma DB atualizado.")

        return db
    else:
        if os.path.exists(persist_directory):
            print ("Carregando Chroma DB")

            db = Chroma(persist_directory=persist_directory, embedding_function=model_embedding)

            return db
        
        else:
            print("Nenhum banco encontrado e nenhum documento fornecido.")

            return None

