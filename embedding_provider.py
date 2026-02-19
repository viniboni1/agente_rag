from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding(embedding_provider = "hugging_face"):
    if embedding_provider == "hugging_face":
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'})
    
    return embedding