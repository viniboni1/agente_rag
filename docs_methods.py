from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def ingest_pdf_documents(directory_path:str):
    paste_path = Path(directory_path)
    
    all_documents = []
    file_path_list = []

    for file in paste_path.iterdir():
        if file.is_file() and file.suffix == ".pdf":
            file_path_list.append(str(file))



    for file_path in file_path_list:
        try:
            loader = PyPDFLoader(file_path)
            document = loader.load()
            all_documents.extend(document)
        
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")   
    
    return all_documents

def get_chunks(document):
    spliter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 200)
    chunks = spliter.split_documents(document)
    return chunks