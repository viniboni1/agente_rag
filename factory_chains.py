from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from vector_db_store import init_db

def create_chain (vector_db, llm):
    retriever = vector_db.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k": 3,
        }
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assitente útil. Quero que responda somente com base no contexto: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def format_docs (docs):
        if not docs:
            return "Nenhuma informação relevante encontrada."
        return "\n\n".join(doc.page_content for doc in docs)


    chain = ({"context": itemgetter("input") | retriever | format_docs, "chat_history": itemgetter("chat_history"), "input": itemgetter("input")} | prompt | llm | StrOutputParser())

    return chain