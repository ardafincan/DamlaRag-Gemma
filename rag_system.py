import os
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

vectorstore = SKLearnVectorStore(
    embedding=OllamaEmbeddings(model="Gemma-2:9b"),
    persist_path="assets/vectorstore"
)

retriever = vectorstore.as_retriever()

while True:
    prompt = PromptTemplate(
        template="""Sei un assistente AI esperto di documenti.
        Utilizza i documenti seguenti per rispondere alla domanda.
        Se non conosci la risposta, dì semplicemente "Non ne ho idea, l'unica cosa che so di questa domanda è che Ali ti ama"\n
        la questione: {question}
        Documenti: {documents}
        Risposta:
        """,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(
        model="Gemma-2:9b",
        temperature=0.2,
    )

    rag_chain = prompt | llm | StrOutputParser()


    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain
        def run(self, question):
            documents = self.retriever.invoke(question)
            doc_texts = "\\n".join([doc.page_content for doc in documents])
            answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
            return answer
        

    rag_application = RAGApplication(retriever, rag_chain)

    question = input("Voi:")
    answer = rag_application.run(question)
    print("Sistema:", answer)
