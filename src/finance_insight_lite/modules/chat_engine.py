import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(vector_db):
    # 1. Initialize the Brain (LLM)
    # Use an active Groq model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",  # Updated model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # 2. Define the System Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and professional.

    {context}

    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)
    
    # 3. Create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 4. Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Create the chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain