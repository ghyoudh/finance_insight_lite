import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


def get_rag_chain(vector_db):
    # Initialize the Brain (LLM)
    # Use an active Groq model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",  # Updated model
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # Define the System Prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and professional.

    IMPORTANT FORMATTING RULES:
    - Always put a SPACE between numbers and currency words (e.g., "104.9 ريال" not "104.9ريال")
    - Always include the page number(s) where you found the information at the end of your answer
    - Format: "Answer text here. (Source: Page X)" or "(Source: Pages X, Y)"

    {context}

    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)

    # Create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Helper function to format documents
    def format_docs(docs):
        # Format documents with page numbers
        formatted = []
        for doc in docs:
            page_num = doc.metadata.get('page', 'Unknown')
            content = doc.page_content
            formatted.append(f"[Page {page_num}] {content}")
        return "\n\n".join(formatted)

    # Create chain that returns both answer and source documents
    rag_chain = RunnableParallel(
        {"answer": (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            ),
        "source_documents": retriever,}
    )

    return rag_chain