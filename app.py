import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import arxiv
import os

@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="google/flan-t5-small", max_length=200, min_length=50, do_sample=False)

    all_chunks = []
    search = arxiv.Search(query="LLM fine-tuning", max_results=2)
    for result in search.results():
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(result.summary)
        all_chunks.extend(chunks)

    vectorstore = FAISS.from_texts(all_chunks, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return embedding_model, summarizer, retriever

embedding_model, summarizer, retriever = load_resources()

# --- Simple UI ---
st.title("Auto-RAG Research Assistant")
query = st.text_input("Ask a research question:")

if query:
    st.write("Searching...")
    print("User query:", query)

    try:
        retrieved_chunks = retriever.get_relevant_documents(query)
        print("Number of retrieved chunks:", len(retrieved_chunks))

        if retrieved_chunks:
            context = " ".join([doc.page_content for doc in retrieved_chunks])
            print("Context (first 200 chars):", context[:200], "...")
            prompted_input = f"Summarize the following in response to the question: '{query}'. {context}"
            summary_output = summarizer(prompted_input, max_length=250, min_length=50, do_sample=False)
            st.subheader("Answer:")
            st.write(summary_output[0]["summary_text"])
            print("Summary generated:", summary_output[0]["summary_text"])
        else:
            st.write("No relevant results found.")
            print("No relevant results found.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"Error during processing: {e}")


# --- Deployment to Streamlit Cloud ---
# 1.  Save your app.py file.
# 2.  Create a requirements.txt file with your dependencies:
#     streamlit
#     transformers
#     langchain-community
#     sentence-transformers
#     arxiv
#     faiss-cpu
# 3.  Go to share.streamlit.io and sign up or log in.
# 4.  Click on "New app" and follow the instructions to deploy.