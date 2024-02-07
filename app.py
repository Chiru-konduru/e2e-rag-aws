import json
import os
import sys
import boto3
import streamlit as st

#We'll use Titan embeddings model to generate embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

#Data Ingestion libraries
import numpy as np 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

#Vector Store libraries
from langchain.vectorstores import faiss

##LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Step 1: creating Bedrock clients
bedrock = boto3.client(service_name = "bedrock-runtime")
bedrock_embedding = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock)

##Step 2: Data Ingestion 
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, 
                                                   chunk_overlap = 1000)
    docs = text_splitter.split_documents(documents)
    return docs
    
## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = faiss.FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama_llm():
    llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client = bedrock,
                  model_kwargs={"max_gen_len":512, "temperature": 0.7, "top_p": 0.9})
    return llm

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


PROMPT = PromptTemplate(template=prompt_template, input_variables =["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type ="similarity", search_kwargs={"k":3}
        ),
        return_source_documents = True,
        chain_type_kwargs={"prompt": PROMPT}
         )

    answer = qa({"query":query})
    return answer['result']

#creating Streamlit application

def main():
    st.set_page_config("ConverseDoc")
    st.header("Explore PDFs and Ask Questions")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update or Create vector Store:")

        if st.button("Vectors update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = faiss.FAISS.load_local("faiss_index", bedrock_embedding)
            llm = get_llama_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()
