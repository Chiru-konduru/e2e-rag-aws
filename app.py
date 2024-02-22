import json
import os
import sys
import boto3
import streamlit as st
import PyPDF2
from io import BytesIO

import shutil
import tempfile

#We'll use Titan embeddings model to generate embeddings

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

#Data Ingestion libraries
import numpy as np 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

#Vector Store libraries
from langchain_community.vectorstores import faiss

##LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.llms import OpenAI

# Global variable to hold the vector store once created
vectorstore_faiss = None

# Step 1: creating/initialize Bedrock clients
# Access the AWS credentials from Streamlit's secrets
aws_access_key_id = st.secrets["aws"]["aws_access_key_id"]
aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"]
aws_default_region = st.secrets["aws"]["aws_default_region"]

# Initialize the Bedrock client with the AWS credentials
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_default_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
#Bedrock client for testing locally
#bedrock = boto3.client(service_name = "bedrock-runtime")

bedrock_embedding = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v1", client = bedrock)


# Define the directory to save uploaded files -local
#path_to_save = 'C:/Users/kchir/OneDrive/Desktop/AWS Projects/e2e-RAG/data'
path_to_save = tempfile.mkdtemp()

os.makedirs(path_to_save, exist_ok=True)  # Create the directory if it does not exist

def clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Remove all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            #if file_path == r"C:/Users/kchir/OneDrive/Desktop/AWS Projects/e2e-RAG/faiss_index":
            if filename == "faiss_index":
                continue
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


# Function to save uploaded files to the local directory
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(path_to_save, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False
    

##Step 2: Data Ingestion 
def data_ingestion():
    loader = PyPDFDirectoryLoader(path_to_save)
    documents = loader.load()

    # Use RecursiveCharacterTextSplitter to split the loaded documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs
    
## Vector Embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = faiss.FAISS.from_documents(
        docs,
        bedrock_embedding
    )
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

# Loading LLM model from Bedrock
def get_llama_llm():
    llm = Bedrock(model_id="meta.llama2-13b-chat-v1", client = bedrock,
                  model_kwargs={"max_gen_len":512, "temperature": 0.7, "top_p": 0.95})
    return llm

# # Loading Gemma model from local server using LM Studio
# def get_gemma_llm():
#     # Initialize the OpenAI client to point to your local server
#     llm_config = OpenAI(base_url="http://localhost:5000/v1", 
#                         api_key="not-needed",
#                         temperature = 0.5,
#                         )
#     llm = llm_config
#     return llm
    


prompt_template = """
 Human: Use the following pieces of context to provide a concise answer to the question at the end. 
If you don't know the answer, just say that you don't know. 
Don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


PROMPT = PromptTemplate(template=prompt_template, input_variables =["context", "question"])


def get_response_llm(llm, vectorstore_faiss, query):

    MAX_TOKENS = 4096

    if len(query) > MAX_TOKENS:
        query = query[:MAX_TOKENS]

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



# Add a function to handle responses and store them
def store_response(key, response):
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    st.session_state.responses[key] = response

# Define a function to clear the session state
def stop_interaction():
    st.session_state['user_has_stopped'] = True
    st.warning('Interaction has been stopped. Refresh the page to start again.')


#creating Streamlit application
def main():
    st.set_page_config("ConverseDoc")
    #st.header("ConverseDoc: Explore PDFs and Ask Questions!")
    st.markdown("<h1 style='text-align: center; color: white;'>ConverseDoc: Explore PDFs and Ask Questions!</h1>", unsafe_allow_html=True)

    # Use session_state to store the vector store and the initial response
    if 'vectorstore_faiss' not in st.session_state:
        st.session_state.vectorstore_faiss = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    if 'user_has_stopped' not in st.session_state:
        st.session_state.user_has_stopped = False

    # Stop interaction if the user has clicked the button
    if st.session_state.user_has_stopped:
        st.warning('You have chosen to stop the interaction. Please refresh the page to reset.')
        return  # Stop further execution of the function
    
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

    if uploaded_files and st.session_state.vectorstore_faiss is None:
        #Clear the directory before saving new files to ensure only uploaded files are processed
        clear_directory(path_to_save)

        for uploaded_file in uploaded_files:
            save_successful = save_uploaded_file(uploaded_file)
            if save_successful:
                st.success(f"File  {uploaded_file.name} saved successfully.")
            else:
                st.error(f"Failed to save file {uploaded_file.name}.")

            # Process saved files using PyPDFDirectoryLoader
        with st.spinner("Processing and vectorizing uploaded files..."):
            docs = data_ingestion()  # Adjust this function to use PyPDFDirectoryLoader as needed
            st.session_state.vectorstore_faiss = get_vector_store(docs)

    # Display previous queries and responses
    for i, (query, response) in enumerate(zip(st.session_state.queries, st.session_state.responses)):
    #for i, (query, response) in enumerate(st.session_state.responses):
        st.markdown(f"**You:** 👤 {query}")
        st.markdown(f"**AI:** 🤖 {response}")


    # Input for user question and button to get answer  
    input_placeholder = st.empty()    
    new_question  = input_placeholder.text_input("Ask a Question from the uploaded PDF Files", 
                                                  key="new_query")
    st.markdown("Note: Please type in your questions in the text box above and click the 'Get Answer' button to get a response.")
    if st.button("Get Answer"):
        if new_question:
            with st.spinner("Fetching answer..."):
                if st.session_state.vectorstore_faiss is None:
                    #Load the vector store; it should be updated with new documents
                    st.session_state.vectorstore_faiss = faiss.FAISS.load_local("faiss_index", bedrock_embedding)
                
                #Ensure the LLM model is loaded
                if st.session_state.llm is None:
                    st.session_state.llm = get_llama_llm()

                #get the response from the LLM model
                if st.session_state.vectorstore_faiss is not None and st.session_state.llm is not None:
                    new_response  = get_response_llm(st.session_state.llm, st.session_state.vectorstore_faiss, new_question)
                    
                    # Append the new question and response to the session state
                    st.session_state.queries.append(new_question)
                    st.session_state.responses.append(new_response)
                    st.markdown(f"👤: {new_question}")
                    st.markdown(f" 🤖: {new_response}")
         
    # Stop interaction button
    if st.button('Stop Interaction', key="stop_interaction_button"):
        st.session_state.user_has_stopped = True
        st.warning('Interaction has been stopped. Refresh the page to start again.')

if __name__ == "__main__":
    main()
