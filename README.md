# e2e-rag-aws
End to End Advanced RAG App using Bedrock and LangChain
# ConverseDoc

ConverseDoc is an advanced web application crafted to redefine user interaction with PDF documents through the power of artificial intelligence. By enabling users to upload a PDF, the application processes its content and facilitates an interactive Q&A based on the document's content. It is an ideal tool for extracting detailed insights from reports, research papers, or any PDF document with efficiency and precision.

## Key Features

- **Single PDF Upload**: Users can upload one PDF document at a time for focused analysis and query processing.
- **Interactive Q&A**: Allows users to ask questions about the uploaded PDF and receive precise, contextually relevant answers.
- **Advanced AI Integration**: Incorporates Amazon Bedrock's Llama model for state-of-the-art text extraction, processing, and question answering capabilities.

## Technology Stack

- **Python**: The foundation of the application, chosen for its extensive library support and versatility.
- **Streamlit**: Powers the user interface, offering a seamless web experience without complex web development requirements.
- **Boto3**: Enables interaction with AWS services, essential for leveraging cloud-based functionalities.
- **PyPDF2**: A library dedicated to PDF file operations, such as reading text from documents.
- **Numpy**: Utilized for complex mathematical operations and managing large, multi-dimensional arrays.
- **LangChain & LangChain Community**: Supports embedding generation and document processing, vital for text analysis and comprehension.
- **FAISS**: Chosen for its unparalleled efficiency in similarity search and clustering of dense vectors, crucial for fast and scalable query processing.

## Why FAISS?

FAISS is selected for its exceptional performance in managing similarity searches and vector clustering, which is critical for navigating through the dense vector space derived from the content of PDF documents. It is noted for:

- Its scalability and performance, capable of handling large datasets efficiently.
- Optimized memory usage, facilitating effective data management.
- Compatibility with the embedding and language model technologies used in ConverseDoc, ensuring a smooth and integrated system architecture.
- Library vs. Service: FAISS is a library rather than a fully managed service. It's designed to be integrated into applications and systems directly, giving developers control over the deployment environment, whether on local machines, on-premises servers, or cloud instances.
- Quantization and Indexing: FAISS emphasizes advanced quantization techniques and offers a wide range of indexing strategies for efficient storage and fast search. This focus allows for handling very large datasets with reduced memory footprints.
- GPU/CPU Optimization: Offers extensive support for both CPU and GPU, enabling high performance for both batch processing and real-time search tasks. Users can choose between or combine both computational resources based on their specific requirements.

## Embedding and Language Model

- **Titan Text Embeddings & LangChain**: Employed for generating robust text embeddings, enhancing the application's content processing capabilities.
- **Llama-13B Chat from Amazon Bedrock Services**: The core language model used for natural language understanding and question answering. Chosen for its cutting-edge performance in generating accurate and relevant responses to user queries.

