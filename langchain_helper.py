
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.document_loaders.json_loader import JSONLoader
from langchain.document_loaders.csv_loader import CSVLoader
import jq
import google.generativeai as genai
import warnings
warnings.filterwarnings("ignore")
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Configure API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_database():
    # Define the jq schema to include more detailed information
    jq_schema = '''.[] | {
    "page_content": "displayName: \( .displayName )\nModel: \( .model )\nBrand: \( .brand )\nPrice: \( .displayPrice )\nElectric: \( .isElectric | tostring )\nRating: \( .avgRating )\nVariants: \( .variants | map(.name + \" - \" + .displayPrice) | join(\"\\n\") )\nGallery: \( .gallery | join(\"\\n\") )",
    "metadata": {
        "image": .image,
        "variants": .variants,
        "gallery": .gallery,
        "displayName": .displayName,
        "model": .model,
        "brand": .brand,
        "displayPrice": .displayPrice,
        "isElectric": .isElectric,
        "avgRating": .avgRating
            }
        }
    '''

    loader = JSONLoader(
        file_path='allData.json',
        jq_schema=jq_schema,
        text_content=False
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")

    # Create FAISS vector store with optimized parameters
    vectordb = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        # faiss_options={"index_flat": {"nlist": 100}}  # Adjust based on your dataset size
    )

    # Save the vector database
    vectordb.save_local("tata_trucks_vectorstore")

    return [vectordb, True]

def get_QNA_chain():
    # Load the vector database
    vectordb = FAISS.load_local(
        "tata_trucks_vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Create a retriever
    retriever = vectordb.as_retriever()

    # Define a simplified and improved prompt template
    # prompt_template = """You are a knowledgeable assistant with access to detailed information about trucks. Your goal is to answer questions as accurately and completely as possible using the provided context.
    #
    # Answer the following question using the context below:
    #
    # QUESTION: {question}
    #
    # CONTEXT: {context}
    #
    # Provide a detailed and specific response based on the context. If the answer is not found in the context, reply exactly with: "I'm only for truck-related query. Rest I'm working upon it."
    # """
    # prompt_template = """
    # You are a knowledgeable assistant with access to detailed information about trucks. Your goal is to provide highly accurate and structured answers using the provided context. If the question is general and not truck-related, please indicate that you are only for truck-related queries.
    #
    # Answer the following question using the context below:
    #
    # ### QUESTION:
    # {question}
    #
    # ### CONTEXT:
    # {context}
    #
    # ### RESPONSE FORMAT:
    # Please structure your response in a clear and detailed manner using the following guidelines:
    #
    # 1. **Heading Level 1: Main Answer**
    #    Provide the most relevant answer to the question using the context. Make sure to highlight important keywords (like truck models, price, brand, etc.) in **bold** or *italic*.
    #
    # 2. **Heading Level 2: Additional Details**
    #    Include any extra information that is relevant but not covered directly in the main answer.
    #
    # 3. **Heading Level 3: Related Images**
    #    If any images are available for the truck or product, include them with proper markdown format and a description.
    #
    # 4. **Heading Level 3: Source**
    #    Provide the source of the data if applicable (e.g., the truck model, price, specifications). Ensure the data is accurate and cited appropriately.
    #
    # **If the answer is not found in the context, reply exactly with:**
    # "I'm only for truck-related queries. Rest I'm working upon it."
    #
    # """
    prompt_template = """
    You are a knowledgeable assistant capable of answering both truck-related queries and general knowledge queries. Your goal is to answer as accurately as possible based on the context provided or by using your own general knowledge for non-truck-related queries.

    Answer the following question using the context below, or use your general knowledge if the question is unrelated to trucks:

    ### QUESTION:
    {question}

    ### CONTEXT:
    {context}

    ### RESPONSE FORMAT:
    Please structure your response in a clear and detailed manner using the following guidelines:

    1. **For Truck-Related Queries:**
        - **Heading Level 1: Main Answer**  
          Provide the most relevant answer to the truck-related question using the context. Make sure to highlight important keywords (like truck models, price, brand, etc.) in **bold** or *italic*.

        - **Heading Level 2: Additional Details**  
          Include any extra information that is relevant but not covered directly in the main answer.

        - **Heading Level 3: Related Images**  
          If any images are available for the truck or product, include them with proper markdown format and a description.

        - **Heading Level 3: Source**  
          Provide the source of the data if applicable (e.g., the truck model, price, specifications). Ensure the data is accurate and cited appropriately.

    2. **For General Knowledge Queries (e.g., personal question, general knowledge questions and apart from the trucks related query, etc.):**
        - Provide a well-structured answer to the general knowledge query using your own capability. You do not need to rely on the truck context for these types of questions.

        **Example Response Structure:**

        - **Heading Level 1: General Knowledge Answer**  
          Provide a concise answer to the query. Use **bold** to highlight key terms and ideas.

        - **Heading Level 2: Further Explanation (Optional)**  
          If needed, provide additional context or explanations to make the answer more comprehensive.

        **For non-truck-related queries, do not use the truck context. If the answer is not found in the context, reply exactly with:**  
        "I'm only for truck-related queries. For general knowledge queries, I will provide my own insights."
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain

# Example usage
if __name__ == "__main__":
    # Create the vector database
    vectordb = create_vector_database()

    # Get the Q&A chain
    qa_chain = get_QNA_chain()

    # Test the chain with a sample question
    question = "What are the variants of the Tata 1815 LPT?"
    answer = qa_chain.run(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

# from langchain.chat_models import ChatOpenAI
# import os
# from dotenv import load_dotenv
# import pandas as pd
# from langchain.document_loaders.json_loader import JSONLoader
# from langchain.document_loaders.csv_loader import CSVLoader
# import jq
# import google.generativeai as genai
# import warnings
# warnings.filterwarnings("ignore")
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
#
# load_dotenv()
#
# # with the OpenAI API Key.
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#
# load_dotenv()  # take environment variables from .env (especially openai api key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#
# llm = ChatOpenAI(
#     openai_api_key=os.getenv("OPENAI_API_KEY"),  # or set via environment variable
#     model_name="gpt-3.5-turbo",
#     temperature=0
# )
# # Use a pre-trained OpenAI Embeddings. model
# huggingFace = HuggingFaceEmbeddings()
#
# # File path for the faiss.
# vector_file_path = "faiss_vectorstore"
#
# def create_vector_database():
#     # loader = CSVLoader(file_path="trucks_data.csv", source_column="prompt")
#     # data = loader.load()
#     loader = JSONLoader(
#         file_path='tata_data.json',
#         jq_schema='''
#             .[] | {
#                 page_content: "displayName:  \(.displayName), Model: \(.model), Brand: \(.brand), Price: \(.displayPrice), Electric: \(.isElectric), Rating: \(.avgRating)",
#                 metadata: {
#                     image: .image,
#                     variants: .variants,
#                     gallery: .gallery,
#                     displayName: .displayName,
#                     model: .model,
#                     brand: .brand,
#                     displayPrice: .displayPrice,
#                     isElectric: .isElectric,
#                     avgRating: .avgRating
#                 }
#             }
#         ''',
#         text_content=False
#     )
#
#
#     documents = loader.load()
#     # print(f"✅ Loaded {len(documents)} documents.")
#
#     # Create FAISS vector store from documents
#     vectordb = FAISS.from_documents(documents=documents, embedding=huggingFace)
#
#     # Optionally: Save the FAISS index to a local file for later use
#     vectordb.save_local(vector_file_path)
#
#     return True
#
# def get_QNA_chain():
#     # Load the vector database from the local folder..
#     vectordb = FAISS.load_local(
#         vector_file_path, huggingFace, allow_dangerous_deserialization=True
#     )
#
#     # Create a retriever for querying the vector database..
#     retriever = vectordb.as_retriever()
#
#     prompt_template = """You are a helpful assistant. Use the context provided below to answer the question as accurately and completely as possible. Your response must be strictly based on the information from the context — do not use outside knowledge or make assumptions.
#
#     When answering:
#     - Include all relevant details found in the context.
#     - Use the exact phrases and wording from the context where possible.
#     - Do not change or summarize unless needed for clarity.
#     - If the answer cannot be found in the context, reply exactly with: "I'm working upon it."
#
#     CONTEXT:
#     {context}
#
#     QUESTION:
#     {question}
#      """
#
#     PROMPT = PromptTemplate(
#         template=prompt_template, input_variables=["context", "question"]
#     )
#
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         input_key="query",
#         chain_type_kwargs={"prompt": PROMPT},
#     )
#     return chain
#
# # if __name__ == "__main__":
# #     create_vector_database()
# #     chain = get_QNA_chain()
# #     print(chain("is 91trucks provide any financial services ??"))
