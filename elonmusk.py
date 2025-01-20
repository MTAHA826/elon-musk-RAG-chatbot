from langchain.chat_models import ChatOpenAI
import openai
from streamlit_mic_recorder import mic_recorder,speech_to_text
import langchain_pinecone
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from bs4 import SoupStrainer
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY=os.environ['PINECONE_API_KEY']

# Initialize embedding and Qdrant

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)
    # of the embeddings you want returned.
    # dimensions=1024
# Qdrant setup
# api_key = os.getenv('qdrant_api_key')
# url = 'https://1328bf7c-9693-4c14-a04c-f342030f3b52.us-east4-0.gcp.cloud.qdrant.io:6333'
# doc_store = QdrantVectorStore.from_existing_collection(
#     embedding=embed,
#     url=url,
#     api_key=api_key,
#     prefer_grpc=True,
#     collection_name="Elon Muske"
# )
pineconedb=PineconeVectorStore.from_existing_index(index_name='project1', embedding=embeddings)
llm = ChatOpenAI(
                model_name='gpt-4o-mini',
                openai_api_key=openai_api_key,
                temperature=0)
# Initialize Google LLM
google_api = st.secrets['google_api_key']
# llm = GoogleGenerativeAI(model="gemini-1.5-flash-002", google_api_key=google_api)

# Setup retriever and chain

num_chunks = 2
retriever = pineconedb.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
# retriever =doc_store.as_retriever(search_type="mmr", search_kwargs={"k": num_chunks})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_str = """
You are a highly knowledgeable and conversational chatbot specializing in providing accurate and insightful information about Elon Musk.
Answer all questions as if you are an expert on his life, career, companies, and achievements. You are trained to answer question related to the provied
context if a user ask question which is different from the context you have to say  :" I am train to answer questions related to Elon Musk only."
Context: {context}
Question: {question}
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)

# Chain setup
query_fetcher = itemgetter("question")
setup = {"question": query_fetcher, "context": query_fetcher | retriever| format_docs }
_chain = setup | _prompt | llm | StrOutputParser()

# Streamlit UI
# Streamlit UI
st.title("Ask Anything About Elon Musk")

# Chat container to display conversation
chat_container = st.container()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def send_input():
    st.session_state.send_input=True
# Input field for queries
with st.container():
    query = st.text_input("Please enter a query", key="query", on_change=send_input)
    send_button = st.button("Send", key="send_btn")  # Single send button
#     audio=mic_recorder(start_prompt="**",stop_prompt="##",key="recorder")
# if audio:
#     st.audio(audio["bytes"])
    text=speech_to_text(language="en",use_container_width=True,just_once=True,key="STT")
if text:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response=_chain.invoke('question':text)
    st.write(response.content)
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))
# Chat logic
if send_button or send_input and query:
    with st.spinner("Processing... Please wait!"):  # Spinner starts here
        response = _chain.invoke({'question': query})  # Generate response
    # Update session state with user query and AI response
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("ai", response))

with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)
