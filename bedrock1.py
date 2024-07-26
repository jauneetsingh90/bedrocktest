import os
import boto3
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.vectorstores.astradb import AstraDB
from langchain.prompts.chat import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Load the environment variables from either Streamlit Secrets or .env file
LOCAL_SECRETS = False

# If running locally, then use .env file, or use local Streamlit Secrets
if LOCAL_SECRETS:
    ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = os.environ["ASTRA_VECTOR_ENDPOINT"]
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]

# If running in Streamlit, then use Streamlit Secrets
else:
    ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    ASTRA_VECTOR_ENDPOINT = st.secrets["ASTRA_VECTOR_ENDPOINT"]
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]

os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

ASTRA_DB_KEYSPACE = "default_keyspace"
ASTRA_DB_COLLECTION = "livspace"

os.environ["LANGCHAIN_PROJECT"] = "blueillusion"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("Started")

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Define the number of docs to retrieve from the vectorstore and memory
top_k_vectorstore = 3
top_k_memory = 3

#############
### Login ###
#############
# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.caption('Using a unique name will keep your content separate from other users.')
            st.text_input('Username', key='username')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if len(st.session_state['username']) > 5:
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 Username must be 6 or more characters')
    return False

def logout():
    del st.session_state.password_correct       
    del st.session_state.user
    del st.session_state.messages
    load_chat_history.clear()
    load_memory.clear()
    load_retriever.clear()

# Cache boto3 session for future runs
@st.cache_resource(show_spinner='Getting the Boto Session...')
def load_boto_client():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    return boto3.client("bedrock-runtime")

@st.cache_resource(show_spinner='Getting the Embedding Model...')
def load_embedding(embedding_model_id):
    return BedrockEmbeddings(credentials_profile_name="357576205424_FIELDOPS_FOPS-DEL", region_name="us-east-1")

# Cache Vector Store for future runs
@st.cache_resource(show_spinner='Getting the Vector Store from Astra DB...')
def load_vectorstore():
    return AstraDB(
        embedding=embedding,
        namespace=ASTRA_DB_KEYSPACE,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
    )

# Cache Retriever for future runs
@st.cache_resource(show_spinner='Getting the retriever...')
def load_retriever():
    return vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

# Cache Chat History for future runs
@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_chat_history(username):
    return AstraDBChatMessageHistory(
        session_id=username,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_VECTOR_ENDPOINT,
        namespace=ASTRA_DB_KEYSPACE,
    )

@st.cache_resource(show_spinner='Getting the Message History from Astra DB...')
def load_memory():
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_memory,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

@st.cache_resource(show_spinner='Getting the Chat Model...')
def load_model(model_id):
    return ChatBedrock(
        credentials_profile_name="357576205424_FIELDOPS_FOPS-DEL",
        region_name="us-east-1",
        model_id=model_id,
        streaming=True,
        model_kwargs={"temperature": 0.2},
    )

@st.cache_data()
def load_prompt():
    template = """You're a helpful fashion assistant tasked to help users shopping for clothes, shoes and accessories. 
You like to help a user find the perfect outfit for a special occasion.
You should also suggest other items to complete the outfit.
Prompt the user with clarifying questions so that you know at least for what occasion they are shopping and their age group.
Do not include any information other than what is provided in the context below.
Include an image of the product taken from the image attribute in the metadata.
Include the price of the product if found in the context.
Include a link to buy each item you recommend if found in the context. Here is a sample buy link:
Buy Now: [Product Name](https://www.blueillusion.com/product-name)
If you don't know the answer, just say 'I do not know the answer'.
If the user has not asked a question related to clothing and Blue Illusion products, you can respond with 'I do not have the products you asked for' and suggest a similar alternative from BlueIllusion context.

Use the following context to answer the question:
{context}

Use the previous chat history to answer the question:
{chat_history}

Question:
{question}

Answer in English"""
    return ChatPromptTemplate.from_messages([("system", template)])

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hi, I'm your personal shopping assistant!")]

st.set_page_config(initial_sidebar_state="collapsed")

if not check_password():
    st.stop()

username = st.session_state.user

with st.sidebar:
    st.image('./public/logo.svg')
    st.text('')
    st.button(f"Logout '{username}'", on_click=logout)

    bedrock_runtime = load_boto_client()
    embedding_model_id = st.selectbox('Embedding Model', [
        'amazon.titan-text-express-v1',
        'amazon.titan-embed-text-v2:0',
    ])
    embedding = load_embedding(embedding_model_id)
    vectorstore = load_vectorstore()
    retriever = load_retriever()
    chat_history = load_chat_history(username)
    memory = load_memory()
    prompt = load_prompt()
    model_id = st.selectbox('LLM Model', [
        'meta.llama2-70b-chat-v1',
        'meta.llama2-13b-chat-v1',
        'amazon.titan-text-express-v1',
        'meta.llama3-1-70b-instruct-v1:0',
        'anthropic.claude-3-sonnet-20240229-v1:0',
        'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'anthropic.claude-3-haiku-20240307-v1:0',
    ])
    model = load_model(model_id)
    with st.form('delete_memory'):
        st.caption('Delete the history in the conversational memory.')
        submitted = st.form_submit_button('Delete chat history')
        if submitted:
            with st.spinner('Deleting chat history...'):
                memory.clear()

# PDF Loader Section
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location
        with open(os.path.join("/tmp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_loader = PyPDFLoader(os.path.join("/tmp", uploaded_file.name))
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs_from_pdf = pdf_loader.load_and_split(text_splitter=splitter)
        
        # Embed documents to vectorstore
        with st.spinner(f"Embedding {uploaded_file.name}..."):
            inserted_ids_from_pdf = vectorstore.add_documents(docs_from_pdf)
            st.success(f"Inserted {len(inserted_ids_from_pdf)} documents from {uploaded_file.name}.")

st.markdown("<style> img {width: 200px;} </style>", unsafe_allow_html=True)

for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

if question := st.chat_input("How can I help you?"):
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        history = memory.load_memory_variables({})
        inputs = RunnableMap({
            'context': lambda x: retriever.get_relevant_documents(x['question']),
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        })
        chain = inputs | prompt | model
        response = chain.invoke({'question': question, 'chat_history': history}, config={'callbacks': [StreamHandler(response_placeholder)], "tags": [username]})
        content = response.content
        response_placeholder.markdown(content)
        memory.save_context({'question': question}, {'answer': content})
        st.session_state.messages.append(AIMessage(content=content))