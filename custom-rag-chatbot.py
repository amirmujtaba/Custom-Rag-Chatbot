import asyncio

# Patch: Streamlit runs in ScriptRunner thread without a loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader, CSVLoader, JSONLoader,UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
import tempfile

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1rem; /* Adjust this value as needed */
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .st-emotion-cache-1mph9ef {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Custom AI Assistant")
# st.subheader("Welcome to Custom AI Assistant. How can I help you?")

st.sidebar.info("We will not save your API key or data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Add llm to the sidebar
st.session_state.llm = st.sidebar.selectbox("Choose LLM", ("OpenAI", "Gemini"), index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")

if(st.session_state.llm == "OpenAI"):
    os.environ["API_KEY"] = st.sidebar.text_input("Enter your OpenAI API Key", value="", max_chars=None, key=None, type="password", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", icon=None, width="stretch")
    if(os.environ["API_KEY"]==""):
        st.error("Please enter your OpenAI API Key")
        st.link_button("Get Your Key", "https://platform.openai.com/api-keys")
        st.stop()
    st.session_state.model = st.sidebar.selectbox("Choose Model", ("gpt-4", "gpt-3.5-turbo"), index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
    llm=ChatOpenAI(api_key=os.environ["API_KEY"],model=st.session_state.model)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ["API_KEY"])

elif(st.session_state.llm == "Gemini"):
    os.environ["API_KEY"] = st.sidebar.text_input("Enter your Gemini API Key", value="", max_chars=None, key=None, type="password", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", icon=None, width="stretch")
    if(os.environ["API_KEY"]==""):
        st.error("Please enter your Gemini API Key")
        st.link_button("Get Your Key", "https://aistudio.google.com/app/api-keys")
        st.stop()
    st.session_state.model = st.sidebar.selectbox("Choose Model", ("gemini-2.5-flash", "gemini-1.5-pro"), index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")
    llm = GoogleGenerativeAI(api_key=os.environ["API_KEY"],model=st.session_state.model)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=os.environ["API_KEY"])

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
vectorstoreIndexCreator = VectorstoreIndexCreator(embedding=embeddings,text_splitter=text_splitter)

# Add system instructions
# system_instructions = st.sidebar.text_input("System Instructions", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", icon=None, width="stretch")

# Add source type selection
source_type = st.sidebar.selectbox("Data source type", ("Both", "Links", "Files"), index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", accept_new_options=False, width="stretch")

all_docs = []

if source_type == "Both" or source_type == "Links":
    if source_type == "Links":
        file_extension=None
        attachments = None
        all_docs = []
        st.session_state.messages=[]

    data_source_links = st.sidebar.text_input("Enter your data source link", value="", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible", icon=None, width="stretch")
    if data_source_links is None or data_source_links == "":
        st.error("Please enter a link or multiple links separated by comma")

    for data_source_link in data_source_links.split(","):
        data_source_link = data_source_link.strip()
        if data_source_link == "":
            continue
        try:
            if st.session_state.messages is None or st.session_state.messages==[]:
                with st.spinner("Please wait...", show_time=False, width="content"):
                    loader = WebBaseLoader(data_source_link)
                    doc = loader.load()
                    all_docs.extend(doc)
            else:
                loader = WebBaseLoader(data_source_link)
                doc = loader.load()
                all_docs.extend(doc)
        except Exception as e:
            st.write(f"Error loading {data_source_link} : {e}")
            st.stop()

if source_type == "Both" or source_type == "Files":
    if source_type == "Files":
        data_source_link = None
        file_extension=None
        attachments = None
        all_docs = []
        st.session_state.messages=[]

    attachments = st.sidebar.file_uploader("Add your data source", type=["txt", "pdf", "docx", "xlsx", "pptx", "csv"], accept_multiple_files=True, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible", width="stretch")
    if attachments is None or attachments == []:
        st.error("Please upload a file or multiple files")
    elif attachments!=[] and len(attachments) > 0:
        data_source_link = None 
        for attachment in attachments:
            file_extension = os.path.splitext(attachment.name)[1]
            with tempfile.NamedTemporaryFile(delete=True, suffix=file_extension) as tmp_file:
                tmp_file.write(attachment.getbuffer())  # Save uploaded content
                temp_file_path = tmp_file.name             # Path to temp file

                try:
                    match file_extension:
                        case ".pdf":
                            loader = PyPDFLoader(temp_file_path)
                        case ".txt":
                            loader = TextLoader(temp_file_path)
                        case ".csv":
                            loader = CSVLoader(temp_file_path)
                        case ".json":
                            loader = JSONLoader(temp_file_path)
                        case _:  # Default case (wildcard)
                            loader = UnstructuredFileLoader(temp_file_path)

                    if st.session_state.messages is None or st.session_state.messages==[]:
                        with st.spinner("Please wait...", show_time=False, width="content"):
                            doc = loader.load()
                            all_docs.extend(doc)
                    else:
                        doc = loader.load()
                        all_docs.extend(doc)
                except Exception as e:
                    st.write(f"Error loading file {temp_file_path} : {e}")
                    st.stop()

if (source_type == "Links" and data_source_links is not None and data_source_links!="") or (source_type == "Files" and attachments is not None and attachments!=[]) or (source_type == "Both" and (data_source_links is not None and data_source_links != "") and (attachments is not None and attachments != [])):
    if "clicked" not in st.session_state:
        st.session_state.clicked = False

    if st.sidebar.button("Submit", type="primary", use_container_width=True):
        st.session_state.clicked = True
        st.session_state.messages = []

    if st.session_state.clicked:    
        try:
            if st.session_state.messages is None or st.session_state.messages==[]:
                with st.spinner("Please wait...", show_time=False, width="content"):
                    indexes = vectorstoreIndexCreator.from_documents(documents=all_docs)
            else:
                indexes = vectorstoreIndexCreator.from_documents(documents=all_docs)
        except Exception as e:
            st.write(f"Error processing data : {e}")
            st.stop()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"][0]["text"])

        system_prompt = """You are a helpful assistant."""    

        if prompt := st.chat_input("Type your question here..."):
            st.session_state.messages.append({"role": "user", 'parts': [{'text': prompt}]})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("ai"):
                try:
                    with st.spinner("preparing answer for you...", show_time=False, width="content"):
                        response = indexes.query(prompt, llm=llm)
                        st.write(response)
                except Exception as e:
                    st.write(f"Error processing query: {e}")
                    st.stop()
            st.session_state.messages.append({"role": "ai", 'parts': [{'text': response}]})
