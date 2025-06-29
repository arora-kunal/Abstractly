import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import torch
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io
import numpy as np
import plotly.express as px
import pandas as pd
import uuid
from io import BytesIO
import base64

# Custom CSS
st.markdown("""
<style>
body { font-family: 'Helvetica', sans-serif; }
.stApp { background-color: #f0f2f6; }
h1 { color: #1e3a8a; font-size: 2.5em; }
.stRadio > label { font-weight: bold; color: #1e3a8a; }
.stTextInput > label { color: #1e3a8a; }
.stButton > button { background-color: #1e3a8a; color: white; border-radius: 8px; padding: 0.5em 1em; }
.stButton > button:hover { background-color: #3b82f6; }
.block-container { padding: 2rem; }
.heatmap-container { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
.toc-container { background-color: #e6f3ff; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize Streamlit app
st.set_page_config(page_title="Abstractly: Smart Research Assistant", layout="wide")
st.title("Abstractly: Smart Assistant for Research Summarization")
st.markdown("Upload PDF (text or scanned) or TXT documents to get summaries, ask questions, or challenge yourself. Features include voice interaction, semantic heatmaps, and cross-document reasoning.")

# Initialize LLM and embeddings
@st.cache_resource
def load_llm():
    model_name = "tiiuae/falcon-rw-1b"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load Falcon model ({model_name}): {str(e)}")
        fallback_model = "sshleifer/tiny-gpt2"
        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            model = AutoModelForCausalLM.from_pretrained(fallback_model)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
            st.warning("Using fallback model: sshleifer/tiny-gpt2")
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as fallback_error:
            st.error(f"Fallback model also failed: {fallback_error}")
            st.stop()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor, model

llm = load_llm()
embeddings = load_embeddings()
whisper_processor, whisper_model = load_whisper()

# Session state for storing
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "memory" not in st.session_state:
    st.session_state.memory = {}
if "questions" not in st.session_state:
    st.session_state.questions = {}
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "toc" not in st.session_state:
    st.session_state.toc = {}
if "user_score" not in st.session_state:
    st.session_state.user_score = 0

# OCR
def extract_text_from_scanned_pdf(file_path):
    images = convert_from_path(file_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

# Get table of contents
def extract_toc(documents):
    toc = []
    for i, doc in enumerate(documents):
        lines = doc.page_content.split("\n")
        for line in lines:
            if line.strip().startswith(("Section", "Chapter", "1.", "2.", "3.")) and len(line.strip()) < 50:
                toc.append((line.strip(), i))
    return toc

# File upload
st.subheader("Upload Documents")
uploaded_files = st.file_uploader("Upload PDF or TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        file_extension = file_key.split(".")[-1].lower()
        file_path = f"temp_{file_key}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Processing document
        try:
            if file_extension == "pdf":
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                except:
                    text = extract_text_from_scanned_pdf(file_path)
                    with open(f"temp_{file_key}.txt", "w") as f:
                        f.write(text)
                    loader = TextLoader(f"temp_{file_key}.txt")
                    documents = loader.load()
                    if os.path.exists(f"temp_{file_key}.txt"):
                        os.remove(f"temp_{file_key}.txt")
            else:
                loader = TextLoader(file_path)
                documents = loader.load()

            # Chunk and embed
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            st.session_state.vector_stores[file_key] = FAISS.from_documents(chunks, embeddings)

            # Generating summary
            full_text = " ".join([doc.page_content for doc in documents])
            summary_prompt = f"Summarize the following document in 150 words or less:\n\n{full_text[:2000]}"
            summary_raw = llm(summary_prompt)
            summary = summary_raw.strip().split("\n")[0].strip()
            st.session_state.summaries[file_key] = summary
            st.session_state.toc[file_key] = extract_toc(documents)
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file_key}: {e}")

    # Displaying summaries and TOC
    for file_key in st.session_state.summaries:
        st.markdown(f"### Summary: {file_key}")
        st.markdown(f"<div class='toc-container'>{st.session_state.summaries[file_key]}</div>", unsafe_allow_html=True)
        if st.session_state.toc[file_key]:
            st.markdown(f"#### Table of Contents: {file_key}")
            for heading, page in st.session_state.toc[file_key]:
                st.markdown(f"- {heading} (Page {page + 1})")

# Voice input
st.subheader("Voice Input")
audio_file = st.file_uploader("Upload an audio file for question input", type=["wav", "mp3"])
if audio_file:
    audio_bytes = audio_file.read()
    try:
        with st.spinner("Transcribing audio..."):
            inputs = whisper_processor(audio_bytes, return_tensors="pt", sampling_rate=16000)
            predicted_ids = whisper_model.generate(inputs["input_features"])
            question = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            st.write(f"Transcribed Question: {question}")
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        question = ""
else:
    question = st.text_input("Or type your question:")

if st.session_state.vector_stores:
    st.markdown("---")
    mode = st.radio("Select Interaction Mode", ["Ask Anything", "Challenge Me"], horizontal=True)
    session_id = st.text_input("Session ID for Collaboration", value=st.session_state.session_id, disabled=True)

    if mode == "Ask Anything":
        st.subheader("Ask Anything")
        if question:
            with st.spinner("Processing..."):
                # Merge vector stores for cross-document reasoning
                merged_store = FAISS.from_documents([], embeddings)
                for file_key, store in st.session_state.vector_stores.items():
                    merged_store.merge_from(store)

                if session_id not in st.session_state.memory:
                    st.session_state.memory[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=merged_store.as_retriever(),
                    memory=st.session_state.memory[session_id]
                )
                result = chain({"question": question})
                answer = result["answer"]
                docs = merged_store.similarity_search(question, k=1)
                snippet = docs[0].page_content[:300] + "..." if len(docs[0].page_content) > 300 else docs[0].page_content

                # Semantic heatmap
                all_docs = [doc for store in st.session_state.vector_stores.values() for doc in store.similarity_search(question, k=10)]
                scores = []
                for store in st.session_state.vector_stores.values():
                    try:
                        score = store.similarity_search_with_score(question, k=1)[0][1]
                    except:
                        score = 0
                    scores.append(score)

                df = pd.DataFrame({"Document": list(st.session_state.vector_stores.keys()), "Relevance Score": scores})
                fig = px.bar(df, x="Document", y="Relevance Score", title="Semantic Relevance Heatmap")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**Answer**: {answer}")
                st.markdown(f"**Justification**: This is supported by the following snippet:\n\n<div class='toc-container'>{snippet}</div>", unsafe_allow_html=True)
                st.markdown(f"**Source**: Page {docs[0].metadata.get('page', 'N/A')}, File {docs[0].metadata.get('source', 'N/A')}")

                # Text-to-speech
                if st.button("Play Answer"):
                    tts = gTTS(answer)
                    tts.save("answer.mp3")
                    audio_file = open("answer.mp3", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    if os.path.exists("answer.mp3"):
                        os.remove("answer.mp3")
    elif mode == "Challenge Me":
        st.subheader("Challenge Me")
        merged_store = FAISS.from_documents([], embeddings)
        for file_key, store in st.session_state.vector_stores.items():
            merged_store.merge_from(store)
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                difficulty = 1 + st.session_state.user_score / 10  # Adaptive difficulty
                question_prompt = f"Generate three logic-based or comprehension-focused questions with difficulty level {difficulty:.1f} based on the document content:\n\n" + " ".join(st.session_state.summaries.values())
                questions = llm(question_prompt).split("\n")[:3]
                st.session_state.questions[session_id] = questions
                st.session_state.user_answers[session_id] = [""] * 3
                st.session_state.feedback[session_id] = [""] * 3

        if session_id in st.session_state.questions:
            for i, q in enumerate(st.session_state.questions[session_id]):
                st.markdown(f"**Question {i+1}**: {q}")
                st.session_state.user_answers[session_id][i] = st.text_input(f"Your answer to Question {i+1}", key=f"answer_{i}_{session_id}")
                if st.button(f"Submit Answer {i+1}", key=f"submit_{i}_{session_id}"):
                    with st.spinner("Evaluating..."):
                        eval_prompt = f"Evaluate the following user answer to the question based on the document:\nQuestion: {q}\nAnswer: {st.session_state.user_answers[session_id][i]}\nDocument Summary: {' '.join(st.session_state.summaries.values())}"
                        feedback = llm(eval_prompt)
                        docs = merged_store.similarity_search(q, k=1)
                        snippet = docs[0].page_content[:300] + "..." if len(docs[0].page_content) > 300 else docs[0].page_content
                        st.session_state.feedback[session_id][i] = f"{feedback}\n\n**Justification**: This evaluation is based on the following snippet:\n\n<div class='toc-container'>{snippet}</div>"
                        # Update user score
                        if "correct" in feedback.lower():
                            st.session_state.user_score += 1
                        st.markdown(st.session_state.feedback[session_id][i], unsafe_allow_html=True)

# Display conversation history
if session_id in st.session_state.memory and st.session_state.memory[session_id].chat_history:
    st.markdown("---")
    st.subheader("Conversation History")
    for msg in st.session_state.memory[session_id].chat_history:
        if msg.type == "human":
            st.markdown(f"**You**: {msg.content}")
        else:
            st.markdown(f"**Abstractly**: {msg.content}")