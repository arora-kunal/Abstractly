# Abstractly: Smart Assistant for Research Summarization

Abstractly is a research-focused assistant that supports intelligent summarization, interactive Q&A, adaptive quizzes, and semantic relevance visualization from uploaded documents. It handles both text-based and scanned PDFs, along with voice-based question input and text-to-speech answers—all powered through a Streamlit interface and HuggingFace models.

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/abstractly.git
cd abstractly
```

### 2. Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate (use the one appropriate for your OS)
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

- **Tesseract OCR**: Required for extracting text from scanned PDFs  
  [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

- **Poppler**: Required for converting PDF pages into images  
  [https://github.com/oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows)

> 📝 Ensure both `tesseract` and `poppler` are installed and available in your system's PATH.

### 5. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 Architecture / Reasoning Flow

### 📂 Input Handling
- Users can upload `.pdf` or `.txt` documents.
- OCR is automatically triggered if a PDF is scanned.
- Audio files (`.wav`, `.mp3`) can be uploaded to ask questions via voice.

### 🔍 Document Processing
- **Text Extraction**:
  - `PyPDFLoader` for text PDFs.
  - `pytesseract` + `pdf2image` for scanned PDFs.
- **Splitting**:
  - `RecursiveCharacterTextSplitter` breaks documents into manageable chunks.
- **Embedding**:
  - Sentence-transformer: `all-MiniLM-L6-v2`.
  - FAISS vector store is used for similarity search.

### 🧠 Language Model
- Uses `tiiuae/falcon-rw-1b` via HuggingFace Pipeline for:
  - Document summarization
  - Answer generation
  - Question generation
  - Answer evaluation
- Falls back to `sshleifer/tiny-gpt2` if the primary model fails to load.

### 🎙️ Audio Pipeline
- **Voice Input**:
  - OpenAI’s `whisper-tiny` handles audio-to-text transcription.
- **Text-to-Speech**:
  - `gTTS` generates playable audio of answers.

### 🧑‍💻 Modes of Operation

- **Ask Anything**:
  - User asks a question.
  - Relevant context is retrieved via similarity search.
  - LLM generates a direct answer with citation and snippet.

- **Challenge Me**:
  - LLM generates three adaptive questions based on document summary.
  - User answers are evaluated and scored.
  - Adaptive difficulty based on user score.

### 📊 Extras
- **Semantic Relevance Heatmap**:
  - Visualizes document relevance to the user’s question using Plotly.
- **Table of Contents**:
  - Extracts and displays headers like "Section", "Chapter", "1." etc.

---

## 📁 Organized Source Code Folder

```
abstractly/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## ✨ Features

- 📄 Handles text and scanned PDFs
- 🧠 Summarizes documents using LLMs
- ❓ Intelligent Q&A with reference snippets
- 🎯 Adaptive quiz mode with scoring
- 🔊 Voice-to-text input and answer playback
- 🔥 Semantic similarity bar charts
- 📑 Table of contents extraction

---

## 👨‍💻 Developed by

**Kunal Arora**  
[GitHub: arora-kunal](https://github.com/arora-kunal)

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
