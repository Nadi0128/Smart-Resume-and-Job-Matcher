# Smart Resume and Job Matcher

An AI-powered Resume and Job Matching System that uses embeddings, semantic search, and Generative AI reasoning to match candidates' resumes with the most relevant job opportunities.

## 🎯 Features

- **Multi-format Support**: Process PDF, DOCX, and TXT resume files
- **Semantic Matching**: Uses SentenceTransformers for contextual understanding beyond keyword matching
- **Structured Extraction**: Automatically extracts skills, education, experience, certifications, and interests
- **LLM-powered Analysis**: Generates human-like explanations for matches using Ollama
- **Batch Matching**: Compare one resume against multiple jobs efficiently using FAISS
- **Actionable Insights**: Provides recommendations and improvement suggestions

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- (Optional) Ollama for LLM features - [Download here](https://ollama.ai)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Smart-Resume-and-Job-Matcher
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set Up Ollama for LLM Features

If you want to use LLM-powered explanations:

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull a model (in a new terminal):
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```

### 5. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### Single Match Mode

1. Go to the **"📄 Single Match"** tab
2. **Upload or paste your resume**:
   - Upload a PDF, DOCX, or TXT file, OR
   - Paste the resume text directly
3. **Upload or paste a job description**:
   - Upload a job description file, OR
   - Paste the job description text
4. Click **"🔍 Analyze Match"**
5. View the results:
   - Match score (0-100%)
   - LLM-generated explanation
   - Skills analysis (matching and missing skills)
   - Improvement suggestions

### Batch Matching Mode

1. Go to the **"📊 Batch Matching"** tab
2. Upload or paste your resume
3. Enter multiple job descriptions, separated by `---` on a new line:
   ```
   Job Description 1
   ---
   Job Description 2
   ---
   Job Description 3
   ```
4. Click **"🔍 Find Top Matches"**
5. View ranked results with similarity scores

## 🏗️ Project Structure

```
Smart-Resume-and-Job-Matcher/
├── app/
│   ├── resume_parser.py      # Resume parsing and structured extraction
│   ├── job_parser.py         # Job description parsing and requirements extraction
│   ├── matcher.py            # Semantic similarity matching with FAISS
│   ├── llm_analyzer.py       # LLM-powered analysis and explanations
│   └── streamlit_app.py      # Main Streamlit application
├── data/
│   └── sample_resumes/       # Sample resume files for testing
├── notebooks/
│   └── experiments.ipynb     # Jupyter notebooks for experimentation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Embedding Models

You can change the embedding model in the Streamlit sidebar:
- `all-MiniLM-L6-v2` (default, fast and efficient)
- `all-mpnet-base-v2` (more accurate, slower)
- `paraphrase-MiniLM-L6-v2` (optimized for similarity)

### LLM Models

If using Ollama, you can select different models:
- `llama2` (default)
- `llama3`
- `mistral`
- `codellama`

Change the Ollama URL if running on a different host/port.

## 🧪 Testing Individual Components

### Test Resume Parser

```bash
python app/resume_parser.py
```

Place a resume file in `data/sample_resumes/` and update the filename in the script.

### Test Job Parser

```python
from app.job_parser import parse_job_description, extract_job_requirements

job_text = "Your job description here..."
cleaned = parse_job_description(job_text)
requirements = extract_job_requirements(job_text)
print(requirements)
```

### Test Matcher

```python
from app.matcher import compute_similarity

resume_text = "Your resume text..."
job_text = "Your job description..."
score = compute_similarity(resume_text, job_text)
print(f"Match score: {score:.2%}")
```

## 📦 Dependencies

- **streamlit**: Web application framework
- **langchain** & **langchain-community**: LLM orchestration
- **sentence-transformers**: Semantic embeddings
- **pypdf**: PDF parsing
- **python-docx**: DOCX parsing
- **faiss-cpu**: Efficient similarity search
- **numpy**: Numerical computations
- **torch** & **transformers**: Deep learning models

## 🐛 Troubleshooting

### "Ollama connection error"
- Make sure Ollama is running: `ollama serve`
- Check the Ollama URL in the sidebar (default: `http://localhost:11434`)
- The app will work without Ollama, but LLM features will be disabled

### "Import errors"
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment

### "PDF parsing errors"
- Some PDFs may have corruption issues - the parser includes error handling
- Try converting the PDF to a different format or using a different PDF

### "Low match scores"
- Ensure both resume and job description have sufficient detail
- The system uses semantic matching, not just keywords
- Review the skills analysis to see what's missing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- SentenceTransformers for semantic embeddings
- LangChain for LLM orchestration
- Streamlit for the web interface
- FAISS for efficient vector search