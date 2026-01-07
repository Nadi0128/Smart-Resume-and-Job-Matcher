# Smart Resume and Job Matcher

An AI-powered Resume and Job Matching System that uses embeddings, semantic search, and Generative AI reasoning to match candidates' resumes with the most relevant job opportunities.

## 🎯 Features

- **Multi-format Support**: Process PDF, DOCX, and TXT resume files
- **Semantic Matching**: Uses SentenceTransformers for contextual understanding beyond keyword matching
- **Structured Extraction**: Automatically extracts skills, education, experience, certifications, and interests
- **LLM-powered Analysis**: Generates human-like explanations for matches using Hugging Face models (Llama-3.1-8B-Instruct via Inference API)
- **Batch Matching**: Compare one resume against multiple jobs efficiently using FAISS
- **Actionable Insights**: Provides recommendations and improvement suggestions

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- **Hugging Face account and API token** for LLM features - [Get your free token here](https://huggingface.co/settings/tokens)
  - The system uses the Llama-3.1-8B-Instruct model via Hugging Face Inference API

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

### 4. Set Up Hugging Face API Token for LLM Features

The application uses Hugging Face models for LLM-powered analysis. You must set up your Hugging Face API token:

1. Create a free account at [https://huggingface.co](https://huggingface.co)
2. Get your API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Set the token as an environment variable:
   ```bash
   # Windows (PowerShell)
   $env:HUGGINGFACE_API_TOKEN="your_token_here"
   
   # Linux/Mac
   export HUGGINGFACE_API_TOKEN="your_token_here"
   ```
   **Alternative**: You can also set `HF_API_KEY` environment variable instead of `HUGGINGFACE_API_TOKEN`.
   Or enter the token directly in the Streamlit app sidebar when prompted.

**Note**: The app uses the `meta-llama/Llama-3.1-8B-Instruct` model by default. You can customize the model in the code by modifying the `model_name` parameter in `app/llm_analyzer.py`.

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

The application uses **Hugging Face Inference API** for LLM-powered explanations:

- **Default Model**: `meta-llama/Llama-3.1-8B-Instruct` (recommended)
- **Other Available Models**: Any Hugging Face model compatible with the Inference API

To use a different model, modify the `model_name` parameter in your code:

```python
from app.llm_analyzer import LLMAnalyzer

# Use default model
analyzer = LLMAnalyzer()

# Or specify a custom model
analyzer = LLMAnalyzer(model_name="meta-llama/Llama-3.1-70B-Instruct")
```

**Requirements**:
- Valid Hugging Face API token (free tier available)
- Token set as `HUGGINGFACE_API_TOKEN` or `HF_API_KEY` environment variable

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
- **huggingface_hub**: Hugging Face Inference API client for LLM-powered analysis
- **sentence-transformers**: Semantic embeddings for resume-job matching
- **pypdf**: PDF parsing and text extraction
- **python-docx**: Microsoft Word document parsing
- **faiss-cpu**: Efficient similarity search for batch matching
- **numpy**: Numerical computations
- **torch** & **transformers**: Deep learning models for embeddings

## 🐛 Troubleshooting

### "Hugging Face API token error"
- Make sure you have a valid Hugging Face API token
- Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Enter the token in the sidebar or set it as `HUGGINGFACE_API_TOKEN` environment variable
- The app will work without the token, but LLM features will be disabled

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

Arturo1s
Nadi0128

## 🙏 Acknowledgments

- SentenceTransformers for semantic embeddings
- Hugging Face for LLM inference
- Streamlit for the web interface
- FAISS for efficient vector search
