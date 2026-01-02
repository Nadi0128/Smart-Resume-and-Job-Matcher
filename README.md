# Smart-Resume-and-Job-Matcher

The Smart Resume and Job Matcher is an intelligent application designed to automatically analyze and match resumes with job descriptions using Natural Language Processing (NLP) and Large Language Models (LLMs).

The system extracts text from resume PDFs, analyzes job descriptions, and computes a semantic compatibility score based on embeddings. In addition, an LLM is used to provide explainable feedback, including missing skills and personalized recommendations to improve resume alignment with the target job.

GitHub Repository Initialization
- Created and initialized a public GitHub repository.
- Cloned the repository locally and set up version control.

# Project Structure Setup
Organized the project into a clear and modular structure:
- app/ for core application logic
- data/ for sample resumes and job descriptions
- notebooks/ for experiments and testing
- Added essential files such as requirements.txt, README.md, and .gitignore.

# Python Environment & Dependencies
- Set up a Python virtual environment.
- Installed required dependencies, including pypdf for PDF processing.

# Resume Parsing Module
- Implemented a resume parser (resume_parser.py) using PyPDF.

# Verifying PDF Resume Parsing
To verify that a resume PDF can be successfully processed, a simple test is included directly in the resume_parser.py file.

Example Test Code
How to Run the Test

Place a PDF resume file in: data/sample_resumes

Run the following command in the terminal:
    - python app/resume_parser.py