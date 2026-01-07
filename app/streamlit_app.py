import streamlit as st
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, skip loading .env file

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration (must be first)
st.set_page_config(
    page_title="Smart Resume and Job Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show loading message immediately
placeholder = st.empty()
with placeholder.container():
    st.info("🔄 Initializing application...")

try:
    from resume_parser import extract_resume_text, extract_structured_info
    from job_parser import parse_job_description, extract_job_requirements
    from matcher import ResumeJobMatcher, get_matcher
    from llm_analyzer import LLMAnalyzer
except Exception as e:
    placeholder.error(f"❌ Error importing modules: {str(e)}")
    st.stop()

# Clear loading message
placeholder.empty()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .match-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .high-match { color: #2ecc71; }
    .medium-match { color: #f39c12; }
    .low-match { color: #e74c3c; }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state with loading indicators
if 'matcher' not in st.session_state:
    try:
        with st.spinner("Loading embedding model (this may take a minute on first run)..."):
            st.session_state.matcher = get_matcher()
        st.success("✅ Embedding model loaded!")
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {str(e)}")
        st.stop()

# Initialize session state for LLM config
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "meta-llama/Llama-3.1-8B-Instruct"
if 'hf_token' not in st.session_state:
    # Try both HUGGINGFACE_API_TOKEN and HF_API_KEY for compatibility
    st.session_state.hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_KEY") or ""

# Main header
st.markdown('<div class="main-header">🔍 Smart Resume and Job Matcher</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
        help="Choose the embedding model for semantic similarity"
    )
    
    # LLM configuration
    use_llm = st.checkbox("Use LLM for Analysis", value=True)
    if use_llm:
        llm_model = st.selectbox(
            "LLM Model",
            [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-chat-hf",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "google/flan-t5-large"
            ],
            help="Hugging Face model for generating explanations",
            index=0
        )
        hf_token = st.text_input(
            "Hugging Face API Token",
            value=st.session_state.hf_token,
            type="password",
            help="Your Hugging Face API token. Get one at https://huggingface.co/settings/tokens"
        )
        
        # Update session state
        st.session_state.llm_model = llm_model
        st.session_state.hf_token = hf_token
        
        # Initialize or reinitialize LLM analyzer if model or token changed
        if 'llm_analyzer' not in st.session_state or \
           st.session_state.get('last_llm_model') != llm_model or \
           st.session_state.get('last_hf_token') != hf_token:
            try:
                st.session_state.llm_analyzer = LLMAnalyzer(model_name=llm_model, hf_token=hf_token if hf_token else None)
                st.session_state.last_llm_model = llm_model
                st.session_state.last_hf_token = hf_token
                if hf_token:
                    st.success("✅ LLM analyzer initialized!")
            except Exception as e:
                st.warning(f"⚠️ LLM analyzer initialization warning: {str(e)}")
                st.session_state.llm_analyzer = None
    else:
        st.session_state.llm_analyzer = None

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📄 Single Match", "📊 Batch Matching", "ℹ️ About"])

with tab1:
    st.header("Match a Resume with a Job Description")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume")
        resume_option = st.radio(
            "Resume Source",
            ["Upload File", "Paste Text"],
            horizontal=True
        )
        
        resume_text = ""
        resume_info = None
        
        if resume_option == "Upload File":
            uploaded_resume = st.file_uploader(
                "Upload Resume",
                type=["pdf", "docx", "txt"],
                help="Upload a PDF, DOCX, or TXT resume file"
            )
            
            if uploaded_resume:
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_resume_{uploaded_resume.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_resume.getbuffer())
                    
                    # Extract text
                    resume_text = extract_resume_text(temp_path)
                    resume_info = extract_structured_info(resume_text)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    st.success("✅ Resume loaded successfully!")
                    
                    with st.expander("View Resume Text"):
                        st.text_area("Resume Content", resume_text, height=200, disabled=True)
                    
                    with st.expander("View Structured Information"):
                        st.json(resume_info)
                        
                except Exception as e:
                    st.error(f"Error processing resume: {str(e)}")
        else:
            resume_text = st.text_area(
                "Paste Resume Text",
                height=300,
                help="Paste the text content of your resume"
            )
            if resume_text:
                resume_info = extract_structured_info(resume_text)
    
    with col2:
        st.subheader("💼 Job Description")
        job_option = st.radio(
            "Job Source",
            ["Upload File", "Paste Text"],
            horizontal=True
        )
        
        job_text = ""
        job_info = None
        
        if job_option == "Upload File":
            uploaded_job = st.file_uploader(
                "Upload Job Description",
                type=["pdf", "docx", "txt"],
                help="Upload a PDF, DOCX, or TXT job description file"
            )
            
            if uploaded_job:
                try:
                    # Save uploaded file temporarily
                    temp_path = f"temp_job_{uploaded_job.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_job.getbuffer())
                    
                    # Extract text
                    job_text = extract_resume_text(temp_path)  # Reuse resume parser
                    job_text = parse_job_description(job_text)
                    job_info = extract_job_requirements(job_text)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    st.success("✅ Job description loaded successfully!")
                    
                    with st.expander("View Job Description"):
                        st.text_area("Job Content", job_text, height=200, disabled=True)
                    
                    with st.expander("View Job Requirements"):
                        st.json(job_info)
                        
                except Exception as e:
                    st.error(f"Error processing job description: {str(e)}")
        else:
            job_text = st.text_area(
                "Paste Job Description",
                height=300,
                help="Paste the job description text"
            )
            if job_text:
                job_text = parse_job_description(job_text)
                job_info = extract_job_requirements(job_text)
    
    # Match button
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        if not resume_text or not job_text:
            st.warning("⚠️ Please provide both resume and job description.")
        else:
            with st.spinner("Computing similarity and generating analysis..."):
                # Compute similarity
                similarity_score = st.session_state.matcher.compute_similarity(
                    resume_text, job_text
                )
                
                # Display match score
                st.markdown("---")
                st.header("📊 Match Results")
                
                # Score display with color coding
                if similarity_score >= 0.7:
                    score_class = "high-match"
                    match_level = "Strong Match"
                elif similarity_score >= 0.5:
                    score_class = "medium-match"
                    match_level = "Moderate Match"
                else:
                    score_class = "low-match"
                    match_level = "Weak Match"
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f'<div class="match-score {score_class}">{similarity_score:.1%}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center; font-size: 1.2rem; margin-bottom: 2rem;">{match_level}</div>', unsafe_allow_html=True)
                
                # Generate LLM explanation
                if use_llm and st.session_state.llm_analyzer:
                    try:
                        with st.spinner("Generating LLM explanation..."):
                            explanation = st.session_state.llm_analyzer.generate_match_explanation(
                                resume_text, job_text, similarity_score,
                                resume_info, job_info
                            )
                        st.subheader("💡 Match Explanation")
                        st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        error_msg = str(e)
                        st.warning(f"Could not generate LLM explanation: {error_msg}")
                        if "loading" in error_msg.lower():
                            st.info("💡 The model is currently loading. Please wait a moment and try again.")
                        elif "rate limit" in error_msg.lower() or "429" in error_msg:
                            st.info("💡 API rate limit reached. Please try again later.")
                        elif "401" in error_msg or "unauthorized" in error_msg.lower():
                            st.info("💡 Authentication error. Please check your Hugging Face API token in the sidebar.")
                        else:
                            st.info("Using fallback explanation.")
                        if st.session_state.llm_analyzer:
                            explanation = st.session_state.llm_analyzer._fallback_explanation(
                                resume_info, job_info, similarity_score
                            )
                            st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
                elif use_llm:
                    # LLM not available, use simple explanation
                    st.subheader("💡 Match Explanation")
                    if resume_info and job_info:
                        explanation = f"Match Score: {similarity_score:.1%}\n\n"
                        common_skills = set(resume_info.get('skills', [])) & set(
                            job_info.get('required_skills', []) + job_info.get('preferred_skills', [])
                        )
                        if common_skills:
                            explanation += f"Key Matching Skills: {', '.join(list(common_skills)[:5])}"
                        st.markdown(f'<div class="info-box">{explanation}</div>', unsafe_allow_html=True)
                
                # Skills analysis
                if resume_info and job_info:
                    st.subheader("🛠️ Skills Analysis")
                    
                    if st.session_state.llm_analyzer:
                        missing_analysis = st.session_state.llm_analyzer.analyze_missing_skills(
                            resume_info.get('skills', []),
                            job_info.get('required_skills', []),
                            job_info.get('preferred_skills', [])
                        )
                    else:
                        # Fallback analysis without LLM
                        resume_skills_lower = [s.lower() for s in resume_info.get('skills', [])]
                        job_required_lower = [s.lower() for s in job_info.get('required_skills', [])]
                        job_preferred_lower = [s.lower() for s in job_info.get('preferred_skills', [])]
                        
                        missing_required = [s for s in job_info.get('required_skills', []) if s.lower() not in resume_skills_lower]
                        missing_preferred = [s for s in job_info.get('preferred_skills', []) if s.lower() not in resume_skills_lower]
                        
                        missing_analysis = {
                            'missing_required': missing_required,
                            'missing_preferred': missing_preferred,
                            'recommendations': [
                                f"Consider highlighting these required skills: {', '.join(missing_required[:5])}" if missing_required else "All required skills are present!",
                                f"These preferred skills could strengthen your application: {', '.join(missing_preferred[:5])}" if missing_preferred else ""
                            ]
                        }
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Matching Skills**")
                        matching_skills = set(resume_info.get('skills', [])) & set(
                            job_info.get('required_skills', []) + job_info.get('preferred_skills', [])
                        )
                        if matching_skills:
                            for skill in list(matching_skills)[:10]:
                                st.success(f"✓ {skill}")
                        else:
                            st.info("No matching skills found.")
                    
                    with col2:
                        st.markdown("**❌ Missing Skills**")
                        if missing_analysis['missing_required']:
                            st.markdown("**Required:**")
                            for skill in missing_analysis['missing_required'][:5]:
                                st.error(f"✗ {skill}")
                        if missing_analysis['missing_preferred']:
                            st.markdown("**Preferred:**")
                            for skill in missing_analysis['missing_preferred'][:5]:
                                st.warning(f"⚠ {skill}")
                        if not missing_analysis['missing_required'] and not missing_analysis['missing_preferred']:
                            st.success("All required skills are present!")
                    
                    # Recommendations
                    if missing_analysis['recommendations']:
                        st.subheader("💬 Recommendations")
                        for rec in missing_analysis['recommendations']:
                            st.info(rec)
                
                # Improvement suggestions
                if use_llm and st.session_state.llm_analyzer:
                    try:
                        suggestions = st.session_state.llm_analyzer.generate_improvement_suggestions(
                            resume_text, job_text
                        )
                        if suggestions:
                            st.subheader("📈 Improvement Suggestions")
                            for i, suggestion in enumerate(suggestions, 1):
                                st.markdown(f"{i}. {suggestion}")
                    except Exception as e:
                        st.debug(f"Could not generate suggestions: {e}")

with tab2:
    st.header("Batch Matching - Match Resume with Multiple Jobs")
    
    # Resume input
    st.subheader("📄 Resume")
    batch_resume_option = st.radio(
        "Resume Source",
        ["Upload File", "Paste Text"],
        horizontal=True,
        key="batch_resume"
    )
    
    batch_resume_text = ""
    
    if batch_resume_option == "Upload File":
        batch_uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"],
            help="Upload a PDF, DOCX, or TXT resume file",
            key="batch_resume_upload"
        )
        
        if batch_uploaded_resume:
            try:
                temp_path = f"temp_batch_resume_{batch_uploaded_resume.name}"
                with open(temp_path, "wb") as f:
                    f.write(batch_uploaded_resume.getbuffer())
                
                batch_resume_text = extract_resume_text(temp_path)
                os.remove(temp_path)
                st.success("✅ Resume loaded!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        batch_resume_text = st.text_area(
            "Paste Resume Text",
            height=200,
            key="batch_resume_text"
        )
    
    # Jobs input
    st.subheader("💼 Job Descriptions")
    batch_jobs_text = st.text_area(
        "Enter Job Descriptions",
        height=400,
        help="Enter multiple job descriptions, separated by '---' or '===' on a new line",
        key="batch_jobs"
    )
    
    if st.button("🔍 Find Top Matches", type="primary", use_container_width=True):
        if not batch_resume_text or not batch_jobs_text:
            st.warning("⚠️ Please provide both resume and job descriptions.")
        else:
            with st.spinner("Processing jobs and computing matches..."):
                # Split jobs
                separators = ['---', '===', '***']
                jobs = [batch_jobs_text]
                for sep in separators:
                    new_jobs = []
                    for job in jobs:
                        new_jobs.extend([j.strip() for j in job.split(sep) if j.strip()])
                    jobs = new_jobs
                
                if len(jobs) == 1:
                    st.info("💡 Tip: Separate multiple job descriptions with '---' on a new line")
                
                # Rank jobs
                matches = st.session_state.matcher.rank_jobs(batch_resume_text, jobs)
                
                st.markdown("---")
                st.header("📊 Top Matches")
                
                for rank, (idx, score, job_text, metadata) in enumerate(matches, 1):
                    with st.expander(f"Rank #{rank} - Match Score: {score:.1%}", expanded=(rank <= 3)):
                        st.markdown(f"**Similarity Score:** {score:.1%}")
                        st.markdown("**Job Description:**")
                        st.text(job_text[:500] + "..." if len(job_text) > 500 else job_text)
                        
                        # Quick analysis
                        job_info = extract_job_requirements(job_text)
                        resume_info = extract_structured_info(batch_resume_text)
                        
                        if resume_info and job_info:
                            matching_skills = set(resume_info.get('skills', [])) & set(
                                job_info.get('required_skills', []) + job_info.get('preferred_skills', [])
                            )
                            if matching_skills:
                                st.markdown(f"**Matching Skills:** {', '.join(list(matching_skills)[:5])}")

with tab3:
    st.header("About Smart Resume and Job Matcher")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application uses **AI-powered semantic matching** to connect candidates with job opportunities.
    Unlike traditional keyword-based matching, this system understands the *meaning* and *context* of
    resumes and job descriptions.
    
    ### 🔧 Key Features
    
    - **Multi-format Support**: Process PDF, DOCX, and TXT files
    - **Semantic Similarity**: Uses SentenceTransformers for contextual understanding
    - **Structured Extraction**: Automatically extracts skills, education, experience, and more
    - **LLM-powered Analysis**: Generates human-like explanations for matches
    - **Batch Matching**: Compare one resume against multiple jobs efficiently
    - **Actionable Insights**: Provides recommendations and improvement suggestions
    
    ### 🛠️ Technology Stack
    
    - **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
    - **Vector Search**: FAISS for efficient similarity search
    - **LLM**: Hugging Face Inference API (Llama2, Mistral, etc.) for explanations
    - **Framework**: Streamlit for the user interface
    - **API**: Hugging Face Hub for LLM inference
    
    ### 📚 How It Works
    
    1. **Parsing**: Extracts and structures information from resumes and job descriptions
    2. **Embedding**: Converts text into high-dimensional vector representations
    3. **Matching**: Computes cosine similarity between resume and job embeddings
    4. **Analysis**: Uses LLM to generate explanations and recommendations
    5. **Ranking**: Sorts matches by relevance score
    
    ### 🚀 Getting Started
    
    1. Upload or paste your resume
    2. Upload or paste a job description
    3. Click "Analyze Match" to see results
    4. Review the match score, explanation, and recommendations
    
    ### 💡 Tips
    
    - For best results, ensure your resume is well-formatted and includes relevant keywords
    - The system works best with detailed job descriptions
    - Use batch matching to quickly compare against multiple opportunities
    - Review the missing skills section to identify areas for improvement
    """)
    
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit, SentenceTransformers, and LangChain**")

