import os
from typing import Dict, List, Optional
from pypdf import PdfReader

# Optional DOCX support
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    docx = None


def extract_resume_text(file_path_or_obj) -> str:
    """
    Extract text from a resume file (PDF, DOCX, or TXT).
    
    Args:
        file_path_or_obj: Path to the resume file (str) or file-like object
        
    Returns:
        Extracted text as string
    """
    # Handle file objects (for backward compatibility)
    if hasattr(file_path_or_obj, 'read'):
        # It's a file object - need to save to temp file or handle differently
        import tempfile
        file_obj = file_path_or_obj
        current_pos = file_obj.tell()
        file_obj.seek(0)
        
        # Try to get filename from file object
        filename = getattr(file_obj, 'name', 'temp_file.pdf')
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        
        file_obj.seek(current_pos)  # Restore position
        
        try:
            result = extract_resume_text(tmp_path)
        finally:
            os.unlink(tmp_path)  # Clean up temp file
        
        return result
    
    # Handle file path (string)
    file_path = file_path_or_obj
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return _extract_from_pdf(file_path)
    elif file_ext == '.docx':
        return _extract_from_docx(file_path)
    elif file_ext in ['.txt', '.text']:
        return _extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def _extract_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def _extract_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    if not DOCX_AVAILABLE:
        raise ImportError(
            "python-docx is not installed. Install it with: pip install python-docx"
        )
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text.strip()


def _extract_from_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_structured_info(resume_text: str) -> Dict[str, any]:
    """
    Extract structured information from resume text.
    Uses simple pattern matching and heuristics.
    
    Args:
        resume_text: Raw text from resume
        
    Returns:
        Dictionary with structured information:
        - skills: List of skills
        - education: List of education entries
        - experience: List of experience entries
        - certifications: List of certifications
        - interests: List of interests/hobbies
    """
    text_lower = resume_text.lower()
    lines = resume_text.split('\n')
    
    # Common skill keywords
    skill_keywords = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'node.js',
        'machine learning', 'deep learning', 'ai', 'data science', 'analytics',
        'aws', 'azure', 'docker', 'kubernetes', 'git', 'agile', 'scrum',
        'power bi', 'tableau', 'excel', 'r', 'tensorflow', 'pytorch',
        'api', 'rest', 'graphql', 'mongodb', 'postgresql', 'mysql'
    ]
    
    # Extract skills
    skills = []
    for keyword in skill_keywords:
        if keyword in text_lower:
            skills.append(keyword.title())
    
    # Extract education (look for common patterns)
    education = []
    edu_keywords = ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'phd', 'diploma']
    in_education_section = False
    current_edu = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in edu_keywords):
            in_education_section = True
            if line.strip():
                current_edu.append(line.strip())
        elif in_education_section and line.strip():
            if len(line.strip()) > 5:  # Likely continuation
                current_edu.append(line.strip())
            else:
                if current_edu:
                    education.append(' '.join(current_edu))
                    current_edu = []
                in_education_section = False
    
    if current_edu:
        education.append(' '.join(current_edu))
    
    # Extract experience (look for work/experience section)
    experience = []
    exp_keywords = ['experience', 'work', 'employment', 'career', 'professional']
    in_experience_section = False
    current_exp = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in exp_keywords):
            in_experience_section = True
        elif in_experience_section and line.strip():
            # Look for date patterns or job titles
            if any(char.isdigit() for char in line) or len(line.strip()) < 50:
                if current_exp:
                    experience.append(' '.join(current_exp))
                    current_exp = []
                if line.strip():
                    current_exp.append(line.strip())
            else:
                current_exp.append(line.strip())
    
    if current_exp:
        experience.append(' '.join(current_exp))
    
    # Extract certifications
    certifications = []
    cert_keywords = ['certification', 'certificate', 'certified', 'license', 'credential']
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in cert_keywords):
            if line.strip():
                certifications.append(line.strip())
    
    # Extract interests
    interests = []
    interest_keywords = ['interest', 'hobby', 'hobbies', 'activities', 'volunteer']
    in_interests_section = False
    
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in interest_keywords):
            in_interests_section = True
        elif in_interests_section and line.strip():
            if len(line.strip()) < 100:  # Likely interest item
                interests.append(line.strip())
            else:
                in_interests_section = False
    
    return {
        'skills': list(set(skills)),  # Remove duplicates
        'education': education[:5],  # Limit to top 5
        'experience': experience[:10],  # Limit to top 10
        'certifications': certifications[:5],
        'interests': interests[:5],
        'raw_text': resume_text
    }

if __name__ == "__main__":
    # Test with file path
    test_file = "data/sample_resumes/test_cv2.pdf"
    if os.path.exists(test_file):
        text = extract_resume_text(test_file)
        print("Texte extrait du fichier pdf ")
        print(text[:1000])
        
        # Test structured extraction
        print("\n" + "="*50)
        print("Structured Information:")
        print("="*50)
        info = extract_structured_info(text)
        for key, value in info.items():
            if key != 'raw_text':
                print(f"\n{key.upper()}:")
                if isinstance(value, list):
                    for item in value[:3]:
                        print(f"  - {item}")
                else:
                    print(f"  {value}")
    else:
        print(f"Test file not found: {test_file}")
        print("Please provide a resume file to test.")
