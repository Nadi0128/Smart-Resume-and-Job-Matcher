import re
from typing import Dict, List, Optional


def parse_job_description(job_text: str) -> str:
    """
    Clean and normalize job description text.
    
    Args:
        job_text: Raw job description text
        
    Returns:
        Cleaned and normalized text
    """
    # Remove extra whitespace
    job_text = re.sub(r'\s+', ' ', job_text.strip())
    # Remove special characters but keep punctuation
    job_text = re.sub(r'[^\w\s.,;:!?()\-/]', '', job_text)
    return job_text


def extract_job_requirements(job_text: str) -> Dict[str, any]:
    """
    Extract structured information from job description.
    
    Args:
        job_text: Job description text
        
    Returns:
        Dictionary with structured information:
        - required_skills: List of required skills
        - preferred_skills: List of preferred skills
        - experience_level: Experience level required
        - education: Education requirements
        - responsibilities: List of responsibilities
    """
    text_lower = job_text.lower()
    
    # Common skill keywords
    all_skills = [
        'python', 'java', 'javascript', 'typescript', 'sql', 'html', 'css', 
        'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
        'machine learning', 'deep learning', 'ai', 'artificial intelligence',
        'data science', 'analytics', 'data analysis', 'data engineering',
        'aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes', 'ci/cd',
        'git', 'agile', 'scrum', 'devops', 'mlops',
        'power bi', 'tableau', 'excel', 'r', 'tensorflow', 'pytorch',
        'api', 'rest', 'graphql', 'microservices', 'mongodb', 'postgresql',
        'mysql', 'nosql', 'redis', 'elasticsearch', 'kafka', 'spark'
    ]
    
    # Extract required skills
    required_skills = []
    preferred_skills = []
    
    for skill in all_skills:
        if skill in text_lower:
            # Check if it's required or preferred
            skill_context = _get_skill_context(job_text, skill)
            if any(word in skill_context.lower() for word in ['required', 'must', 'essential', 'need']):
                required_skills.append(skill.title())
            elif any(word in skill_context.lower() for word in ['preferred', 'nice', 'bonus', 'plus']):
                preferred_skills.append(skill.title())
            else:
                required_skills.append(skill.title())  # Default to required
    
    # Extract experience level
    experience_level = None
    exp_patterns = {
        'entry': ['entry', 'junior', '0-2', '0-1', 'fresh', 'graduate'],
        'mid': ['mid', '2-5', '3-5', 'intermediate'],
        'senior': ['senior', '5+', 'lead', 'principal', 'architect'],
        'executive': ['executive', 'director', 'vp', 'vice president', 'c-level']
    }
    
    for level, patterns in exp_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            experience_level = level
            break
    
    # Extract education requirements
    education = []
    edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma', 'bs', 'ms', 'mba']
    for keyword in edu_keywords:
        if keyword in text_lower:
            education.append(keyword.title())
    
    # Extract responsibilities (look for bullet points or numbered lists)
    responsibilities = []
    lines = job_text.split('\n')
    in_responsibilities = False
    
    responsibility_keywords = ['responsibility', 'duties', 'role', 'will', 'tasks']
    
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in responsibility_keywords):
            in_responsibilities = True
        elif in_responsibilities:
            # Look for bullet points or numbered items
            if re.match(r'^[\-\•\*]\s+', line) or re.match(r'^\d+[\.\)]\s+', line):
                responsibilities.append(line.strip())
            elif line.strip() and len(line.strip()) > 20:
                responsibilities.append(line.strip())
    
    return {
        'required_skills': list(set(required_skills)),
        'preferred_skills': list(set(preferred_skills)),
        'experience_level': experience_level,
        'education': list(set(education)),
        'responsibilities': responsibilities[:10],  # Limit to top 10
        'raw_text': job_text
    }


def _get_skill_context(text: str, skill: str, context_window: int = 50) -> str:
    """Get context around a skill mention in the text."""
    text_lower = text.lower()
    skill_lower = skill.lower()
    
    if skill_lower in text_lower:
        index = text_lower.find(skill_lower)
        start = max(0, index - context_window)
        end = min(len(text), index + len(skill) + context_window)
        return text[start:end]
    return ""
