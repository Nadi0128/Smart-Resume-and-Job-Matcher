from typing import Dict, List, Optional
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, skip loading .env file

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class LLMAnalyzer:
    """
    LLM-based analyzer for generating explanations and insights
    about resume-job matches.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", hf_token: Optional[str] = None):
        """
        Initialize LLM analyzer.
        
        Args:
            model_name: Name of the Hugging Face model to use (default: "meta-llama/Llama-3.1-8B-Instruct")
            hf_token: Hugging Face API token (default: from HUGGINGFACE_API_TOKEN or HF_API_KEY env var)
        """
        self.model_name = model_name
        # Try both HUGGINGFACE_API_TOKEN and HF_API_KEY for compatibility
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_KEY")
        
        if not HUGGINGFACE_AVAILABLE:
            print("Warning: huggingface_hub not available. Install with: pip install huggingface_hub")
            self.client = None
        elif not self.hf_token:
            print("Warning: Hugging Face API token not provided. Set HUGGINGFACE_API_TOKEN environment variable or pass hf_token parameter.")
            print("Get your token from: https://huggingface.co/settings/tokens")
            self.client = None
        else:
            try:
                self.client = InferenceClient(model=model_name, token=self.hf_token)
            except Exception as e:
                print(f"Warning: Could not initialize Hugging Face client. Using fallback mode. Error: {e}")
                self.client = None
    
    def generate_match_explanation(
        self, 
        resume_text: str, 
        job_text: str, 
        similarity_score: float,
        resume_info: Optional[Dict] = None,
        job_info: Optional[Dict] = None
    ) -> str:
        """
        Generate an explanation for why a resume matches a job.
        
        Args:
            resume_text: Full resume text
            job_text: Full job description text
            similarity_score: Similarity score between resume and job
            resume_info: Structured resume information (optional)
            job_info: Structured job information (optional)
            
        Returns:
            Explanation text
        """
        if not self.client:
            return self._fallback_explanation(resume_info, job_info, similarity_score)
        
        prompt = f"""You are an expert career advisor and recruiter. Analyze the match between a candidate's resume and a job description.

Resume Summary:
{resume_text[:2000]}

Job Description:
{job_text[:2000]}

Similarity Score: {similarity_score:.2%}

Provide a detailed explanation (2-3 paragraphs) explaining:
1. Why this candidate is a good match for this position
2. Key strengths and relevant experience/skills
3. Any potential gaps or areas for improvement

Be specific and reference actual skills, experiences, or qualifications mentioned in both documents."""
        
        try:
            # Use chat_completion API (works with instruct/chat models)
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
                top_p=0.95
            )
            # Extract content from response
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating LLM explanation: {error_msg}")
            # Check for specific error types
            if "Model" in error_msg and ("loading" in error_msg.lower() or "not ready" in error_msg.lower()):
                return self._fallback_explanation(resume_info, job_info, similarity_score) + "\n\n(Note: Model is loading, please try again in a moment)"
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return self._fallback_explanation(resume_info, job_info, similarity_score) + "\n\n(Note: API rate limit reached, please try again later)"
            elif "401" in error_msg or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                return self._fallback_explanation(resume_info, job_info, similarity_score) + "\n\n(Note: Invalid API token. Please check your Hugging Face token)"
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return self._fallback_explanation(resume_info, job_info, similarity_score) + "\n\n(Note: Access denied. This model may require special access. Try a different model.)"
            else:
                return self._fallback_explanation(resume_info, job_info, similarity_score)
    
    def analyze_missing_skills(
        self,
        resume_skills: List[str],
        job_required_skills: List[str],
        job_preferred_skills: List[str]
    ) -> Dict[str, List[str]]:
        """
        Analyze missing skills and provide recommendations.
        
        Args:
            resume_skills: Skills found in resume
            job_required_skills: Required skills for the job
            job_preferred_skills: Preferred skills for the job
            
        Returns:
            Dictionary with missing_required, missing_preferred, and recommendations
        """
        resume_skills_lower = [s.lower() for s in resume_skills]
        job_required_lower = [s.lower() for s in job_required_skills]
        job_preferred_lower = [s.lower() for s in job_preferred_skills]
        
        missing_required = [s for s in job_required_skills if s.lower() not in resume_skills_lower]
        missing_preferred = [s for s in job_preferred_skills if s.lower() not in resume_skills_lower]
        
        # Generate recommendations
        recommendations = []
        if missing_required:
            recommendations.append(
                f"Consider highlighting or acquiring these required skills: {', '.join(missing_required[:5])}"
            )
        if missing_preferred:
            recommendations.append(
                f"These preferred skills could strengthen your application: {', '.join(missing_preferred[:5])}"
            )
        
        if not recommendations:
            recommendations.append("Your skills align well with the job requirements!")
        
        return {
            'missing_required': missing_required,
            'missing_preferred': missing_preferred,
            'recommendations': recommendations
        }
    
    def generate_improvement_suggestions(
        self,
        resume_text: str,
        job_text: str
    ) -> List[str]:
        """
        Generate suggestions to improve resume alignment with job.
        
        Args:
            resume_text: Resume text
            job_text: Job description text
            
        Returns:
            List of improvement suggestions
        """
        if not self.client:
            return self._fallback_suggestions()
        
        prompt = f"""You are a career coach. Review this resume against a job description and provide 3-5 specific, actionable suggestions to improve the resume's alignment with the job.

Resume:
{resume_text[:1500]}

Job Description:
{job_text[:1500]}

Provide suggestions as a numbered list. Be specific and actionable."""
        
        try:
            # Use chat_completion API
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.7,
                top_p=0.95
            )
            
            # Extract content from response
            suggestions_text = response.choices[0].message.content
            
            # Parse suggestions into list
            suggestions = [
                s.strip() 
                for s in suggestions_text.split('\n') 
                if s.strip() and (s.strip()[0].isdigit() or s.strip().startswith('-') or s.strip().startswith('•'))
            ]
            return suggestions[:5] if suggestions else self._fallback_suggestions()
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return self._fallback_suggestions()
    
    def _fallback_explanation(
        self,
        resume_info: Optional[Dict],
        job_info: Optional[Dict],
        similarity_score: float
    ) -> str:
        """Generate a fallback explanation without LLM."""
        explanation = f"Match Score: {similarity_score:.1%}\n\n"
        
        if resume_info and job_info:
            common_skills = set(resume_info.get('skills', [])) & set(
                job_info.get('required_skills', []) + job_info.get('preferred_skills', [])
            )
            if common_skills:
                explanation += f"Key Matching Skills: {', '.join(list(common_skills)[:5])}\n\n"
        
        if similarity_score > 0.7:
            explanation += "This is a strong match. The candidate's profile aligns well with the job requirements."
        elif similarity_score > 0.5:
            explanation += "This is a moderate match. There is good alignment with some areas for improvement."
        else:
            explanation += "This is a weaker match. Consider highlighting more relevant skills and experience."
        
        return explanation
    
    def _fallback_suggestions(self) -> List[str]:
        """Generate fallback suggestions without LLM."""
        return [
            "Highlight relevant skills and experience more prominently",
            "Use keywords from the job description in your resume",
            "Quantify achievements and results where possible",
            "Tailor your resume summary to match the job requirements",
            "Include any relevant certifications or training"
        ]

