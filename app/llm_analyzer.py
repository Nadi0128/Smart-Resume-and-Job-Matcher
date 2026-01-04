from typing import Dict, List, Optional
import os

try:
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older langchain versions
        from langchain.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False


class LLMAnalyzer:
    """
    LLM-based analyzer for generating explanations and insights
    about resume-job matches.
    """
    
    def __init__(self, model_name: str = "llama2", base_url: Optional[str] = None):
        """
        Initialize LLM analyzer.
        
        Args:
            model_name: Name of the Ollama model to use (default: "llama2")
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name
        
        if not LANGCHAIN_AVAILABLE:
            print("Warning: LangChain not available. Install with: pip install langchain langchain-community")
            self.llm = None
        else:
            try:
                self.llm = Ollama(model=model_name, base_url=self.base_url)
            except Exception as e:
                print(f"Warning: Could not initialize Ollama. Using fallback mode. Error: {e}")
                print("Make sure Ollama is running: ollama serve")
                self.llm = None
    
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
        if not self.llm:
            return self._fallback_explanation(resume_info, job_info, similarity_score)
        
        prompt_template = PromptTemplate(
            input_variables=["resume_text", "job_text", "similarity_score"],
            template="""
You are an expert career advisor and recruiter. Analyze the match between a candidate's resume and a job description.

Resume Summary:
{resume_text}

Job Description:
{job_text}

Similarity Score: {similarity_score:.2%}

Provide a detailed explanation (2-3 paragraphs) explaining:
1. Why this candidate is a good match for this position
2. Key strengths and relevant experience/skills
3. Any potential gaps or areas for improvement

Be specific and reference actual skills, experiences, or qualifications mentioned in both documents.
"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            explanation = chain.run(
                resume_text=resume_text[:2000],  # Limit length
                job_text=job_text[:2000],
                similarity_score=similarity_score
            )
            return explanation.strip()
        except Exception as e:
            print(f"Error generating LLM explanation: {e}")
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
        if not self.llm:
            return self._fallback_suggestions()
        
        prompt_template = PromptTemplate(
            input_variables=["resume_text", "job_text"],
            template="""
You are a career coach. Review this resume against a job description and provide 3-5 specific, actionable suggestions to improve the resume's alignment with the job.

Resume:
{resume_text}

Job Description:
{job_text}

Provide suggestions as a numbered list. Be specific and actionable.
"""
        )
        
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            suggestions_text = chain.run(
                resume_text=resume_text[:1500],
                job_text=job_text[:1500]
            )
            
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

