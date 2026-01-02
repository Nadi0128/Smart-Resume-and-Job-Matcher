def parse_job_description(job_text: str) -> str:
    """
    Clean and normalize job description text.
    """
    job_text = job_text.strip()
    job_text = job_text.replace("\n", " ")
    return job_text
