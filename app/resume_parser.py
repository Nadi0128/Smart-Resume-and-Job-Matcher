from pypdf import PdfReader


def extract_resume_text(pdf_file):
    """
    Extract text from a PDF resume file.
    """
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip()

if __name__ == "__main__":
    with open("data/sample_resumes/test.pdf", "rb") as f:
        text = extract_resume_text(f)

    print("Texte extrait du fichier pdf ")
    print(text[:1000])
