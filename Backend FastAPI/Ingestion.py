import fitz


def extract_text_from_PDF(file_bytes: bytes) -> str:
    """Pull text from pdf using fitz into raw text"""
    doc = fitz.open(file_bytes, filetype="pdf")
    text = ""
    
    for page in doc:
        text += page.get_text()
    return text

def break_down_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    
    while 0 < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        start += chunk_size + overlap
        
    return chunks
    