import re
from pypdf import PdfReader

def clean_page(text):
    # 1. Split into lines
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            continue
        # Skip running header
        if "Francisco Cândido Xavier - Nosso Lar" in stripped:
            continue
        # Skip page numbers (just digits)
        if stripped.isdigit():
            continue
        cleaned_lines.append(line)
        
    # Reassemble cleaned lines
    cleaned_text = "\n".join(cleaned_lines)
    
    # 2. Reassemble hyphenated words at line breaks (e.g. apren-\ndizes -> aprendizes)
    cleaned_text = re.sub(r'-\s*\n\s*', '', cleaned_text)
    
    # 3. Reconstruct paragraphs
    # Paragraph breaks are marked by punctuation (. or ? or !) followed by spaces and a newline.
    # We replace those newlines with a special token, then replace all other single newlines with spaces,
    # and finally replace the special token with double newlines.
    
    # Temporarily replace paragraph-end newlines with <PARA_BREAK>
    # We match . or ? or ! followed by optional spaces, then newline
    cleaned_text = re.sub(r'([.?!])\s*\n', r'\1<PARA_BREAK>', cleaned_text)
    
    # Replace all remaining newlines with space
    cleaned_text = cleaned_text.replace('\n', ' ')
    
    # Replace double spaces with single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Restore paragraph breaks
    cleaned_text = cleaned_text.replace('<PARA_BREAK>', '\n\n')
    
    # Split into actual paragraphs
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    return paragraphs

# Test on page 5
reader = PdfReader("book.pdf")
page_text = reader.pages[5].extract_text()
paragraphs = clean_page(page_text)

print(f"--- PAGE 5 PARAGRAPHS ({len(paragraphs)} found) ---")
for i, p in enumerate(paragraphs[:5], 1):
    print(f"[{i}]: {p}\n")
