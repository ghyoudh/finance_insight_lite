import fitz  # PyMuPDF
import os

def pdf_to_markdown(pdf_path, output_dir="data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    md_text = ""

    for page in doc:
        text = page.get_text("text")
        md_text += text + "\n\n"

    file_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    save_path = os.path.join(output_dir, file_name)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Markdown file saved to: {save_path}")
    return md_text
