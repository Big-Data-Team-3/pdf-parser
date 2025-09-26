import argparse
from docling.document_converter import DocumentConverter
import os

def convert_from_docling(pdf_src):
    converter = DocumentConverter()
    result = converter.convert(pdf_src)    # returns a result with document

    doc = result.document
    return doc

def save_from_docling(doc, format, target_folder, filename):
    if(format=="md"):
        md = doc.export_to_markdown()
        os.makedirs(os.path.join(target_folder, 'docling'), exist_ok=True)
        output_file = os.path.join(target_folder, 'docling', filename)
        with open(output_file, "w") as f:
            f.write(md)
    elif(format=="json"):
        os.makedirs(os.path.join(target_folder, 'docling'), exist_ok=True)
        output_file = os.path.join(target_folder, 'docling', filename)
        doc.save_as_json(output_file)  # Assuming this writes directly to the file
    else:
        raise ValueError(f"Unsupported format: {format}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Convert PDF to markdown and JSON using Docling")
    parser.add_argument("--input-dir", required=True, help="Input directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Output directory for markdown and JSON files")
    args = parser.parse_args()

    for file in os.listdir(args.input_dir):
        if file.endswith(".pdf"):
            base_name = os.path.splitext(file)[0]
            target_folder = os.path.join(args.output_dir, base_name)
            doc = convert_from_docling(os.path.join(args.input_dir, file))
            save_from_docling(doc, "md", target_folder, file.replace(".pdf", ".md"))
            save_from_docling(doc, "json", target_folder, file.replace(".pdf", ".json"))
    print("Conversion complete")