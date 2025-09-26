import argparse
from docling.document_converter import DocumentConverter
import os

def convert_from_docling(pdf_src):
    converter = DocumentConverter()
    result = converter.convert(pdf_src)    # returns a result with document

    doc = result.document
    return doc

def save_from_docling(doc, format, path, filename):
    if(format=="md"):
        md = doc.export_to_markdown()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output_file = os.path.join(path, filename)
        with open(output_file, "w") as f:
            f.write(md)
    elif(format=="json"):
        output_file = os.path.join(path, filename)
        js = doc.save_as_json(output_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(js)
    else:
        raise ValueError(f"Unsupported format: {format}")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Convert PDF to markdown and JSON using Docling")
    parser.add_argument("--input-dir", required=True, help="Input directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Output directory for markdown and JSON files")
    args = parser.parse_args()

    for file in os.listdir(args.input_dir):
        if file.endswith(".pdf"):
            doc = convert_from_docling(os.path.join(args.input_dir, file))
            save_from_docling(doc, "md", args.output_dir, file.replace(".pdf", ".md"))
            save_from_docling(doc, "json", args.output_dir, file.replace(".pdf", ".json"))
    print("Conversion complete")