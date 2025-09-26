# PDF Parser Python API

This document describes the **API-level usage and logic** for the `pdf_parser` library, focusing on the main `PDFParser` class and its options. This is intended for both users and developers.

---

## Overview

The `pdf_parser` library provides a unified, programmatic interface for extracting structured text, word boxes, and layout information from PDF files. It supports batch processing, multiple output formats, and is designed for integration with DVC and other data pipelines.

---

## Main API: `PDFParser`

### Class Signature

```python
from pdf_parser import PDFParser

class PDFParser:
    def __init__(self, table_settings: dict = None):
        ...
    def parse(
        self,
        input_source: str | list[str],
        output_dir: str,
        output_format: str = "json",
        year: str = None,
        table_settings: dict = None
    ) -> list[dict]:
        """
        Unified method to process PDF input from either a single file, list of files, or directory.

        Args:
            input_source: Path to a PDF file, a list of PDF file paths, or a directory containing PDFs.
            output_dir: Directory to save processed outputs.
            output_format: Format to save outputs ("json", "txt", or "markdown"). Default is "json".
            year: Optional year to filter PDFs if input_source is a directory.
            table_settings: Optional dictionary of table extraction settings.

        Returns:
            List of processing statistics/metadata for each PDF.
        """
```

---

## Usage Examples

### 1. Process a Single PDF

```python
from pdf_parser import PDFParser

parser = PDFParser()
stats = parser.parse(
    input_source="data/raw/MSFT/10-K/PDFs/MSFT_10-K_20230727_000095017023035122.pdf",
    output_dir="data/parsed/MSFT/2023",
    output_format="json"
)
```

### 2. Process a List of PDFs

```python
from pdf_parser import PDFParser

pdf_files = [
    "data/raw/MSFT/10-K/PDFs/file1.pdf",
    "data/raw/MSFT/10-K/PDFs/file2.pdf"
]
parser = PDFParser()
stats = parser.parse(
    input_source=pdf_files,
    output_dir="data/parsed/MSFT/2023",
    output_format="markdown"
)
```

### 3. Process All PDFs in a Directory (optionally filter by year)

```python
from pdf_parser import PDFParser

parser = PDFParser()
stats = parser.parse(
    input_source="data/raw/MSFT/10-K/PDFs",
    output_dir="data/parsed/MSFT/2023",
    output_format="txt",
    year="2023"
)
```

---

## Output Files

For each PDF, the following files may be generated in the output directory:

- `*_extracted.txt` / `*_extracted.json` / `*_extracted.md`: Extracted text in the requested format.
- `*_metadata.json`: Extraction and quality metadata.
- `*_wordboxes.json`: Word-level bounding box data.
- `*_layout.json`: Page and document layout information.

---

## Internal Logic Flow

- The `parse` method determines the type of `input_source` (file, list, or directory).
- It dispatches to the appropriate internal batch or single-file processing method.
- For each PDF:
    - The file is opened and processed page by page.
    - Text is extracted using `pdfplumber` (with OCR fallback if needed).
    - Word boxes and layout are extracted and analyzed.
    - Outputs are saved in the requested format.
    - Metadata and statistics are returned for each file.

---

## Customization

- **Output Format:** Choose between `"json"`, `"txt"`, or `"markdown"` for the main extracted text.
- **Table Extraction:** Pass custom `table_settings` if you want to tweak table extraction (see code for options).
- **Year Filtering:** When processing a directory, use the `year` argument to filter files by year in the filename.

---

## Extensibility

- The parser is modular: you can extend or swap out internal methods for custom extraction, layout analysis, or evaluation.
- For advanced evaluation (e.g., WER against XBRL), see the internal `_evaluation.py` module.
- All helpers are private methods of the `PDFParser` class and are not part of the public API.

---

## Example: DVC Integration

You can use this parser in a DVC pipeline step by calling the `parse` method from a script, ensuring reproducible, tracked outputs.

---

## Notes

- Only the `parse` method of the `PDFParser` class is considered public API. All other methods are internal and may change.
- The parser is designed for batch, script, and programmatic use—not for interactive CLI use.

---
