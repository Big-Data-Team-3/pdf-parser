import os
import shutil
from pdf_parser import PDFParser

def test_parse_single_pdf(tmp_path):
    # Setup: create output directory and use a sample PDF path
    output_dir = tmp_path / "parsed"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Replace this with a real PDF path in your test data
    sample_pdf = "data/raw/MSFT/10-K/PDFs/MSFT_10-K_20230727_000095017023035122.pdf"
    if not os.path.exists(sample_pdf):
        print(f"Test PDF not found: {sample_pdf}")
        return

    parser = PDFParser(table_extractor="tabula")
    stats = parser.parse(
        input_source=sample_pdf,
        output_dir=str(output_dir),
        output_format="json"
    )
    # Check that output files exist
    base_name = os.path.splitext(os.path.basename(sample_pdf))[0]
    expected_json = output_dir / f"{base_name}_extracted.json"
    expected_metadata = output_dir / f"{base_name}_metadata.json"
    expected_wordboxes = output_dir / f"{base_name}_wordboxes.json"
    expected_layout = output_dir / f"{base_name}_layout.json"
    assert expected_json.exists(), "Extracted JSON file not found"
    assert expected_metadata.exists(), "Metadata file not found"
    assert expected_wordboxes.exists(), "Wordboxes file not found"
    assert expected_layout.exists(), "Layout file not found"
    print("Single PDF parse test passed.")

def test_parse_directory(tmp_path):
    # Setup: create output directory and use a sample PDF directory
    output_dir = tmp_path / "parsed_dir"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_pdf_dir = "data/raw/MSFT/10-K/PDFs"
    if not os.path.isdir(sample_pdf_dir):
        print(f"Test PDF directory not found: {sample_pdf_dir}")
        return

    parser = PDFParser(table_extractor="tabula")
    stats = parser.parse(
        input_source=sample_pdf_dir,
        output_dir=str(output_dir),
        output_format="txt"
    )
    # Check that at least one output file exists
    found = any(f.name.endswith("_extracted.txt") for f in output_dir.iterdir())
    assert found, "No extracted TXT files found in output directory"
    print("Directory parse test passed.")

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_parse_single_pdf(tmp_path=tmpdir)
        test_parse_directory(tmp_path=tmpdir)
