# src/pdf_parser/parser.py

#region imports
import os
import glob
import json
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import sys

from pdf_parser._structures import WordBox, PageLayout
#endregion

#region functions

class PDFParser:
    def __init__(self):
        pass
    

    def parse(self,
        input_source: str,
        output_dir: str,
        output_format: str = "json",
        year: str = None,
        table_settings: dict = None
    ) -> list:
        """
        Unified method to process PDF input from either a single file, list of files, or directory.
        
        Args:
            input_source: Path to PDF file, list of PDF paths, or directory containing PDFs
            output_dir: Directory to save processed outputs
            output_format: Format to save outputs (default: "json") 
            year: Optional year to filter PDFs if input_source is directory
            table_settings: Optional dictionary of table extraction settings
        
        Returns:
            List of processing statistics/metadata for each PDF
        """
        # Handle single PDF file
        if isinstance(input_source, str) and input_source.lower().endswith('.pdf'):
            return [self._process_pdf(
                pdf_path=input_source,
                output_dir=output_dir, 
                output_format=output_format,
                table_settings=table_settings
            )]
        # Handle list of PDF files
        if isinstance(input_source, (list, tuple)):
            return self._process_pdfs(
                pdf_files=input_source,
                output_dir=output_dir,
                output_format=output_format, 
                table_settings=table_settings
            )
        # Handle directory of PDFs
        if isinstance(input_source, str) and os.path.isdir(input_source):
            return self._process_pdf_dir(
                input_dir=input_source,
                output_dir=output_dir,
                output_format=output_format,
                year=year,
                table_settings=table_settings
            )
        raise ValueError(
            "Input source must be a PDF file path, list of PDF paths, or directory path"
        )


    
    def _convert_output_format(out_base: str, output_format: str):
        """
        Convert the extracted text output to the requested format (json, markdown, txt).
        """
        txt_file = f"{out_base}_extracted.txt"
        json_file = f"{out_base}_extracted.json"
        md_file = f"{out_base}_extracted.md"
        if not os.path.exists(txt_file):
            print(f"Text file {txt_file} not found, skipping format conversion.")
            return
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        if output_format == "json":
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({"text": text}, f, indent=2, ensure_ascii=False)
        elif output_format == "markdown":
            with open(md_file, "w", encoding="utf-8") as f:
                f.write("# Extracted Text\n\n")
                f.write(text)
        elif output_format == "txt":
            pass  # Already in txt
        print(f"Output written in {output_format} format.")

    def _process_single_pdf(pdf_path, output_dir, filename, table_settings, output_format="json"):
        """
        Process a single PDF file with quality assessment, word box extraction, and layout analysis.
        Saves outputs to disk and returns processing statistics.
        Only writes the extracted text in the requested output_format.
        """
        import os
        from datetime import datetime

        # Set up output paths
        base_name = filename.rsplit('.', 1)[0]
        output_paths = {}
        if output_format == "txt":
            output_paths["text"] = os.path.join(output_dir, f"{base_name}_extracted.txt")
        elif output_format == "json":
            output_paths["text"] = os.path.join(output_dir, f"{base_name}_extracted.json")
        elif output_format == "markdown":
            output_paths["text"] = os.path.join(output_dir, f"{base_name}_extracted.md")
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        # Always save these
        metadata_file = os.path.join(output_dir, f"{base_name}_metadata.json")
        wordboxes_file = os.path.join(output_dir, f"{base_name}_wordboxes.json")
        layout_file = os.path.join(output_dir, f"{base_name}_layout.json")
        tables_dir = os.path.join(output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        # Optionally skip if already processed
        if all(os.path.exists(f) for f in [*output_paths.values(), metadata_file, wordboxes_file, layout_file]):
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)

        file_stats = {
            'filename': filename,
            'total_pages': 0,
            'pdfplumber_pages': 0,
            'ocr_pages': 0,
            'poor_quality_pages': 0,
            'total_word_boxes': 0,
            'processing_time': 0,
            'page_details': []
        }
        start_time = datetime.now()

        extracted_pages = []
        all_word_boxes = []
        all_page_layouts = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                file_stats['total_pages'] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    # --- Table extraction (optional) ---
                    # _extract_and_integrate_tables(pdf_path, tables_dir, page_num-1, table_settings)

                    # --- Page extraction ---
                    page_result = _extract_page_with_quality_check(page, page_num)
                    extracted_pages.append(page_result)
                    all_word_boxes.extend(page_result['word_boxes'])
                    all_page_layouts.append(page_result['page_layout'])
                    file_stats['page_details'].append(page_result['metadata'])

                    # Update stats
                    if page_result['metadata']['method'] == 'pdfplumber':
                        file_stats['pdfplumber_pages'] += 1
                    else:
                        file_stats['ocr_pages'] += 1
                    if page_result['metadata']['quality_flag']:
                        file_stats['poor_quality_pages'] += 1
                    file_stats['total_word_boxes'] += len(page_result['word_boxes'])

            # Save outputs
            _save_extracted_content(extracted_pages, output_paths["text"], metadata_file, file_stats, output_format)
            _save_word_boxes_and_layout(all_word_boxes, all_page_layouts, wordboxes_file, layout_file, file_stats)

        except Exception as e:
            file_stats['error'] = str(e)

        file_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        return file_stats

    def _process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        output_format: str = "json",
        table_settings: dict = None
    ) -> dict:
        """
        Process a single PDF file and save outputs in the specified format and directory.
        Returns statistics or metadata about the processing.
        """

        filename = os.path.basename(pdf_path)
        base_name = filename.rsplit('.', 1)[0]
        out_base = os.path.join(output_dir, base_name)

        # Use the main processing function to extract and save outputs
        file_stats = self._process_single_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            filename=filename,
            table_settings=table_settings,
            output_format=output_format
        )

        # Convert extracted text to the requested format (if not already in that format)
        self._convert_output_format(out_base, output_format)

        return file_stats

    def _process_pdfs(
        self,
        pdf_files: list,
        output_dir: str,
        output_format: str = "json",
        table_settings: dict = None
    ) -> list:
        """
        Process a list of PDF files.
        Returns a list of stats/metadata for each file.
        """
        results = []
        for pdf_path in pdf_files:
            result = self._process_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                output_format=output_format,
                table_settings=table_settings
            )
            results.append(result)
        return results

    def _process_pdf_dir(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = "json",
        year: str = None,
        table_settings: dict = None
    ) -> list:
        """
        Process all PDF files in a directory (optionally filter by year).
        Returns a list of stats/metadata for each file.
        """
        import glob, os
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        if year:
            pdf_files = [f for f in pdf_files if year in os.path.basename(f)]
        return self._process_pdfs(
            pdf_files=pdf_files,
            output_dir=output_dir,
            output_format=output_format,
            table_settings=table_settings
        )


#endregion

#region main
def _extract_page_with_quality_check(page, page_num):
    """
    Extract text and word boxes from a single page with quality assessment and OCR fallback.
    Returns a dict with keys: 'text', 'word_boxes', 'page_layout', 'metadata'.
    """
    # TODO: Implement extraction logic (pdfplumber, OCR fallback, layout analysis, etc.)
    raise NotImplementedError("Implement _extract_page_with_quality_check")

def _save_extracted_content(extracted_pages, output_file, metadata_file, file_stats, output_format):
    """
    Save extracted text and metadata with page-level granularity in the requested format.
    """
    # Save extracted text in the requested format
    if output_format == "txt":
        with open(output_file, 'w', encoding='utf-8') as f:
            for page_data in extracted_pages:
                f.write(page_data['text'])
                f.write("\n\n")
    elif output_format == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "pages": [
                        {
                            "page_number": page_data['metadata']['page_number'],
                            "text": page_data['text']
                        }
                        for page_data in extracted_pages
                    ]
                },
                f, indent=2, ensure_ascii=False
            )
    elif output_format == "markdown":
        with open(output_file, 'w', encoding='utf-8') as f:
            for page_data in extracted_pages:
                page_num = page_data['metadata']['page_number']
                f.write(f"# Page {page_num}\n\n")
                f.write(page_data['text'])
                f.write("\n\n")
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    # Save metadata as JSON
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(file_stats, f, indent=2, ensure_ascii=False)

def _save_word_boxes_and_layout(all_word_boxes, all_page_layouts, wordboxes_file, layout_file, file_stats):
    """
    Save word boxes and layout data as JSON files.
    """
    # TODO: Implement saving logic for word boxes and layout
    raise NotImplementedError("Implement _save_word_boxes_and_layout")

def _extract_with_ocr(page):
    """
    Extract text from a PDF page using Tesseract OCR as a fallback.
    Returns the extracted text as a string.
    """
    try:
        # Convert page to image (high resolution for better OCR)
        page_image = page.to_image(resolution=300)
        pil_image = page_image.original

        # Use pytesseract for OCR
        text = pytesseract.image_to_string(pil_image, lang='eng')
        return text.strip()
    except Exception as e:
        print(f"      ❌ OCR failed: {e}")
        return ""

def _calculate_quality_score(metrics, method):
    """
    Calculate a quality score (0-100) for the extracted text.
    """
    score = 0

    # Character count scoring (0-30 points)
    char_count = metrics.get('char_count', 0)
    if 200 <= char_count <= 5000:
        score += 30
    elif 100 <= char_count < 200 or 5000 < char_count <= 8000:
        score += 20
    elif 50 <= char_count < 100 or 8000 < char_count <= 10000:
        score += 10

    # Word count scoring (0-25 points)
    word_count = metrics.get('word_count', 0)
    if 50 <= word_count <= 1000:
        score += 25
    elif 25 <= word_count < 50 or 1000 < word_count <= 1500:
        score += 15
    elif 10 <= word_count < 25:
        score += 10

    # Line count scoring (0-20 points)
    line_count = metrics.get('line_count', 0)
    if 10 <= line_count <= 100:
        score += 20
    elif 5 <= line_count < 10 or 100 < line_count <= 150:
        score += 15
    elif 2 <= line_count < 5:
        score += 10

    # Method bonus (0-25 points)
    if method == 'pdfplumber':
        score += 25
    else:  # tesseract or other OCR
        score += 15

    return min(score, 100)

def _analyze_page_layout(word_boxes, page_width, page_height):
    """
    Analyze page layout and determine reading order with comprehensive metrics.
    Returns a dictionary with layout analysis.
    """
    if not word_boxes:
        return {
            'columns': 0,
            'rows': 0,
            'text_density': 0,
            'layout_type': 'empty',
            'reading_order': [],
            'avg_font_size': 0,
            'font_size_variance': 0,
            'aspect_ratio': 0,
            'text_flow_analysis': {}
        }

    # Calculate text density
    total_text_area = sum(box.area for box in word_boxes)
    page_area = page_width * page_height
    text_density = total_text_area / page_area if page_area > 0 else 0

    # Determine layout type based on word distribution
    x_positions = [box.center_x for box in word_boxes]
    y_positions = [box.center_y for box in word_boxes]

    # Simple column detection
    x_sorted = sorted(set(x_positions))
    column_gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
    avg_gap = sum(column_gaps) / len(column_gaps) if column_gaps else 0

    # Estimate number of columns
    estimated_columns = max(1, int(page_width / (avg_gap + 50)) if avg_gap > 0 else 1)

    # Determine reading order
    reading_order = _determine_reading_order(word_boxes, estimated_columns)

    # Classify layout type
    layout_type = _classify_layout_type(word_boxes, estimated_columns, text_density)

    # Calculate additional metrics
    font_sizes = [box.fontsize for box in word_boxes if box.fontsize is not None]
    avg_font_size = np.mean(font_sizes) if font_sizes else 0
    font_size_variance = np.var(font_sizes) if len(font_sizes) > 1 else 0

    x_spread = max(x_positions) - min(x_positions) if x_positions else 0
    y_spread = max(y_positions) - min(y_positions) if y_positions else 0
    aspect_ratio = x_spread / y_spread if y_spread > 0 else 0

    return {
        'columns': estimated_columns,
        'rows': len(set(y_positions)),
        'text_density': text_density,
        'layout_type': layout_type,
        'reading_order': reading_order,
        'avg_font_size': avg_font_size,
        'font_size_variance': font_size_variance,
        'aspect_ratio': aspect_ratio,
        'text_flow_analysis': {
            'x_spread': x_spread,
            'y_spread': y_spread,
            'word_count': len(word_boxes),
            'unique_x_positions': len(set(x_positions)),
            'unique_y_positions': len(set(y_positions))
        }
    }

def _extract_word_boxes_with_layout(page, page_num, method='pdfplumber'):
    """
    Extract word boxes with comprehensive layout analysis including document positioning.
    Returns (word_boxes, text_blocks).
    """
    word_boxes = []
    text_blocks = []

    try:
        if method == 'pdfplumber':
            # Try basic word extraction
            words = page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False
            )
            for idx, word in enumerate(words):
                if not all(attr in word for attr in ['text', 'x0', 'y0', 'x1', 'y1']):
                    continue
                word_box = WordBox(
                    text=word.get('text', ''),
                    x0=float(word.get('x0', 0)),
                    y0=float(word.get('y0', 0)),
                    x1=float(word.get('x1', 0)),
                    y1=float(word.get('y1', 0)),
                    width=float(word.get('x1', 0)) - float(word.get('x0', 0)),
                    height=float(word.get('y1', 0)) - float(word.get('y0', 0)),
                    fontname=word.get('fontname'),
                    fontsize=word.get('size'),
                    fontcolor=word.get('fontcolor'),
                    doctop=word.get('doctop'),
                    upright=word.get('upright'),
                    top=word.get('top'),
                    bottom=word.get('bottom'),
                    left=word.get('left'),
                    right=word.get('right'),
                    page_number=page_num,
                    word_index=idx
                )
                word_boxes.append(word_box)
            try:
                text_blocks = page.extract_text_simple()
            except Exception:
                text_blocks = []
        else:  # OCR method
            # You can implement a more detailed OCR word box extraction if needed
            word_boxes = []  # Placeholder for OCR word box extraction
            text_blocks = []
    except Exception as e:
        print(f"      ❌ Word box extraction failed for page {page_num}: {e}")
        return [], []

    return word_boxes, text_blocks

def _analyze_document_layout(page_layouts):
    """
    Analyze overall document layout across all pages.
    Returns a dictionary with document-level layout statistics.
    """
    import numpy as np

    if not page_layouts:
        return {'document_type': 'empty', 'analysis': {}}

    # Collect statistics across all pages
    layout_types = [layout.layout_type for layout in page_layouts]
    column_counts = [layout.estimated_columns for layout in page_layouts]
    text_densities = [layout.text_density for layout in page_layouts]
    font_sizes = [layout.average_font_size for layout in page_layouts if layout.average_font_size > 0]

    # Analyze document characteristics
    most_common_layout = max(set(layout_types), key=layout_types.count) if layout_types else 'unknown'
    avg_columns = np.mean(column_counts) if column_counts else 1
    avg_text_density = np.mean(text_densities) if text_densities else 0
    avg_font_size = np.mean(font_sizes) if font_sizes else 0

    # Determine document type
    if most_common_layout in ['single_column', 'narrow_single_column']:
        document_type = 'single_column_document'
    elif most_common_layout in ['two_column', 'mixed_formatting_two_column']:
        document_type = 'two_column_document'
    elif most_common_layout in ['multi_column', 'dense_multi_column']:
        document_type = 'multi_column_document'
    else:
        document_type = 'mixed_layout_document'

    return {
        'document_type': document_type,
        'most_common_layout': most_common_layout,
        'average_columns': avg_columns,
        'average_text_density': avg_text_density,
        'average_font_size': avg_font_size,
        'layout_distribution': {layout: layout_types.count(layout) for layout in set(layout_types)},
        'column_distribution': {cols: column_counts.count(cols) for cols in set(column_counts)},
        'total_pages': len(page_layouts),
        'pages_with_content': len([layout for layout in page_layouts if layout.word_boxes])
    }

def _determine_reading_order(word_boxes, estimated_columns):
    """
    Determine reading order of words (top-to-bottom, left-to-right), optionally column-aware.
    Returns a list of word indices in reading order.
    """
    if not word_boxes:
        return []

    # Simple top-to-bottom, left-to-right sorting
    def simple_reading_order():
        sorted_boxes = sorted(word_boxes, key=lambda box: (box.y0, box.x0))
        return [box.word_index for box in sorted_boxes]

    # Column-aware reading order
    def column_aware_reading_order():
        if estimated_columns <= 1:
            return simple_reading_order()
        page_width = max(box.x1 for box in word_boxes) if word_boxes else 0
        column_width = page_width / estimated_columns
        column_groups = [[] for _ in range(estimated_columns)]
        for box in word_boxes:
            column_idx = min(int(box.center_x / column_width), estimated_columns - 1)
            column_groups[column_idx].append(box)
        reading_order = []
        for column in column_groups:
            column_sorted = sorted(column, key=lambda box: box.y0)
            reading_order.extend([box.word_index for box in column_sorted])
        return reading_order

    # Advanced reading order with line detection (optional, not used by default)
    # def advanced_reading_order():
    #     ...

    if estimated_columns > 1:
        return column_aware_reading_order()
    else:
        return simple_reading_order()

def _classify_layout_type(word_boxes, estimated_columns, text_density):
    """
    Classify the layout type based on word distribution and density.
    """
    if not word_boxes:
        return 'empty'

    # Analyze word distribution
    x_positions = [box.center_x for box in word_boxes]
    y_positions = [box.center_y for box in word_boxes]

    # Calculate spreads and statistics
    x_spread = max(x_positions) - min(x_positions) if x_positions else 0
    y_spread = max(y_positions) - min(y_positions) if y_positions else 0
    aspect_ratio = x_spread / y_spread if y_spread > 0 else 0

    # Analyze font size distribution
    font_sizes = [box.fontsize for box in word_boxes if box.fontsize is not None]
    avg_font_size = np.mean(font_sizes) if font_sizes else 12
    font_size_variance = np.var(font_sizes) if len(font_sizes) > 1 else 0

    # Analyze text density patterns
    density_thresholds = {
        'very_sparse': 0.05,
        'sparse': 0.15,
        'normal': 0.35,
        'dense': 0.55,
        'very_dense': 0.75
    }

    # Classify based on multiple criteria
    if text_density < density_thresholds['very_sparse']:
        return 'very_sparse'
    elif text_density < density_thresholds['sparse']:
        return 'sparse'
    elif estimated_columns == 1:
        if aspect_ratio < 0.3:
            return 'narrow_single_column'
        elif font_size_variance > 50:  # High variance in font sizes
            return 'mixed_formatting_single_column'
        else:
            return 'single_column'
    elif estimated_columns == 2:
        if font_size_variance > 50:
            return 'mixed_formatting_two_column'
        else:
            return 'two_column'
    elif estimated_columns >= 3:
        if text_density > density_thresholds['dense']:
            return 'dense_multi_column'
        else:
            return 'multi_column'
    elif x_spread < y_spread * 0.4:
        return 'narrow_column'
    elif font_size_variance > 100:  # Very high variance
        return 'complex_mixed_layout'
    elif text_density > density_thresholds['very_dense']:
        return 'very_dense_layout'
    else:
        return 'mixed_layout'
#endregion