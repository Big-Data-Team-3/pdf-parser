# src/pdf_parser/parser.py

#region imports
import os
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse
import sys
import tabula
import hashlib

from pdf_parser._structures import WordBox, PageLayout
#endregion

#region functions

class PDFParser:
    def __init__(self, table_extractor="pdfplumber", table_settings=None):
        """
        Args:
            table_extractor: "pdfplumber", "tabula", or "both"
            table_settings: dict of settings for table extraction (passed to tabula or pdfplumber)
        """
        self.table_extractor = table_extractor
        self.table_settings = table_settings or {}

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
    
    def _stats(self):
        '''
        Generate performance metrics and stats for the output of the parser from the PDF file.
        '''
        return {
            'total_pages': 0,
            'total_tables': 0,
            'pdfplumber_pages': 0,
            'ocr_pages': 0,
            'poor_quality_pages': 0,
            'total_word_boxes': 0,
            'processing_time': 0,
            'page_details': []
        }
    

    
    @staticmethod
    def _convert_output_format(out_base: str, output_format: str):
        """
        Convert the extracted text output to the requested format (json, markdown, txt).
        """

        pass

    def _process_single_pdf(self, pdf_path, output_dir, filename, table_settings, output_format="json", force=False):
        """
        Process a single PDF file with quality assessment, word box extraction, and layout analysis.
        Saves outputs to disk and returns processing statistics.
        Only writes the extracted text in the requested output_format.
        """
        

        # Set up output paths
        base_name = filename.rsplit('.', 1)[0]
        pdf_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        output_paths = {}
        if output_format == "txt":
            output_paths["text"] = os.path.join(output_dir, f"{base_name}_extracted.txt")
        elif output_format == "json":
            output_paths["text"] = os.path.join(pdf_output_dir, f"{base_name}_extracted.json")
        elif output_format == "markdown":
            output_paths["text"] = os.path.join(output_dir, f"{base_name}_extracted.md")
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        # Always save these
        metadata_file = os.path.join(pdf_output_dir, f"{base_name}_metadata.json")
        wordboxes_file = os.path.join(pdf_output_dir, f"{base_name}_wordboxes.json")
        layout_file = os.path.join(pdf_output_dir, f"{base_name}_layout.json")
        tables_dir = os.path.join(pdf_output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

        # --- Table extraction ---
        if self.table_extractor in ("tabula", "both"):
            self._extract_tables_with_tabula(pdf_path, tables_dir)
        if self.table_extractor in ("pdfplumber", "both"):
            self._extract_tables_with_pdfplumber(pdf_path, tables_dir, table_settings)

        # # Optionally skip if already processed
        # if all(os.path.exists(f) for f in [*output_paths.values(), metadata_file, wordboxes_file, layout_file]):
        #     with open(metadata_file, "r", encoding="utf-8") as f:
        #         return json.load(f)

        # --- Text extraction ---
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
        print(f"--------------------------------")
        print(f"Starting text extraction...")
        extracted_pages = []
        all_word_boxes = []
        all_page_layouts = []
        print(f"--------------------------------")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                file_stats['total_pages'] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    # --- Table extraction (optional) ---
                    # _extract_and_integrate_tables(pdf_path, tables_dir, page_num-1, table_settings)

                    # --- Page extraction ---
                    try:
                        page_result = self._extract_page_with_quality_check(page, page_num)
                    except Exception as e:
                        print(f"      ❌ Page {page_num}: {e}")
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
            self._save_txt_file(extracted_pages, pdf_output_dir, file_stats)
            self._save_extracted_content(extracted_pages, output_paths["text"], metadata_file, file_stats, output_format)
            self._save_word_boxes_and_layout(all_word_boxes, all_page_layouts, wordboxes_file, layout_file, file_stats)

        except Exception as e:
            file_stats['error'] = str(e)

        file_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        return file_stats


    def _save_txt_file(self, extracted_pages, output_dir, file_stats):
        txt_output_path = os.path.join(output_dir, "extracted.txt")
        with open(txt_output_path, 'w', encoding='utf-8') as f:

            f.write(f"# Extracted Text from {file_stats['filename']}\n")
            f.write(f"# Processing Date: {datetime.now().isoformat()}\n")
            f.write(f"# Total Pages: {file_stats['total_pages']}\n")
            f.write(f"# PDFplumber Pages: {file_stats['pdfplumber_pages']}\n")
            f.write(f"# OCR Pages: {file_stats['ocr_pages']}\n")
            f.write(f"# Poor Quality Pages: {file_stats['poor_quality_pages']}\n\n")
            
            for page_data in extracted_pages:
                page_num = page_data['metadata']['page_number']
                method = page_data['metadata']['method']
                quality_score = page_data['metadata']['quality_score']
                
                f.write(f"\n{'='*80}\n")
                f.write(f"PAGE {page_num} | Method: {method} | Quality Score: {quality_score}\n")
                f.write(f"{'='*80}\n\n")
                f.write(page_data['text'])
                f.write(f"\n\n")

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
        pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
        if year:
            pdf_files = [f for f in pdf_files if year in os.path.basename(f)]
        return self._process_pdfs(
            pdf_files=pdf_files,
            output_dir=output_dir,
            output_format=output_format,
            table_settings=table_settings
        )

    def _extract_tables_with_tabula(self, pdf_path, tables_dir):
        """
        Extract tables from a PDF using tabula-py, save as CSVs, and generate metadata.
        """
        import tabula
        import hashlib
        import json
        import os
        from datetime import datetime
        import pandas as pd

        def compute_file_hash(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()

        def compute_table_id(file_id, page_number, table_index, mode, params_hash):
            content = f"{file_id}_{page_number}_{table_index}_{mode}_{params_hash}"
            return hashlib.sha256(content.encode()).hexdigest()

        def is_valid_table(df):
            if df is None or df.empty:
                return False, "empty_table"
            if df.shape[1] <= 1:
                return False, "single_column"
            has_numbers = df.map(lambda x: str(x).replace(".", "", 1).isdigit()).any().any()
            if has_numbers:
                return True, "has_numbers"
            elif df.shape[1] > 1:
                return True, "multi_column"
            else:
                return False, "no_numbers_single_column"

        os.makedirs(tables_dir, exist_ok=True)
        metadata_dir = os.path.join(tables_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        print(f"--------------------------------")
        print(f"Extracting tables with tabula from {pdf_path}...")
        print(f"Using stream mode...")
        start_time = datetime.now()
        tables = tabula.read_pdf(
            pdf_path,
            pages="all",
            multiple_tables=True,
            stream=True
        )
        end_time = datetime.now()
        print(f"Time taken to extract tables: {end_time - start_time}")
        table_extraction_time = (end_time - start_time).total_seconds()
        file_id = compute_file_hash(pdf_path)
        output_files = []
        tables_metadata = []
        valid_table_count = 0

        for i, table in enumerate(tables, start=1):
            is_valid, validation_reason = is_valid_table(table)
            csv_path = os.path.join(tables_dir, f"tabula_table_{i}.csv") if is_valid else None
            if is_valid:
                table.to_csv(csv_path, index=False)
                output_files.append(csv_path)
                valid_table_count += 1

            # Table metadata
            params_hash = hashlib.sha256("stream_mode".encode()).hexdigest()[:8]
            table_id = compute_table_id(file_id, i, i, "stream", params_hash)
            csv_hash = compute_file_hash(csv_path) if csv_path and os.path.exists(csv_path) else None
            num_rows = len(table) if table is not None else 0
            num_columns = len(table.columns) if table is not None and not table.empty else 0
            numeric_cell_ratio = 0.0
            empty_cell_ratio = 0.0
            if table is not None and not table.empty:
                total_cells = num_rows * num_columns
                if total_cells > 0:
                    numeric_cells = table.map(lambda x: str(x).replace(".", "", 1).isdigit()).sum().sum()
                    numeric_cell_ratio = numeric_cells / total_cells
                    empty_cells = table.isnull().sum().sum()
                    empty_cell_ratio = empty_cells / total_cells

            table_meta = {
                "table_id": table_id,
                "file_id": file_id,
                "table_index": i,
                "csv_path": csv_path,
                "csv_sha256": csv_hash,
                "page_number": i,  # Tabula doesn't always provide page info
                "num_rows": num_rows,
                "num_columns": num_columns,
                "is_valid_table": is_valid,
                "validation_reason": validation_reason,
                "numeric_cell_ratio": round(numeric_cell_ratio, 3),
                "empty_cell_ratio": round(empty_cell_ratio, 3),
                "extraction_timestamp": datetime.now(timezone.utc).isoformat() + "Z"
            }
            tables_metadata.append(table_meta)

        # File-level metadata
        file_metadata = {
            "file_id": file_id,
            "source_path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "file_size_bytes": os.path.getsize(pdf_path),
            "extraction_timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "tabula_version": getattr(tabula, '__version__', 'unknown'),
            "mode": "stream",
            "tables_found": len(tables),
            "tables_valid": valid_table_count,
            "output_dir": tables_dir,
            "processing_status": "success"
        }

        # Save metadata files
        with open(os.path.join(metadata_dir, "files.jsonl"), "w") as f:
            f.write(json.dumps(file_metadata) + "\n")
        with open(os.path.join(metadata_dir, "tables.jsonl"), "w") as f:
            for table_meta in tables_metadata:
                f.write(json.dumps(table_meta) + "\n")

    def _extract_tables_with_pdfplumber(self, pdf_path, tables_dir, table_settings):
        """
        Extract tables from a PDF using pdfplumber and save as CSVs.
        """
        import pdfplumber
        import pandas as pd

        os.makedirs(tables_dir, exist_ok=True)
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables(table_settings or {})
                for idx, table in enumerate(tables):
                    if not table or not table[0]:
                        continue
                    df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                    csv_path = os.path.join(tables_dir, f"pdfplumber_table_{page_num+1}_{idx+1}.csv")
                    df.to_csv(csv_path, index=False)

    def _extract_page_with_quality_check(self, page, page_num):
        """
        Extract text and word boxes from a single page with quality assessment and OCR fallback.
        Returns a dict with keys: 'text', 'word_boxes', 'page_layout', 'metadata'.
        """
        # Try pdfplumber first
        text = page.extract_text()
        
        # Quality assessment metrics
        char_count = len(text) if text else 0
        word_count = len(text.split()) if text else 0
        line_count = len(text.split('\n')) if text else 0
        
        # Quality thresholds (adjust based on your needs)
        min_chars_per_page = 100
        max_chars_per_page = 10000
        min_words_per_page = 20
        
        quality_metrics = {
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'char_density': char_count / (page.width * page.height) if hasattr(page, 'width') else 0,
            'quality_score': 0
        }
        
        # Determine quality and extraction method
        if not text or char_count < min_chars_per_page:
            # Poor extraction, try OCR
            print(f"    ⚠️  Page {page_num}: Poor pdfplumber extraction ({char_count} chars), trying OCR...")
            text = self._extract_with_ocr(page)
            method = 'tesseract'
            quality_flag = True
            
            # Recalculate metrics for OCR text
            quality_metrics.update({
                'char_count': len(text) if text else 0,
                'word_count': len(text.split()) if text else 0,
                'line_count': len(text.split('\n')) if text else 0
            })
            
        elif char_count > max_chars_per_page:
            # Suspiciously long text (might be garbled)
            print(f"    ⚠️  Page {page_num}: Suspiciously long text ({char_count} chars)")
            method = 'pdfplumber'
            quality_flag = True
            
        else:
            # Good extraction
            method = 'pdfplumber'
            quality_flag = False
        
        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(quality_metrics, method)
        quality_metrics['quality_score'] = quality_score
        
        # Extract word boxes and perform layout analysis
        word_boxes, text_blocks = PDFParser._extract_word_boxes_with_layout(page, page_num, method)
        
        # Analyze page layout
        layout_analysis = PDFParser._analyze_page_layout(word_boxes, page.width, page.height)
        
        # Create PageLayout object
        from pdf_parser._structures import PageLayout
        page_layout = PageLayout(
            page_number=page_num,
            page_width=page.width,
            page_height=page.height,
            word_boxes=word_boxes,
            text_blocks=text_blocks,
            reading_order=layout_analysis['reading_order'],
            layout_analysis=layout_analysis
        )
        
        # Create metadata
        metadata = {
            'page_number': page_num,
            'method': method,
            'quality_flag': quality_flag,
            'quality_score': quality_score,
            'extraction_timestamp': datetime.now().isoformat(),
            'word_count': len(word_boxes),
            'layout_type': layout_analysis['layout_type'],
            'estimated_columns': layout_analysis['columns'],
            'text_density': layout_analysis['text_density'],
            **quality_metrics
        }
        
        return {
            'text': text or "",
            'word_boxes': word_boxes,
            'page_layout': page_layout,
            'metadata': metadata
        }

    @staticmethod
    def _save_extracted_content(extracted_pages, output_file, metadata_file, file_stats, output_format):
        """
        Save extracted text and metadata with page-level granularity in the requested format.
        """
        # Save extracted text in the requested format
        
        if output_format == "json":
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

    def _save_word_boxes_and_layout(self, all_word_boxes, all_page_layouts, wordboxes_file, layout_file, file_stats):
        """
        Save word boxes and layout data as JSON files.
        """
        # Save word boxes data
        word_boxes_data = {
            'file_info': {
                'filename': file_stats['filename'],
                'processing_date': datetime.now().isoformat(),
                'total_word_boxes': len(all_word_boxes),
                'total_pages': file_stats['total_pages']
            },
            'word_boxes': [box.to_dict() for box in all_word_boxes],
            'statistics': {
                'total_word_boxes': len(all_word_boxes),
                'avg_word_boxes_per_page': len(all_word_boxes) / file_stats['total_pages'] if file_stats['total_pages'] > 0 else 0,
                'pages_with_word_boxes': len([layout for layout in all_page_layouts if layout.word_boxes])
            }
        }
        # Save word boxes data
        with open(wordboxes_file, 'w', encoding='utf-8') as f:
            json.dump(word_boxes_data, f, indent=2, ensure_ascii=False)


        # Save layout data
        layout_data = {
            'file_info': {
                'filename': file_stats['filename'],
                'processing_date': datetime.now().isoformat(),
                'total_pages': file_stats['total_pages']
            },
            'page_layouts': [layout.to_dict() for layout in all_page_layouts],
            'document_analysis': self._analyze_document_layout(all_page_layouts)
        }
        
        with open(layout_file, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        print(f"  📦 Saved word boxes: {os.path.basename(wordboxes_file)}")
        print(f"  📐 Saved layout data: {os.path.basename(layout_file)}")
        print(f"  📊 Total word boxes: {len(all_word_boxes)}")

    def _extract_with_ocr(self, page):
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

    def _calculate_quality_score(self, metrics, method):
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

    @staticmethod
    def _extract_word_boxes_ocr(page, page_num):
        import pytesseract
        word_boxes = []
        try:
            page_image = page.to_image(resolution=300)
            pil_image = page_image.original
            ocr_data = pytesseract.image_to_data(
                pil_image,
                lang='eng',
                output_type=pytesseract.Output.DICT
            )
            
            for idx, (level, text, conf, x, y, w, h) in enumerate(zip(
                ocr_data['level'],
                ocr_data['text'],
                ocr_data['conf'],
                ocr_data['left'],
                ocr_data['top'],
                ocr_data['width'],
                ocr_data['height']
            )):
                # Only process word-level entries (level 5) with valid confidence and non-empty text
                if level == 5 and conf > 30 and text.strip():
                    try:
                        x0 = float(x)
                        x1 = float(x) + float(w)
                        top = float(y)
                        bottom = float(y) + float(h)
                        width = float(w)
                    except (TypeError, ValueError):
                        continue  # skip this word if any value is invalid
                    
                    word_box = WordBox(
                        text=text.strip(),
                        x0=x0,
                        x1=x1,
                        top=top,
                        bottom=bottom,
                        width=width,
                        confidence=float(conf),
                        page_number=page_num,
                        word_index=idx
                    )
                    word_boxes.append(word_box)
        except Exception as e:
            print(f"      ❌ OCR word box extraction failed for page {page_num}: {e}")
        return word_boxes
    
    #region layout analysis
    @staticmethod
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
        total_text_area = sum((box.x1 - box.x0) * (box.bottom - box.top) for box in word_boxes)
        page_area = page_width * page_height
        text_density = total_text_area / page_area if page_area > 0 else 0

        # Determine layout type based on word distribution
        x_positions = [(box.x0 + box.x1) / 2 for box in word_boxes]
        y_positions = [(box.top + box.bottom) / 2 for box in word_boxes]

        # Simple column detection
        x_sorted = sorted(set(x_positions))
        column_gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
        avg_gap = sum(column_gaps) / len(column_gaps) if column_gaps else 0

        # Estimate number of columns
        estimated_columns = max(1, int(page_width / (avg_gap + 50)) if avg_gap > 0 else 1)

        # Determine reading order
        reading_order = PDFParser._determine_reading_order(word_boxes, estimated_columns)

        # Classify layout type
        layout_type = PDFParser._classify_layout_type(word_boxes, estimated_columns, text_density)


        x_spread = max(x_positions) - min(x_positions) if x_positions else 0
        y_spread = max(y_positions) - min(y_positions) if y_positions else 0
        aspect_ratio = x_spread / y_spread if y_spread > 0 else 0

        return {
            'columns': estimated_columns,
            'rows': len(set(y_positions)),
            'text_density': text_density,
            'layout_type': layout_type,
            'reading_order': reading_order,
            'aspect_ratio': aspect_ratio,
            'text_flow_analysis': {
                'x_spread': x_spread,
                'y_spread': y_spread,
                'word_count': len(word_boxes),
                'unique_x_positions': len(set(x_positions)),
                'unique_y_positions': len(set(y_positions))
            }
        }

    @staticmethod
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
                    if not all(attr in word for attr in ['text', 'x0', 'x1', 'top', 'bottom']):
                        continue
                    word_box = WordBox(
                        text=word.get('text', ''),
                        x0=float(word.get('x0', 0)),
                        x1=float(word.get('x1', 0)),
                        top=word.get('top'),
                        bottom=word.get('bottom'),
                        doctop=word.get('doctop'),
                        upright=word.get('upright'),
                        page_number=page_num,
                        word_index=idx,
                        width=float(word.get('x1', 0)) - float(word.get('x0', 0)),
                    )
                    word_boxes.append(word_box)
                try:
                    text_blocks = page.extract_text_simple()
                except Exception:
                    text_blocks = []
            else:  # OCR method
                # You can implement a more detailed OCR word box extraction if needed
                word_boxes = PDFParser._extract_word_boxes_ocr(page, page_num)
                text_blocks = []
        except Exception as e:
            print(f"      ❌ Word box extraction failed for page {page_num}: {e}")
            return [], []

        return word_boxes, text_blocks

    def _analyze_document_layout(self, page_layouts):
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

    @staticmethod
    def _determine_reading_order(word_boxes, estimated_columns):
        """
        Determine reading order of words (top-to-bottom, left-to-right), optionally column-aware.
        Returns a list of word indices in reading order.
        """
        if not word_boxes:
            return []

        # Simple top-to-bottom, left-to-right sorting
        def simple_reading_order():
            sorted_boxes = sorted(word_boxes, key=lambda box: (box.top, box.x0))
            return [box.word_index for box in sorted_boxes]

        # Column-aware reading order
        def column_aware_reading_order():
            if estimated_columns <= 1:
                return simple_reading_order()
            page_width = max(box.x1 for box in word_boxes) if word_boxes else 0
            column_width = page_width / estimated_columns
            column_groups = [[] for _ in range(estimated_columns)]
            for box in word_boxes:
                center_x = (box.x0 + box.x1) / 2
                column_idx = min(int(center_x / column_width), estimated_columns - 1)
                column_groups[column_idx].append(box)
            reading_order = []
            for column in column_groups:
                column_sorted = sorted(column, key=lambda box: box.top)
                reading_order.extend([box.word_index for box in column_sorted])
            return reading_order

        # Advanced reading order with line detection (optional, not used by default)
        # def advanced_reading_order():
        #     ...

        if estimated_columns > 1:
            return column_aware_reading_order()
        else:
            return simple_reading_order()

    @staticmethod
    def _classify_layout_type(word_boxes, estimated_columns, text_density):
        """
        Classify the layout type based on word distribution and density.
        """
        if not word_boxes:
            return 'empty'

        # Analyze word distribution
        x_positions = [(box.x0 + box.x1) / 2 for box in word_boxes]
        y_positions = [(box.top + box.bottom) / 2 for box in word_boxes]

        # Calculate spreads and statistics
        x_spread = max(x_positions) - min(x_positions) if x_positions else 0
        y_spread = max(y_positions) - min(y_positions) if y_positions else 0
        aspect_ratio = x_spread / y_spread if y_spread > 0 else 0


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
            else:
                return 'single_column'
        elif estimated_columns == 2:
            return 'two_column'
        elif estimated_columns >= 3:
            if text_density > density_thresholds['dense']:
                return 'dense_multi_column'
            else:
                return 'multi_column'
        elif x_spread < y_spread * 0.4:
            return 'narrow_column'
        elif text_density > density_thresholds['very_dense']:
            return 'very_dense_layout'
        else:
            return 'mixed_layout'

#endregion

#region test call

if __name__ == "__main__":
    parser = PDFParser(table_extractor="tabula")
    args = argparse.ArgumentParser()
    args.add_argument("--input-dir", required=True)
    args.add_argument("--output-dir", required=True)
    args.add_argument("--output-format", required=True)
    args = args.parse_args()
    parser.parse(
        input_source=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.output_format
    )



#endregion