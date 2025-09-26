# Layout analysis
'''
We need to perform layout analysis on the set of files extracted from the PDF.
These are the output files from the in-built parser.

- extracted text JSON
- extracted text TXT
- layout JSON
- wordboxes JSON
- metadata JSON

There is also another folder in the output directory called "tables" which contains the tables extracted from the PDF, using the tabula library.

the tables folder contains the following files:
- bunch of csv files extracted from the PDF
- metadata files for the whole file and the tables connecting the generated tables to their place in the document, inside the metadata subfolder.


--------------------------------

Our goal right now is to perform layout analysis on the set of files extracted from the PDF.
We need to follow the steps below:

1. Load all the necessary files from the output directory
2. Preprocess the data to be used for layout analysis
3. Perform inference on the data using the layoutlmv3 model.
4. Save the results in the output directory's subfolder called 'lmv3' which will be created if it doesn't exist.

--------------------------------

We will use the LayoutLMv3 model for this task.
'''

# region imports
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import sys
from PIL import Image
# endregion

# region functions
def load_data(target_folder):
    """
    Load required data files from the parsed document directory.
    This is the output of the parser.py file.
        
        Args:
            target_folder: Path to the parsed document folder containing
                extracted JSON files and tables metadata.
                
        Returns:
            Dictionary containing:
            - layout_data: List of layout elements from layout.json
            - file_jsonl: List of entries from files.jsonl
            - tables_jsonl: List of entries from tables.jsonl
            - file_metadata: Dictionary from metadata.json
    """
    target_folder = Path(target_folder)
    data = {
        'layout_data': [],
        'file_jsonl': [],
        'tables_jsonl': [],
        'file_metadata': {}
    }

    # Find layout file (ends with _layout.json)
    layout_files = list(target_folder.glob('*_layout.json'))
    if not layout_files:
        raise FileNotFoundError(f"No layout file found in {target_folder}")
    with open(layout_files[0], 'r') as f:
        data['layout_data'] = json.load(f)['page_layouts']

    # Find and load metadata files
    metadata_files = list(target_folder.glob('*_metadata.json'))
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            data['file_metadata'] = json.load(f)

    # Load JSONL files from tables/metadata subdirectory
    tables_meta_dir = target_folder / 'tables' / 'metadata'
    
    # Load files.jsonl
    files_jsonl = tables_meta_dir / 'files.jsonl'
    if files_jsonl.exists():
        with open(files_jsonl, 'r') as f:
            data['file_jsonl'] = [json.loads(line) for line in f]

    # Load tables.jsonl
    tables_jsonl = tables_meta_dir / 'tables.jsonl'
    if tables_jsonl.exists():
        with open(tables_jsonl, 'r') as f:
            data['tables_jsonl'] = [json.loads(line) for line in f]

    return data

def preprocess_data(loaded_data, use_layout_parsing:bool=False):
    '''
    Preprocess the data to be used for layout analysis, by performing validation and formatting.
    Steps:
    1. Validate the data by checking if the required files are present.
        - layout_data
        - file_jsonl
        - tables_jsonl
        - file_metadata
    2. Format the data to be used for layout analysis.
    3. Return the formatted data.
    '''
    def validate_data(data):
        '''
        Validate the data by checking if the required files are present.
        '''
        # check if the object is a dict and has the required keys
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        if 'layout_data' not in data:
            raise ValueError("layout_data is required")
        if 'file_jsonl' not in data:
            raise ValueError("file_jsonl is required")
        if 'tables_jsonl' not in data:
            raise ValueError("tables_jsonl is required")
        if 'file_metadata' not in data:
            raise ValueError("file_metadata is required")
        return True
    
    def _extract_layout_elements(word_boxes, page_width, page_height, layout_analysis):
        '''
        Extract layout elements from word boxes.
        Groups words into logical elements like paragraphs, headers, lists, etc.
        '''
        if not word_boxes:
            return []
        
        elements = []
        
        # Sort words by reading order (top to bottom, left to right)
        sorted_words = sorted(word_boxes, key=lambda w: (w.get('top', 0), w.get('x0', 0)))
        
        # Group words into lines based on vertical proximity
        lines = []
        current_line = [sorted_words[0]] if sorted_words else []
        line_threshold = page_height * 0.02  # 2% of page height
        
        for word in sorted_words[1:]:
            if current_line:
                prev_word = current_line[-1]
                # Check if word is on same line (similar y-coordinate)
                if abs(word.get('top', 0) - prev_word.get('top', 0)) <= line_threshold:
                    current_line.append(word)
                else:
                    lines.append(current_line)
                    current_line = [word]
        
        if current_line:
            lines.append(current_line)
        
        # Group lines into elements (paragraphs, headers, etc.)
        element_id = 0
        current_element = None
        
        for line_words in lines:
            if not line_words:
                continue
                
            # Calculate line properties
            line_text = ' '.join(word.get('text', '') for word in line_words)
            line_bbox = _calculate_line_bbox(line_words, page_width, page_height)
            
            # Classify element type based on properties
            element_type = _classify_element_type(line_words, line_text, layout_analysis)
            
            # Create new element or extend current one
            if (current_element is None or 
                current_element['element_type'] != element_type or
                element_type in ['header', 'title']):  # Headers/titles are always separate
                
                # Save previous element
                if current_element:
                    elements.append(current_element)
                
                # Start new element
                element_id += 1
                current_element = {
                    "element_id": f"elem_{element_id}",
                    "element_type": element_type,
                    "text": line_text.strip(),
                    "bbox": line_bbox,
                    "properties": {
                        "line_count": 1,
                        "word_count": len(line_words),
                        "confidence": _calculate_average_confidence(line_words)
                    }
                }
            else:
                # Extend current element
                current_element['text'] += ' ' + line_text.strip()
                current_element['bbox'] = _merge_bboxes(current_element['bbox'], line_bbox)
                current_element['properties']['line_count'] += 1
                current_element['properties']['word_count'] += len(line_words)
        
        # Add final element
        if current_element:
            elements.append(current_element)
        
        return elements


    def _calculate_average_confidence(words):
        '''
        Calculate the average confidence of the words in the line.
        '''
        if not words:
            return 1.0
        
        confidences = []
        for word in words:
            conf = word.get('confidence')
            if conf is not None and isinstance(conf, (int, float)):
                confidences.append(float(conf))
            else:
                confidences.append(1.0)  # Default confidence for missing values
        
        return sum(confidences) / len(confidences) if confidences else 1.0

    def _analyze_layout_distribution(layout_parsing_data):
        '''
        Analyze the layout distribution across the document.
        '''
        element_types = {}
        total_elements = 0
        
        # Count element types across all pages
        for page in layout_parsing_data.get("document_structure", []):
            for element in page.get("elements", []):
                elem_type = element.get("element_type", "unknown")
                element_types[elem_type] = element_types.get(elem_type, 0) + 1
                total_elements += 1
        
        # Calculate distribution percentages
        distribution = {}
        if total_elements > 0:
            for elem_type, count in element_types.items():
                distribution[elem_type] = {
                    "count": count,
                    "percentage": round((count / total_elements) * 100, 2)
                }
        
        return {
            "total_elements": total_elements,
            "element_types": element_types,
            "distribution": distribution,
            "dominant_type": max(element_types.items(), key=lambda x: x[1])[0] if element_types else "none"
        }
    
    # Helper functions for layout element extraction
    def _calculate_line_bbox(words, page_width, page_height):
        """Calculate normalized bounding box for a line of words."""
        if not words:
            return [0, 0, 0, 0]
        
        min_x = min(w.get('x0', 0) for w in words)
        max_x = max(w.get('x1', w.get('x0', 0) + w.get('width', 0)) for w in words)
        min_y = min(w.get('top', 0) for w in words)
        max_y = max(w.get('bottom', w.get('top', 0) + 10) for w in words)
        
        # Normalize to 0-1000 scale
        return [
            min(1000, max(0, int((min_x / page_width) * 1000))),
            min(1000, max(0, int((min_y / page_height) * 1000))),
            min(1000, max(0, int((max_x / page_width) * 1000))),
            min(1000, max(0, int((max_y / page_height) * 1000)))
        ]

    def _is_likely_tabular_data(text):
        """Detect if text is likely tabular financial data."""
        # Count dollar signs and numeric patterns
        dollar_count = text.count('$')
        comma_separated_numbers = len([x for x in text.split() if ',' in x and any(c.isdigit() for c in x)])
        
        # Look for multiple columns of aligned data
        lines = text.split('\n')
        if len(lines) > 3:  # Multi-line content
            # Check for consistent patterns across lines
            lines_with_numbers = sum(1 for line in lines if any(c.isdigit() for c in line))
            if lines_with_numbers > len(lines) * 0.6:  # 60% of lines have numbers
                return True
        
        # Financial data indicators
        if dollar_count >= 4 and comma_separated_numbers >= 6:
            return True
            
        # Look for balance sheet / income statement patterns
        financial_keywords = ['assets', 'liabilities', 'equity', 'revenue', 'income', 'cash', 'debt']
        keyword_matches = sum(1 for keyword in financial_keywords if keyword in text.lower())
        if keyword_matches >= 3 and dollar_count >= 2:
            return True
            
        return False

    def _classify_element_type(line_words, line_text, layout_analysis):
        """Classify the type of layout element based on text and formatting."""
        text = line_text.strip()
        
        # Simple heuristics for element classification
        if len(text) < 5:
            return "fragment"
        elif text.isupper() and len(text) < 100:
            return "header"
        elif any(char.isdigit() for char in text[:10]) and ('.' in text[:10] or ')' in text[:10]):
            return "list_item"
        elif len(text) < 150 and text.endswith('.') and text.count('.') == 1:
            return "title"
        elif any(keyword in text.lower() for keyword in ['table', 'schedule', 'exhibit']):
            return "table_reference"
        # NEW: Enhanced table detection
        elif _is_likely_tabular_data(text):
            return "table"
        else:
            return "paragraph"

    
    def _merge_bboxes(bbox1, bbox2):
        """Merge two bounding boxes."""
        return [
            min(bbox1[0], bbox2[0]),  # min x
            min(bbox1[1], bbox2[1]),  # min y
            max(bbox1[2], bbox2[2]),  # max x
            max(bbox1[3], bbox2[3])   # max y
        ]    

    def format_data(data, use_layout_parsing):
        '''
        Format the data for LayoutLMv3 model input.
        Transforms the loaded data into the format expected by LayoutLMv3:
        - Normalized bounding boxes (0-1000 scale)
        - Clean text tokens
        - Page-level structure
        
        Args:
            data: Dictionary containing layout_data, file_jsonl, tables_jsonl, file_metadata
            
        Returns:
            Dictionary formatted for LayoutLMv3 input with normalized coordinates
        '''
        
        if use_layout_parsing:
            # We will preprocess for layout parsing
            # This involves creating a structure suitable for document layout understanding
            # including logical reading order, element classification, and hierarchical structure
            
            layout_parsing_data = {
                "file_info": {
                    "filename": data['file_metadata'].get('filename', 'unknown'),
                    "total_pages": len(data['layout_data']) if data['layout_data'] else 0,
                    "document_type": "financial_report"  # Based on 10-K context
                },
                "document_structure": [],
                "layout_elements": [],
                "reading_order": []
            }
            
            # Process each page for layout parsing
            for page_idx, page_layout in enumerate(data['layout_data']):
                page_number = page_layout.get('page_number', page_idx + 1)
                page_width = page_layout.get('page_width', 612.0)
                page_height = page_layout.get('page_height', 792.0)
                word_boxes = page_layout.get('word_boxes', [])
                layout_analysis = page_layout.get('layout_analysis', {})
                
                # Group words into logical elements (paragraphs, headers, etc.)
                layout_elements = _extract_layout_elements(
                    word_boxes, page_width, page_height, layout_analysis
                )
                
                # Create page structure for layout parsing
                page_structure = {
                    "page_number": page_number,
                    "page_dimensions": {
                        "width": float(page_width),
                        "height": float(page_height)
                    },
                    "layout_type": layout_analysis.get('layout_type', 'single_column'),
                    "columns": layout_analysis.get('columns', 1),
                    "text_density": layout_analysis.get('text_density', 0),
                    "elements": layout_elements,
                    "reading_order": layout_analysis.get('reading_order', [])
                }
                
                layout_parsing_data["document_structure"].append(page_structure)
                layout_parsing_data["layout_elements"].extend(layout_elements)
            
            # Add table information for layout understanding
            if data['tables_jsonl']:
                layout_parsing_data["tables"] = []
                for table in data['tables_jsonl']:
                    table_element = {
                        "element_type": "table",
                        "page_number": table.get('page_number', 0),
                        "table_id": table.get('table_id', ''),
                        "dimensions": {
                            "rows": table.get('num_rows', 0),
                            "columns": table.get('num_columns', 0)
                        },
                        "properties": {
                            "is_valid": table.get('is_valid_table', False),
                            "validation_reason": table.get('validation_reason', ''),
                            "numeric_ratio": table.get('numeric_cell_ratio', 0),
                            "empty_ratio": table.get('empty_cell_ratio', 0)
                        },
                        "csv_path": table.get('csv_path', '')
                    }
                    layout_parsing_data["tables"].append(table_element)
            
            # Add document-level analysis
            layout_parsing_data["document_analysis"] = {
                "total_pages": len(layout_parsing_data["document_structure"]),
                "total_elements": len(layout_parsing_data["layout_elements"]),
                "layout_distribution": _analyze_layout_distribution(layout_parsing_data),
                "processing_metadata": {
                    "processing_time": data['file_metadata'].get('processing_time', 0),
                    "extraction_method": "hybrid_pdfplumber_ocr"
                }
            }
            
            return layout_parsing_data
        
        else:
            # We will preprocess directly for LayoutLMv3
            layoutlmv3_data = {
                "file_info": {
                    "filename": data['file_metadata'].get('filename', 'unknown'),
                    "total_pages": len(data['layout_data']) if data['layout_data'] else 0
                    },
                    "page_layouts": []
            }

            # Process each page for LayoutLMv3
            for page_layout in data['layout_data']:
                page_number = page_layout.get('page_number', 0)
                page_width = page_layout.get('page_width', 612.0)
                page_height = page_layout.get('page_height', 792.0)
                word_boxes = page_layout.get('word_boxes', [])
                
                # Transform word boxes to LayoutLMv3 format
                transformed_words = []
                for word_box in word_boxes:
                    # Clean text and skip empty entries
                    text = word_box.get('text', '').strip()
                    if not text:
                        continue
                    
                    # Get bounding box coordinates
                    x0 = word_box.get('x0', 0)
                    y0 = word_box.get('top', word_box.get('y0', 0))  # Use 'top' if available
                    x1 = word_box.get('x1', 0)
                    y1 = word_box.get('bottom', word_box.get('y1', 0))  # Use 'bottom' if available
                    
                    # Handle case where coordinates might be missing
                    if x1 == 0:
                        x1 = x0 + word_box.get('width', 0)
                    
                    # Normalize coordinates to 0-1000 range (LayoutLMv3 standard)
                    normalized_bbox = [
                        min(1000, max(0, int((x0 / page_width) * 1000))),
                        min(1000, max(0, int((y0 / page_height) * 1000))),
                        min(1000, max(0, int((x1 / page_width) * 1000))),
                        min(1000, max(0, int((y1 / page_height) * 1000)))
                    ]
                    
                    # Ensure bbox is valid (x1 > x0, y1 > y0)
                    if normalized_bbox[2] <= normalized_bbox[0]:
                        normalized_bbox[2] = min(1000, normalized_bbox[0] + 10)
                    if normalized_bbox[3] <= normalized_bbox[1]:
                        normalized_bbox[3] = min(1000, normalized_bbox[1] + 10)
                    
                    # Create word entry for LayoutLMv3
                    word_entry = {
                        "text": text,
                        "bbox": normalized_bbox
                    }
                    
                    transformed_words.append(word_entry)
                
                # Create page layout entry
                page_entry = {
                    "page_number": page_number,
                    "page_width": float(page_width),
                    "page_height": float(page_height), 
                    "words": transformed_words
                }
                
                layoutlmv3_data["page_layouts"].append(page_entry)
            
            # Add metadata for tracking
            layoutlmv3_data["processing_info"] = {
                "total_pages": len(layoutlmv3_data["page_layouts"]),
                "total_words": sum(len(page["words"]) for page in layoutlmv3_data["page_layouts"]),
                "source_metadata": {
                    "processing_time": data['file_metadata'].get('processing_time', 0),
                    "total_word_boxes": data['file_metadata'].get('total_word_boxes', 0)
                }
            }
            
            # Add table information for context
            if data['tables_jsonl']:
                layoutlmv3_data["tables_info"] = []
                for table in data['tables_jsonl']:
                    table_info = {
                        "page_number": table.get('page_number', 0),
                        "table_id": table.get('table_id', ''),
                        "num_rows": table.get('num_rows', 0),
                        "num_columns": table.get('num_columns', 0),
                        "is_valid": table.get('is_valid_table', False)
                    }
                    layoutlmv3_data["tables_info"].append(table_info)
            
            return layoutlmv3_data
        
    status = validate_data(loaded_data)
    if not status:
        raise ValueError("Data is not valid")
    # format the data to be used for layout analysis
    formatted_data = format_data(loaded_data, use_layout_parsing)
    return formatted_data


    


def perform_inference(preprocessed_data, use_gpu:bool=False, layout_parsing:bool=False):
    '''
    Perform inference on the data using the LayoutLMv3 model.
    
    Args:
        preprocessed_data: Formatted data following formatted_data.json schema
        use_gpu: Whether to use GPU for inference
        layout_parsing: Whether to perform layout parsing before LayoutLMv3 inference
        
    Returns:
        Dictionary containing inference results with token classifications and layout analysis
    '''
    
    def _analyze_element_layout(element, page_structure):
        """Analyze layout properties of an element within page context."""
        bbox = element.get("bbox", [0, 0, 0, 0])
        page_dims = page_structure.get("page_dimensions", {"width": 612, "height": 792})
        
        # Calculate layout properties
        width_ratio = (bbox[2] - bbox[0]) / 1000  # Normalized width
        height_ratio = (bbox[3] - bbox[1]) / 1000  # Normalized height
        x_position = bbox[0] / 1000  # Left margin ratio
        y_position = bbox[1] / 1000  # Top position ratio
        
        # Determine layout characteristics
        layout_analysis = {
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "x_position": x_position,
            "y_position": y_position,
            "is_full_width": width_ratio > 0.8,
            "is_left_aligned": x_position < 0.2,
            "is_centered": 0.3 < x_position < 0.7,
            "is_right_aligned": x_position > 0.8,
            "vertical_position": "top" if y_position < 0.3 else "middle" if y_position < 0.7 else "bottom"
        }
        
        return layout_analysis

    def _determine_semantic_role(element, page_structure):
        """Determine the semantic role of an element in document structure."""
        element_type = element.get("element_type", "paragraph")
        text = element.get("text", "").lower()
        layout_context = element.get("layout_context", {})
        
        # Enhanced semantic classification
        if element_type == "header":
            if layout_context.get("is_centered", False):
                return "main_title"
            elif layout_context.get("y_position", 0) < 0.2:
                return "page_header"
            else:
                return "section_header"
        
        elif element_type == "table":  # ADD THIS
            return "financial_table"
        
        elif element_type == "paragraph":
            if any(keyword in text for keyword in ["table", "schedule", "exhibit"]):
                return "table_caption"
            elif len(text) < 100:
                return "short_text"
            else:
                return "body_text"
        
        elif element_type == "list_item":
            return "enumerated_item"
        
        elif element_type == "table_reference":
            return "table_link"
        
        return element_type

    def _calculate_reading_order(element, all_elements):
        """Calculate reading order position within the page."""
        element_bbox = element.get("bbox", [0, 0, 0, 0])
        
        # Sort elements by reading order (top to bottom, left to right)
        sorted_elements = sorted(all_elements, key=lambda e: (e.get("bbox", [0, 0, 0, 0])[1], e.get("bbox", [0, 0, 0, 0])[0]))
        
        try:
            return sorted_elements.index(element) + 1
        except ValueError:
            return len(all_elements)

    def _perform_document_layout_analysis(all_elements):
        """Perform comprehensive document-level layout analysis."""
        if not all_elements:
            return {}
        
        # Analyze element distribution
        element_types = {}
        semantic_roles = {}
        layout_patterns = {
            "full_width_elements": 0,
            "multi_column_indicators": 0,
            "centered_elements": 0,
            "left_aligned_elements": 0
        }
        
        for element in all_elements:
            # Count element types
            elem_type = element.get("element_type", "unknown")
            element_types[elem_type] = element_types.get(elem_type, 0) + 1
            
            # Count semantic roles
            semantic_role = element.get("semantic_role", "unknown")
            semantic_roles[semantic_role] = semantic_roles.get(semantic_role, 0) + 1
            
            # Analyze layout patterns
            layout_context = element.get("layout_context", {})
            if layout_context.get("is_full_width", False):
                layout_patterns["full_width_elements"] += 1
            if layout_context.get("is_centered", False):
                layout_patterns["centered_elements"] += 1
            if layout_context.get("is_left_aligned", False):
                layout_patterns["left_aligned_elements"] += 1
        
        # Determine document layout characteristics
        total_elements = len(all_elements)
        layout_characteristics = {
            "predominant_alignment": "left" if layout_patterns["left_aligned_elements"] > total_elements * 0.6 else "mixed",
            "layout_complexity": "simple" if len(element_types) <= 3 else "complex",
            "structure_type": "formal" if semantic_roles.get("section_header", 0) > 5 else "informal"
        }
        
        return {
            "total_elements": total_elements,
            "element_type_distribution": element_types,
            "semantic_role_distribution": semantic_roles,
            "layout_patterns": layout_patterns,
            "layout_characteristics": layout_characteristics,
            "document_flow": _analyze_document_flow(all_elements)
        }

    def _analyze_document_flow(all_elements):
        """Analyze the flow and structure of the document."""
        flow_analysis = {
            "reading_flow": "top_to_bottom",
            "section_breaks": 0,
            "column_changes": 0,
            "hierarchical_depth": 0
        }
        
        # Analyze hierarchical structure
        headers = [e for e in all_elements if e.get("element_type") == "header"]
        flow_analysis["hierarchical_depth"] = len(set(h.get("semantic_role") for h in headers))
        
        # Count section breaks (large vertical gaps)
        for i in range(1, len(all_elements)):
            prev_element = all_elements[i-1]
            curr_element = all_elements[i]
            
            prev_y = prev_element.get("bbox", [0, 0, 0, 0])[3]  # bottom of previous
            curr_y = curr_element.get("bbox", [0, 0, 0, 0])[1]  # top of current
            
            # Large gap indicates section break
            if curr_y - prev_y > 50:  # Threshold for section break
                flow_analysis["section_breaks"] += 1
        
        return flow_analysis

    def _perform_page_inference(processor, model, device, words, boxes, id2label):
        """
        Perform LayoutLMv3 inference on a single page.
        
        Args:
            processor: LayoutLMv3 processor
            model: LayoutLMv3 model
            device: torch device
            words: List of word strings
            boxes: List of bounding boxes [x0, y0, x1, y1] normalized to 0-1000
            id2label: Label mapping dictionary
            
        Returns:
            List of predictions for each word
        """
        try:
            # Prepare inputs for LayoutLMv3
            # Create a dummy image (white background) since we're not using OCR
            dummy_image = Image.new('RGB', (1000, 1000), color='white')
            
            # Process the inputs
            encoding = processor(
                dummy_image,
                words,
                boxes=boxes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            for key in encoding:
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].to(device)
            
            # Perform inference
            with torch.no_grad():
                outputs = model(**encoding)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class = predictions.argmax(dim=-1)
            
            # Convert predictions to labels
            predicted_labels = []
            confidence_scores = []
            
            # Get the actual sequence length (excluding padding)
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            
            for i, (token_id, attention) in enumerate(zip(input_ids, attention_mask)):
                if attention == 0:  # Skip padding tokens
                    continue
                    
                if i < len(predicted_token_class[0]):
                    pred_id = predicted_token_class[0][i].item()
                    confidence = predictions[0][i][pred_id].item()
                    
                    predicted_labels.append(id2label.get(pred_id, "O"))
                    confidence_scores.append(float(confidence))
                else:
                    predicted_labels.append("O")
                    confidence_scores.append(0.0)
            
            # Map predictions back to original words
            word_predictions = []
            word_idx = 0
            
            for i, (label, confidence) in enumerate(zip(predicted_labels, confidence_scores)):
                # Skip special tokens ([CLS], [SEP], etc.)
                if i > 0 and word_idx < len(words):
                    word_predictions.append({
                        "word": words[word_idx],
                        "bbox": boxes[word_idx],
                        "predicted_label": label,
                        "confidence": confidence
                    })
                    word_idx += 1
                    
                if word_idx >= len(words):
                    break
            
            return word_predictions
            
        except Exception as e:
            print(f"⚠️ Error during inference for page: {e}")
            # Return fallback predictions
            return [
                {
                    "word": word,
                    "bbox": bbox,
                    "predicted_label": "O",
                    "confidence": 0.0
                }
                for word, bbox in zip(words, boxes)
            ]

    def _analyze_document_predictions(page_predictions, id2label):
        """
        Analyze predictions across the entire document.
        """
        label_counts = {}
        total_words = 0
        high_confidence_predictions = 0
        
        for page_pred in page_predictions:
            for pred in page_pred.get("predictions", []):
                label = pred.get("predicted_label", "O")
                confidence = pred.get("confidence", 0.0)
                
                label_counts[label] = label_counts.get(label, 0) + 1
                total_words += 1
                
                if confidence > 0.8:
                    high_confidence_predictions += 1
        
        # Calculate percentages
        label_percentages = {
            label: (count / total_words * 100) if total_words > 0 else 0
            for label, count in label_counts.items()
        }
        
        return {
            "total_words": total_words,
            "label_distribution": label_counts,
            "label_percentages": label_percentages,
            "high_confidence_ratio": high_confidence_predictions / total_words if total_words > 0 else 0,
            "detected_structures": [label for label, count in label_counts.items() if count > 10 and label != "O"]
        }

    def _combine_rule_based_and_ml_predictions(rule_based_elements, ml_predictions, element_word_mapping):
        """
        Combine rule-based element analysis with ML predictions.
        
        Args:
            rule_based_elements: Elements with rule-based analysis
            ml_predictions: Word-level ML predictions from LayoutLMv3
            element_word_mapping: List mapping each word to its element index
        
        Returns:
            Enhanced elements with both rule-based and ML insights
        """
        enhanced_elements = []
        
        for element_idx, element in enumerate(rule_based_elements):
            # Get ML predictions for this element's words
            element_ml_predictions = []
            for word_idx, pred in enumerate(ml_predictions):
                if word_idx < len(element_word_mapping) and element_word_mapping[word_idx] == element_idx:
                    element_ml_predictions.append(pred)
            
            # Analyze ML predictions for this element
            ml_analysis = _analyze_element_ml_predictions(element_ml_predictions)
            
            # Combine rule-based and ML insights
            enhanced_element = {
                **element,
                "ml_predictions": element_ml_predictions,
                "ml_analysis": ml_analysis,
                "confidence_score": ml_analysis.get("avg_confidence", 0.0),
                "predicted_labels": ml_analysis.get("dominant_labels", []),
                "hybrid_classification": _determine_hybrid_classification(
                    element.get("semantic_role", "unknown"),
                    ml_analysis.get("dominant_label", "O")
                )
            }
            
            enhanced_elements.append(enhanced_element)
        
        return enhanced_elements

    def _analyze_element_ml_predictions(predictions):
        """Analyze ML predictions for a single element."""
        if not predictions:
            return {"avg_confidence": 0.0, "dominant_labels": [], "label_distribution": {}}
        
        # Calculate average confidence
        avg_confidence = sum(p.get("confidence", 0.0) for p in predictions) / len(predictions)
        
        # Count label occurrences
        label_counts = {}
        for pred in predictions:
            label = pred.get("predicted_label", "O")
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find dominant labels
        dominant_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        dominant_label = dominant_labels[0][0] if dominant_labels else "O"
        
        return {
            "avg_confidence": avg_confidence,
            "dominant_label": dominant_label,
            "dominant_labels": [label for label, count in dominant_labels[:3]],
            "label_distribution": label_counts,
            "total_words": len(predictions)
        }

    def _determine_hybrid_classification(rule_based_role, ml_label):
        """Combine rule-based and ML classifications for final element type."""
        # Priority mapping: rule-based takes precedence, ML provides refinement
        if rule_based_role in ["main_title", "page_header", "section_header"]:
            if "HEADER" in ml_label:
                return f"confirmed_{rule_based_role}"
            else:
                return f"potential_{rule_based_role}"
        
        elif rule_based_role == "financial_table":  # ADD THIS
            if "TABLE" in ml_label:
                return "confirmed_financial_table"
            else:
                return "rule_based_financial_table"
        
        elif rule_based_role == "body_text":
            if ml_label in ["B-ANSWER", "I-ANSWER"]:
                return "detailed_content"
            elif "TABLE" in ml_label:
                return "table_content"
            else:
                return "body_text"
        
        elif rule_based_role == "table_caption":
            if "TABLE" in ml_label:
                return "confirmed_table_caption"
            else:
                return "potential_table_caption"
        
        else:
            return f"{rule_based_role}_ml_{ml_label.lower()}"
    
    # Set device
    print("--------------------------------")
    print("Setting up device...")
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print("--------------------------------")
    
    # Initialize processor and model
    print("📦 Loading LayoutLMv3 model and processor...")
    try:
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=13)
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load LayoutLMv3 model: {e}")
    print("--------------------------------")
    # Define label mappings for token classification
    id2label = {
        0: "O",           # Outside
        1: "B-HEADER",    # Beginning of header
        2: "I-HEADER",    # Inside header
        3: "B-QUESTION",  # Beginning of question
        4: "I-QUESTION",  # Inside question
        5: "B-ANSWER",    # Beginning of answer
        6: "I-ANSWER",    # Inside answer
        7: "B-TABLE",     # Beginning of table
        8: "I-TABLE",     # Inside table
        9: "B-LIST",      # Beginning of list
        10: "I-LIST",     # Inside list
        11: "B-FOOTER",   # Beginning of footer
        12: "I-FOOTER"    # Inside footer
    }
    
    if layout_parsing:
        # Perform inference with layout parsing approach
        print("🔄 Performing layout parsing + LayoutLMv3 inference...")
        
        inference_results = {
            "inference_type": "layout_parsing",
            "file_info": preprocessed_data.get("file_info", {}),
            "document_structure": preprocessed_data.get("document_structure", []),
            "enhanced_elements": [],
            "model_predictions": []
        }
        
        # Process each page with element-level understanding
        all_elements = []

        # Process each page with enhanced layout understanding
        for page_structure in preprocessed_data.get("document_structure", []):
            page_number = page_structure.get("page_number", 0)
            elements = page_structure.get("elements", [])
            
            print(f"📄 Processing page {page_number} with {len(elements)} elements")

            # STEP 1: Rule-based element analysis
            page_elements = []
            
            # STEP 2: Prepare for ML inference - extract words and boxes from elements
            page_words = []
            page_boxes = []
            element_word_mapping = []  # Track which words belong to which elements
            
            for element_idx, element in enumerate(elements):
                # Rule-based analysis
                element_analysis = _analyze_element_layout(element, page_structure)
                
                enhanced_element = {
                    **element,
                    "page_number": page_number,
                    "layout_context": element_analysis,
                    "semantic_role": _determine_semantic_role(element, page_structure),
                    "reading_order": _calculate_reading_order(element, elements)
                }
                page_elements.append(enhanced_element)
                
                # Prepare words for ML inference
                element_text = element.get("text", "")
                element_bbox = element.get("bbox", [0, 0, 0, 0])
                
                # Split element text into words for LayoutLMv3
                words = element_text.split()
                for i, word in enumerate(words):
                    page_words.append(word)
                    # Approximate word-level bounding boxes within element
                    word_bbox = [
                        element_bbox[0] + i * ((element_bbox[2] - element_bbox[0]) // max(len(words), 1)),
                        element_bbox[1],
                        element_bbox[0] + (i + 1) * ((element_bbox[2] - element_bbox[0]) // max(len(words), 1)),
                        element_bbox[3]
                    ]
                    page_boxes.append(word_bbox)
                    element_word_mapping.append(element_idx)  # Track which element this word belongs to
            
            # STEP 3: ML Inference - run LayoutLMv3 on all words
            if page_words:
                print(f"🧠 Running LayoutLMv3 inference on {len(page_words)} words...")
                page_predictions = _perform_page_inference(
                    processor, model, device, page_words, page_boxes, id2label
                )
                
                # STEP 4: Combine rule-based + ML results
                # Map ML predictions back to elements
                enhanced_elements_with_ml = _combine_rule_based_and_ml_predictions(
                    page_elements, page_predictions, element_word_mapping
                )
                
                inference_results["model_predictions"].append({
                    "page_number": page_number,
                    "predictions": page_predictions,
                    "element_count": len(elements),
                    "word_count": len(page_words)
                })
                
                all_elements.extend(enhanced_elements_with_ml)
            else:
                # No words to process, just add rule-based elements
                all_elements.extend(page_elements)




        # Document-level layout analysis
        inference_results["enhanced_elements"] = all_elements
        inference_results["layout_analysis"] = _perform_document_layout_analysis(all_elements)
        
        print(f"✅ Layout parsing completed: {len(all_elements)} elements analyzed")
        
    else:
        # Direct LayoutLMv3 inference
        print("🔄 Performing direct LayoutLMv3 inference...")
        
        inference_results = {
            "inference_type": "direct_layoutlmv3",
            "file_info": preprocessed_data.get("file_info", {}),
            "processing_info": preprocessed_data.get("processing_info", {}),
            "page_predictions": [],
            "document_analysis": {}
        }
        
        # Process each page from formatted data
        print("--------------------------------")
        for page_layout in preprocessed_data.get("page_layouts", []):
            page_number = page_layout.get("page_number", 0)
            words = page_layout.get("words", [])
            
            if not words:
                continue
                
            # Extract text and bounding boxes
            page_words = [word["text"] for word in words]
            page_boxes = [word["bbox"] for word in words]
            
            # Perform inference
            page_predictions = _perform_page_inference(
                processor, model, device, page_words, page_boxes, id2label
            )
            
            inference_results["page_predictions"].append({
                "page_number": page_number,
                "page_width": page_layout.get("page_width", 612.0),
                "page_height": page_layout.get("page_height", 792.0),
                "total_words": len(page_words),
                "predictions": page_predictions
            })
        
        # Add document-level analysis
        inference_results["document_analysis"] = _analyze_document_predictions(
            inference_results["page_predictions"], id2label
        )
    
    # Add tables information if available
    if "tables_info" in preprocessed_data:
        inference_results["tables_info"] = preprocessed_data["tables_info"]
    
    print(f"✅ Inference completed successfully for {len(inference_results.get('page_predictions', inference_results.get('model_predictions', [])))} pages")
    
    return inference_results

def save_inference_results(inference_data, output_dir:str):
    '''
    Save the results in the output directory's subfolder called 'lmv3' which will be created if it doesn't exist.
    '''
    # save the inference data
    os.makedirs(os.path.join(output_dir, 'lmv3'), exist_ok=True)
    with open(os.path.join(output_dir, 'lmv3', 'inference_data.json'), 'w', encoding='utf-8') as f:
        json.dump(inference_data, f, indent=2, ensure_ascii=False)
# endregion

# region main
if __name__ == "__main__":
    # load the data from a given target folder
    args = argparse.ArgumentParser()
    args.add_argument("--input-dir", required=True)
    args.add_argument("--output-dir", required=True)
    args.add_argument("--output-format", required=True)
    args = args.parse_args()
    # Loop through all the PDF files in the input directory to make target_folders and seqeuntially generate output JSON files
    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in input directory")
        sys.exit(1)
    
    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        target_folder = os.path.join(args.output_dir, base_name)
        loaded_data = load_data(target_folder)
        # perform data preprocessing
        preprocessed_data = preprocess_data(loaded_data, use_layout_parsing=True)
        # perform inference
        inference_data = perform_inference(preprocessed_data, layout_parsing=True)
        # save the results
        save_inference_results(inference_data, output_dir=target_folder)
# endregion