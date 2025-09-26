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
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
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
        
        # Process each page with enhanced layout understanding
        for page_structure in preprocessed_data.get("document_structure", []):
            page_number = page_structure.get("page_number", 0)
            elements = page_structure.get("elements", [])
            
            # Convert layout elements to LayoutLMv3 format for inference
            page_words = []
            page_boxes = []
            
            for element in elements:
                element_text = element.get("text", "")
                element_bbox = element.get("bbox", [0, 0, 0, 0])
                
                # Split element text into words for token-level analysis
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
            
            if page_words:
                # Perform LayoutLMv3 inference on the page
                page_predictions = _perform_page_inference(
                    processor, model, device, page_words, page_boxes, id2label
                )
                
                inference_results["model_predictions"].append({
                    "page_number": page_number,
                    "predictions": page_predictions
                })
        
        # Enhance elements with model predictions
        inference_results["enhanced_elements"] = _enhance_elements_with_predictions(
            preprocessed_data.get("document_structure", []),
            inference_results["model_predictions"]
        )
        
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

def _enhance_elements_with_predictions(document_structure, model_predictions):
    """
    Enhance layout elements with model predictions.
    """
    enhanced_elements = []
    
    for page_structure in document_structure:
        page_number = page_structure.get("page_number", 0)
        
        # Find corresponding predictions
        page_preds = None
        for pred_page in model_predictions:
            if pred_page.get("page_number") == page_number:
                page_preds = pred_page.get("predictions", [])
                break
        
        if page_preds:
            for element in page_structure.get("elements", []):
                enhanced_element = element.copy()
                enhanced_element["model_predictions"] = []
                
                # Match predictions to elements based on text content
                element_text = element.get("text", "")
                for pred in page_preds:
                    if pred.get("word", "") in element_text:
                        enhanced_element["model_predictions"].append(pred)
                
                enhanced_elements.append(enhanced_element)
    
    return enhanced_elements

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
    target_folder = "data/parsed/MSFT_10-K_20230727_000095017023035122"
    loaded_data = load_data(target_folder)
    # perform data preprocessing
    preprocessed_data = preprocess_data(loaded_data, use_layout_parsing=True)
    # perform inference
    inference_data = perform_inference(preprocessed_data, layout_parsing=True)
    # save the results
    save_inference_results(inference_data, output_dir=target_folder)
# endregion