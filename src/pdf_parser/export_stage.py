"""
Export Stage - Consolidate outputs from text extraction, tabula, and markdown exporter
Provides unified access to all PDF parsing results
"""

import json
import os
import csv
import glob
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import nbformat
    from nbconvert import PythonExporter
except ImportError:
    nbformat = None
    PythonExporter = None

try:
    from .markdown_exporter import export_markdown_from_inference_file
except ImportError:
    # Handle case when run as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from markdown_exporter import export_markdown_from_inference_file


@dataclass
class ExportSummary:
    """Summary of all available exports for a document"""
    document_name: str
    text_extraction_available: bool
    tabula_tables_count: int
    markdown_export_available: bool
    inference_data_available: bool
    total_size_bytes: int
    export_timestamp: str


class ExportStage:
    """Consolidate and export outputs from all PDF parsing stages"""
    
    def __init__(self, parsed_data_dir: str):
        """
        Initialize export stage with parsed data directory
        
        Args:
            parsed_data_dir: Base directory containing all parsed outputs
        """
        self.parsed_data_dir = Path(parsed_data_dir)
        self.text_extraction_dir = self.parsed_data_dir / "MSFT"
        self.tabula_output_dir = self.parsed_data_dir / "tabula_output"
        self.wer_metrics = self._load_wer_metrics()
        
    def get_text_extraction_outputs(self) -> Dict[str, Any]:
        """
        Read and process text extraction outputs (.txt files)
        
        Returns:
            Dictionary containing text extraction results
        """
        text_outputs = {
            "extraction_method": "pdfplumber + OCR fallback",
            "files_found": [],
            "total_files": 0,
            "total_size_bytes": 0,
            "content": {}
        }
        
        # Search for text files in the MSFT directory
        txt_files = list(self.text_extraction_dir.glob("**/*.txt"))
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_size = txt_file.stat().st_size
                
                file_info = {
                    "path": str(txt_file),
                    "size_bytes": file_size,
                    "content": content,
                    "line_count": len(content.split('\n')),
                    "char_count": len(content),
                    "word_count": len(content.split())
                }
                
                text_outputs["files_found"].append(txt_file.name)
                text_outputs["content"][txt_file.name] = file_info
                text_outputs["total_size_bytes"] += file_size
                
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
        
        text_outputs["total_files"] = len(txt_files)
        return text_outputs
    
    def get_tabula_outputs(self) -> Dict[str, Any]:
        """
        Read and process tabula CSV outputs and metadata
        
        Returns:
            Dictionary containing tabula extraction results
        """
        tabula_outputs = {
            "extraction_method": "tabula-py stream mode",
            "metadata": {},
            "tables": {},
            "summary": {
                "total_tables": 0,
                "valid_tables": 0,
                "total_size_bytes": 0,
                "pages_with_tables": set()
            }
        }
        
        # Read metadata files
        files_jsonl = self.tabula_output_dir / "metadata" / "files.jsonl"
        tables_jsonl = self.tabula_output_dir / "metadata" / "tables.jsonl"
        
        if files_jsonl.exists():
            with open(files_jsonl, 'r') as f:
                tabula_outputs["metadata"]["file_info"] = json.loads(f.read().strip())
        
        if tables_jsonl.exists():
            table_metadata = []
            with open(tables_jsonl, 'r') as f:
                for line in f:
                    table_info = json.loads(line.strip())
                    table_metadata.append(table_info)
                    
                    # Load actual table data if CSV exists
                    if table_info.get("csv_path") and table_info["is_valid_table"]:
                        csv_path = self.tabula_output_dir / Path(table_info["csv_path"]).name
                        if csv_path.exists():
                            try:
                                if pd is None:
                                    raise ImportError("pandas not available")
                                df = pd.read_csv(csv_path)
                                table_data = {
                                    "metadata": table_info,
                                    "data": df.to_dict('records'),
                                    "shape": df.shape,
                                    "columns": list(df.columns),
                                    "preview": df.head(3).to_dict('records')
                                }
                                tabula_outputs["tables"][f"table_{table_info['table_index']}"] = table_data
                                tabula_outputs["summary"]["total_size_bytes"] += csv_path.stat().st_size
                                tabula_outputs["summary"]["pages_with_tables"].add(table_info["page_number"])
                                
                            except Exception as e:
                                print(f"Error reading CSV {csv_path}: {e}")
            
            tabula_outputs["metadata"]["tables"] = table_metadata
            tabula_outputs["summary"]["total_tables"] = len(table_metadata)
            tabula_outputs["summary"]["valid_tables"] = len([t for t in table_metadata if t["is_valid_table"]])
            tabula_outputs["summary"]["pages_with_tables"] = list(tabula_outputs["summary"]["pages_with_tables"])
        
        return tabula_outputs
    
    def get_markdown_export(self, inference_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate markdown export using the markdown exporter
        
        Args:
            inference_data_path: Path to inference_data.json file
            
        Returns:
            Dictionary containing markdown export results
        """
        markdown_output = {
            "export_method": "StructuredMarkdownExporter with ML predictions",
            "inference_data_path": inference_data_path,
            "markdown_content": None,
            "content_length": 0,
            "export_successful": False,
            "error": None
        }
        
        # Look for inference data file if not provided
        if not inference_data_path:
            # Search for inference_data.json in the parsed directory
            inference_files = list(self.parsed_data_dir.glob("**/lmv3/inference_data.json"))
            if inference_files:
                inference_data_path = str(inference_files[0])
                markdown_output["inference_data_path"] = inference_data_path
        
        if inference_data_path and os.path.exists(inference_data_path):
            try:
                markdown_content = export_markdown_from_inference_file(inference_data_path)
                markdown_output["markdown_content"] = markdown_content
                markdown_output["content_length"] = len(markdown_content)
                markdown_output["export_successful"] = True
                
                # Extract some metadata from the content
                lines = markdown_content.split('\n')
                markdown_output["line_count"] = len(lines)
                markdown_output["page_count"] = markdown_content.count("## Page ")
                markdown_output["table_count"] = markdown_content.count("```")  # Rough estimate
                
            except Exception as e:
                markdown_output["error"] = str(e)
                print(f"Error generating markdown export: {e}")
        else:
            markdown_output["error"] = f"Inference data file not found: {inference_data_path}"
        
        return markdown_output
    
    def _load_wer_metrics(self) -> Dict[str, Any]:
        """
        Load WER evaluation metrics from the evaluation notebook
        
        Returns:
            Dictionary containing WER metrics or empty dict if unavailable
        """
        wer_metrics = {}
        
        try:
            # Look for the evaluation notebook
            notebook_path = Path(__file__).parent.parent.parent / "notebooks" / "evaluation_wer.ipynb"
            
            if not notebook_path.exists() or nbformat is None:
                return {}
            
            # Read and parse the notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Look for the aggregate_metrics output in the notebook
            for cell in nb.cells:
                if cell.cell_type == 'code' and 'outputs' in cell:
                    for output in cell.outputs:
                        if hasattr(output, 'text') and 'AGGREGATE METRICS' in str(output.text):
                            # Parse the metrics from the output text
                            lines = str(output.text).split('\n')
                            for line in lines:
                                if ':' in line and any(metric in line for metric in ['mean_wer', 'mean_cer', 'mean_text_similarity']):
                                    parts = line.strip().split(': ')
                                    if len(parts) == 2:
                                        key = parts[0]
                                        try:
                                            value = float(parts[1])
                                            wer_metrics[key] = value
                                        except ValueError:
                                            wer_metrics[key] = parts[1]
                            break
            
            # If we found metrics, add metadata
            if wer_metrics:
                wer_metrics['source'] = 'evaluation_wer.ipynb'
                wer_metrics['evaluation_pages'] = 4  # From notebook: pages 61-64
                
        except Exception as e:
            print(f"Warning: Could not load WER metrics from notebook: {e}")
        
        return wer_metrics
    
    def generate_unified_export(self, output_path: Optional[str] = None, 
                              include_content: bool = True) -> Dict[str, Any]:
        """
        Generate a unified export combining all outputs
        
        Args:
            output_path: Optional path to save the unified export JSON
            include_content: Whether to include full content or just summaries
            
        Returns:
            Dictionary containing all export results
        """
        print("Collecting text extraction outputs...")
        text_outputs = self.get_text_extraction_outputs()
        
        print("Collecting tabula outputs...")
        tabula_outputs = self.get_tabula_outputs()
        
        print("Generating markdown export...")
        markdown_outputs = self.get_markdown_export()
        
        # Create unified export structure
        unified_export = {
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "parsed_data_directory": str(self.parsed_data_dir),
                "export_stage_version": "1.0.0"
            },
            "wer_evaluation": self.wer_metrics if self.wer_metrics else None,
            "summary": {
                "text_extraction": {
                    "files_count": text_outputs["total_files"],
                    "total_size_bytes": text_outputs["total_size_bytes"],
                    "available": text_outputs["total_files"] > 0
                },
                "tabula_extraction": {
                    "tables_count": tabula_outputs["summary"]["total_tables"],
                    "valid_tables_count": tabula_outputs["summary"]["valid_tables"],
                    "pages_with_tables": len(tabula_outputs["summary"]["pages_with_tables"]),
                    "total_size_bytes": tabula_outputs["summary"]["total_size_bytes"],
                    "available": tabula_outputs["summary"]["total_tables"] > 0
                },
                "markdown_export": {
                    "content_length": markdown_outputs["content_length"],
                    "page_count": markdown_outputs.get("page_count", 0),
                    "available": markdown_outputs["export_successful"]
                }
            },
            "outputs": {}
        }
        
        # Add detailed outputs based on include_content flag
        if include_content:
            unified_export["outputs"]["text_extraction"] = text_outputs
            unified_export["outputs"]["tabula_extraction"] = tabula_outputs
            unified_export["outputs"]["markdown_export"] = markdown_outputs
        else:
            # Include only metadata and summaries
            unified_export["outputs"]["text_extraction"] = {
                "extraction_method": text_outputs["extraction_method"],
                "files_found": text_outputs["files_found"],
                "total_files": text_outputs["total_files"],
                "total_size_bytes": text_outputs["total_size_bytes"]
            }
            unified_export["outputs"]["tabula_extraction"] = {
                "extraction_method": tabula_outputs["extraction_method"],
                "metadata": tabula_outputs["metadata"],
                "summary": tabula_outputs["summary"]
            }
            unified_export["outputs"]["markdown_export"] = {
                k: v for k, v in markdown_outputs.items() 
                if k != "markdown_content"  # Exclude large content
            }
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(unified_export, f, indent=2, ensure_ascii=False)
            print(f"Unified export saved to: {output_path}")
        
        return unified_export
    
    def create_export_summary(self) -> ExportSummary:
        """Create a high-level summary of available exports"""
        text_outputs = self.get_text_extraction_outputs()
        tabula_outputs = self.get_tabula_outputs()
        markdown_outputs = self.get_markdown_export()
        
        total_size = (text_outputs["total_size_bytes"] + 
                     tabula_outputs["summary"]["total_size_bytes"])
        
        return ExportSummary(
            document_name="MSFT_10-K_Analysis",
            text_extraction_available=text_outputs["total_files"] > 0,
            tabula_tables_count=tabula_outputs["summary"]["total_tables"],
            markdown_export_available=markdown_outputs["export_successful"],
            inference_data_available=markdown_outputs["inference_data_path"] is not None,
            total_size_bytes=total_size,
            export_timestamp=datetime.now().isoformat()
        )


def main():
    """CLI interface for export stage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export and consolidate PDF parsing results")
    parser.add_argument("--parsed-dir", default="/Users/RiyanshiKedia/Desktop/pdf-parser/data/parsed",
                       help="Directory containing parsed outputs")
    parser.add_argument("--output", "-o", help="Output file for unified export JSON")
    parser.add_argument("--summary-only", action="store_true",
                       help="Generate summary only (exclude full content)")
    parser.add_argument("--format", choices=["json", "summary"], default="json",
                       help="Output format")
    
    args = parser.parse_args()
    
    # Initialize export stage
    export_stage = ExportStage(args.parsed_dir)
    
    if args.format == "summary":
        # Generate and display summary
        summary = export_stage.create_export_summary()
        print("\n=== Export Summary ===")
        print(f"Document: {summary.document_name}")
        print(f"Text Extraction Available: {summary.text_extraction_available}")
        print(f"Tabula Tables Found: {summary.tabula_tables_count}")
        print(f"Markdown Export Available: {summary.markdown_export_available}")
        print(f"Inference Data Available: {summary.inference_data_available}")
        print(f"Total Size: {summary.total_size_bytes:,} bytes")
        print(f"Export Timestamp: {summary.export_timestamp}")
    else:
        # Generate unified export
        include_content = not args.summary_only
        unified_export = export_stage.generate_unified_export(
            output_path=args.output,
            include_content=include_content
        )
        
        print("\n=== Export Results ===")
        print(f"Text Extraction: {unified_export['summary']['text_extraction']['files_count']} files")
        print(f"Tabula Tables: {unified_export['summary']['tabula_extraction']['tables_count']} tables")
        print(f"Markdown Export: {'✓' if unified_export['summary']['markdown_export']['available'] else '✗'}")
        
        if not args.output:
            print(f"\nUnified export data available in memory ({len(str(unified_export)):,} characters)")


if __name__ == "__main__":
    main()

# Add direct call for testing/demonstration
def run_export():
    """Run export stage with default settings"""
    export_stage = ExportStage("/Users/RiyanshiKedia/Desktop/pdf-parser/data/parsed")
    return export_stage.generate_unified_export()

# Uncomment the line below to run automatically
run_export()
