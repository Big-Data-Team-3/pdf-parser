"""
Export Stage - Simplified PDF parsing results consolidator
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import nbformat
except ImportError:
    nbformat = None


class ExportStage:
    """Consolidate and export outputs from all PDF parsing stages"""
    
    def __init__(self, parsed_data_dir: str):
        """Initialize with parsed data directory"""
        self.parsed_data_dir = Path(parsed_data_dir)
        self.tabula_output_dir = os.path.join(self.parsed_data_dir, "tables")
        
    def get_text_files(self) -> Dict[str, Any]:
        """Get text extraction files summary"""
        txt_files = list(self.parsed_data_dir.glob("*/extracted.txt"))
        return {
            "count": len(txt_files),
            "files": [str(f) for f in txt_files],
            "total_size": sum(os.path.getsize(f) for f in txt_files if os.path.exists(f))
        }
    
    def get_tabula_summary(self) -> Dict[str, Any]:
        """Get tabula extraction summary"""
        tables_jsonl_array = list(self.parsed_data_dir.glob("*/tables/metadata/tables.jsonl"))
        
        if len(tables_jsonl_array) == 0:
            return {"count": 0, "valid_tables": 0}
        
        table_count = 0
        valid_count = 0
        
        for tables_jsonl in tables_jsonl_array:
            with open(tables_jsonl, 'r') as f:
                for line in f:
                    table_info = json.loads(line.strip())
                    table_count += 1
                    if table_info.get("is_valid_table"):
                        valid_count += 1
        return {
            "count": table_count,
            "valid_tables": valid_count,
        }
    
    def get_inference_files(self) -> Dict[str, Any]:
        """Check for inference data files"""
        inference_files = list(self.parsed_data_dir.glob("*/lmv3/inference_data.json"))
        return {
            "available": len(inference_files) > 0,
            "count": len(inference_files),
            "paths": [str(f) for f in inference_files]
        }
        
    def export(self) -> Dict[str, Any]:
        """Generate simplified export summary"""
        
        text_files = self.get_text_files()
        tabula_data = self.get_tabula_summary()
        inference_data = self.get_inference_files()
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "parsed_directory": str(self.parsed_data_dir),
            "text_files": text_files,
            "tabula_tables": tabula_data,
            "inference_data": inference_data,
        }
        
        return export_data
    
    def summary(self) -> None:
        """Print export summary"""
        data = self.export()
        print(f"\n=== Export Summary ===")
        print(f"Text files: {data['text_files']['count']}")
        print(f"Tabula tables: {data['tabula_tables']['count']} ({data['tabula_tables']['valid_tables']} valid)")
        print(f"Timestamp: {data['timestamp']}")


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export PDF parsing results")
    parser.add_argument("--parsed-dir", default="data/parsed", help="Parsed data directory")
    
    args = parser.parse_args()
    
    export_stage = ExportStage(args.parsed_dir)
    export_stage.summary()

# def test(format="summary"):
#     export_stage = ExportStage("data/parsed")
#     parsed_dir = "data/parsed"
#     if format == "summary":
#         export_stage.summary()
#     else:
#         export_stage.export()

if __name__ == "__main__":
    main()