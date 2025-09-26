"""
Structured Markdown Exporter for Layout Analysis Results
Converts LayoutLMv3 inference output to well-structured markdown (Enhanced Output Only)
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict


class StructuredMarkdownExporter:
    """Convert layout analysis results to structured markdown (Enhanced mode only)"""
    
    def __init__(self, inference_data: Dict[str, Any]):
        """
        Initialize with inference data from LayoutLMv3 analysis
        
        Args:
            inference_data: The JSON output from layout analysis matching the schema
        """
        self.data = inference_data
        self.enhanced_elements = inference_data.get('enhanced_elements', [])
    
    def export_enhanced_markdown(self) -> str:
        """Export with enhanced structure using ML predictions"""
        markdown_content = []
        
        # Document header with analysis summary
        file_info = self.data.get('file_info', {})
        layout_analysis = self.data.get('layout_analysis', {})
        
        markdown_content.append(f"# {file_info.get('filename', 'Document')}\n")
        markdown_content.append(f"*Document Type: {file_info.get('document_type', 'Unknown')}*\n")
        markdown_content.append(f"*Total Pages: {file_info.get('total_pages', 0)}*\n")
        
        # Document structure summary
        element_dist = layout_analysis.get('element_type_distribution', {})
        if element_dist:
            markdown_content.append("\n## Document Structure\n")
            for elem_type, count in element_dist.items():
                if count > 0:
                    markdown_content.append(f"- {elem_type.title()}: {count}\n")
        
        markdown_content.append("\n---\n")
        
        # Process enhanced elements grouped by page
        pages_content = defaultdict(list)
        
        for element in self.enhanced_elements:
            page_num = element.get('page_number', 1)
            pages_content[page_num].append(element)
        
        # Sort and process each page
        for page_num in sorted(pages_content.keys()):
            elements = pages_content[page_num]
            
            # Sort by reading order
            elements.sort(key=lambda e: e.get('reading_order', 0))
            
            markdown_content.append(f"\n## Page {page_num}\n")
            
            # Convert elements to markdown
            for element in elements:
                element_md = self._convert_enhanced_element_to_markdown(element)
                if element_md:
                    markdown_content.append(element_md)
            
            markdown_content.append("\n---\n")
        
        return "\n".join(markdown_content)

    def _convert_enhanced_element_to_markdown(self, element: Dict) -> str:
        """Convert enhanced element with ML predictions to markdown"""
        text = element.get('text', '').strip()
        if not text:
            return ""
        
        # PRIORITY 0: Rule-based financial table detection (override ML when needed)
        if self._is_financial_table_content(text):
            return self._format_as_table(text)
        
        # Use hybrid classification for best accuracy
        hybrid_class = element.get('hybrid_classification', '')
        semantic_role = element.get('semantic_role', '')
        ml_analysis = element.get('ml_analysis', {})
        dominant_label = ml_analysis.get('dominant_label', '')
        element_type = element.get('element_type', 'paragraph')
        
        # PRIORITY 1: ML predictions (most accurate)
        if 'TABLE' in dominant_label:
            return self._format_as_table(text)
        elif 'HEADER' in dominant_label:
            return f"### {text}\n\n"
        elif 'LIST' in dominant_label:
            return f"- {text}\n"
        
        # PRIORITY 2: Hybrid classification
        elif 'table' in hybrid_class.lower():
            return self._format_as_table(text)
        elif 'title' in hybrid_class.lower() or semantic_role == 'title':
            level = 2 if 'main' in hybrid_class.lower() else 3
            return f"{'#' * level} {text}\n\n"
        elif 'header' in hybrid_class.lower():
            return f"### {text}\n\n"
        elif 'list' in hybrid_class.lower():
            return f"- {text}\n"
        
        # PRIORITY 3: Semantic role and element type  
        elif semantic_role == 'enumerated_item':
            return f"1. {text}\n"
        elif len(text.split()) < 10:  # Short text
            return f"**{text}**  \n"
        else:
            # Regular paragraph with proper spacing
            return f"{text}\n\n"

    def _is_financial_table_content(self, text: str) -> bool:
        """Detect financial table content that ML might miss."""
        # Strong indicators of financial statements
        dollar_count = text.count('$')
        
        # Financial statement structure patterns
        financial_keywords = [
            'assets', 'liabilities', 'equity', 'cash', 'debt', 'revenue', 
            'income', 'expenses', 'current assets', 'stockholders', 'retained earnings'
        ]
        
        # Count financial keywords
        keyword_matches = sum(1 for keyword in financial_keywords 
                            if keyword.lower() in text.lower())
        
        # Count numeric patterns (numbers with commas)
        import re
        numeric_pattern = r'\b\d{1,3}(,\d{3})+\b'  # Matches numbers like 34,704
        numeric_matches = len(re.findall(numeric_pattern, text))
        
        # Multi-line structured data
        lines = text.split('\n')
        lines_with_dollars = sum(1 for line in lines if '$' in line)
        
        # Detection criteria
        if (dollar_count >= 4 and 
            keyword_matches >= 3 and 
            numeric_matches >= 4):
            return True
            
        # Alternative: High concentration of financial data
        if (dollar_count >= 6 and 
            numeric_matches >= 6 and
            lines_with_dollars >= 3):
            return True
            
        # Alternative: Balance sheet specific
        if ('total assets' in text.lower() and 
            'liabilities' in text.lower() and 
            dollar_count >= 3):
            return True
            
        return False

    def _format_as_table(self, text: str) -> str:
        """Format tabular text as markdown table."""
        # Try to parse the financial data into a proper markdown table
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return f"\n```\n{text}\n```\n\n"  # Code block for short tables
        
        # For financial statements, try to create a proper table
        if self._is_financial_statement(text):
            return self._format_financial_table(text)
        else:
            return f"\n```\n{text}\n```\n\n"  # Code block fallback

    def _is_financial_statement(self, text: str) -> bool:
        """Check if text is a financial statement."""
        financial_indicators = ['$', 'assets', 'liabilities', 'revenue', 'cash', 'debt', 'equity']
        return sum(1 for indicator in financial_indicators if indicator.lower() in text.lower()) >= 3

    def _format_financial_table(self, text: str) -> str:
        """Format financial statement text as a markdown table."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Simple table formatting for financial data
        table_lines = ["\n| Item | 2023 | 2022 |", "|------|------|------|"]
        
        for line in lines:
            # Extract financial line items (this is a simplified approach)
            if '$' in line and any(char.isdigit() for char in line):
                # Parse line into components (item name and values)
                parts = line.split('$')
                if len(parts) >= 2:
                    item = parts[0].strip()
                    values = '$'.join(parts[1:])
                    # Split values by spaces or common separators
                    value_parts = [v.strip() for v in values.replace(',', '').split() if v.strip()]
                    if len(value_parts) >= 2:
                        table_lines.append(f"| {item} | ${value_parts[0]} | ${value_parts[1]} |")
        
        if len(table_lines) > 2:  # Has actual data rows
            return "\n" + "\n".join(table_lines) + "\n\n"
        else:
            return f"\n```\n{text}\n```\n\n"  # Fallback to code block

def export_markdown_from_inference_file(inference_file_path: str, output_file_path: str = None) -> str:
    """
    Export enhanced markdown from an inference data file
    
    Args:
        inference_file_path: Path to the inference_data.json file
        output_file_path: Optional path to save the markdown file
        
    Returns:
        The generated enhanced markdown content
    """
    with open(inference_file_path, 'r', encoding='utf-8') as f:
        inference_data = json.load(f)
    
    exporter = StructuredMarkdownExporter(inference_data)
    markdown_content = exporter.export_enhanced_markdown()
    
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Enhanced structured markdown saved to: {output_file_path}")
    
    return markdown_content

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export enhanced structured markdown from layout analysis")
    parser.add_argument("inference_file", nargs='?', help="Path to inference_data.json file")
    parser.add_argument("-o", "--output", help="Output markdown file path")
    args = parser.parse_args()

    if args.inference_file:
        markdown = export_markdown_from_inference_file(args.inference_file, args.output)
        print("Enhanced structured markdown generated:")
        print(f"Length: {len(markdown):,} characters")
        if not args.output:
            print("\nPreview (first 300 chars):")
            print(markdown[:300] + "..." if len(markdown) > 300 else markdown)

    else:
        print("Usage:")
        print("  # Export from specific file")
        print("  python src/pdf_parser/markdown_exporter.py path/to/inference_data.json -o output.md")