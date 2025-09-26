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
        
        # Use hybrid classification for best accuracy
        hybrid_class = element.get('hybrid_classification', '')
        semantic_role = element.get('semantic_role', '')
        ml_analysis = element.get('ml_analysis', {})
        dominant_label = ml_analysis.get('dominant_label', '')
        
        # Enhanced classification logic
        if 'title' in hybrid_class.lower() or semantic_role == 'title':
            level = 2 if 'main' in hybrid_class.lower() else 3
            return f"{'#' * level} {text}\n\n"
            
        elif 'header' in hybrid_class.lower() or 'HEADER' in dominant_label:
            return f"### {text}\n\n"
            
        elif 'list' in hybrid_class.lower() or 'LIST' in dominant_label:
            return f"- {text}\n"
            
        elif 'table' in hybrid_class.lower() or 'TABLE' in dominant_label:
            return f"\n*Table Reference: {text}*\n\n"
            
        elif semantic_role == 'enumerated_item':
            return f"1. {text}\n"
            
        elif len(text.split()) < 10:  # Short text
            return f"**{text}**  \n"
            
        else:
            # Regular paragraph with proper spacing
            return f"{text}\n\n"


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


def test_enhanced_markdown_export():
    """
    Test function for enhanced markdown export only
    """
    print("🧪 Testing Enhanced Markdown Export")
    print("=" * 40)
    
    # Path to your existing inference data
    test_inference_path = "data/parsed/MSFT_10-K_20230727_000095017023035122/lmv3/inference_data.json"
    
    # Check if the test file exists
    if not os.path.exists(test_inference_path):
        print(f"❌ Test file not found: {test_inference_path}")
        import glob
        inference_files = glob.glob("data/parsed/*/lmv3/inference_data.json")
        print("Available inference files:")
        for f in inference_files[:3]:
            print(f"  - {f}")
        return False
    
    try:
        print(f"📂 Loading: {test_inference_path}")
        
        # Load and export
        enhanced_markdown = export_markdown_from_inference_file(test_inference_path)
        
        print(f"✅ Enhanced markdown generated: {len(enhanced_markdown):,} characters")
        
        # Save test output
        test_output_dir = "test_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        output_file = os.path.join(test_output_dir, "enhanced_output.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_markdown)
        
        print(f"💾 Saved to: {output_file}")
        
        # Show preview
        print(f"\n📖 Preview (first 500 chars):")
        print("-" * 50)
        print(enhanced_markdown[:500])
        if len(enhanced_markdown) > 500:
            print("...")
        print("-" * 50)
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


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