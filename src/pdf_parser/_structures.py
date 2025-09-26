from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

# Internal data structures for pdf_parser. Not part of the public API.

@dataclass
class WordBox:
    """Data class for word box information with comprehensive positioning data"""
    text: str
    x0: float
    x1: float
    width: float
    confidence: Optional[float] = None
    page_number: int = 0
    word_index: int = 0
    doctop: Optional[float] = None  # Document-level top position (page offset)
    upright: Optional[bool] = None  # Text orientation (True = normal, False = rotated)
    top: Optional[float] = None     # Page-relative top position
    bottom: Optional[float] = None  # Page-relative bottom position

    def to_dict(self):
        return asdict(self)

    @property
    def center_x(self):
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self):
        return (self.y0 + self.y1) / 2

    @property
    def area(self):
        height = self.bottom - self.top
        return self.width * height

    @property
    def is_rotated(self):
        """Check if text is rotated (not upright)"""
        return self.upright is False

    @property
    def document_position(self):
        """Get document-level position including page offset"""
        if self.doctop is not None:
            return {
                'document_top': self.doctop,
                'page_relative_top': self.top,
                'page_number': self.page_number
            }
        return None

@dataclass
class PageLayout:
    """Data class for page layout information with comprehensive analysis"""
    page_number: int
    page_width: float
    page_height: float
    word_boxes: List[WordBox]
    text_blocks: List[Dict]
    reading_order: List[int]
    layout_analysis: Dict

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'page_number': self.page_number,
            'page_width': self.page_width,
            'page_height': self.page_height,
            'word_boxes': [box.to_dict() for box in self.word_boxes],
            'text_blocks': self.text_blocks,
            'reading_order': self.reading_order,
            'layout_analysis': self.layout_analysis
        }

    @property
    def total_words(self):
        """Total number of words on the page"""
        return len(self.word_boxes)

    @property
    def text_density(self):
        """Text density as percentage of page area covered by text"""
        if not self.word_boxes:
            return 0.0
        # Calculate area manually instead of using box.area
        total_text_area = sum((box.x1 - box.x0) * (box.bottom - box.top) 
                            for box in self.word_boxes 
                            if box.top is not None and box.bottom is not None)
        page_area = self.page_width * self.page_height
        return (total_text_area / page_area) * 100 if page_area > 0 else 0.0

    @property
    def average_font_size(self):
        """Average font size across all words"""
        # Use fontsize if it exists, otherwise skip
        font_sizes = [getattr(box, 'fontsize', None) for box in self.word_boxes]
        font_sizes = [fs for fs in font_sizes if fs is not None]
        return np.mean(font_sizes) if font_sizes else 0.0

    @property
    def layout_type(self):
        """Get the classified layout type"""
        return self.layout_analysis.get('layout_type', 'unknown')

    @property
    def estimated_columns(self):
        """Get the estimated number of columns"""
        return self.layout_analysis.get('columns', 1)

    def get_words_by_column(self, column_index):
        """Get words belonging to a specific column"""
        if self.estimated_columns <= 1:
            return self.word_boxes
        column_width = self.page_width / self.estimated_columns
        column_start = column_index * column_width
        column_end = (column_index + 1) * column_width
        return [box for box in self.word_boxes if column_start <= box.center_x < column_end]

    def get_words_by_line(self, line_tolerance=10):
        """Group words into lines based on Y-coordinate"""
        if not self.word_boxes:
            return []
        lines = []
        current_line = []
        for box in sorted(self.word_boxes, key=lambda b: b.y0):
            if not current_line or not any(abs(box.y0 - line_box.y0) < line_tolerance for line_box in current_line):
                if current_line:
                    lines.append(sorted(current_line, key=lambda b: b.x0))
                current_line = [box]
            else:
                current_line.append(box)
        if current_line:
            lines.append(sorted(current_line, key=lambda b: b.x0))
        return lines

    def get_rotated_words(self):
        """Get all rotated words on the page"""
        return [box for box in self.word_boxes if box.is_rotated]

    def get_high_confidence_words(self, min_confidence=80):
        """Get words with confidence above threshold (for OCR results)"""
        return [box for box in self.word_boxes if box.confidence is not None and box.confidence >= min_confidence]

    def analyze_text_flow(self):
        """Analyze the text flow pattern on the page"""
        if not self.word_boxes:
            return {'flow_type': 'empty', 'flow_score': 0}
        reading_order_boxes = [self.word_boxes[i] for i in self.reading_order if i < len(self.word_boxes)]
        flow_violations = 0
        for i in range(len(reading_order_boxes) - 1):
            current = reading_order_boxes[i]
            next_box = reading_order_boxes[i + 1]
            if next_box.y0 < current.y0 - 20:  # Next word is much higher
                flow_violations += 1
            elif next_box.x0 > current.x1 + 100:  # Next word is far to the right
                flow_violations += 1
        flow_score = max(0, 100 - (flow_violations / len(reading_order_boxes)) * 100)
        if flow_score > 90:
            flow_type = 'excellent'
        elif flow_score > 75:
            flow_type = 'good'
        elif flow_score > 50:
            flow_type = 'fair'
        else:
            flow_type = 'poor'
        return {
            'flow_type': flow_type,
            'flow_score': flow_score,
            'violations': flow_violations,
            'total_transitions': len(reading_order_boxes) - 1
        }
