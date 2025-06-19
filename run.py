import os
import re
import json
import base64
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from dotenv import load_dotenv

# PDF Processing
import pymupdf4llm
import PyPDF2
from PIL import Image

# AI Processing
from groq import Groq

# HTML Generation
from jinja2 import Template
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a document section with metadata"""
    title: str
    content: str
    section_type: str  # header, content, list, table, requirements, numerical, etc.
    level: int = 1
    subsections: List['DocumentSection'] = None
    bullets: List[str] = None
    numbers: List[Dict[str, Any]] = None  # For numerical data
    requirements: List[str] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []
        if self.bullets is None:
            self.bullets = []
        if self.numbers is None:
            self.numbers = []
        if self.requirements is None:
            self.requirements = []


@dataclass
class DocumentStructure:
    """Represents the overall document structure"""
    title: str
    subtitle: str = ""
    executive_summary: str = ""
    sections: List[DocumentSection] = None
    metadata: Dict[str, Any] = None
    key_metrics: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = []
        if self.metadata is None:
            self.metadata = {}
        if self.key_metrics is None:
            self.key_metrics = []


class PDFExtractor:
    """Handles PDF content extraction and preprocessing"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF using multiple methods"""
        try:
            # Method 1: Using pymupdf4llm (better for complex layouts)
            try:
                text = pymupdf4llm.to_markdown(pdf_path)
                if text and len(text.strip()) > 100:
                    logger.info("Successfully extracted text using pymupdf4llm")
                    return text
            except Exception as e:
                logger.warning(f"pymupdf4llm extraction failed: {e}")
            
            # Method 2: Using PyPDF2 as fallback
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            if text and len(text.strip()) > 100:
                logger.info("Successfully extracted text using PyPDF2")
                return text
            else:
                raise Exception("Insufficient text extracted from PDF")
                
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common extraction issues
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\r\n', '\n')  # Normalize line endings
        
        return text.strip()


class ContentAnalyzer:
    """Analyzes PDF content using Groq AI to extract structure and meaning"""
    
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
    
    def analyze_document_structure(self, text: str) -> DocumentStructure:
        """Analyze document and extract structured information"""
        
        # Split into manageable chunks if text is too long
        max_chunk_size = 15000  # Adjust based on model limits
        if len(text) > max_chunk_size:
            chunks = self._split_text_into_chunks(text, max_chunk_size)
            structures = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")
                structure = self._analyze_chunk(chunk, is_partial=True)
                structures.append(structure)
            
            # Merge structures
            return self._merge_structures(structures)
        else:
            return self._analyze_chunk(text, is_partial=False)
    
    def _split_text_into_chunks(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks while preserving section boundaries"""
        chunks = []
        current_chunk = ""
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _analyze_chunk(self, text: str, is_partial: bool = False) -> DocumentStructure:
        """Analyze a single chunk of text"""
        
        prompt = self._create_analysis_prompt(text, is_partial)
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            return self._parse_analysis_result(result)
            
        except Exception as e:
            logger.error(f"Failed to analyze content with Groq: {e}")
            # Fallback to basic structure extraction
            return self._create_fallback_structure(text)
    
    def _create_analysis_prompt(self, text: str, is_partial: bool) -> str:
        """Create enhanced prompt for document analysis"""
        
        partial_note = " (Note: This is a partial document chunk)" if is_partial else ""
        
        return f"""
        Analyze the following document content and extract its structure with enhanced formatting{partial_note}.
        
        Please provide a JSON response with the following structure:
        {{
            "title": "Main document title",
            "subtitle": "Document subtitle if any",
            "executive_summary": "Brief summary of the document (2-3 sentences)",
            "key_metrics": [
                {{
                    "label": "Metric name",
                    "value": "Metric value",
                    "type": "percentage|number|currency|time",
                    "context": "Brief context"
                }}
            ],
            "sections": [
                {{
                    "title": "Section title",
                    "content": "Section content summary",
                    "section_type": "header|content|list|table|requirements|numerical|conclusion",
                    "level": 1,
                    "bullets": ["bullet point 1", "bullet point 2"],
                    "numbers": [
                        {{
                            "label": "Number description",
                            "value": "123",
                            "unit": "units",
                            "type": "metric|stat|requirement"
                        }}
                    ],
                    "requirements": ["requirement 1", "requirement 2"],
                    "subsections": [
                        {{
                            "title": "Subsection title",
                            "content": "Subsection content",
                            "section_type": "content",
                            "level": 2,
                            "bullets": [],
                            "numbers": [],
                            "requirements": []
                        }}
                    ]
                }}
            ],
            "metadata": {{
                "document_type": "technical|business|academic|requirements|other",
                "complexity": "low|medium|high",
                "primary_topics": ["topic1", "topic2"],
                "has_requirements": true|false,
                "has_metrics": true|false,
                "has_lists": true|false
            }}
        }}
        
        Enhanced Guidelines:
        1. Extract clear hierarchical structure from the document
        2. Identify and separate bullet points into the "bullets" array
        3. Extract numerical data, metrics, and statistics into "numbers" array
        4. Identify requirements, specifications, or action items into "requirements" array
        5. Extract key metrics that appear at document level
        6. Classify section types more granularly (requirements, numerical, etc.)
        7. Preserve technical terms and important data
        8. Keep content summaries comprehensive but concise
        9. Look for patterns like "Requirements:", "Specifications:", numbered lists
        10. Identify percentage values, currency amounts, dates, and measurements
        
        Document Content:
        {text}
        
        Please respond with valid JSON only.
        """
    
    def _parse_analysis_result(self, result: str) -> DocumentStructure:
        """Parse the AI analysis result into DocumentStructure"""
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = result
            
            data = json.loads(json_str)
            
            # Convert to DocumentStructure
            sections = []
            for section_data in data.get('sections', []):
                subsections = []
                for subsection_data in section_data.get('subsections', []):
                    subsection_data.setdefault('bullets', [])
                    subsection_data.setdefault('numbers', [])
                    subsection_data.setdefault('requirements', [])
                    subsections.append(DocumentSection(**subsection_data))
                
                section_data['subsections'] = subsections
                section_data.setdefault('bullets', [])
                section_data.setdefault('numbers', [])
                section_data.setdefault('requirements', [])
                sections.append(DocumentSection(**section_data))
            
            return DocumentStructure(
                title=data.get('title', 'Untitled Document'),
                subtitle=data.get('subtitle', ''),
                executive_summary=data.get('executive_summary', ''),
                sections=sections,
                metadata=data.get('metadata', {}),
                key_metrics=data.get('key_metrics', [])
            )
            
        except Exception as e:
            logger.error(f"Failed to parse analysis result: {e}")
            # Return a basic structure if parsing fails
            return DocumentStructure(
                title="Document Analysis Failed",
                executive_summary="Unable to analyze document structure automatically.",
                sections=[DocumentSection(
                    title="Raw Content",
                    content=result[:1000] + "..." if len(result) > 1000 else result,
                    section_type="content"
                )]
            )
    
    def _merge_structures(self, structures: List[DocumentStructure]) -> DocumentStructure:
        """Merge multiple document structures from chunks"""
        if not structures:
            return DocumentStructure(title="Empty Document")
        
        # Use the first structure as base
        merged = structures[0]
        
        # Merge sections from other structures
        for structure in structures[1:]:
            merged.sections.extend(structure.sections)
            merged.key_metrics.extend(structure.key_metrics)
        
        # Update executive summary to be more comprehensive
        summaries = [s.executive_summary for s in structures if s.executive_summary]
        if summaries:
            merged.executive_summary = " ".join(summaries)
        
        return merged
    
    def _create_fallback_structure(self, text: str) -> DocumentStructure:
        """Create a basic structure when AI analysis fails"""
        lines = text.split('\n')
        title = lines[0] if lines else "Document"
        
        # Try to identify sections by looking for headers
        sections = []
        current_section = None
        
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
                
            # Simple heuristic for headers (lines with fewer than 100 chars, no periods)
            if len(line) < 100 and not line.endswith('.') and (line.isupper() or line.startswith('#')):
                if current_section:
                    sections.append(current_section)
                current_section = DocumentSection(
                    title=line,
                    content="",
                    section_type="header"
                )
            elif current_section:
                current_section.content += line + " "
        
        if current_section:
            sections.append(current_section)
        
        return DocumentStructure(
            title=title,
            sections=sections if sections else [DocumentSection(
                title="Content",
                content=text,
                section_type="content"
            )]
        )


class HTMLGenerator:
    """Generates beautiful, interactive HTML from structured document data"""
    
    def __init__(self):
        self.template = self._create_enhanced_html_template()
    
    def generate_html(self, structure: DocumentStructure) -> str:
        """Generate enhanced HTML from document structure"""
        
        # Prepare template variables
        template_vars = {
            'title': structure.title,
            'subtitle': structure.subtitle,
            'executive_summary': structure.executive_summary,
            'sections': structure.sections,
            'metadata': structure.metadata,
            'key_metrics': structure.key_metrics,
            'has_timeline': self._detect_timeline(structure),
            'has_metrics': self._detect_metrics(structure),
            'has_requirements': structure.metadata.get('has_requirements', False),
            'has_lists': structure.metadata.get('has_lists', False),
            'color_scheme': self._determine_color_scheme(structure)
        }
        
        # Render template
        template = Template(self.template)
        html = template.render(**template_vars)
        
        return self._optimize_html(html)
    
    def _detect_timeline(self, structure: DocumentStructure) -> bool:
        """Detect if document contains timeline information"""
        timeline_keywords = ['timeline', 'phase', 'milestone', 'schedule', 'implementation']
        text = structure.title + " " + structure.executive_summary
        for section in structure.sections:
            text += " " + section.title + " " + section.content
        
        return any(keyword in text.lower() for keyword in timeline_keywords)
    
    def _detect_metrics(self, structure: DocumentStructure) -> bool:
        """Detect if document contains metrics/KPIs"""
        if structure.key_metrics:
            return True
            
        metric_patterns = [r'\d+%', r'>\s*\d+', r'<\s*\d+', r'\d+:\d+', 'KPI', 'metric']
        text = " ".join([s.content for s in structure.sections])
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in metric_patterns)
    
    def _determine_color_scheme(self, structure: DocumentStructure) -> str:
        """Determine appropriate color scheme based on content"""
        doc_type = structure.metadata.get('document_type', 'business')
        
        schemes = {
            'technical': 'blue-tech',
            'business': 'blue-business',
            'academic': 'green-academic',
            'medical': 'red-medical',
            'requirements': 'purple-requirements'
        }
        
        return schemes.get(doc_type, 'blue-business')
    
    def _create_enhanced_html_template(self) -> str:
        """Create the enhanced HTML template with interactive elements"""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            {% if color_scheme == 'blue-tech' %}
            --primary: #1e3c72;
            --secondary: #2a5298;
            --accent: #3498db;
            --gradient: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            {% elif color_scheme == 'green-academic' %}
            --primary: #27ae60;
            --secondary: #2ecc71;
            --accent: #16a085;
            --gradient: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            {% elif color_scheme == 'red-medical' %}
            --primary: #c0392b;
            --secondary: #e74c3c;
            --accent: #e67e22;
            --gradient: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
            {% elif color_scheme == 'purple-requirements' %}
            --primary: #8e44ad;
            --secondary: #9b59b6;
            --accent: #e74c3c;
            --gradient: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
            {% else %}
            --primary: #2c3e50;
            --secondary: #34495e;
            --accent: #3498db;
            --gradient: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            {% endif %}
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 20px auto;
            background: white;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            border-radius: 20px;
            overflow: hidden;
        }

        .header {
            background: var(--gradient);
            color: white;
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
            opacity: 0.3;
            animation: float 20s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(1deg); }
        }

        .header h1 {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 15px;
            position: relative;
            z-index: 2;
        }

        .header h2 {
            font-size: 1.4rem;
            font-weight: 300;
            opacity: 0.9;
            position: relative;
            z-index: 2;
        }

        .content {
            padding: 40px;
        }

        /* Key Metrics Dashboard */
        .metrics-dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .metric-card {
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            transform: translateY(0);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        .executive-summary {
            background: linear-gradient(45deg, #f8f9ff, #e8f2ff);
            border-left: 5px solid var(--accent);
            padding: 30px;
            margin-bottom: 40px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.1);
            position: relative;
        }

        .executive-summary::before {
            content: '\\f05a';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 1.5rem;
            color: var(--accent);
            opacity: 0.3;
        }

        .executive-summary h3 {
            color: var(--primary);
            font-size: 1.8rem;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section {
            margin-bottom: 50px;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeInUp 0.6s ease forwards;
        }

        .section:nth-child(odd) { animation-delay: 0.1s; }
        .section:nth-child(even) { animation-delay: 0.2s; }

        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .section-header {
            background: var(--gradient);
            color: white;
            padding: 20px;
            border-radius: 10px 10px 0 0;
            display: flex;
            align-items: center;
            gap: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .section-header:hover {
            transform: translateX(5px);
        }

        .section-header.collapsible::after {
            content: '\\f107';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            margin-left: auto;
            transition: transform 0.3s ease;
        }

        .section-header.collapsed::after {
            transform: rotate(-90deg);
        }

        .section-number {
            background: rgba(255,255,255,0.2);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2rem;
        }

        .section-content {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 0 0 10px 10px;
            border: 1px solid #e9ecef;
            border-top: none;
        }

        .section-content.collapsed {
            display: none;
        }

        /* Enhanced Lists */
        .bullet-list {
            list-style: none;
            padding: 0;
            margin: 20px 0;
        }

        .bullet-list li {
            position: relative;
            padding: 10px 0 10px 30px;
            margin-bottom: 8px;
            transition: all 0.3s ease;
        }

        .bullet-list li::before {
            content: '\\f105';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            left: 0;
            top: 10px;
            color: var(--accent);
            font-size: 1.1rem;
        }

        .bullet-list li:hover {
            background: rgba(52, 152, 219, 0.05);
            border-radius: 5px;
            padding-left: 35px;
        }

        .requirements-list {
            background: linear-gradient(45deg, #fff5f5, #fef2f2);
            border: 1px solid #fecaca;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }

        .requirements-list h4 {
            color: #dc2626;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .requirements-list h4::before {
            content: '\\f058';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
        }

        .requirements-list ul {
            list-style: none;
            padding: 0;
        }

        .requirements-list li {
            position: relative;
            padding: 8px 0 8px 25px;
            margin-bottom: 5px;
        }

        .requirements-list li::before {
            content: '\\f00c';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            left: 0;
            top: 8px;
            color: #16a34a;
            font-size: 0.9rem;
        }

        /* Numerical Data Blocks */
        .numbers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .number-block {
            background: linear-gradient(45deg, #ffffff, #f8fafc);
            border: 2px solid var(--accent);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .number-block:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }

        .number-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--primary);
            display: block;
            margin-bottom: 5px;
        }

        .number-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 5px;
        }

        .number-unit {
            font-size: 0.8rem;
            color: var(--accent);
            font-weight: 500;
        }

        .subsection {
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
            border-left: 3px solid var(--accent);
        }

        .subsection h4 {
            color: var(--primary);
            font-size: 1.3rem;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--accent);
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }

        .subsection h4::before {
            content: '\\f0da';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            font-size: 1rem;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .feature-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 4px solid var(--accent);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }

        .conclusion {
            background: var(--gradient);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-top: 50px;
        }

        /* Interactive Elements */
        .toggle-all {
            background: var(--accent);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }

        .toggle-all:hover {
            background: var(--primary);
            transform: translateY(-2px);
        }

        .search-bar {
            width: 100%;
            padding: 15px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            margin-bottom: 20px;
            transition: border-color 0.3s ease;
        }

        .search-bar:focus {
            outline: none;
            border-color: var(--accent);
        }

        .highlighted {
            background: yellow;
            padding: 2px 4px;
            border-radius: 3px;
        }

        /* Progress Bar */
        .progress-bar {
            width: 100%;
            height: 4px;
            background: #e9ecef;
            border-radius: 2px;
            overflow: hidden;
            margin-bottom: 30px;
        }

        .progress-fill {
            height: 100%;
            background: var(--gradient);
            width: 0%;
            transition: width 2s ease;
            animation: progressAnimation 3s ease-in-out;
        }

        @keyframes progressAnimation {
            0% { width: 0%; }
            100% { width: 100%; }
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 10px;
            }
            
            .header {
                padding: 30px 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .content {
                padding: 20px;
            }
            
            .feature-grid, .metrics-dashboard, .numbers-grid {
                grid-template-columns: 1fr;
            }

            .section-header {
                padding: 15px;
            }

            .section-content {
                padding: 20px;
            }
        }

        /* Print Styles */
        @media print {
            body {
                background: white;
            }
            
            .container {
                box-shadow: none;
                margin: 0;
            }
            
            .section-header {
                background: #f8f9fa !important;
                color: #2c3e50 !important;
            }
            
            .toggle-all, .search-bar {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            {% if subtitle %}
            <h2>{{ subtitle }}</h2>
            {% endif %}
        </div>

        <div class="content">
            <!-- Progress Bar -->
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>

            <!-- Search Bar -->
            <input type="text" class="search-bar" placeholder="Search document content..." id="searchBar">

            <!-- Toggle All Button -->
            <button class="toggle-all" onclick="toggleAllSections()">
                <i class="fas fa-eye"></i> Toggle All Sections
            </button>

            <!-- Key Metrics Dashboard -->
            {% if key_metrics %}
            <div class="metrics-dashboard">
                {% for metric in key_metrics %}
                <div class="metric-card" title="{{ metric.context }}">
                    <span class="metric-value">{{ metric.value }}</span>
                    <span class="metric-label">{{ metric.label }}</span>
                </div>
                {% endfor %}
            </div>
            {% endif %}

            {% if executive_summary %}
            <div class="executive-summary">
                <h3><i class="fas fa-info-circle"></i> Executive Summary</h3>
                <p>{{ executive_summary }}</p>
            </div>
            {% endif %}

            {% for section in sections %}
            <div class="section" data-section="{{ loop.index }}">
                <div class="section-header collapsible" onclick="toggleSection({{ loop.index }})">
                    <div class="section-number">{{ loop.index }}</div>
                    <h3>{{ section.title }}</h3>
                </div>
                <div class="section-content" id="section-{{ loop.index }}">
                    {% if section.content %}
                    <p>{{ section.content }}</p>
                    {% endif %}
                    
                    <!-- Bullet Points -->
                    {% if section.bullets %}
                    <ul class="bullet-list">
                        {% for bullet in section.bullets %}
                        <li>{{ bullet }}</li>
                        {% endfor %}
                    </ul>
                    {% endif %}
                    
                    <!-- Requirements -->
                    {% if section.requirements %}
                    <div class="requirements-list">
                        <h4><i class="fas fa-clipboard-check"></i> Requirements</h4>
                        <ul>
                            {% for requirement in section.requirements %}
                            <li>{{ requirement }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                    
                    <!-- Numerical Data -->
                    {% if section.numbers %}
                    <div class="numbers-grid">
                        {% for number in section.numbers %}
                        <div class="number-block" title="{{ number.type }}">
                            <span class="number-value">{{ number.value }}</span>
                            <div class="number-label">{{ number.label }}</div>
                            {% if number.unit %}
                            <div class="number-unit">{{ number.unit }}</div>
                            {% endif %}
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    <!-- Subsections -->
                    {% if section.subsections %}
                    {% for subsection in section.subsections %}
                    <div class="subsection">
                        <h4>{{ subsection.title }}</h4>
                        {% if subsection.content %}
                        <p>{{ subsection.content }}</p>
                        {% endif %}
                        
                        <!-- Subsection Bullets -->
                        {% if subsection.bullets %}
                        <ul class="bullet-list">
                            {% for bullet in subsection.bullets %}
                            <li>{{ bullet }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        <!-- Subsection Requirements -->
                        {% if subsection.requirements %}
                        <div class="requirements-list">
                            <h4><i class="fas fa-clipboard-check"></i> Requirements</h4>
                            <ul>
                                {% for requirement in subsection.requirements %}
                                <li>{{ requirement }}</li>
                                {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                        
                        <!-- Subsection Numbers -->
                        {% if subsection.numbers %}
                        <div class="numbers-grid">
                            {% for number in subsection.numbers %}
                            <div class="number-block" title="{{ number.type }}">
                                <span class="number-value">{{ number.value }}</span>
                                <div class="number-label">{{ number.label }}</div>
                                {% if number.unit %}
                                <div class="number-unit">{{ number.unit }}</div>
                                {% endif %}
                            </div>
                            {% endfor %}
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <script>
        // Section Toggle Functionality
        let allExpanded = true;

        function toggleSection(sectionNum) {
            const content = document.getElementById(`section-${sectionNum}`);
            const header = content.previousElementSibling;
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                header.classList.remove('collapsed');
            } else {
                content.classList.add('collapsed');
                header.classList.add('collapsed');
            }
        }

        function toggleAllSections() {
            const contents = document.querySelectorAll('.section-content');
            const headers = document.querySelectorAll('.section-header.collapsible');
            const button = document.querySelector('.toggle-all');
            
            if (allExpanded) {
                contents.forEach(content => content.classList.add('collapsed'));
                headers.forEach(header => header.classList.add('collapsed'));
                button.innerHTML = '<i class="fas fa-eye-slash"></i> Expand All Sections';
                allExpanded = false;
            } else {
                contents.forEach(content => content.classList.remove('collapsed'));
                headers.forEach(header => header.classList.remove('collapsed'));
                button.innerHTML = '<i class="fas fa-eye"></i> Toggle All Sections';
                allExpanded = true;
            }
        }

        // Search Functionality
        document.getElementById('searchBar').addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const sections = document.querySelectorAll('.section');
            
            // Clear previous highlights
            clearHighlights();
            
            if (searchTerm.length > 2) {
                sections.forEach(section => {
                    const content = section.querySelector('.section-content');
                    const text = content.textContent.toLowerCase();
                    
                    if (text.includes(searchTerm)) {
                        section.style.display = 'block';
                        highlightText(content, searchTerm);
                        // Expand section if collapsed
                        content.classList.remove('collapsed');
                        section.querySelector('.section-header').classList.remove('collapsed');
                    } else {
                        section.style.opacity = '0.3';
                    }
                });
            } else {
                // Reset all sections
                sections.forEach(section => {
                    section.style.display = 'block';
                    section.style.opacity = '1';
                });
            }
        });

        function highlightText(element, searchTerm) {
            const walker = document.createTreeWalker(
                element,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node);
            }

            textNodes.forEach(textNode => {
                const text = textNode.textContent;
                const regex = new RegExp(`(${searchTerm})`, 'gi');
                if (regex.test(text)) {
                    const highlightedText = text.replace(regex, '<span class="highlighted">$1</span>');
                    const span = document.createElement('span');
                    span.innerHTML = highlightedText;
                    textNode.parentNode.replaceChild(span, textNode);
                }
            });
        }

        function clearHighlights() {
            const highlights = document.querySelectorAll('.highlighted');
            highlights.forEach(highlight => {
                const parent = highlight.parentNode;
                parent.replaceChild(document.createTextNode(highlight.textContent), highlight);
                parent.normalize();
            });
            
            // Reset section opacity
            document.querySelectorAll('.section').forEach(section => {
                section.style.opacity = '1';
            });
        }

        // Smooth scrolling for metric cards
        document.querySelectorAll('.metric-card').forEach(card => {
            card.addEventListener('click', function() {
                this.style.transform = 'scale(1.1)';
                setTimeout(() => {
                    this.style.transform = 'translateY(-5px)';
                }, 200);
            });
        });

        // Intersection Observer for animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animationDelay = '0s';
                    entry.target.style.animationPlayState = 'running';
                }
            });
        }, observerOptions);

        // Observe all sections
        document.querySelectorAll('.section').forEach(section => {
            observer.observe(section);
        });

        // Print functionality
        function printDocument() {
            // Expand all sections before printing
            if (!allExpanded) {
                toggleAllSections();
            }
            window.print();
        }

        // Add print button dynamically
        const printButton = document.createElement('button');
        printButton.innerHTML = '<i class="fas fa-print"></i> Print Document';
        printButton.className = 'toggle-all';
        printButton.style.marginLeft = '10px';
        printButton.onclick = printDocument;
        document.querySelector('.toggle-all').parentNode.insertBefore(printButton, document.querySelector('.toggle-all').nextSibling);

        // Initialize progress bar animation
        window.addEventListener('load', () => {
            setTimeout(() => {
                document.querySelector('.progress-fill').style.width = '100%';
            }, 500);
        });
    </script>
</body>
</html>
        """
    
    def _optimize_html(self, html: str) -> str:
        """Optimize the generated HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove empty elements
        for element in soup.find_all():
            if not element.get_text(strip=True) and not element.name in ['br', 'hr', 'img', 'input']:
                element.decompose()
        
        return str(soup)


class PDFToHTMLConverter:
    """Main converter class that orchestrates the entire conversion process"""
    
    def __init__(self, groq_api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the converter
        
        Args:
            groq_api_key: Your Groq API key (if not provided, will load from .env)
            model: Groq model to use for analysis
        """
        # Load API key from environment if not provided
        if groq_api_key is None:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not found. Please provide it as a parameter or set it in your .env file."
                )
        
        self.extractor = PDFExtractor()
        self.analyzer = ContentAnalyzer(groq_api_key, model)
        self.generator = HTMLGenerator()
        
        logger.info("Enhanced PDF to HTML Converter initialized")
    
    def convert_pdf_to_html(self, pdf_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert PDF to beautiful, interactive HTML
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the HTML file
            
        Returns:
            Generated HTML string
        """
        try:
            logger.info(f"Starting enhanced conversion of {pdf_path}")
            
            # Step 1: Extract text from PDF
            logger.info("Extracting text from PDF...")
            raw_text = self.extractor.extract_text_from_pdf(pdf_path)
            processed_text = self.extractor.preprocess_text(raw_text)
            
            if len(processed_text) < 100:
                raise ValueError("Insufficient text content extracted from PDF")
            
            # Step 2: Analyze document structure with enhanced features
            logger.info("Analyzing document structure with enhanced AI...")
            structure = self.analyzer.analyze_document_structure(processed_text)
            
            # Step 3: Generate enhanced HTML
            logger.info("Generating beautiful, interactive HTML...")
            html_output = self.generator.generate_html(structure)
            
            # Step 4: Save if output path provided
            if output_path:
                self.save_html(output_path, html_output)
                logger.info(f"Enhanced HTML saved to {output_path}")
            
            logger.info("Enhanced conversion completed successfully")
            return html_output
            
        except Exception as e:
            logger.error(f"Enhanced conversion failed: {e}")
            raise
    
    def save_html(self, output_path: str, html_content: str):
        """Save HTML content to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"Failed to save HTML: {e}")
            raise
    
    def batch_convert(self, pdf_directory: str, output_directory: str) -> List[str]:
        """
        Convert multiple PDFs in a directory with enhanced features
        
        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Directory to save HTML files
            
        Returns:
            List of generated HTML file paths
        """
        pdf_dir = Path(pdf_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                logger.info(f"Converting {pdf_file.name} with enhanced features")
                
                html_filename = pdf_file.stem + ".html"
                output_path = output_dir / html_filename
                
                html_content = self.convert_pdf_to_html(str(pdf_file), str(output_path))
                generated_files.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Failed to convert {pdf_file.name}: {e}")
                continue
        
        logger.info(f"Enhanced batch conversion completed. {len(generated_files)} files converted.")
        return generated_files

    def convert_with_custom_config(self, pdf_path: str, config: Dict[str, Any]) -> str:
        """
        Convert PDF with custom configuration
        
        Args:
            pdf_path: Path to the PDF file
            config: Custom configuration dictionary
                - color_scheme: Custom color scheme
                - enable_search: Enable/disable search functionality
                - enable_collapsible: Enable/disable collapsible sections
                - custom_css: Additional CSS styles
                
        Returns:
            Generated HTML string
        """
        # Store original template
        original_template = self.generator.template
        
        try:
            # Modify generator based on config
            if config.get('custom_css'):
                self.generator.template = self.generator.template.replace(
                    '</style>',
                    f"{config['custom_css']}\n</style>"
                )
            
            # Convert with enhanced features
            html_output = self.convert_pdf_to_html(pdf_path)
            
            return html_output
            
        finally:
            # Restore original template
            self.generator.template = original_template


# Example usage and testing functions
def example_usage():
    """Example of how to use the enhanced converter"""
    
    # Initialize converter (API key loaded from .env file)
    try:
        converter = PDFToHTMLConverter()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set GROQ_API_KEY in your .env file or pass it as a parameter:")
        print("converter = PDFToHTMLConverter(groq_api_key='your_api_key_here')")
        return
    
    # Convert single PDF with enhanced features
    try:
        html_output = converter.convert_pdf_to_html(
            pdf_path="Ai Agent Business Requirements.pdf",
            output_path="enhanced_document.html"
        )
        print("Enhanced conversion successful!")
        print(f"HTML length: {len(html_output)} characters")
        print("Features included:")
        print("- Interactive collapsible sections")
        print("- Search functionality")
        print("- Enhanced bullet points and numbered lists")
        print("- Numerical data blocks")
        print("- Requirements highlighting")
        print("- Key metrics dashboard")
        print("- Mobile responsive design")
        
    except Exception as e:
        print(f"Enhanced conversion failed: {e}")
    
    # # Example with custom configuration
    # try:
    #     custom_config = {
    #         'color_scheme': 'purple-requirements',
    #         'custom_css': '''
    #             .custom-highlight {
    #                 background: linear-gradient(45deg, #ff6b6b, #feca57);
    #                 padding: 10px;
    #                 border-radius: 5px;
    #                 margin: 10px 0;
    #             }
    #         '''
    #     }
        
    #     html_output = converter.convert_with_custom_config(
    #         pdf_path="Ai Agent Business Requirements.pdf",
    #         config=custom_config
    #     )
        
    #     converter.save_html("custom_styled_document.html", html_output)
    #     print("Custom styled conversion completed!")
        
    # except Exception as e:
    #     print(f"Custom conversion failed: {e}")
    
    # # Batch convert multiple PDFs
    # try:
    #     generated_files = converter.batch_convert(
    #         pdf_directory="./pdfs/",
    #         output_directory="./enhanced_html_output/"
    #     )
    #     print(f"Enhanced batch conversion completed: {len(generated_files)} files")
        
    # except Exception as e:
    #     print(f"Enhanced batch conversion failed: {e}")



if __name__ == "__main__":
    
    # Run example usage
    example_usage()