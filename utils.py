import json
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import streamlit as st
import io

def get_file_stats(chunks):
    """Calculate statistics for processed document chunks"""
    if not chunks:
        return {
            "chunks": 0,
            "words": 0,
            "chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    total_chars = sum(len(chunk) for chunk in chunks)
    total_words = sum(len(chunk.split()) for chunk in chunks)
    chunk_sizes = [len(chunk) for chunk in chunks]
    
    return {
        "chunks": len(chunks),
        "words": total_words,
        "chars": total_chars,
        "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
        "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
        "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
        "avg_words_per_chunk": total_words // len(chunks) if chunks else 0
    }

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def clean_text_for_display(text, max_length=500):
    """Clean and truncate text for display"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text

def export_chat_history(history, format_type="json"):
    """Export chat history in different formats"""
    if not history:
        return None
    
    fmt = format_type.lower()
    if fmt == "json":
        return export_as_json(history)
    elif fmt == "pdf":
        return export_as_pdf(history)
    elif fmt in ["markdown", "md"]:
        return export_as_markdown(history)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")

def get_text_complexity_score(text):
    """Calculate text complexity score"""
    if not text:
        return 0
    
    sentences = len(re.split(r'[.!?]+', text))
    words = len(text.split())
    characters = len(text)
    
    if sentences == 0:
        return 0
    
    # Flesch Reading Ease approximation
    avg_sentence_length = words / sentences
    avg_syllables_per_word = estimate_syllables(text) / words if words > 0 else 0
    
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Normalize to 0-100 scale
    return max(0, min(100, score))

def estimate_syllables(text):
    """Estimate syllable count in text"""
    words = text.lower().split()
    syllable_count = 0
    
    for word in words:
        word = re.sub(r'[^a-z]', '', word)
        if len(word) == 0:
            continue
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        # Every word has at least 1 syllable
        syllables = max(1, syllables)
        syllable_count += syllables
    
    return syllable_count

def format_confidence_level(confidence):
    """Format confidence level for display"""
    if confidence >= 0.9:
        return "🟢 Very High", "#28a745"
    elif confidence >= 0.7:
        return "🔵 High", "#007bff"
    elif confidence >= 0.5:
        return "🟡 Medium", "#ffc107"
    elif confidence >= 0.3:
        return "🟠 Low", "#fd7e14"
    else:
        return "🔴 Very Low", "#dc3545"

def create_progress_bar(current, total, width=30):
    """Create a text-based progress bar"""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = int((current / total) * width)
    bar = "=" * progress + "-" * (width - progress)
    percentage = int((current / total) * 100)
    
    return f"[{bar}] {percentage}%"

def truncate_text(text, max_length=100, suffix="..."):
    """Truncate text intelligently"""
    if not text or len(text) <= max_length:
        return text
    
    # Try to break at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can break at a reasonable point
        return text[:last_space] + suffix
    else:
        return text[:max_length] + suffix

def detect_document_language(text):
    """Detect document language (simple heuristic)"""
    if not text:
        return "unknown"
    
    # Simple language detection based on common words
    english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'}
    spanish_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para'}
    french_words = {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se'}
    
    words = set(text.lower().split()[:100])  # Check first 100 words
    
    english_score = len(words.intersection(english_words))
    spanish_score = len(words.intersection(spanish_words))
    french_score = len(words.intersection(french_words))
    
    if english_score >= spanish_score and english_score >= french_score:
        return "english"
    elif spanish_score >= french_score:
        return "spanish"
    elif french_score > 0:
        return "french"
    else:
        return "unknown"

def sanitize_filename(filename):
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 255 - len(ext) - 1
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename or "unnamed_file"

def extract_keywords(text, max_keywords=10):
    """Extract key terms from text"""
    if not text:
        return []
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Common stop words
    stop_words = {
        'that', 'with', 'have', 'this', 'will', 'you', 'from', 'they', 'know',
        'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come',
        'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take',
        'than', 'them', 'well', 'were', 'what', 'your', 'also', 'back', 'call',
        'came', 'each', 'even', 'find', 'going', 'great', 'high', 'keep', 'last',
        'left', 'life', 'live', 'made', 'most', 'move', 'must', 'need', 'next',
        'only', 'open', 'place', 'right', 'same', 'seem', 'still', 'those',
        'under', 'used', 'using', 'want', 'water', 'ways', 'work', 'world',
        'would', 'years', 'young'
    }
    
    # Filter words and count frequency
    word_counts = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]

def format_file_info(file_info):
    """Format file information for display"""
    if not file_info:
        return "No file information available"
    
    name = file_info.get('name', 'Unknown')
    size = file_info.get('size', '0 B')
    file_type = file_info.get('type', 'Unknown')
    chunks = file_info.get('chunks', 0)
    uploaded_at = file_info.get('uploaded_at', 'Unknown')
    
    try:
        # Format timestamp
        if uploaded_at != 'Unknown':
            dt = datetime.fromisoformat(uploaded_at)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M')
        else:
            formatted_time = 'Unknown'
    except:
        formatted_time = str(uploaded_at)
    
    return f"""
📄 **{name}**
📊 Size: {size} | Type: {file_type}
🔢 Chunks: {chunks} | Uploaded: {formatted_time}
    """.strip()

def create_download_link(data, filename, mime_type):
    """Create a download link for data"""
    import base64
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def validate_file_upload(uploaded_file):
    """Comprehensive file upload validation"""
    if not uploaded_file:
        return False, "No file provided"
    
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024
    if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size:
        return False, f"File too large. Maximum size: 10MB. Your file: {format_file_size(uploaded_file.size)}"
    
    # Check file extension
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_extension = '.' + uploaded_file.name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        return False, f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
    
    # Check filename
    if len(uploaded_file.name) > 255:
        return False, "Filename too long (max 255 characters)"
    
    # Check for potentially malicious filenames
    dangerous_patterns = ['../', '..\\', '<script', 'javascript:', 'data:']
    filename_lower = uploaded_file.name.lower()
    for pattern in dangerous_patterns:
        if pattern in filename_lower:
            return False, "Potentially unsafe filename detected"
    
    return True, "File is valid"

def get_system_info():
    """Get system information for debugging"""
    import platform
    import sys
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "streamlit_version": st.__version__ if hasattr(st, '__version__') else "Unknown",
        "timestamp": datetime.now().isoformat()
    }

def export_as_json(history):
    """Export chat history as JSON"""
    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_conversations": len(history),
        "conversations": []
    }
    
    for i, (question, answer, context, metadata) in enumerate(history, 1):
        conversation = {
            "id": i,
            "question": question,
            "answer": answer,
            "metadata": metadata,
            "context_preview": context[:200] + "..." if len(context) > 200 else context
        }
        export_data["conversations"].append(conversation)
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def export_as_pdf(history):
    """Export chat history as PDF"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        question_style = ParagraphStyle(
            'Question',
            parent=styles['Normal'],
            fontSize=12,
            textColor=HexColor('#1B4F72'),
            fontName='Helvetica-Bold',
            leftIndent=20,
            spaceBefore=20,
            spaceAfter=10
        )
        
        answer_style = ParagraphStyle(
            'Answer',
            parent=styles['Normal'],
            fontSize=11,
            textColor=HexColor('#2C3E50'),
            leftIndent=20,
            rightIndent=20,
            spaceBefore=10,
            spaceAfter=20
        )
        
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#7F8C8D'),
            leftIndent=20,
            spaceAfter=15
        )
        
        # Build PDF content
        content = []
        
        # Title
        content.append(Paragraph("🤖 AI Document Assistant - Chat History", title_style))
        content.append(Spacer(1, 20))
        
        # Export info
        export_info = f"Exported on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>Total Conversations: {len(history)}"
        content.append(Paragraph(export_info, styles['Normal']))
        content.append(Spacer(1, 30))
        
        # Chat history
        for i, (question, answer, context, metadata) in enumerate(history, 1):
            # Question
            content.append(Paragraph(f"<b>Question {i}:</b> {clean_html(question)}", question_style))
            
            # Answer
            content.append(Paragraph(f"<b>Answer:</b> {clean_html(answer)}", answer_style))
            
            # Metadata
            if metadata:
                confidence = metadata.get('confidence', 0)
                source_file = metadata.get('source_file', 'Unknown')
                timestamp = metadata.get('timestamp', 'Unknown')
                
                meta_text = f"Confidence: {confidence:.1%} | Source: {source_file} | Time: {format_timestamp(timestamp)}"
                content.append(Paragraph(meta_text, metadata_style))
            
            # Add separator line
            if i < len(history):
                content.append(Spacer(1, 10))
                content.append(Paragraph("<hr/>", styles['Normal']))
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def clean_html(text):
    """Clean text for HTML/PDF display"""
    if not text:
        return ""
    
    # Escape HTML characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#x27;")
    
    # Convert markdown-like formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
    
    return text

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if not timestamp:
        return "Unknown"
    
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
        
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp)

def validate_question(question):
    """Validate user question"""
    if not question or not question.strip():
        return False, "Question cannot be empty"
    
    if len(question.strip()) < 3:
        return False, "Question is too short"
    
    if len(question) > 1000:
        return False, "Question is too long (max 1000 characters)"
    
    # Check for spam or repetitive content
    if len(set(question.lower().split())) < len(question.split()) * 0.3:
        return False, "Question appears to be spam or repetitive"
    
    return True, "Valid question"

def generate_sample_questions(document_type="general"):
    """Generate sample questions based on document type"""
    samples = {
        "general": [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the important conclusions?",
            "Are there any specific recommendations?",
            "What methodology was used?"
        ],
        "academic": [
            "What is the research question or hypothesis?",
            "What methodology was used in this study?",
            "What are the main findings?",
            "What are the limitations of this research?",
            "What future research is suggested?"
        ],
        "technical": [
            "What are the system requirements?",
            "How does the implementation work?",
            "What are the key features?",
            "Are there any known issues or limitations?",
            "What are the configuration options?"
        ],
        "business": [
            "What are the key business objectives?",
            "What is the market analysis?",
            "What are the financial projections?",
            "What are the risks and mitigation strategies?",
            "What are the next steps or recommendations?"
        ]
    }
    
    return samples.get(document_type, samples["general"])

def calculate_reading_time(text):
    """Calculate estimated reading time"""
    if not text:
        return "0 minutes"
    
    # Average reading speed: 200 words per minute
    words = len(text.split())
    minutes = words / 200
    
    if minutes < 1:
        return "< 1 minute"
    elif minutes < 60:
        hours = int(minutes / 60)
        remaining_minutes = int(minutes % 60)
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
        else:
            return f"{int(minutes)} minute{'s' if int(minutes) != 1 else ''}"
    else:
        hours = int(minutes / 60)
        remaining_minutes = int(minutes % 60)
        return f"{hours} hour{'s' if hours > 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"

def get_document_insights(text):
    """Get comprehensive document insights"""
    if not text:
        return {}
    
    insights = {
        "word_count": len(text.split()),
        "character_count": len(text),
        "paragraph_count": len(text.split('\n\n')),
        "reading_time": calculate_reading_time(text),
        "complexity_score": get_text_complexity_score(text),
        "language": detect_document_language(text),
        "keywords": extract_keywords(text),
        "estimated_pages": len(text) / 2500,  # Approximate chars per page
    }
    
    return insights

def create_summary_stats(stats):
    """Create formatted summary statistics"""
    if not stats:
        return "No statistics available"
    
    formatted_stats = f"""
    📊 **Document Statistics**
    
    **Content Overview:**
    - 📝 Words: {stats.get('word_count', 0):,}
    - 🔤 Characters: {stats.get('character_count', 0):,}
    - 📄 Paragraphs: {stats.get('paragraph_count', 0):,}
    - ⏱️ Reading Time: {stats.get('reading_time', 'Unknown')}
    - 📖 Estimated Pages: {stats.get('estimated_pages', 0):.1f}
    
    **Analysis:**
    - 🌐 Language: {stats.get('language', 'Unknown').title()}
    - 🧠 Complexity Score: {stats.get('complexity_score', 0):.1f}/100
    
    **Keywords:** {', '.join(stats.get('keywords', [])[:5]) if stats.get('keywords') else 'None detected'}
    """
    
    return formatted_stats.strip()

def handle_processing_error(error, context="Processing"):
    """Handle and format processing errors"""
    error_msg = str(error)
    
    # Common error patterns and user-friendly messages
    error_patterns = {
        "file not found": "The file could not be found. Please check the file path.",
        "permission denied": "Permission denied. Please check file permissions.",
        "memory": "The file is too large to process. Try with a smaller file.",
        "encoding": "The file encoding is not supported. Please use UTF-8 encoded files.",
        "pdf": "Error processing PDF. The file may be corrupted or password-protected.",
        "docx": "Error processing Word document. The file may be corrupted.",
        "timeout": "Processing timed out. The file may be too large or complex."
    }
    
    user_friendly_msg = f"{context} failed: "
    
    for pattern, message in error_patterns.items():
        if pattern in error_msg.lower():
            user_friendly_msg += message
            break
    else:
        user_friendly_msg += "An unexpected error occurred. Please try again."
    
    return user_friendly_msg

def validate_document_content(content):
    """Validate document content before processing"""
    if not content:
        return False, "Document is empty"
    
    if len(content.strip()) < 50:
        return False, "Document content is too short for meaningful analysis"
    
    # Check for reasonable text content (not just whitespace or special chars)
    text_chars = sum(1 for char in content if char.isalnum() or char.isspace())
    if text_chars < len(content) * 0.5:
        return False, "Document contains too many non-text characters"
    
    return True, "Document content is valid"

def create_chat_context(question, history, max_context_length=2000):
    """Create context for chat from previous conversations"""
    if not history:
        return ""
    
    context_parts = []
    current_length = 0
    
    # Add most recent conversations first
    for q, a, _, _ in reversed(history):
        entry = f"Previous Q: {q}\nPrevious A: {a[:200]}...\n"
        if current_length + len(entry) > max_context_length:
            break
        context_parts.insert(0, entry)
        current_length += len(entry)
    
    return "\n".join(context_parts) if context_parts else ""

def optimize_text_for_search(text, max_length=5000):
    """Optimize text for search/retrieval operations"""
    if not text or len(text) <= max_length:
        return text
    
    # Try to find good breaking points (paragraph boundaries, sentences)
    paragraphs = text.split('\n\n')
    optimized = ""
    
    for paragraph in paragraphs:
        if len(optimized) + len(paragraph) > max_length:
            # If adding this paragraph would exceed limit, try to add sentences
            sentences = paragraph.split('. ')
            for sentence in sentences:
                if len(optimized) + len(sentence) + 2 > max_length:
                    break
                optimized += sentence + ". "
            break
        optimized += paragraph + "\n\n"
    
    return optimized.strip()

def get_file_type_icon(file_extension):
    """Get appropriate icon for file type"""
    icons = {
        '.pdf': '📄',
        '.docx': '📝',
        '.doc': '📝',
        '.txt': '📄',
        '.md': '📝',
        '.html': '🌐',
        '.rtf': '📝'
    }
    return icons.get(file_extension.lower(), '📄')

def create_backup_data(data, backup_type="chat_history"):
    """Create backup data with metadata"""
    backup = {
        "backup_type": backup_type,
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "data": data,
        "checksum": hash(str(data))  # Simple checksum for data integrity
    }
    return backup

def restore_backup_data(backup_data):
    """Restore data from backup with validation"""
    try:
        if not isinstance(backup_data, dict):
            return None, "Invalid backup format"
        
        if "data" not in backup_data:
            return None, "Backup data is missing"
        
        # Verify checksum if available
        if "checksum" in backup_data:
            current_checksum = hash(str(backup_data["data"]))
            if current_checksum != backup_data["checksum"]:
                return None, "Backup data may be corrupted"
        
        return backup_data["data"], "Backup restored successfully"
    
    except Exception as e:
        return None, f"Error restoring backup: {str(e)}"

def log_user_interaction(action, details=None):
    """Log user interactions for analytics (privacy-compliant)"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details or {},
        "session_id": hash(datetime.now().date())  # Daily session identifier
    }
    
    # In a real implementation, you would send this to your analytics service
    # For now, we'll just return the log entry for potential local storage
    return log_entry

def export_as_markdown(history):
    """Export chat history as Markdown string"""
    lines = [
        "# Atlas Document Assistant - Chat History",
        f"Exported on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        f"Total Conversations: {len(history)}",
        "---",
        ""
    ]
    for i, (question, answer, context, metadata) in enumerate(history, 1):
        lines.append(f"## Question {i}: {question}")
        lines.append("")
        lines.append(f"### Answer:")
        lines.append(answer)
        lines.append("")
        if metadata:
            confidence = metadata.get("confidence", 0)
            source_file = metadata.get("source_file", "Unknown")
            response_time = metadata.get("response_time", 0)
            lines.append(f"*Confidence: {confidence:.1%} | Source: {source_file} | Response time: {response_time:.2f}s*")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)

def is_conversational_query(query):
    """
    Detect if the user query is a simple greeting, thank you, or feedback response.
    """
    cleaned = re.sub(r'[^\w\s]', '', query.lower()).strip()
    words = cleaned.split()
    
    if not words:
        return False
        
    # Common conversational words
    conversational_words = {
        "hi", "hello", "hey", "greetings", "hola",
        "thanks", "thank", "thankyou", "thx", "thanku",
        "good", "nice", "great", "awesome", "perfect", "okay", "ok", "cool",
        "bye", "goodbye", "exit"
    }
    
    # Phrases
    conversational_phrases = [
        "thank you", "thank you so much", "thanks a lot", "you are welcome", "youre welcome",
        "good job", "well done", "sounds good", "sounds great", "makes sense"
    ]
    
    # Check if the query is a single conversational word
    if len(words) <= 2:
        if all(w in conversational_words for w in words):
            return True
            
    # Check exact match for phrases
    if cleaned in conversational_phrases:
        return True
        
    return False