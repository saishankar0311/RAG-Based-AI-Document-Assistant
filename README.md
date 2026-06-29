# 🤖RAG Based AI Chatbot - AI Document Assistant

Project Explanation:
This AI-Powered Document Chat Assistant is a sophisticated question-answering system that enables users to upload documents (PDF, DOCX, TXT) and engage in intelligent conversations about their content. The application processes documents by extracting text, cleaning and preprocessing it, then splitting it into semantic chunks for efficient retrieval. It utilizes advanced Retrieval-Augmented Generation (RAG) architecture, combining vector embeddings with traditional keyword search for hybrid document retrieval. When users ask questions, the system searches through the document chunks using both semantic similarity (via sentence transformers) and keyword matching, retrieves the most relevant context, and generates comprehensive answers using Mistral AI's language model. The interface features a clean, single-section design with real-time chat functionality, confidence scoring for responses, and automatic source attribution, making it ideal for document analysis, research assistance, and content exploration.
Technical Stack & Keywords
Frontend & UI

Streamlit - Python web framework for rapid UI development, handles user interactions and real-time updates
HTML/CSS - Custom styling for chat bubbles, animations, and responsive design elements
JavaScript (implicit) - Streamlit's built-in reactivity for dynamic content updates

Document Processing

PyPDF2 - PDF text extraction library, handles multi-page document parsing and content retrieval
python-docx - Microsoft Word document processing, extracts text from DOCX files including tables
Text Preprocessing - Regular expressions (re module) for cleaning, normalizing, and structuring raw text
Chunking Algorithm - Overlapping text segmentation with sentence boundary detection for context preservation

AI & Machine Learning

Sentence Transformers - Neural network models for semantic text embeddings, converts text to vector representations
all-MiniLM-L6-v2 - Lightweight transformer model optimized for sentence similarity tasks
Mistral AI API - Large language model for natural language generation and question answering
RAG (Retrieval-Augmented Generation) - Architecture combining information retrieval with generative AI

Vector Search & Indexing

FAISS - Facebook's similarity search library for efficient nearest neighbor retrieval in high-dimensional spaces
Scikit-learn - Machine learning library providing TF-IDF vectorization and cosine similarity calculations
TF-IDF (Term Frequency-Inverse Document Frequency) - Statistical measure for keyword-based document relevance
Hybrid Search - Combines semantic embeddings with traditional keyword matching for improved accuracy

Data Management

NumPy - Numerical computing library for vector operations and mathematical computations
Session State Management - Streamlit's persistent storage for maintaining chat history and document data
Metadata Tracking - Document source attribution, chunk indexing, and confidence scoring

APIs & Communication

Requests - HTTP library for API communication with Mistral AI services
python-dotenv - Environment variable management for secure API key storage
Error Handling - Retry logic, timeout management, and user-friendly error messages

Architecture Patterns

Modular Design - Separated concerns with distinct files for document loading, vector storage, and QA processing
Real-time Processing - Streaming document upload with progress indicators and immediate feedback
Confidence Scoring - Similarity score aggregation for response reliability assessment



# To download requiremnts
pip install -r requirements.txt

# To run
streamlit run app.py"# RAG-Based-AI-Document-Assistant" 
