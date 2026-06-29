# Atlas Document Assistant - Newly Implemented Features

This document provides a detailed breakdown of the features and updates newly implemented in the project. These additions are designed to improve response precision, interface responsiveness, source tracking clarity, and local execution accessibility.

---

## 1. Hierarchical Parent-Child Chunking

To resolve the classic trade-off in RAG between retrieving highly specific search matches (which require small text segments) and providing sufficient context to the LLM (which requires larger text blocks), we implemented a hierarchical indexing strategy.

- **How it works**:
  - The document loader first segments the cleaned text into large segments representing broad paragraphs or themes, called **Parent Chunks**.
  - It then splits each parent chunk into smaller overlapping segments, called **Child Chunks**.
  - The vector store embeds only the child chunks, adding them to the FAISS indexing system. Each child chunk metadata block maintains a direct reference mapping back to its parent context.
  - When you submit a question, semantic search identifies the best matching child chunks. However, the system retrieves and forwards the corresponding parent chunks to the LLM. This provides the LLM with deep context to synthesize complete, well-formed answers, while keeping embedding searches highly localized.

---

## 2. Local LLM Integration (Ollama Support)

To facilitate local offline usage and privacy-focused operations, the QA engine now supports local models running via Ollama in addition to cloud-based Mistral API endpoints.

- **How it works**:
  - Users can select their active provider from the Configuration panel.
  - Selecting Ollama reveals settings for specifying the custom endpoint host address and model names.
  - The QA engine establishes local server-sent stream streams, enabling processing of local models directly on your hardware.

---

## 3. Real-Time Response Streaming

Rather than forcing you to wait in front of a loading spinner for the entire response to generate, chat answers are now streamed in real time.

- **How it works**:
  - The QA engine leverages HTTP streaming endpoints. It decodes incoming chunks on-the-fly.
  - The UI updates token-by-token using placeholder containers, showing a typewriter effect.
  - This significantly improves perceived response latency, aligning the application with modern conversational assistant standards.

---

## 4. Multi-Thread Session Management

You can now keep separate conversations going at the same time without cross-talk or losing history.

- **How it works**:
  - A thread selector and controls are added to the sidebar.
  - You can create new threads, rename active threads, or delete threads you no longer need.
  - Each thread preserves its chat history in isolation, and the UI adapts automatically when you toggle threads.

---

## 5. Interactive Citation Previews

Transparency is essential for RAG systems. The assistant now renders detailed source attribution drawers beneath every response.

- **How it works**:
  - An expandable panel lists each retrieved source file.
  - Within the panel, tabbed views let you inspect the **Child Snippet** (the exact text that triggered the semantic database match) side-by-side with the **Parent Context** (the surrounding text that was fed into the LLM context window).
  - This allows you to verify the response and trace the AI's logic step-by-step.

---

## 6. Search Document Filtering

When working with many documents, you might want to query specific files instead of searching the entire database.

- **How it works**:
  - The sidebar displays a checklist of all uploaded files.
  - You can check or uncheck individual files.
  - The vector store uses this checklist to filter keyword and semantic search results dynamically, ensuring the retrieved context only comes from allowed documents.

---

## 7. Markdown Document Ingestion Support

In addition to PDF, DOCX, and TXT files, the loader now natively parses Markdown files.

- **How it works**:
  - The document validator and parser recognize Markdown extension endings and mime types.
  - They process markdown headings and layouts, ensuring structure is preserved.

---

## 8. Detailed Document Insights Dashboard

A dedicated tab provides analytical breakdowns of uploaded documents.

- **How it works**:
  - The dashboard calculates structural metrics (Word Count, Estimated Pages, Complexity Index).
  - It uses word frequency metrics to highlight key terms.
  - Estimates reading time based on typical reading speeds.

---

## 9. Advanced Settings & Exports Controls

The configuration panel gives you precise control over LLM parameters and data exports.

- **How it works**:
  - Sliders let you set the Temperature, Top-P, and Max Prediction tokens for responses.
  - You can download your current chat thread in three formats:
    - **JSON**: A structured export including metadata, timestamp, confidence scores, and source maps.
    - **Markdown**: A clean, formatted document containing the conversation log.
    - **PDF**: A document format suited for printing and sharing.

---

## 10. Document Comparison (Multi-Document Diffing)

To make it easy to analyze changes and compare different source documents side-by-side (such as draft contracts or project readmes), we integrated a custom multi-document comparison workflow.

- **How it works**:
  - In the Chat Workspace, toggle to "Document Comparison Mode".
  - Select "Document A" and "Document B" from your uploaded document files.
  - When you ask a question (such as "What are the core differences in pricing or liabilities?"), the system triggers independent search queries on both documents separately. This guarantees equal representation in the retrieved context.
  - Chunks from both sources are formatted with clean headers (`=== SOURCE DOCUMENT A ===` and `=== SOURCE DOCUMENT B ===`) and fed into a custom comparison prompt template.
  - The template guides the LLM to output comparisons clearly, utilizing a side-by-side Markdown comparison table.
  - Citations are grouped into two separate tabs matching Document A and Document B for easy verification.

---

## 11. Interactive PDF Viewer with Page Jump

To allow users to view actual source PDF documents directly within the interface in a synchronized manner, we built a custom double-column PDF viewer layout.

- **How it works**:
  - When you upload a PDF file, it is automatically cached inside the project directory under `data/uploaded_pdfs/`.
  - When the document is processed, each chunk's position in the text is matched backwards to identify its exact page number inside the PDF, which is stored in the vector index's search metadata.
  - In the chat bubbles, expanding any source citation reveals a button labeled `📖 Open [file] on Page N`.
  - Clicking this button opens a side-by-side workspace: the left side displays the chat thread, and the right side opens the interactive PDF viewer.
  - The viewer automatically jumps to the cited page using native browser URL parameters (`#page=N`) and provides manual page navigation buttons (`Prev` and `Next`).

---

## 12. Automatic State Persistence & Caching

To prevent losing uploaded documents, vector store segments, and chat threads upon page reload, we integrated a full disk persistence framework.

- **How it works**:
  - The application automatically tracks state mutations (such as file uploads, thread creations/deletions, history clears, and assistant replies).
  - On every mutation, the app serializes the current state:
    - **Vector Store**: Saved to `data/vector_store.pkl` (with the FAISS index written separately to `data/vector_store.pkl.faiss`).
    - **Session Metadata**: Stores the uploaded files registry list, thread dictionary (with message history logs), active thread ID, and active slider configuration inside `data/app_metadata.pkl`.
  - When the app is initialized (e.g. on page refresh or startup), it checks for these files and automatically re-populates the vector index and session state, allowing for a completely seamless experience.
