from .nodes import LlamaCPPModelLoader, LlamaCPPOptions, LlamaCPPEngine, LlamaCPPMemoryCleanup, FireRedOCREngine, DoclingLayoutAnalyzer, DoclingLayoutMarkdownEngine, PDFLoader, DocLayoutYOLOLoader, DocLayoutMarkdownEngine

NODE_CLASS_MAPPINGS = {
    "LlamaCPPModelLoader": LlamaCPPModelLoader,
    "LlamaCPPOptions": LlamaCPPOptions,
    "LlamaCPPEngine": LlamaCPPEngine,
    "LlamaCPPMemoryCleanup": LlamaCPPMemoryCleanup,
    "FireRedOCREngine": FireRedOCREngine,
    "DoclingLayoutAnalyzer": DoclingLayoutAnalyzer,
    "DoclingLayoutMarkdownEngine": DoclingLayoutMarkdownEngine,
    "PDFLoader": PDFLoader,
    "DocLayoutYOLOLoader": DocLayoutYOLOLoader,
    "DocLayoutMarkdownEngine": DocLayoutMarkdownEngine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPPModelLoader": "Llama CPP Model Loader",
    "LlamaCPPOptions": "Llama CPP Options",
    "LlamaCPPEngine": "Llama CPP Engine",
    "LlamaCPPMemoryCleanup": "Llama CPP Memory Cleanup",
    "FireRedOCREngine": "FireRed OCR Engine",
    "DoclingLayoutAnalyzer": "Docling Layout Analyzer",
    "DoclingLayoutMarkdownEngine": "Docling Layout Markdown Engine",
    "PDFLoader": "PDF Loader",
    "DocLayoutYOLOLoader": "DocLayout YOLO Loader",
    "DocLayoutMarkdownEngine": "DocLayout Markdown Engine",
}
