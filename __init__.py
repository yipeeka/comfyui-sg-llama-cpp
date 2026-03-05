from .nodes import LlamaCPPModelLoader, LlamaCPPOptions, LlamaCPPEngine, LlamaCPPMemoryCleanup, FireRedOCREngine, PDFLoader

NODE_CLASS_MAPPINGS = {
    "LlamaCPPModelLoader": LlamaCPPModelLoader,
    "LlamaCPPOptions": LlamaCPPOptions,
    "LlamaCPPEngine": LlamaCPPEngine,
    "LlamaCPPMemoryCleanup": LlamaCPPMemoryCleanup,
    "FireRedOCREngine": FireRedOCREngine,
    "PDFLoader": PDFLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCPPModelLoader": "Llama CPP Model Loader",
    "LlamaCPPOptions": "Llama CPP Options",
    "LlamaCPPEngine": "Llama CPP Engine",
    "LlamaCPPMemoryCleanup": "Llama CPP Memory Cleanup",
    "FireRedOCREngine": "FireRed OCR Engine",
    "PDFLoader": "PDF Loader",
}
