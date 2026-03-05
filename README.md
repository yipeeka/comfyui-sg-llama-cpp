# comfyui-sg-llama-cpp

ComfyUI custom node that acts as a llama-cpp-python wrapper, with support for vision models and document OCR. Allows generating text responses from prompts using llama.cpp.

![Screenshot](assets/node_preview.png)

## Features

- Load and use GGUF models (including vision models)
- Generate text prompts using llama.cpp
- Support for multi-modal inputs (multiple images per request)
- Vision pipeline powered by `MTMDChatHandler` (llama-cpp-python ≥ v0.3.28)
- Thinking/reasoning model support with `enable_thinking` toggle and `strip_thinking` output
- Concurrent multimodal image decoding (ThreadPoolExecutor)
- Memory management options
- **PDF Loader**: Convert PDF pages to ComfyUI IMAGE batches (requires `pymupdf`)
- **FireRed-OCR**: Dedicated document OCR node (PDF/image → Markdown, with LaTeX & HTML table support)

## Installation

1. Install the required dependency/wheel from:
   ```
   https://github.com/JamePeng/llama-cpp-python/releases
   ```

2. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/sebagallo/comfyui-sg-llama-cpp
   ```

3. Install Python dependencies:
   ```bash
   pip install pymupdf
   ```

4. Restart ComfyUI.

## Node Reference

### LlamaCPPModelLoader
Loads GGUF model files and prepares them for use.

**Inputs**
- **Required**:
  - `model_name`: Select the GGUF model file to load.
- **Optional**:
  - `chat_format`: Chat template to use (default: `llama-2`). Vision formats are prefixed with `vision-` (e.g. `vision-qwen3vl`, `vision-qwen35`, `vision-llava15`).
  - `mmproj_model_name`: Multi-modal projector model for vision (default: `None`). Required for vision inference.

**Outputs**
- `model`: The loaded Llama model object.

> **Note**: At startup, detected vision handlers are printed to the console:
> `[LlamaCPP] Detected vision handlers: ['vision-gemma3', 'vision-llava15', 'vision-qwen35', ...]`

---

### LlamaCPPOptions
Configures advanced parameters for the model.

**Inputs**
- **Optional**:
  - `n_gpu_layers`: Layers to offload to GPU VRAM. `0` = CPU only, `-1` = all layers on GPU (default: `-1`).
  - `n_ctx`: Context window size. Larger = more memory (default: `2048`). For OCR/vision, use `32768`+.
  - `n_threads`: CPU threads to use. `-1` = auto (default: `-1`).
  - `n_threads_batch`: Threads for batch processing. `-1` = auto (default: `-1`).
  - `n_batch`: Batch size (default: `512`).
  - `n_ubatch`: Micro-batch size (default: `512`).
  - `main_gpu`: Main GPU ID (default: `0`).
  - `offload_kqv`: Offload KV cache to GPU VRAM. Faster but uses more VRAM; disable if VRAM is tight (default: `Enabled`).
  - `numa`: NUMA affinity (default: `Disabled`).
  - `use_mmap`: Memory-mapped file loading (default: `Enabled`).
  - `use_mlock`: Lock memory-mapped files in RAM (default: `Disabled`).
  - `verbose`: Verbose llama.cpp logging (default: `Disabled`).
  - `vision_use_gpu`: Use GPU for vision encoder (default: `Enabled`).
  - `vision_image_min_tokens`: Minimum image tokens, `-1` for default (default: `-1`).
  - `vision_image_max_tokens`: Maximum image tokens, `-1` for default (default: `-1`).
  - `vision_enable_thinking`: Enable thinking mode for MiniCPM-V 4.5 (default: `Disabled`).

**Outputs**
- `options`: A configuration dictionary.

---

### LlamaCPPEngine
The main generation node.

**Inputs**
- **Required**:
  - `model`: The model from `LlamaCPPModelLoader`.
  - `prompt`: The text prompt.
- **Optional**:
  - `image`: Input image(s) for vision models. Accepts a full IMAGE batch — all images are sent to the model in a single request (requires `vision-*` chat format + mmproj).
  - `options`: Options from `LlamaCPPOptions`.
  - `system_prompt`: System instruction (default: empty).
  - `memory_cleanup`: Memory strategy after generation (default: `close`).
    - `close`: Free the model after each run.
    - `backend_free`: Free model + llama backend.
    - `full_cleanup`: Free model + backend + CUDA cache.
    - `persistent`: Keep model loaded between runs (fastest for repeated use). Automatically reloads if the model changes.
  - `response_format`: `text` or `json_object` (default: `text`).
  - `enable_thinking`: Enable thinking/reasoning output for thinking models (Qwen3, Qwen3.5, QwQ, etc.). When disabled, injects an empty `<think></think>` prefix to force the model to skip reasoning (default: `Enabled`).
  - `max_tokens`: Maximum tokens to generate (default: `512`).
  - `temperature`: Sampling temperature (default: `0.2`).
  - `top_p`: Nucleus sampling (default: `0.95`).
  - `top_k`: Top-k sampling (default: `100`).
  - `repeat_penalty`: Repetition penalty (default: `1.0`).
  - `seed`: Random seed, `-1` for random (default: `-1`).
  - `strip_thinking`: Strip `<think>...</think>` blocks from the `response` output. Also handles truncated thinking (no closing tag). The thinking content is available in the `thinking` output (default: `Disabled`).

**Outputs**
- `response`: The generated text (thinking stripped if `strip_thinking` is enabled).
- `thinking`: The extracted thinking/reasoning content (empty if `strip_thinking` is disabled or model produced none).

> **Tip for thinking models**: If output is only `<think>` content and gets cut off, it means `max_tokens` was exhausted before the model finished reasoning. Solutions:
> - Increase `n_ctx` (e.g. 32768) and/or reduce `max_tokens`
> - Set `enable_thinking` to `Disabled` to skip reasoning entirely
> - Enable `strip_thinking` to always extract the final answer even from partial output

---

### LlamaCPPMemoryCleanup
Utility to manually free resources at a specific point in the workflow.

**Inputs**
- **Required**:
  - `memory_cleanup`: Cleanup mode (`close`, `backend_free`, `full_cleanup`, `persistent`).
- **Optional**:
  - `passthrough`: Any input to pass through (allows chaining).

**Outputs**
- `passthrough`: The input passed through unmodified.

---

### PDFLoader
Renders PDF pages as a ComfyUI `IMAGE` batch. Use this to feed PDFs into any vision node.

Requires `pymupdf` (`pip install pymupdf`).

**Inputs**
- **Required**:
  - `pdf_path`: Absolute path to the PDF file. Both forward (`/`) and back (`\`) slashes work on Windows.
- **Optional**:
  - `dpi`: Render resolution (default: `150`, range: 72–600). Higher = sharper but more memory.
  - `page_start`: First page to load, 1-indexed (default: `1`).
  - `page_end`: Last page to load, 0 = all pages (default: `0`).

**Outputs**
- `images`: All rendered pages as a single `IMAGE` batch `(B, H, W, C)`. Pages of different sizes are zero-padded to the largest dimensions.
- `page_count`: Number of pages loaded.

**Example workflows**
```
PDF Loader → FireRed OCR Engine → Show Text
PDF Loader → Llama CPP Engine   → Show Text
PDF Loader → Preview Image
```

---

### FireRedOCREngine
Dedicated OCR node for **FireRed-OCR GGUF** models. Converts document/page images to structured Markdown using a built-in expert OCR prompt. Each image in the batch is processed individually to avoid context overflow.

**Inputs**
- **Required**:
  - `model`: Model from `LlamaCPPModelLoader` (must have `mmproj` selected and a `vision-*` chat format).
  - `image`: Document or page image batch to OCR. Use `PDFLoader` to load PDF files.
- **Optional**:
  - `options`: Options from `LlamaCPPOptions`. Set `n_ctx` to at least `32768` for documents.
  - `custom_prompt`: Override the built-in OCR prompt. Leave empty to use the official FireRed-OCR prompt.
  - `max_tokens`: Maximum tokens to generate (default: `8192` — documents need long output).
  - `temperature`: Sampling temperature (default: `0.1` — lower = more deterministic).
  - `memory_cleanup`: Memory strategy after generation (default: `close`).
  - `seed`: Random seed, `-1` for random (default: `-1`).

**Outputs**
- `markdown`: The recognized document in Markdown format (tables as HTML, formulas as LaTeX). Multiple pages are separated by `<!-- Page N -->` comments.

**Quick setup**
1. Download from [mradermacher/FireRed-OCR-GGUF](https://huggingface.co/mradermacher/FireRed-OCR-GGUF):
   - `FireRed-OCR.Q4_K_M.gguf` (or any quant)
   - `FireRed-OCR.mmproj-Q8_0.gguf`
2. In `LlamaCPPModelLoader`: set `mmproj_model_name` and pick the `vision-qwen25vl` (or similar) chat format.
3. Set `n_ctx` to `32768`+ in `LlamaCPPOptions`.
4. Connect: `PDF Loader → LlamaCPPModelLoader → FireRed OCR Engine → Show Text`

---

## Custom Model Folders

By default, the node loads GGUF models from ComfyUI's `text_encoders` folder. You can optionally specify additional folders with a `config.json` file.

### Configuration

Create `config.json` in the custom nodes directory:

```json
{
  "model_folders": [
    "C:\\Users\\YourUsername\\models",
    "D:\\AI\\LLM\\models",
    "/home/user/models"
  ]
}
```

- The file is **optional** — the node works without it
- Non-existent paths are automatically filtered out
- Models from all folders appear in the model selection dropdown
- See `config.example.json` for examples

---

## Vision Models

Vision inference requires:
1. A vision GGUF model (the LLM backbone)
2. A matching `mmproj` GGUF file (the visual encoder projector)
3. Selecting a `vision-*` chat format in `LlamaCPPModelLoader`

Supported handlers are auto-detected at startup from the installed llama-cpp-python version. Supported models include: LLaVA 1.5/1.6, Qwen2.5-VL, Qwen3-VL, Qwen3.5, MiniCPM-V 2.6/4.5, Gemma3, GLM-4V, Moondream, LFM2-VL, and more.

**Multiple images**: All images in a ComfyUI IMAGE batch are sent to the model in a single request. Use `PDF Loader` or the native `Load Images` node to build multi-image batches.

> **n_threads for vision**: The vision pipeline uses multi-threaded image decoding. If `n_threads` is set to `-1` (auto), the node automatically sets it to `os.cpu_count()` to avoid a crash in `ThreadPoolExecutor`.

### FireRed-OCR

[FireRed-OCR](https://github.com/FireRedTeam/FireRed-OCR) is a Qwen3-VL-2B-based SOTA document OCR model. Use the dedicated **FireRed OCR Engine** node for best results — it pre-bakes the expert OCR prompt automatically.

| File | Description |
|---|---|
| `FireRed-OCR.Q4_K_M.gguf` | Main model (recommended quant) |
| `FireRed-OCR.mmproj-Q8_0.gguf` | Visual projector |

Download: [mradermacher/FireRed-OCR-GGUF](https://huggingface.co/mradermacher/FireRed-OCR-GGUF)

---

## Requirements

- llama-cpp-python ≥ v0.3.30 (from https://github.com/JamePeng/llama-cpp-python)
- pymupdf (optional, for PDF support — `pip install pymupdf`)

## License

This project is licensed under the GNU AGPLv3 License - see the [LICENSE](LICENSE) file for details.

## Repository

https://github.com/sebagallo/comfyui-sg-llama-cpp
