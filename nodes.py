import os
import gc
import inspect
import torch
import base64
import io
import json
from PIL import Image
from torchvision.transforms import ToPILImage
from llama_cpp import Llama, llama_backend_free
from llama_cpp.llama_chat_format import (
    Llava15ChatHandler,
    LlamaChatCompletionHandlerRegistry,
)
# v0.3.28+: MTMDChatHandler is the new base class for all multimodal handlers
try:
    from llama_cpp.llama_chat_format import MTMDChatHandler as _MTMDChatHandler
except ImportError:
    _MTMDChatHandler = None  # Older versions don't have this
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from typing import Dict, Any, List, Type, Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    pass

# Register layout YOLO models directory
layout_yolo_dir = os.path.join(folder_paths.models_dir, "doclayout_yolo")
if not os.path.exists(layout_yolo_dir):
    os.makedirs(layout_yolo_dir, exist_ok=True)
folder_paths.add_model_folder_path("doclayout_yolo", layout_yolo_dir)

# Config file path
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

def load_config() -> Dict[str, Any]:
    """Load config from config.json, return empty dict if not found or invalid."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_user_model_folders() -> List[str]:
    """Get user-specified model folders from config."""
    config = load_config()
    return config.get('model_folders', [])

def get_merged_model_folders() -> List[str]:
    """Merge ComfyUI text_encoders folders with user folders."""
    try:
        comfy_folders = folder_paths.get_folder_paths("text_encoders")
    except:
        comfy_folders = []
    user_folders = get_user_model_folders()
    all_folders = comfy_folders + user_folders
    # Filter out non-existent paths
    return [f for f in all_folders if os.path.exists(f)]

def scan_gguf_models_in_folders() -> List[str]:
    """Scan merged folders for GGUF model files."""
    folders = get_merged_model_folders()
    model_list = []
    for folder in folders:
        try:
            files = os.listdir(folder)
            model_list.extend([f for f in files if f.lower().endswith('.gguf')])
        except:
            pass  # Skip inaccessible folders
    return model_list

def find_model_path(model_name: str) -> str:
    """Find full path to model in merged folders."""
    folders = get_merged_model_folders()
    for folder in folders:
        path = os.path.join(folder, model_name)
        if os.path.exists(path):
            return path
    return None

# Global LLM instance for persistence
_global_llm = None

def _convert_image_to_data_uri(image_tensor: torch.Tensor, batch_index: int = 0) -> str:
    """Convert a ComfyUI image tensor to a base64 data URI for vision models."""
    try:
        to_pil = ToPILImage()
        # ComfyUI images are (B, H, W, C), select batch_index and permute to (C, H, W)
        idx = min(batch_index, image_tensor.shape[0] - 1)
        pil_img = to_pil(image_tensor[idx].permute(2, 0, 1))
        # Convert to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        data_uri = f"data:image/png;base64,{img_base64}"
        # Clean up memory
        buffered.close()
        del pil_img
        return data_uri

    except Exception as e:
        raise ValueError(f"Failed to convert image tensor to data URI: {str(e)}")


def _get_handler_params(handler_class: type) -> set:
    """Walk the full MRO to collect all explicitly-named __init__ params."""
    params = set()
    for klass in type.mro(handler_class):
        if klass is object:
            continue
        init = klass.__dict__.get('__init__')
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
            for name, p in sig.parameters.items():
                if name == 'self':
                    continue
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                params.add(name)
        except (ValueError, TypeError):
            continue
    return params


def _model_identity(model: dict, llama_kwargs: dict) -> tuple:
    """Return a hashable identity tuple for the current model config."""
    return (
        model.get("model_path"),
        model.get("chat_format"),
        model.get("mmproj_model_path"),
    )


# Track currently-loaded model identity for persistent mode
_global_llm_identity = None


def _pdf_to_image_tensors(pdf_path: str, dpi: int = 150) -> torch.Tensor:
    """Render each PDF page and return a stacked ComfyUI IMAGE tensor (B, H, W, C) float32."""
    import fitz  # pymupdf
    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        t = torch.frombuffer(bytearray(pix.samples), dtype=torch.uint8)
        t = t.reshape(pix.height, pix.width, 3).float() / 255.0
        pages.append(t)
    doc.close()
    if not pages:
        raise RuntimeError("PDF has no pages.")
    # Pad all pages to the same size (largest H × W) before stacking
    max_h = max(p.shape[0] for p in pages)
    max_w = max(p.shape[1] for p in pages)
    padded = []
    for p in pages:
        h, w, c = p.shape
        if h < max_h or w < max_w:
            canvas = torch.zeros(max_h, max_w, c, dtype=p.dtype)
            canvas[:h, :w] = p
            p = canvas
        padded.append(p)
    return torch.stack(padded)  # (B, H, W, C)


def _cleanup_global_llm(mode: str):
    """Helper function to cleanup the global LLM based on mode."""
    global _global_llm
    if mode == "persistent":
        return  # No cleanup

    # Common cleanup for all non-persistent modes
    if _global_llm is not None:
        # Clean up chat_handler if it exists (for vision models)
        if hasattr(_global_llm, "chat_handler") and _global_llm.chat_handler is not None:
            try:
                # v0.3.30+: close() safely handles _exit_stack internally (even on init failure)
                # No need to manually close _exit_stack or clip_model attributes.
                _global_llm.chat_handler.close()
            except Exception:
                pass  # Ignore cleanup errors for chat_handler

            if hasattr(_global_llm, "chat_handler"):
                del _global_llm.chat_handler

            gc.collect()
        try:
            _global_llm.close()
        except AttributeError:
            pass  # close() method may not be available in all versions
        del _global_llm
        _global_llm = None
        gc.collect()

    # Backend free cleanup for backend_free and full_cleanup modes
    if mode in ["backend_free", "full_cleanup"]:
        try:
            llama_backend_free()
            gc.collect()
        except (NameError, AttributeError):
            pass  # llama_backend_free may not be available

    # Full torch cleanup for full_cleanup mode
    if mode == "full_cleanup":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()


# Mapping of chat formats to vision chat handlers (dynamically discovered)
import llama_cpp.llama_chat_format as lcf

# v0.3.28+: MTMDChatHandler is the new base; Llava15ChatHandler kept for backward compat.
_base_vision_classes = tuple(filter(None, [
    getattr(lcf, 'MTMDChatHandler', None),
    getattr(lcf, 'Llava15ChatHandler', None),
]))

VISION_HANDLERS = {}
if _base_vision_classes:
    for name, obj in inspect.getmembers(lcf):
        if (
            inspect.isclass(obj)
            and issubclass(obj, _base_vision_classes)
            and obj not in _base_vision_classes
        ):
            vision_name = f"vision-{name.lower().replace('chathandler', '')}"
            VISION_HANDLERS[vision_name] = obj

print(f"[LlamaCPP] Detected vision handlers: {sorted(VISION_HANDLERS.keys())}")

class LlamaCPPModelLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        model_list = scan_gguf_models_in_folders()

        # Filter models based on criteria (case-insensitive)
        model_name_list = [f for f in model_list if 'mmproj' not in f.lower() and 'draft' not in f.lower()]
        mmproj_list = [f for f in model_list if 'mmproj' in f.lower()]

        # Add "None" option to make mmproj truly optional
        if mmproj_list:
            mmproj_list.insert(0, "None")
        else:
            mmproj_list = ["None"]

        # Available chat formats from llama-cpp-python
        base_chat_formats = sorted(list(LlamaChatCompletionHandlerRegistry()._chat_handlers.keys()))
        vision_formats = list(VISION_HANDLERS.keys())
        chat_formats = sorted(base_chat_formats + vision_formats)

        return {
            "required": {
                "model_name": (model_name_list if model_name_list else ["No GGUF models found"], {"tooltip": "Select GGUF model file"}),
            },
            "optional": {
                "chat_format": (chat_formats, {"default": "llama-2", "tooltip": "Chat format template"}),
                "mmproj_model_name": (mmproj_list, {"default": "None", "tooltip": "Multi-modal projector model for vision (select 'None' to disable)"}),
            }
        }

    RETURN_TYPES = ("LLAMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "LlamaCPP"

    def load_model(self, model_name: str, chat_format: str = "llama-2", mmproj_model_name: str = "None") -> tuple:
        try:
            model_path = find_model_path(model_name)

            if model_path is None:
                raise FileNotFoundError(f"Model file not found: {model_name}")

            if not model_name.lower().endswith('.gguf'):
                raise ValueError(f"Selected file is not a GGUF model: {model_name}")

            model_info = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Handle mmproj if provided and not "None"
            if mmproj_model_name and mmproj_model_name != "None":
                mmproj_model_path = find_model_path(mmproj_model_name)
                if mmproj_model_path is None:
                    raise FileNotFoundError(f"Multi-modal projector model not found: {mmproj_model_name}")
                model_info["mmproj_model_path"] = mmproj_model_path

            # Return model info dict
            return (model_info,)

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")


class LlamaCPPOptions(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "optional": {
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100, "tooltip": "Number of GPU layers to use"}),
                "n_ctx": ("INT", {"default": 2048, "min": 0, "max": 262144, "tooltip": "Context window size (0 for max)"}),
                "n_threads": ("INT", {"default": -1, "min": -1, "max": 256, "tooltip": "Number of threads (-1 for auto)"}),
                "n_threads_batch": ("INT", {"default": -1, "min": -1, "max": 256, "tooltip": "Number of threads per batch (-1 for auto)"}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 16384, "tooltip": "Batch size"}),
                "n_ubatch": ("INT", {"default": 512, "min": 1, "max": 16384, "tooltip": "Micro batch size"}),
                "main_gpu": ("INT", {"default": 0, "min": 0, "max": 100, "tooltip": "GPU ID for main device"}),
                "offload_kqv": (IO.BOOLEAN, {"default": True, "tooltip": "Enable offloading of K/Q/V tensors to GPU", "label_on": "Enabled", "label_off": "Disabled"}),
                "numa": (IO.BOOLEAN, {"default": False, "tooltip": "Enable NUMA affinity", "label_on": "Enabled", "label_off": "Disabled"}),
                "use_mmap": (IO.BOOLEAN, {"default": True, "tooltip": "Enable memory-mapped files", "label_on": "Enabled", "label_off": "Disabled"}),
                "use_mlock": (IO.BOOLEAN, {"default": False, "tooltip": "Enable lock for memory-mapped files", "label_on": "Enabled", "label_off": "Disabled"}),
                "verbose": (IO.BOOLEAN, {"default": False, "tooltip": "Enable verbose logging", "label_on": "Enabled", "label_off": "Disabled"}),
                "vision_use_gpu": (IO.BOOLEAN, {"default": True, "tooltip": "Vision: Enable GPU for vision handler", "label_on": "Enabled", "label_off": "Disabled"}),
                "vision_image_min_tokens": ("INT", {"default": -1, "min": -1, "max": 16384, "tooltip": "Vision: Minimum image tokens (-1 for default)"}),
                "vision_image_max_tokens": ("INT", {"default": -1, "min": -1, "max": 16384, "tooltip": "Vision: Maximum image tokens (-1 for default)"}),
                "vision_enable_thinking": (IO.BOOLEAN, {"default": False, "tooltip": "Vision: Enable thinking (MiniCPMv45)", "label_on": "Enabled", "label_off": "Disabled"}),
            }
        }

    RETURN_TYPES = ("LLAMA_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "get_options"
    CATEGORY = "LlamaCPP"

    def get_options(self, **kwargs) -> tuple:
        try:
            # Filter out None/empty values
            options = {k: v for k, v in kwargs.items() if v is not None and v != ""}

            return (options,)

        except Exception as e:
            raise RuntimeError(f"Failed to process options: {str(e)}")


class LlamaCPPEngine(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": ("LLAMA_MODEL", {"tooltip": "Loaded Llama model from Model Loader"}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "User prompt for chat completion"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input image(s) / video frames batch for vision models"}),
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Options node"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "System prompt"}),
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "close", "tooltip": "Memory cleanup method after generation"}),
                "response_format": (["text", "json_object"], {"default": "text", "tooltip": "Output format (json_object forces valid JSON)"}),
                "enable_thinking": (IO.BOOLEAN, {"default": True, "tooltip": "Enable thinking/reasoning (Qwen3, QwQ, etc.). False injects empty <think> block to skip reasoning.", "label_on": "Enabled", "label_off": "Disabled"}),
                "is_video": (IO.BOOLEAN, {"default": False, "tooltip": "Treat the IMAGE input as a video frame batch: uniformly sample video_max_frames and send them all to the vision model.", "label_on": "Video", "label_off": "Image"}),
                "video_max_frames": ("INT", {"default": 8, "min": 1, "max": 256, "tooltip": "Maximum number of frames to sample from the video batch (uniform sampling)."}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 262144, "tooltip": "Maximum tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Sampling temperature"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Top-p sampling"}),
                "top_k": ("INT", {"default": 100, "min": 0, "max": 400, "tooltip": "Top-k sampling"}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01, "tooltip": "Repeat penalty"}),
                "seed": ("INT", {"default": -1, "min": -1, "tooltip": "Random seed (-1 for random)"}),
                "strip_thinking": (IO.BOOLEAN, {"default": False, "tooltip": "Strip <think>...</think> blocks from output (thinking models like QwQ, Qwen3)", "label_on": "Strip", "label_off": "Keep"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "thinking")
    FUNCTION = "generate"
    CATEGORY = "LlamaCPP"

    def generate(self, model: Dict[str, Any], prompt: str, image: torch.Tensor = None, options: Dict[str, Any] = None, system_prompt: str = "", memory_cleanup: str = "close", response_format: str = "text", enable_thinking: bool = True, is_video: bool = False, video_max_frames: int = 8, max_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95, top_k: int = 40, repeat_penalty: float = 1.1, seed: int = -1, strip_thinking: bool = False) -> tuple:
        global _global_llm
        try:
            # Validate inputs
            if not isinstance(model, dict) or "model_path" not in model:
                raise ValueError("Invalid model data received from Model Loader")

            options = options or {}
            if not isinstance(options, dict):
                raise ValueError("Invalid options data received from Options node")

            if not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            # Extract model info
            model_path = model["model_path"]
            chat_format = model["chat_format"]

            # Determine if vision handler is enabled
            vision_enabled = "mmproj_model_path" in model and chat_format.startswith("vision-")

            # Create messages for chat completion
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt.strip()})

            # Create user message content
            user_content = prompt.strip()

            # If vision is enabled and image is provided, handle image or video frames
            if vision_enabled and image is not None:
                total_frames = image.shape[0]
                if is_video and total_frames > 1:
                    # Uniform frame sampling
                    if total_frames <= video_max_frames:
                        indices = list(range(total_frames))
                    else:
                        import numpy as np
                        indices = [int(round(i)) for i in np.linspace(0, total_frames - 1, video_max_frames)]
                    print(f"[LlamaCPP] Video mode: sampling {len(indices)} frames from {total_frames} total (indices: {indices})")
                else:
                    indices = list(range(total_frames))
                user_content = [
                    {"type": "image_url", "image_url": _convert_image_to_data_uri(image, i)}
                    for i in indices
                ] + [{"type": "text", "text": prompt.strip()}]

            messages.append({"role": "user", "content": user_content})

            # Prepare Llama initialization parameters
            llama_kwargs = {
                "model_path": model_path,
                "chat_format": chat_format,
            }

            # Add options
            for k, v in options.items():
                if not k.startswith("vision_"):
                    llama_kwargs[k] = v

            # Handle vision models: use chat_handler based on chat_format
            if vision_enabled:
                # Workaround: MTMDChatHandler._process_mtmd_prompt computes
                # max_workers = min(llama.n_threads, len(image_urls))
                # which crashes ThreadPoolExecutor when n_threads is -1 (auto).
                # Ensure n_threads is always a positive integer for vision mode.
                if llama_kwargs.get("n_threads", -1) <= 0:
                    llama_kwargs["n_threads"] = max(1, os.cpu_count() or 4)

                handler_class = VISION_HANDLERS.get(chat_format, Llava15ChatHandler)
                handler_params = _get_handler_params(handler_class)

                handler_kwargs = {}
                # Always pass clip_model_path and verbose (present in all known handlers)
                handler_kwargs["clip_model_path"] = model["mmproj_model_path"]
                handler_kwargs["verbose"] = options.get("verbose", False)

                # Process vision_ prefixed options — only include if handler supports them
                for k, v in options.items():
                    if k.startswith("vision_"):
                        param_name = k.replace("vision_", "")
                        if param_name in handler_params:
                            handler_kwargs[param_name] = v

                chat_handler = handler_class(**handler_kwargs)

                # Inject enable_thinking into vision handler's extra_template_arguments
                if hasattr(chat_handler, "extra_template_arguments"):
                    chat_handler.extra_template_arguments["enable_thinking"] = enable_thinking

                llama_kwargs["chat_handler"] = chat_handler
                # Remove chat_format when using vision handler
                llama_kwargs.pop("chat_format", None)

            # Pre-generation LLM management
            # persistent mode: only skip reload if the model identity matches
            current_identity = _model_identity(model, llama_kwargs)
            if memory_cleanup == "persistent":
                global _global_llm_identity
                if _global_llm is None or _global_llm_identity != current_identity:
                    _cleanup_global_llm("close")
                    _global_llm = Llama(**llama_kwargs)
                    _global_llm_identity = current_identity
            else:  # close or backend_free - always cleanup existing and create new
                _cleanup_global_llm(memory_cleanup)
                _global_llm = Llama(**llama_kwargs)

            # Prepare response_format parameter
            response_format_param = {"type": response_format} if response_format is not None else None

            # For text models: implement enable_thinking=false via partial assistant prefix trick.
            # Pre-fills an empty <think> block so the model skips reasoning entirely.
            # Equivalent to --chat-template-kwargs '{"enable_thinking":false}' in llama-server.
            completion_messages = list(messages)
            if not enable_thinking and not vision_enabled:
                completion_messages.append({
                    "role": "assistant",
                    "content": "<think>\n\n</think>\n\n"
                })

            # Generate response using global LLM
            response = _global_llm.create_chat_completion(
                messages=completion_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                seed=seed,
                response_format=response_format_param,
            )

            # Extract the response text
            if not response or "choices" not in response or not response["choices"]:
                raise RuntimeError("No response generated by the model")

            response_text = response["choices"][0]["message"]["content"] or ""

            # Strip <think>...</think> blocks if requested
            # Also handles truncated output (think block never closed due to max_tokens)
            # Also handles "orphan" </think>: model emits reasoning text before </think>
            # without a matching <think> (the opening was injected via the prefilled
            # assistant message, so it doesn't appear in response_text).
            thinking_text = ""
            if strip_thinking:
                import re
                # Case 1: bare </think> at the start (no opening <think> in response_text).
                # Everything before the first </think> is thinking content.
                orphan_match = re.match(r"^(.*?)</think>\s*", response_text, re.DOTALL)
                if orphan_match and "<think>" not in response_text[:orphan_match.end()]:
                    thinking_text = orphan_match.group(1).strip()
                    response_text = response_text[orphan_match.end():]

                # Case 2: extract all complete <think>...</think> blocks
                think_blocks = re.findall(r"<think>(.*?)</think>", response_text, re.DOTALL)
                if think_blocks:
                    extra = "\n\n".join(b.strip() for b in think_blocks)
                    thinking_text = (thinking_text + "\n\n" + extra).strip() if thinking_text else extra
                # Remove complete think blocks
                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
                # Case 3: dangling opening tag (truncated output, no closing tag)
                response_text = re.sub(r"<think>.*$", "", response_text, flags=re.DOTALL)
                response_text = response_text.strip()

            # Post-generation memory cleanup
            _cleanup_global_llm(memory_cleanup)

        except Exception as e:
            print(f"LlamaCPP Engine Error: {str(e)}")
            raise RuntimeError(f"LlamaCPP Engine Error: {str(e)}") from e

        return (response_text, thinking_text)


class LlamaCPPMemoryCleanup(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "close", "tooltip": "Memory cleanup method"}),
            },
            "optional": {
                "passthrough": (IO.ANY, {"tooltip": "Any input to pass through"}),
            }
        }

    RETURN_TYPES = (IO.ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "cleanup"
    CATEGORY = "LlamaCPP"

    def cleanup(self, memory_cleanup: str, passthrough=None) -> tuple:
        try:
            _cleanup_global_llm(memory_cleanup)
        except Exception as e:
            print(f"LlamaCPP Memory Cleanup Error: {str(e)}")

        return (passthrough,)


# ---------------------------------------------------------------------------
# FireRed-OCR prompt (from conv_for_infer.py — FireRedTeam/FireRed-OCR)
# ---------------------------------------------------------------------------
_FIRERED_OCR_PROMPT = """You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

        1. Text Processing:
        - Accurately recognize all text content in the PDF image without guessing or inferring.
        - Convert the recognized text into Markdown format.
        - Maintain the original document structure, including headings, paragraphs, lists, etc.

        2. Mathematical Formula Processing:
        - Convert all mathematical formulas to LaTeX format.
        - Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)
        - Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]

        3. Table Processing:
        - Convert tables to HTML format.
        - Wrap the entire table with <table> and </table>.

        4. Figure Handling:
        - Ignore figures content in the PDF image. Do not attempt to describe or convert images.

        5. Output Format:
        - Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
        - For complex layouts, try to maintain the original document's structure and format as closely as possible.

        Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."""


class PDFLoader(ComfyNodeABC):
    """Load a PDF file and render all pages as a ComfyUI IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "pdf_path": ("STRING", {"default": "", "tooltip": "Absolute path to the PDF file (forward or back slashes both work on Windows)"}),
            },
            "optional": {
                "dpi": ("INT", {"default": 150, "min": 72, "max": 600, "tooltip": "Render resolution. Higher = sharper image, more VRAM/memory needed."}),
                "page_start": ("INT", {"default": 1, "min": 1, "tooltip": "First page to load (1-indexed)"}),
                "page_end": ("INT", {"default": 0, "min": 0, "tooltip": "Last page to load (0 = all pages)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "page_count")
    FUNCTION = "load_pdf"
    CATEGORY = "LlamaCPP"

    def load_pdf(self, pdf_path: str, dpi: int = 150, page_start: int = 1, page_end: int = 0) -> tuple:
        pdf_path = pdf_path.strip()
        if not pdf_path:
            raise ValueError("pdf_path is empty.")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError("pymupdf is required. Install with: pip install pymupdf")

        # Slice page range
        import fitz as _fitz
        doc = _fitz.open(pdf_path)
        total = doc.page_count
        doc.close()

        start = max(0, page_start - 1)          # convert 1-indexed → 0-indexed
        end = (total if page_end == 0 else min(page_end, total))
        if start >= end:
            raise ValueError(f"Invalid page range: start={page_start}, end={page_end}, total={total}")

        # Render selected pages
        import fitz as _fitz2
        doc = _fitz2.open(pdf_path)
        scale = dpi / 72.0
        mat = _fitz2.Matrix(scale, scale)
        pages = []
        for i in range(start, end):
            pix = doc[i].get_pixmap(matrix=mat, colorspace=_fitz2.csRGB, alpha=False)
            t = torch.frombuffer(bytearray(pix.samples), dtype=torch.uint8)
            t = t.reshape(pix.height, pix.width, 3).float() / 255.0
            pages.append(t)
        doc.close()
        print(f"[PDFLoader] Loaded {len(pages)} page(s) from '{pdf_path}' at {dpi} DPI")

        # Pad to common size and stack
        max_h = max(p.shape[0] for p in pages)
        max_w = max(p.shape[1] for p in pages)
        padded = []
        for p in pages:
            h, w, c = p.shape
            if h < max_h or w < max_w:
                canvas = torch.zeros(max_h, max_w, c, dtype=p.dtype)
                canvas[:h, :w] = p
                p = canvas
            padded.append(p)
        images = torch.stack(padded)  # (B, H, W, C)
        return (images, len(pages))


class FireRedOCREngine(ComfyNodeABC):
    """Dedicated OCR node for FireRed-OCR GGUF models (Qwen3-VL-based).

    Download the model + mmproj from:
      https://huggingface.co/mradermacher/FireRed-OCR-GGUF
        - FireRed-OCR.Q4_K_M.gguf  (or any quant)
        - FireRed-OCR.mmproj-Q8_0.gguf

    In the Model Loader, set chat_format to the Qwen2.5-VL vision handler
    (e.g. 'vision-qwen25vl' or 'vision-qwen25vlchathandler' depending on
    your llama-cpp-python version — check the loader dropdown).
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": ("LLAMA_MODEL", {"tooltip": "FireRed-OCR model loaded via Llama CPP Model Loader (must include mmproj)"}),
                "image": ("IMAGE", {"tooltip": "Document/page image(s) to OCR. Use PDFLoader to convert PDF pages."}),
            },
            "optional": {
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Llama CPP Options node"}),
                "custom_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override the built-in OCR prompt. Leave empty to use the FireRed-OCR default prompt.",
                }),
                "max_tokens": ("INT", {"default": 8192, "min": 128, "max": 65536, "tooltip": "Maximum tokens (documents need long output)"}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Lower = more deterministic OCR"}),
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "close", "tooltip": "Memory cleanup after generation"}),
                "seed": ("INT", {"default": -1, "min": -1, "tooltip": "Random seed (-1 for random)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown",)
    FUNCTION = "ocr"
    CATEGORY = "LlamaCPP"

    def ocr(
        self,
        model: Dict[str, Any],
        image: torch.Tensor,
        options: Dict[str, Any] = None,
        custom_prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        memory_cleanup: str = "close",
        seed: int = -1,
    ) -> tuple:
        global _global_llm
        try:
            if not isinstance(model, dict) or "model_path" not in model:
                raise ValueError("Invalid model data — connect a Llama CPP Model Loader")

            if "mmproj_model_path" not in model:
                raise ValueError("FireRed-OCR requires a mmproj model. Set mmproj_model_name in the Model Loader.")

            chat_format = model.get("chat_format", "")
            if not chat_format.startswith("vision-"):
                raise ValueError(
                    f"chat_format '{chat_format}' does not look like a vision handler. "
                    "Select a 'vision-*' format in the Model Loader (e.g. vision-qwen25vl)."
                )

            options = options or {}
            model_path = model["model_path"]
            prompt_text = custom_prompt.strip() if custom_prompt.strip() else _FIRERED_OCR_PROMPT

            # Split image batch into individual page tensors
            page_images = [image[i].unsqueeze(0) for i in range(image.shape[0])]

            # Build Llama kwargs from options
            llama_kwargs = {"model_path": model_path}
            for k, v in options.items():
                if not k.startswith("vision_"):
                    llama_kwargs[k] = v

            # Vision handler
            if llama_kwargs.get("n_threads", -1) <= 0:
                llama_kwargs["n_threads"] = max(1, os.cpu_count() or 4)

            handler_class = VISION_HANDLERS.get(chat_format, Llava15ChatHandler)
            handler_params = _get_handler_params(handler_class)

            handler_kwargs = {
                "clip_model_path": model["mmproj_model_path"],
                "verbose": options.get("verbose", False),
            }
            for k, v in options.items():
                if k.startswith("vision_"):
                    param_name = k.replace("vision_", "")
                    if param_name in handler_params:
                        handler_kwargs[param_name] = v

            chat_handler = handler_class(**handler_kwargs)
            llama_kwargs["chat_handler"] = chat_handler

            # LLM lifecycle
            current_identity = _model_identity(model, llama_kwargs)
            if memory_cleanup == "persistent":
                global _global_llm_identity
                if _global_llm is None or _global_llm_identity != current_identity:
                    _cleanup_global_llm("close")
                    _global_llm = Llama(**llama_kwargs)
                    _global_llm_identity = current_identity
            else:
                _cleanup_global_llm(memory_cleanup)
                _global_llm = Llama(**llama_kwargs)

            # OCR each page individually and collect results
            page_results = []
            for page_idx, page_tensor in enumerate(page_images):
                print(f"[FireRedOCR] Processing page {page_idx + 1}/{len(page_images)}...")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": _convert_image_to_data_uri(page_tensor, 0)}
                        ] + [{"type": "text", "text": prompt_text}],
                    }
                ]
                response = _global_llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                )
                if not response or "choices" not in response or not response["choices"]:
                    raise RuntimeError(f"No response generated for page {page_idx + 1}")
                page_results.append(response["choices"][0]["message"]["content"] or "")

            # Join pages with separator
            if len(page_results) == 1:
                markdown_text = page_results[0]
            else:
                markdown_text = "\n\n".join(
                    f"<!-- Page {i + 1} -->\n{text}" for i, text in enumerate(page_results)
                )
            _cleanup_global_llm(memory_cleanup)

        except Exception as e:
            print(f"[FireRedOCR] Error: {str(e)}")
            raise RuntimeError(f"FireRed-OCR Error: {str(e)}") from e

        return (markdown_text,)




class DocLayoutYOLOLoader(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        # Get list of .pt files in doclayout_yolo folder
        model_list = folder_paths.get_filename_list("doclayout_yolo")
        model_list = [f for f in model_list if f.endswith('.pt')]

        return {
            "required": {
                "model_name": (
                    model_list + ["juliozhao/DocLayout-YOLO-DocStructBench (auto-download)"],
                    {"tooltip": "Select a YOLOv10 .pt model from models/doclayout_yolo or auto-download the default one."}
                )
            }
        }

    RETURN_TYPES = ("YOLO_MODEL",)
    RETURN_NAMES = ("yolo_model",)
    FUNCTION = "load_yolo"
    CATEGORY = "LlamaCPP"

    def load_yolo(self, model_name: str) -> tuple:
        try:
            from doclayout_yolo import YOLOv10
        except ImportError:
            raise ImportError("Please install doclayout-yolo: pip install doclayout-yolo")

        if model_name == "juliozhao/DocLayout-YOLO-DocStructBench (auto-download)":
            try:
                import huggingface_hub
            except ImportError:
                raise ImportError("Please install huggingface_hub to use the auto-download feature: pip install huggingface_hub")
                
            repo_id = "juliozhao/DocLayout-YOLO-DocStructBench"
            filename = "doclayout_yolo_docstructbench_imgsz1024.pt"
            local_dir = folder_paths.get_folder_paths("doclayout_yolo")[0]
            print(f"[DocLayout-YOLO] Downloading {filename} from {repo_id} to {local_dir}...")
            # Download directly to the comfyui models folder
            model_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        else:
            model_path = folder_paths.get_full_path("doclayout_yolo", model_name)
            if not model_path:
                raise FileNotFoundError(f"Model not found: {model_name}")

        print(f"[DocLayout-YOLO] Loading model from {model_path}...")
        model = YOLOv10(model_path)
        return (model,)


class DocLayoutMarkdownEngine(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Document page image (single image or batch)."}),
                "yolo_model": ("YOLO_MODEL", {"tooltip": "Loaded DocLayout-YOLO model."}),
                "llm_model": ("LLAMA_MODEL", {"tooltip": "Loaded Llama VLM model for OCR (e.g., Qwen2.5-VL)."}),
            },
            "optional": {
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Llama CPP Options node"}),
                "max_tokens": ("INT", {"default": 2048, "min": 128, "max": 65536, "tooltip": "Max tokens per OCR generation"}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "persistent"}),
                "seed": ("INT", {"default": -1, "min": -1}),
                "yolo_conf": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 1.0, "step": 0.05, "tooltip": "YOLO confidence threshold"}),
                "yolo_imgsz": ("INT", {"default": 1024, "min": 512, "max": 4096, "step": 32, "tooltip": "YOLO inference image size"}),
                "enable_thinking": (IO.BOOLEAN, {"default": False, "tooltip": "Enable reasoning <think> (useful for some models)"}),
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Extract the text in this image exactly as written. Do not add explanations. Keep formatting where possible.",
                    "tooltip": "Prompt used for extracting Plain Text, Titles, and Captions."
                }),
                "formula_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this mathematical formula to LaTeX. Output ONLY the latex code mathematically formatted inside $$ blocks. Do not add markdown backticks.",
                    "tooltip": "Prompt used for parsing mathematical formulas."
                }),
                "table_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this table to Markdown format. Output ONLY the table structure without any surrounding text.",
                    "tooltip": "Prompt used for extracting Tables."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("markdown",)
    FUNCTION = "process_layout"
    CATEGORY = "LlamaCPP"

    def process_layout(
        self,
        image: torch.Tensor,
        yolo_model: Any,
        llm_model: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        memory_cleanup: str = "persistent",
        seed: int = -1,
        yolo_conf: float = 0.2,
        yolo_imgsz: int = 1024,
        enable_thinking: bool = False,
        text_prompt: str = "Extract the text in this image exactly as written. Do not add explanations. Keep formatting where possible.",
        formula_prompt: str = "Convert this mathematical formula to LaTeX. Output ONLY the latex code mathematically formatted inside $$ blocks. Do not add markdown backticks.",
        table_prompt: str = "Convert this table to Markdown format. Output ONLY the table structure without any surrounding text.",
    ) -> tuple:
        import cv2
        import numpy as np

        # Validate inputs
        if not hasattr(yolo_model, "predict"):
            raise ValueError("yolo_model does not appear to be a valid YOLO object.")
        if not isinstance(llm_model, dict) or "model_path" not in llm_model:
            raise ValueError("Invalid llm_model data.")
        if "mmproj_model_path" not in llm_model:
            raise ValueError("LLM must be a vision model with mmproj loaded.")
            
        chat_format = llm_model.get("chat_format", "")
        if not chat_format.startswith("vision-"):
            raise ValueError("Select a 'vision-*' format in the Model Loader.")

        options = options or {}
        
        # --- LLM Engine Setup (extracted from LlamaCPPEngine) ---
        global _global_llm
        llama_kwargs = {"model_path": llm_model["model_path"]}
        for k, v in options.items():
            if not k.startswith("vision_"):
                llama_kwargs[k] = v

        if llama_kwargs.get("n_threads", -1) <= 0:
            llama_kwargs["n_threads"] = max(1, os.cpu_count() or 4)

        handler_class = VISION_HANDLERS.get(chat_format, Llava15ChatHandler)
        handler_params = _get_handler_params(handler_class)
        handler_kwargs = {
            "clip_model_path": llm_model["mmproj_model_path"],
            "verbose": options.get("verbose", False),
        }
        for k, v in options.items():
            if k.startswith("vision_"):
                param_name = k.replace("vision_", "")
                if param_name in handler_params:
                    handler_kwargs[param_name] = v

        chat_handler = handler_class(**handler_kwargs)
        if hasattr(chat_handler, "extra_template_arguments"):
            chat_handler.extra_template_arguments["enable_thinking"] = enable_thinking
        llama_kwargs["chat_handler"] = chat_handler

        current_identity = _model_identity(llm_model, llama_kwargs)
        if memory_cleanup == "persistent":
            global _global_llm_identity
            if _global_llm is None or _global_llm_identity != current_identity:
                _cleanup_global_llm("close")
                _global_llm = Llama(**llama_kwargs)
                _global_llm_identity = current_identity
        else:
            _cleanup_global_llm(memory_cleanup)
            _global_llm = Llama(**llama_kwargs)

        # Helper function for LLM inference
        def _run_vlm(crop_tensor: torch.Tensor, prompt: str) -> str:
            messages = list()
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": _convert_image_to_data_uri(crop_tensor, 0)},
                {"type": "text", "text": prompt}
            ]})
            completion_messages = list(messages)
            if not enable_thinking:
                completion_messages.append({
                    "role": "assistant",
                    "content": "<think>\n\n</think>\n\n"
                })
            response = _global_llm.create_chat_completion(
                messages=completion_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            text = response["choices"][0]["message"]["content"] or ""
            # Strip fake think blocks if enable_thinking was false
            if not enable_thinking:
                import re
                orphan_match = re.match(r"^(.*?)</think>\s*", text, re.DOTALL)
                if orphan_match and "<think>" not in text[:orphan_match.end()]:
                    text = text[orphan_match.end():]
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
                text = text.strip()
            return text

        # Function to process single page
        def _process_page(page_tensor: torch.Tensor) -> str:
            # 1. Image preparation for YOLO
            # PyTorch Tensor (1, H, W, C) float32 -> Numpy (H, W, C) uint8
            pil_img = ToPILImage()(page_tensor.squeeze(0).permute(2, 0, 1))
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 2. YOLO prediction
            print("[DocLayoutMarkdown] Running YOLO Layout Prediction...")
            det_res = yolo_model.predict(cv_img, imgsz=yolo_imgsz, conf=yolo_conf, device="cpu")[0]
            
            # Extract boxes
            # doclayout-yolo assumes 0-indexed classes. Typical map: 
            # 0: title, 1: plain text, 2: abandon, 3: figure, 4: figure caption, 
            # 5: table, 6: table caption, 7: table footnote, 8: isolate_formula, 9: formula caption
            names = det_res.names

            boxes_data = []
            for box in det_res.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                label = names[cls_id]

                # We can group table elements, figure elements etc.
                if label == "abandon":
                    continue
                    
                boxes_data.append({
                    "bbox": (int(x1), int(max(0.0, float(y1)-5)), int(x2), int(float(y2)+5)),
                    "label": label,
                    "conf": conf,
                    "center_y": (float(y1) + float(y2)) / 2,
                    "center_x": (float(x1) + float(x2)) / 2
                })

            # 3. Sort boxes in a reading order
            # Basic sort: top-to-bottom, left-to-right (handles simple multi-column somewhat poorly, but ok for now)
            # Group into text lines loosely by center_y
            boxes_data.sort(key=lambda b: b["center_y"])
            line_margin = 20
            lines = []
            current_line = []
            
            for b in boxes_data:
                if not current_line:
                    current_line.append(b)
                else:
                    avg_y: float = sum(float(x["center_y"]) for x in current_line) / len(current_line)
                    if abs(b["center_y"] - avg_y) < line_margin:
                        current_line.append(b)
                    else:
                        lines.append(current_line)
                        current_line = [b]
            if current_line:
                lines.append(current_line)
                
            sorted_boxes = []
            for line in lines:
                line.sort(key=lambda b: b["center_x"])
                sorted_boxes.extend(line)

            # 4. Process each region
            md_elements = []
            for i, region in enumerate(sorted_boxes):
                x1, y1, x2, y2 = region["bbox"]
                label = region["label"]
                conf = region["conf"]
                print(f"[DocLayoutMarkdown] Processing region {i+1}/{len(sorted_boxes)}: {label} ({conf:.2f})")

                # Crop tensor
                # Ensure bounds
                h, w, _ = page_tensor.shape[1:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Expand tensor back to B,H,W,C for VLM
                crop_tensor = page_tensor[:, y1:y2, x1:x2, :]

                if label in ["title", "plain text", "figure caption", "table caption", "table footnote", "formula caption"]:
                    # Text Extraction
                    text = _run_vlm(crop_tensor, text_prompt)
                    if label == "title":
                        md_elements.append(f"## {text.strip()}")
                    elif label in ["figure caption", "table caption"]:
                        md_elements.append(f"*{text.strip()}*")
                    else:
                        md_elements.append(text.strip())

                elif label == "isolate_formula":
                    # Formula Extraction
                    latex = _run_vlm(crop_tensor, formula_prompt)
                    # Automatically wrap in block equations if not already
                    latex_text = latex.strip()
                    if not latex_text.startswith("$$"):
                        latex_text = f"$${latex_text}$$"
                    md_elements.append(latex_text)

                elif label == "table":
                    # Table Extraction
                    table_md = _run_vlm(crop_tensor, table_prompt)
                    md_elements.append(table_md.strip())

                elif label == "figure":
                    # Image formatting directly to base64
                    data_uri = _convert_image_to_data_uri(crop_tensor, 0)
                    md_elements.append(f"![Figure]({data_uri})")

            return "\n\n".join(md_elements)

        # Iterate over batch
        try:
            page_results = []
            batch_size = image.shape[0]
            for i in range(batch_size):
                page_tensor = image[i:i+1] # (1, H, W, C)
                print(f"[DocLayoutMarkdown] ================= Page {i+1}/{batch_size} =================")
                res = _process_page(page_tensor)
                page_results.append(res)
                
            final_markdown = "\n\n---\n\n".join(page_results) if batch_size > 1 else page_results[0]
            
            _cleanup_global_llm(memory_cleanup)
            return (final_markdown,)
            
        except Exception as e:
            print(f"DocLayoutMarkdown Engine Error: {str(e)}")
            raise RuntimeError(f"DocLayoutMarkdown Engine Error: {str(e)}") from e
