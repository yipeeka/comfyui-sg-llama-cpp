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


def _convert_image_to_pil(image_tensor: torch.Tensor, batch_index: int = 0) -> Image.Image:
    """Convert a ComfyUI image tensor to a PIL image."""
    try:
        to_pil = ToPILImage()
        idx = min(batch_index, image_tensor.shape[0] - 1)
        return to_pil(image_tensor[idx].permute(2, 0, 1))
    except Exception as e:
        raise ValueError(f"Failed to convert image tensor to PIL image: {str(e)}")


def _convert_pil_to_data_uri(pil_img: Image.Image) -> str:
    """Convert a PIL image to a base64 data URI."""
    try:
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        buffered.close()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        raise ValueError(f"Failed to convert PIL image to data URI: {str(e)}")


def _convert_pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to ComfyUI IMAGE tensor shape (1, H, W, C)."""
    try:
        import numpy as np

        arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Failed to convert PIL image to tensor: {str(e)}")


def _image_batch_to_docling_stream(image_tensor: torch.Tensor, name: str = "document.tiff"):
    """Convert a ComfyUI IMAGE batch to a multi-page TIFF DocumentStream for Docling."""
    try:
        from io import BytesIO
        from docling.datamodel.base_models import DocumentStream
    except ImportError as e:
        raise ImportError("docling is required. Install with: pip install docling") from e

    pil_pages = [_convert_image_to_pil(image_tensor, i).convert("RGB") for i in range(int(image_tensor.shape[0]))]
    if not pil_pages:
        raise ValueError("image batch is empty")

    buf = BytesIO()
    first, rest = pil_pages[0], pil_pages[1:]
    first.save(
        buf,
        format="TIFF",
        save_all=True,
        append_images=rest,
        compression="tiff_deflate",
    )
    buf.seek(0)
    return DocumentStream(name=name, stream=buf)


def _split_paged_output(text: str) -> List[str]:
    """Split <!-- Page N --> joined output back into page-sized chunks."""
    import re

    text = (text or "").strip()
    if not text:
        return []

    pattern = r"(?:^|\n)\s*<!--\s*Page\s+\d+\s*-->\s*\n?"
    parts = re.split(pattern, text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts if parts else [text]


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


class DoclingLayoutAnalyzer(ComfyNodeABC):
    """Analyze document layout with Docling and emit a JSON summary."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Document page image batch. PDFLoader output works well here."}),
            },
            "optional": {
                "document_name": ("STRING", {"default": "document", "tooltip": "Logical document name for Docling conversion."}),
                "do_ocr": (IO.BOOLEAN, {"default": False, "tooltip": "Enable Docling OCR inside layout analysis.", "label_on": "Enabled", "label_off": "Disabled"}),
                "do_table_structure": (IO.BOOLEAN, {"default": True, "tooltip": "Enable Docling table-structure analysis.", "label_on": "Enabled", "label_off": "Disabled"}),
                "images_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Image scale used by Docling page rendering."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layout_json",)
    FUNCTION = "analyze"
    CATEGORY = "LlamaCPP"

    def analyze(
        self,
        image: torch.Tensor,
        document_name: str = "document",
        do_ocr: bool = False,
        do_table_structure: bool = True,
        images_scale: float = 2.0,
    ) -> tuple:
        try:
            try:
                from docling.document_converter import DocumentConverter, ImageFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
            except ImportError as e:
                raise ImportError("docling is required. Install with: pip install docling") from e

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = do_ocr
            pipeline_options.do_table_structure = do_table_structure
            pipeline_options.generate_page_images = True
            pipeline_options.images_scale = images_scale

            converter = DocumentConverter(
                allowed_formats=[InputFormat.IMAGE],
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                },
            )

            stream = _image_batch_to_docling_stream(image, name=f"{document_name or 'document'}.tiff")
            result = converter.convert(stream)
            doc = result.document

            layout = []
            for item, _level in doc.iterate_items():
                prov = item.prov[0] if getattr(item, "prov", None) else None
                bbox = getattr(prov, "bbox", None)
                layout.append({
                    "type": type(item).__name__,
                    "label": str(getattr(item, "label", "")),
                    "page_no": getattr(prov, "page_no", None),
                    "text_preview": (getattr(item, "text", "") or "")[:200],
                    "bbox": {
                        "l": getattr(bbox, "l", None),
                        "t": getattr(bbox, "t", None),
                        "r": getattr(bbox, "r", None),
                        "b": getattr(bbox, "b", None),
                    } if bbox is not None else None,
                })

            payload = {
                "document_name": document_name,
                "page_count": len(getattr(doc, "pages", []) or []),
                "items": layout,
            }
            return (json.dumps(payload, indent=2, ensure_ascii=False),)

        except Exception as e:
            print(f"[DoclingLayoutAnalyzer] Error: {str(e)}")
            raise RuntimeError(f"Docling Layout Analyzer Error: {str(e)}") from e


class DoclingLayoutMarkdownEngine(ComfyNodeABC):
    """Hybrid Markdown engine: Docling for layout, external VLM for recognition."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Document page image batch."}),
                "llm_model": ("LLAMA_MODEL", {"tooltip": "Loaded vision LLM model for OCR/recognition."}),
            },
            "optional": {
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Llama CPP Options node"}),
                "document_name": ("STRING", {"default": "document", "tooltip": "Logical document name for Docling conversion."}),
                "do_ocr": (IO.BOOLEAN, {"default": False, "tooltip": "Enable Docling OCR during layout analysis.", "label_on": "Enabled", "label_off": "Disabled"}),
                "do_table_structure": (IO.BOOLEAN, {"default": True, "tooltip": "Enable Docling table-structure analysis.", "label_on": "Enabled", "label_off": "Disabled"}),
                "images_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Image scale used by Docling page rendering."}),
                "max_tokens": ("INT", {"default": 2048, "min": 128, "max": 65536, "tooltip": "Max tokens per region extraction"}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "persistent"}),
                "seed": ("INT", {"default": -1, "min": -1}),
                "enable_thinking": (IO.BOOLEAN, {"default": False, "tooltip": "Enable reasoning <think> for the vision model"}),
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Extract the text in this image exactly as written. Do not add explanations. Keep formatting where possible.",
                    "tooltip": "Prompt for text-like regions."
                }),
                "title_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Extract this heading/title exactly as written. Output only the title text.",
                    "tooltip": "Prompt for titles and section headers."
                }),
                "formula_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this mathematical formula to LaTeX. Output ONLY the latex code mathematically formatted inside $$ blocks. Do not add markdown backticks.",
                    "tooltip": "Prompt for mathematical formulas."
                }),
                "table_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this table to Markdown format. Output ONLY the table structure without any surrounding text.",
                    "tooltip": "Prompt for table regions."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("markdown", "figure_images", "layout_json")
    FUNCTION = "process_layout"
    CATEGORY = "LlamaCPP"

    def process_layout(
        self,
        image: torch.Tensor,
        llm_model: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        document_name: str = "document",
        do_ocr: bool = False,
        do_table_structure: bool = True,
        images_scale: float = 2.0,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        memory_cleanup: str = "persistent",
        seed: int = -1,
        enable_thinking: bool = False,
        text_prompt: str = "Extract the text in this image exactly as written. Do not add explanations. Keep formatting where possible.",
        title_prompt: str = "Extract this heading/title exactly as written. Output only the title text.",
        formula_prompt: str = "Convert this mathematical formula to LaTeX. Output ONLY the latex code mathematically formatted inside $$ blocks. Do not add markdown backticks.",
        table_prompt: str = "Convert this table to Markdown format. Output ONLY the table structure without any surrounding text.",
    ) -> tuple:
        try:
            try:
                from docling.document_converter import DocumentConverter, ImageFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
            except ImportError as e:
                raise ImportError("docling is required. Install with: pip install docling") from e

            if not isinstance(llm_model, dict) or "model_path" not in llm_model:
                raise ValueError("Invalid llm_model data.")
            if "mmproj_model_path" not in llm_model:
                raise ValueError("LLM must be a vision model with mmproj loaded.")

            chat_format = llm_model.get("chat_format", "")
            if not chat_format.startswith("vision-"):
                raise ValueError("Select a 'vision-*' format in the Model Loader.")

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = do_ocr
            pipeline_options.do_table_structure = do_table_structure
            pipeline_options.generate_page_images = True
            pipeline_options.images_scale = images_scale

            converter = DocumentConverter(
                allowed_formats=[InputFormat.IMAGE],
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                },
            )
            stream = _image_batch_to_docling_stream(image, name=f"{document_name or 'document'}.tiff")
            result = converter.convert(stream)
            doc = result.document

            options = options or {}
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

            def _run_vlm_pil(pil_img: Image.Image, prompt: str) -> str:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": _convert_pil_to_data_uri(pil_img)},
                        {"type": "text", "text": prompt},
                    ],
                }]
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
                if not enable_thinking:
                    import re
                    orphan_match = re.match(r"^(.*?)</think>\s*", text, re.DOTALL)
                    if orphan_match and "<think>" not in text[:orphan_match.end()]:
                        text = text[orphan_match.end():]
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
                    text = text.strip()
                return text

            page_elements = {}
            layout_items = []
            figure_crops = []

            for item, _level in doc.iterate_items():
                prov = item.prov[0] if getattr(item, "prov", None) else None
                page_no = int(getattr(prov, "page_no", 1) or 1)
                if page_no not in page_elements:
                    page_elements[page_no] = []

                bbox = getattr(prov, "bbox", None)
                label_str = str(getattr(item, "label", "")).lower()
                type_name = type(item).__name__

                layout_items.append({
                    "type": type_name,
                    "label": str(getattr(item, "label", "")),
                    "page_no": page_no,
                    "text_preview": (getattr(item, "text", "") or "")[:200],
                    "bbox": {
                        "l": getattr(bbox, "l", None),
                        "t": getattr(bbox, "t", None),
                        "r": getattr(bbox, "r", None),
                        "b": getattr(bbox, "b", None),
                    } if bbox is not None else None,
                })

                try:
                    crop = item.get_image(doc)
                except Exception:
                    crop = None

                if crop is None:
                    continue

                type_name_lower = type_name.lower()
                if "picture" in type_name_lower or "figure" in label_str:
                    figure_crops.append(_convert_pil_to_tensor(crop))
                    fig_num = len(figure_crops)
                    page_elements[page_no].append(f"![Figure {fig_num}]({_convert_pil_to_data_uri(crop)})")
                    continue

                if "table" in type_name_lower or "table" in label_str:
                    text = _run_vlm_pil(crop, table_prompt).strip()
                    if text:
                        page_elements[page_no].append(text)
                    continue

                if "formula" in label_str or "equation" in label_str:
                    latex = _run_vlm_pil(crop, formula_prompt).strip()
                    if latex and not latex.startswith("$$"):
                        latex = f"$${latex}$$"
                    if latex:
                        page_elements[page_no].append(latex)
                    continue

                if "sectionheader" in type_name_lower or "title" in label_str or "header" in label_str:
                    title = _run_vlm_pil(crop, title_prompt).strip()
                    if title:
                        page_elements[page_no].append(f"## {title}")
                    continue

                text = _run_vlm_pil(crop, text_prompt).strip()
                if text:
                    page_elements[page_no].append(text)

            sorted_pages = sorted(page_elements.keys())
            page_markdown = ["\n\n".join(page_elements[p]).strip() for p in sorted_pages]
            final_markdown = "\n\n---\n\n".join(page_markdown) if len(page_markdown) > 1 else (page_markdown[0] if page_markdown else "")

            if figure_crops:
                max_h = max(int(t.shape[1]) for t in figure_crops)
                max_w = max(int(t.shape[2]) for t in figure_crops)
                padded = []
                for t in figure_crops:
                    h, w = int(t.shape[1]), int(t.shape[2])
                    if h < max_h or w < max_w:
                        canvas = torch.zeros(1, max_h, max_w, t.shape[3], dtype=t.dtype, device=t.device)
                        canvas[:, :h, :w, :] = t
                        padded.append(canvas)
                    else:
                        padded.append(t)
                figure_images = torch.cat(padded, dim=0)
            else:
                figure_images = torch.zeros(1, 1, 1, 3, dtype=torch.float32)

            _cleanup_global_llm(memory_cleanup)

            payload = {
                "document_name": document_name,
                "page_count": len(getattr(doc, "pages", []) or []),
                "items": layout_items,
            }
            return (final_markdown, figure_images, json.dumps(payload, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"[DoclingLayoutMarkdown] Error: {str(e)}")
            raise RuntimeError(f"Docling Layout Markdown Error: {str(e)}") from e




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
                "processing_mode": (
                    ["per_region", "full_page_ocr"],
                    {
                        "default": "per_region",
                        "tooltip": (
                            "per_region: LLM processes each YOLO-detected region individually. "
                            "full_page_ocr: LLM receives the whole page image once (faster); "
                            "YOLO only locates figures which are extracted and embedded as images."
                        ),
                    },
                ),
                "max_page_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "[full_page_ocr] Longest edge (px) to which the full page is downscaled before sending to the LLM. Reduces image token count and prevents KV-cache overflow.",
                }),
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
                    "tooltip": "[per_region] Prompt for Plain Text, Titles, and Captions."
                }),
                "formula_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this mathematical formula to LaTeX. Output ONLY the latex code mathematically formatted inside $$ blocks. Do not add markdown backticks.",
                    "tooltip": "[per_region] Prompt for mathematical formulas."
                }),
                "table_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this table to Markdown format. Output ONLY the table structure without any surrounding text.",
                    "tooltip": "[per_region] Prompt for tables."
                }),
                "page_ocr_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Convert this document page to Markdown. Preserve headings, paragraphs, lists, tables (as Markdown tables), and formulas (as LaTeX inside $$ blocks). Do not add any commentary.",
                    "tooltip": "[full_page_ocr] Prompt sent with the whole page image to the LLM."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("markdown", "figure_images")
    FUNCTION = "process_layout"
    CATEGORY = "LlamaCPP"

    def process_layout(
        self,
        image: torch.Tensor,
        yolo_model: Any,
        llm_model: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        processing_mode: str = "per_region",
        max_page_size: int = 1024,
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
        page_ocr_prompt: str = "Convert this document page to Markdown. Preserve headings, paragraphs, lists, tables (as Markdown tables), and formulas (as LaTeX inside $$ blocks). Do not add any commentary.",
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
        def _process_page(page_tensor: torch.Tensor) -> tuple:
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

            # 3. Sort boxes in reading order (column-aware)
            # Strategy:
            #   a) Separate "spanning" boxes (width > 60% of page) from "column" boxes.
            #      Spanning boxes (full-width titles, bylines) would fill the coverage
            #      histogram and prevent column gap detection.
            #   b) Use only column boxes to find vertical gutters (zero-coverage runs).
            #   c) Assign column boxes to detected columns; sort each column top→bottom.
            #   d) Merge spans + column content in reading order by interleaving on y1.
            if not boxes_data:
                sorted_boxes = []
            else:
                img_w = cv_img.shape[1]
                span_threshold = img_w * 0.6  # boxes wider than this are "spanning"

                spanning = []  # full-width elements (headers, bylines)
                col_boxes = []  # proper column content
                for b in boxes_data:
                    bx1, _, bx2, _ = b["bbox"]
                    if (bx2 - bx1) >= span_threshold:
                        spanning.append(b)
                    else:
                        col_boxes.append(b)

                # Build coverage histogram from narrow (column) boxes only
                coverage = [0] * (img_w + 1)
                for b in col_boxes:
                    bx1, _, bx2, _ = b["bbox"]
                    for px in range(max(0, bx1), min(img_w, bx2)):
                        coverage[px] += 1

                # Find column boundaries via zero-coverage gutters (≥ 1% page width)
                min_gap = max(10, img_w // 100)
                col_splits = [0]
                in_gap = False
                gap_start = 0
                for px in range(img_w):
                    if coverage[px] == 0:
                        if not in_gap:
                            in_gap = True
                            gap_start = px
                    else:
                        if in_gap:
                            gap_end = px
                            if gap_end - gap_start >= min_gap:
                                col_splits.append((gap_start + gap_end) // 2)
                            in_gap = False
                col_splits.append(img_w)

                col_bounds = [(col_splits[i], col_splits[i + 1]) for i in range(len(col_splits) - 1)]
                columns = [[] for _ in col_bounds]

                # Assign each column box to the column containing its center_x
                for b in col_boxes:
                    cx = b["center_x"]
                    assigned = False
                    for ci, (cl, cr) in enumerate(col_bounds):
                        if cl <= cx < cr:
                            columns[ci].append(b)
                            assigned = True
                            break
                    if not assigned:
                        columns[-1].append(b)

                # Sort each column top→bottom, then merge left→right
                col_sorted = []
                for col in columns:
                    col.sort(key=lambda b: b["bbox"][1])
                    col_sorted.extend(col)

                # Sort spanning elements top→bottom
                spanning.sort(key=lambda b: b["bbox"][1])

                # Interleave spanning boxes back into col_sorted by y1.
                # A spanning box is inserted before the first column box whose
                # y1 is greater than the spanning box's y1.
                sorted_boxes = []
                span_iter = iter(spanning)
                next_span = next(span_iter, None)
                for cb in col_sorted:
                    while next_span is not None and next_span["bbox"][1] <= cb["bbox"][1]:
                        sorted_boxes.append(next_span)
                        next_span = next(span_iter, None)
                    sorted_boxes.append(cb)
                # Append any remaining spanning boxes (e.g. footer-spanning elements)
                if next_span is not None:
                    sorted_boxes.append(next_span)
                for s in span_iter:
                    sorted_boxes.append(s)

            # figure_idx is shared across pages via a mutable container so that
            # figure numbers are globally unique within a batch.
            # _process_page returns (markdown_str, [crop_tensor, ...])
            def _embed_figure(crop_tensor: torch.Tensor, fig_index: int) -> str:
                """Return a markdown image tag with base64 src and a Figure-N label."""
                data_uri = _convert_image_to_data_uri(crop_tensor, 0)
                return f"![Figure {fig_index}]({data_uri})"

            figure_crops: list = []  # collected crop tensors for this page

            _PAGE_OCR_SENTINEL = "__PAGE_OCR__"
            md_elements = []
            page_ocr_inserted = False

            for i, region in enumerate(sorted_boxes):
                x1, y1, x2, y2 = region["bbox"]
                label = region["label"]
                conf = region["conf"]
                print(f"[DocLayoutMarkdown] Processing region {i+1}/{len(sorted_boxes)}: {label} ({conf:.2f})")

                h, w, _ = page_tensor.shape[1:4]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop_tensor = page_tensor[:, y1:y2, x1:x2, :]

                if processing_mode == "full_page_ocr":
                    if label == "figure":
                        figure_crops.append(crop_tensor)
                        fig_num = page_figure_offset + len(figure_crops)
                        md_elements.append(_embed_figure(crop_tensor, fig_num))
                    else:
                        if not page_ocr_inserted:
                            md_elements.append(_PAGE_OCR_SENTINEL)
                            page_ocr_inserted = True
                else:
                    if label in ["title", "plain text", "figure caption", "table caption", "table footnote", "formula caption"]:
                        text = _run_vlm(crop_tensor, text_prompt)
                        if label == "title":
                            md_elements.append(f"## {text.strip()}")
                        elif label in ["figure caption", "table caption"]:
                            md_elements.append(f"*{text.strip()}*")
                        else:
                            md_elements.append(text.strip())

                    elif label == "isolate_formula":
                        latex = _run_vlm(crop_tensor, formula_prompt)
                        latex_text = latex.strip()
                        if not latex_text.startswith("$$"):
                            latex_text = f"$${latex_text}$$"
                        md_elements.append(latex_text)

                    elif label == "table":
                        table_md = _run_vlm(crop_tensor, table_prompt)
                        md_elements.append(table_md.strip())

                    elif label == "figure":
                        figure_crops.append(crop_tensor)
                        fig_num = page_figure_offset + len(figure_crops)
                        md_elements.append(_embed_figure(crop_tensor, fig_num))

            if processing_mode == "full_page_ocr" and page_ocr_inserted:
                print("[DocLayoutMarkdown] full_page_ocr: running single full-page LLM call...")
                # Reset KV cache so leftover state from any prior per-region calls
                # (or previous pages) doesn't block allocation for the full-page image.
                if hasattr(_global_llm, "reset"):
                    _global_llm.reset()

                # Downscale the page image if it's larger than max_page_size on its
                # longest edge — this is the primary cause of the KV-cache OOM.
                ocr_tensor = page_tensor  # (1, H, W, C)
                ph, pw = int(page_tensor.shape[1]), int(page_tensor.shape[2])
                longest = max(ph, pw)
                if longest > max_page_size:
                    scale = max_page_size / longest
                    new_h, new_w = max(1, int(ph * scale)), max(1, int(pw * scale))
                    # torch.nn.functional.interpolate expects (N, C, H, W)
                    import torch.nn.functional as F
                    ocr_tensor = F.interpolate(
                        page_tensor.permute(0, 3, 1, 2).float(),
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    print(f"[DocLayoutMarkdown] full_page_ocr: scaled page {ph}x{pw} → {new_h}x{new_w}")

                page_text = _run_vlm(ocr_tensor, page_ocr_prompt)
                md_elements = [
                    page_text.strip() if el == _PAGE_OCR_SENTINEL else el
                    for el in md_elements
                ]

            return "\n\n".join(md_elements), figure_crops

        # Iterate over batch
        try:
            page_results = []
            all_figure_crops: list = []  # flat list of (1,H,W,C) tensors across all pages
            batch_size = image.shape[0]
            for i in range(batch_size):
                page_tensor = image[i:i+1]  # (1, H, W, C)
                print(f"[DocLayoutMarkdown] ================= Page {i+1}/{batch_size} =================")
                page_figure_offset = len(all_figure_crops)  # figure index offset for this page
                md_text, page_figures = _process_page(page_tensor)
                page_results.append(md_text)
                all_figure_crops.extend(page_figures)

            final_markdown = "\n\n---\n\n".join(page_results) if batch_size > 1 else page_results[0]

            # Build figure_images tensor — pad all crops to the same H×W
            if all_figure_crops:
                max_h = max(int(t.shape[1]) for t in all_figure_crops)
                max_w = max(int(t.shape[2]) for t in all_figure_crops)
                padded = []
                for t in all_figure_crops:  # t is (1, h, w, C)
                    h, w = int(t.shape[1]), int(t.shape[2])
                    if h < max_h or w < max_w:
                        canvas = torch.zeros(1, max_h, max_w, t.shape[3], dtype=t.dtype, device=t.device)
                        canvas[:, :h, :w, :] = t
                        padded.append(canvas)
                    else:
                        padded.append(t)
                figure_images = torch.cat(padded, dim=0)  # (N, H, W, C)
            else:
                # No figures detected — return a 1×1 transparent placeholder
                figure_images = torch.zeros(1, 1, 1, 3, dtype=torch.float32)

            _cleanup_global_llm(memory_cleanup)
            return (final_markdown, figure_images)
            
        except Exception as e:
            print(f"DocLayoutMarkdown Engine Error: {str(e)}")
            raise RuntimeError(f"DocLayoutMarkdown Engine Error: {str(e)}") from e
