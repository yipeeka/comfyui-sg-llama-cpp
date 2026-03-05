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
from typing import Dict, Any, List, Type

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

def _convert_image_to_data_uri(image_tensor: torch.Tensor) -> str:
    """Convert a ComfyUI image tensor to a base64 data URI for vision models."""
    try:
        to_pil = ToPILImage()
        # ComfyUI images are (B, H, W, C), select first batch and permute to (C, H, W)
        pil_img = to_pil(image_tensor[0].permute(2, 0, 1))
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
                "image": ("IMAGE", {"tooltip": "Input image for vision models"}),
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Options node"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "System prompt"}),
                "memory_cleanup": (["close", "backend_free", "full_cleanup", "persistent"], {"default": "close", "tooltip": "Memory cleanup method after generation"}),
                "response_format": (["text", "json_object"], {"default": "text", "tooltip": "Output format (json_object forces valid JSON)"}),
                "enable_thinking": (IO.BOOLEAN, {"default": True, "tooltip": "Enable thinking/reasoning (Qwen3, QwQ, etc.). False injects empty <think> block to skip reasoning.", "label_on": "Enabled", "label_off": "Disabled"}),
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

    def generate(self, model: Dict[str, Any], prompt: str, image: torch.Tensor = None, options: Dict[str, Any] = None, system_prompt: str = "", memory_cleanup: str = "close", response_format: str = "text", enable_thinking: bool = True, max_tokens: int = 128, temperature: float = 0.8, top_p: float = 0.95, top_k: int = 40, repeat_penalty: float = 1.1, seed: int = -1, strip_thinking: bool = False) -> tuple:
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

            # If vision is enabled and image is provided, convert image and create structured content
            if vision_enabled and image is not None:
                image_data_uri = _convert_image_to_data_uri(image)
                user_content = [
                    {"type": "image_url", "image_url": image_data_uri},
                    {"type": "text", "text": prompt.strip()}
                ]

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

                # v0.3.30+: MTMDChatHandler uses **kwargs to RAISE on unexpected params.
                # Walk the full MRO to collect all explicitly-named __init__ params across
                # the entire inheritance chain. Only pass params that are explicitly declared.
                handler_params = set()
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
                            handler_params.add(name)
                    except (ValueError, TypeError):
                        continue

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
            if memory_cleanup == "persistent":
                if _global_llm is None:
                    _global_llm = Llama(**llama_kwargs)
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
            thinking_text = ""
            if strip_thinking:
                import re
                # Extract all complete think blocks
                think_blocks = re.findall(r"<think>(.*?)</think>", response_text, re.DOTALL)
                thinking_text = "\n\n".join(b.strip() for b in think_blocks)
                # Remove complete think blocks
                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
                # Remove any dangling opening tag (truncated output, no closing tag)
                response_text = re.sub(r"<think>.*$", "", response_text, flags=re.DOTALL)
                response_text = response_text.strip()

            # Post-generation memory cleanup
            _cleanup_global_llm(memory_cleanup)

        except Exception as e:
            response_text = f"Error generating response: {str(e)}"
            thinking_text = ""
            print(f"LlamaCPP Engine Error: {str(e)}")

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
                "image": ("IMAGE", {"tooltip": "Document/page image to OCR"}),
            },
            "optional": {
                "options": ("LLAMA_OPTIONS", {"tooltip": "Model options from Llama CPP Options node"}),
                "custom_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override the built-in OCR prompt. Leave empty to use the FireRed-OCR default prompt.",
                }),
                "n_ctx": ("INT", {"default": 32768, "min": 4096, "max": 131072, "tooltip": "Context window — must fit image tokens + prompt + output. VL images alone can eat 4k+ tokens."}),
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
        n_ctx: int = 32768,
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

            # Build message with image + OCR prompt
            prompt_text = custom_prompt.strip() if custom_prompt.strip() else _FIRERED_OCR_PROMPT
            image_data_uri = _convert_image_to_data_uri(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": image_data_uri},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # Llama init kwargs
            llama_kwargs = {
                "model_path": model_path,
            }
            for k, v in options.items():
                if not k.startswith("vision_"):
                    llama_kwargs[k] = v

            # Vision handler — same logic as LlamaCPPEngine
            if llama_kwargs.get("n_threads", -1) <= 0:
                llama_kwargs["n_threads"] = max(1, os.cpu_count() or 4)

            handler_class = VISION_HANDLERS.get(chat_format, Llava15ChatHandler)

            handler_params = set()
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
                        handler_params.add(name)
                except (ValueError, TypeError):
                    continue

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
            if memory_cleanup == "persistent":
                if _global_llm is None:
                    _global_llm = Llama(**llama_kwargs)
            else:
                _cleanup_global_llm(memory_cleanup)
                _global_llm = Llama(**llama_kwargs)

            response = _global_llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )

            if not response or "choices" not in response or not response["choices"]:
                raise RuntimeError("No response generated")

            markdown_text = response["choices"][0]["message"]["content"] or ""

            _cleanup_global_llm(memory_cleanup)

        except Exception as e:
            markdown_text = f"FireRed-OCR Error: {str(e)}"
            print(f"[FireRedOCR] Error: {str(e)}")

        return (markdown_text,)
