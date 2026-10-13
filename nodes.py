"""CapRL-Video captioning nodes for ComfyUI.

Two execution paths:
- subprocess (default): calls an external Python env (`python_bin`) so the
  heavy VLM stack (transformers>=4.57, qwen-vl-utils) never touches ComfyUI's
  own environment and VRAM is fully released when the child process exits.
- in-process: set `python_bin` to empty string; requires ComfyUI's env to
  satisfy requirements.txt. The model is cached in-process between calls.
"""

import os
import subprocess
import tempfile
import time
import urllib.request

import folder_paths
import numpy as np
import torch
from PIL import Image

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caprl_caption.py")

MEDIA_EXTS = (
    ".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
)
CAPTION_MODES = ["auto", "timeline", "video", "image", "prompt"]
CAPTION_LANGS = ["en", "zh"]

# in-process model cache: {model_dir: (model, processor)}
_MODEL_CACHE = {}


MEDIA_NONE = "none"

def _list_input_media():
    try:
        files = sorted(
            f for f in os.listdir(folder_paths.get_input_directory())
            if f.lower().endswith(MEDIA_EXTS)
        )
    except OSError:
        files = []
    # 哨兵项：image 输入端口接管时/文件为空时占位（combo 不接受空串）
    return [MEDIA_NONE] + files


def _save_image_tensor(image: torch.Tensor) -> str:
    """把 IMAGE tensor 存成临时 PNG，返回文件路径。"""
    arr = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    fd, path = tempfile.mkstemp(prefix="caprl_", suffix=".png")
    os.close(fd)
    Image.fromarray(arr).save(path)
    return path


def _run_subprocess(media_path: str, mode: str, fps: float, max_tokens: int,
                    model: str, python_bin: str, prompt: str, language: str) -> str:
    out_txt = media_path + ".caption.txt"
    cmd = [
        python_bin, SCRIPT_PATH,
        "--model", model,
        "--input", media_path,
        "--mode", mode,
        "--fps", str(fps),
        "--max-new-tokens", str(max_tokens),
        "--language", language,
        "--out-text", out_txt,
    ]
    if prompt:
        cmd += ["--prompt", prompt]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"CapRL subprocess failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    with open(out_txt, "r", encoding="utf-8") as f:
        caption = f.read()
    os.remove(out_txt)
    return caption.strip()


def _run_inprocess(media_path: str, mode: str, fps: float, max_tokens: int,
                   model_dir: str, prompt: str, language: str) -> str:
    try:
        import transformers  # noqa: F401
        if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 57):
            raise ImportError(
                f"transformers {transformers.__version__} < 4.57; "
                "Qwen3-VL requires >=4.57. Use python_bin subprocess mode "
                "or upgrade this environment."
            )
        sys_path = os.path.dirname(SCRIPT_PATH)
        import sys
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        import caprl_caption as caprl
    except ImportError as e:
        raise ImportError(
            f"CapRL in-process deps missing ({e}). "
            "Set python_bin to a prepared env instead (see README)."
        )

    if model_dir not in _MODEL_CACHE:
        _MODEL_CACHE[model_dir] = caprl.load_model(model_dir)
    model_obj, processor = _MODEL_CACHE[model_dir]

    is_video = caprl.guess_is_video(media_path)
    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "timeline" if is_video else "image"
    resolved_prompt = prompt or caprl.PROMPTS[language][resolved_mode]
    messages = caprl.build_messages(media_path, is_video, resolved_prompt, fps)
    return caprl.run_caption(model_obj, processor, messages, max_tokens).strip()


class CapRLCaption:
    """图片/视频 → CapRL 反推文本。输出 STRING（同图链）+ 可选存 caption.txt 到 output。"""

    CATEGORY = "CapRL"
    FUNCTION = "caption"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("caption", "caption_file")
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, media=None, **kwargs):
        # 队列代理会把上传的 URL 下载成 dl_<md5>.ext 文件名回填——
        # 该值不在 combo 静态列表里，改用"文件存在于 input 目录"做校验
        if media in (None, "", MEDIA_NONE):
            return True
        path = os.path.join(folder_paths.get_input_directory(), media)
        if os.path.isfile(path):
            return True
        return f"CapRLCaption: input 目录不存在文件 {media}"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media": (_list_input_media(), {
                    "tooltip": "ComfyUI input 目录下的图片或视频文件名；"
                               "接入 image 输入端口时此项被忽略",
                }),
                "mode": (CAPTION_MODES, {"default": "auto"}),
                "language": (CAPTION_LANGS, {
                    "default": "en",
                    "tooltip": "内置模板的输出语言；Wan 等英文 prompt 域建议 en，"
                               "H3/中文场景可选 zh（custom_prompt 非空时此参数无效）",
                }),
                "fps": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "model": ("STRING", {
                    "default": os.environ.get("CAPRL_MODEL", "internlm/CapRL-Video-4B"),
                    "tooltip": "HF/ModelScope 模型 ID 或本地目录",
                }),
                "python_bin": ("STRING", {
                    "default": os.environ.get("CAPRL_PYTHON", ""),
                    "tooltip": "独立推理环境的 python 路径；留空=在 ComfyUI 进程内推理",
                }),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "save_txt": ("BOOLEAN", {"default": True}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    def caption(self, media, mode, language, fps, max_tokens, model, python_bin,
                custom_prompt, save_txt, image=None):
        if image is not None:
            media_path = _save_image_tensor(image)
            if mode == "auto":
                mode = "image"
        else:
            if not media or media == MEDIA_NONE:
                raise ValueError("CapRLCaption: media 未选择且未接入 image 输入")
            media_path = os.path.join(folder_paths.get_input_directory(), media)
            if not os.path.isfile(media_path):
                raise FileNotFoundError(f"input 目录不存在文件: {media}")

        if python_bin.strip():
            text = _run_subprocess(media_path, mode, fps, max_tokens,
                                   model, python_bin.strip(), custom_prompt.strip(),
                                   language)
        else:
            text = _run_inprocess(media_path, mode, fps, max_tokens,
                                  model, custom_prompt.strip(), language)

        caption_file = ""
        ui = {"text": [text]}
        if save_txt:
            out_dir = folder_paths.get_output_directory()
            fname = f"caprl_caption_{int(time.time())}.txt"
            caption_file = os.path.join(out_dir, fname)
            with open(caption_file, "w", encoding="utf-8") as f:
                f.write(text)
            # 声明为文件输出：队列代理按 images/gifs/videos/audio 键收集上传
            # → txt URL 进 pending_params → 串行下游 CapRLLoadCaption.source 占位符
            ui["images"] = [{"filename": fname, "subfolder": "", "type": "output"}]

        return {"ui": ui, "result": (text, caption_file)}


class CapRLLoadCaption:
    """url / 本地路径 / 纯文本 → STRING。用于把上游任务的 caption.txt URL
    转成下游 prompt 节点可用的字符串（串行链路占位符放在 url 输入上）。"""

    CATEGORY = "CapRL"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "http(s) URL / 本地文件路径 / 直接粘贴的文本",
                }),
                "timeout": ("INT", {"default": 30, "min": 5, "max": 300}),
            },
        }

    def load(self, source, timeout):
        src = (source or "").strip()
        if not src:
            return ("",)
        if src.startswith(("http://", "https://")):
            with urllib.request.urlopen(src, timeout=timeout) as resp:
                return (resp.read().decode("utf-8").strip(),)
        if os.path.isfile(src):
            with open(src, "r", encoding="utf-8") as f:
                return (f.read().strip(),)
        return (src,)


NODE_CLASS_MAPPINGS = {
    "CapRLCaption": CapRLCaption,
    "CapRLLoadCaption": CapRLLoadCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CapRLCaption": "CapRL Caption (Image/Video → Text)",
    "CapRLLoadCaption": "CapRL Load Caption (URL/Path/Text → String)",
}
