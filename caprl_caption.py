#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapRL-Video-4B 通用反推工具（图片/视频 → 结构化文字剧本）

定位（严格遵循"CapRL 是眼睛不是手"的边界）：
- 输出只进"理解/检索/打标"链路 + 作为下游生成工作流的【全局提示词】素材
- 生成端的时间轴由物理约束控制（pose/depth 图），文本不承担逐秒控制职责

输入支持：
- 本地路径或 http(s) URL；图片(jpg/png/webp)与视频(mp4/webm/mov)自动识别
- 视频走 Qwen3-VL video 接口（fps 采样），图片走 image 接口

三种模式（--mode）：
- image  : 官方图像模板 "Please describe this image in detail."
- video  : 官方视频模板 "Please describe this video in detail."
- timeline: 官方时间戳剧本模板（[mm:ss - mm:ss] 结构化逐段输出，串行链路默认用这个）

输出（二选一或都给）：
- --out-text  : 纯文本剧本（一行）
- --out-json  : {"caption":..., "type":..., "source":..., "mode":..., "generated_at":...}

串行工作流接入（与项目 precondition 机制零适配成本）：
- 本脚本输出的 caption 字符串可直接写入子任务 pending_params，
  经 button_param_map 映射后，由云端 PHP 守护 / 本地 precondition_processor.py
  的 _replace_comfyui_placeholders_in_json() 替换下游 ComfyUI 工作流里的
  {{caprl_caption}} 占位符（正整数张数/seed 同理，见 runninghub_api.mod.php 惯例）
- 下游工作流模板示例：WanVideoTextEncodeCached.positive_prompt 写成
  "the person from the reference image ..., {{caprl_caption}}, natural lighting"
- JSON 输出方便调度层读取多字段；纯文本输出可直接做占位符替换值

用法：
  python caprl_caption.py --input /root/ComfyUI/input/222222.mp4 --mode timeline --out-json out.json
  python caprl_caption.py --input https://xx/img.jpg --mode image --out-text cap.txt
  python caprl_caption.py --input a.mp4 --mode timeline --placeholder-name caprl_caption \
      --out-json out.json --template downstream_template.json --output-template filled.json

运行环境：ComfyUI 容器内 /data/node1_local/caprl_venv（--system-site-packages 复用 torch，
transformers>=4.57 装在 venv 内不影响 ComfyUI 自身环境）。
"""

import argparse
import datetime
import json
import mimetypes
import os
import sys
import tempfile
import urllib.request

# ---------- 官方 prompt 模板（源自 internlm/CapRL-Video-4B README，勿改动语义） ----------
IMAGE_PROMPT = "Please describe this image in detail."

VIDEO_PROMPT = "Please describe this video in detail."

IMAGE_PROMPT_ZH = "请详细描述这张图片的内容。"

VIDEO_PROMPT_ZH = "请详细描述这段视频的内容。"

TIMELINE_PROMPT = (
    "You are a video captioner. Describe this video in strictly chronological order and in great detail. \n\n"
    "Your description should cover:\n"
    "1. **Key Visuals:** Detailed appearance of main characters (clothing, colors, gender) and prominent objects.\n"
    "2. **Actions & Timestamps:** Specific movements and interactions (e.g., instead of \"he moved,\" say "
    "\"he walked quickly towards the door\"). **Crucially, you must anchor distinct actions, events, and state "
    "changes with exact timestamps using the format `[mm:ss]` or `[mm:ss - mm:ss]`.**\n"
    "3. **Sequence:** Clearly state the order of events from start to finish, explicitly noting when actions "
    "happen simultaneously or sequentially.\n"
    "4. **Setting:** Describe the background environment and lighting.\n\n"
    "Goal: Provide a comprehensive and temporally precise description so that someone who cannot see this video "
    "can visualize it perfectly and map out an exact mental timeline of all events.\n\n"
    "Example Format:\n"
    "\"[00:00 - 00:05] A man wearing a red jacket stands by a window in a dimly lit room. "
    "[00:06] He suddenly turns around and [00:08 - 00:12] walks quickly towards a wooden desk."
)

# 中文版逐段剧本模板：时间戳格式不变（[mm:ss] 与英文版一致），仅描述语言切换为中文
TIMELINE_PROMPT_ZH = (
    "你是一名视频内容分析师。请严格按时间顺序、详尽地描述这段视频，全部使用中文输出。\n\n"
    "描述必须覆盖：\n"
    "1. **关键画面**：主要人物的外貌细节（服装、颜色、性别）与显著物体。\n"
    "2. **动作与时间戳**：具体动作和互动（不要写\"他动了\"，要写\"他快步走向门边\"）。"
    "**必须为每个不同的动作、事件和状态变化标注精确时间戳，格式为 `[mm:ss]` 或 `[mm:ss - mm:ss]`。**\n"
    "3. **事件顺序**：明确说明事件从头至尾的先后顺序，注明哪些动作同时发生、哪些先后发生。\n"
    "4. **场景环境**：描述背景环境与光线。\n\n"
    "目标：提供一份全面且时间精确的描述，让看不到视频的人也能在脑海中还原完整画面和精确时间线。\n\n"
    "示例格式：\n"
    "\"[00:00 - 00:05] 一名穿红色外套的男子站在光线昏暗的房间窗边。"
    "[00:06] 他突然转身，[00:08 - 00:12] 快步走向一张木质书桌。\""
)

# 生成端可直接消费的全局提示词模板（Wan/H3 均不支持时间戳文本：
# 只描述主体/服装/环境/光影/画风，禁动作词与时间词——时间轴由 pose/depth 物理约束）
PROMPT_MODE_PROMPT = (
    "Describe this video as a single ready-to-use text-to-video generation prompt. "
    "Rules: describe ONLY the subject appearance, clothing and materials, environment and background, "
    "lighting, camera framing and cinematic style. "
    "Do NOT mention any actions, movements, gestures, temporal order, or timestamps. "
    "Output one paragraph of comma-separated phrases, under 80 words, English only, no preamble."
)

# 中文生成提示词模板：同样的"静态外观描述"约束，输出中文逗号分隔短语（供 H3 等中文友好的生成模型使用；
# Wan 生态建议仍用 en——其 text encoder 对英文短语响应更稳定）
PROMPT_MODE_PROMPT_ZH = (
    "请把这段视频概括为一段可直接用于视频生成的中文提示词。"
    "规则：只描述主体外貌、服装与材质、环境与背景、光线、镜头景别和画面风格。"
    "不要提及任何动作、移动、手势、时间顺序或时间戳。"
    "输出一段以逗号分隔的短语，80字以内，不要任何开场白。"
)

# 按语言索引的模板表：{language: {mode: prompt}}
PROMPTS = {
    "en": {
        "image": IMAGE_PROMPT,
        "video": VIDEO_PROMPT,
        "timeline": TIMELINE_PROMPT,
        "prompt": PROMPT_MODE_PROMPT,
    },
    "zh": {
        "image": IMAGE_PROMPT_ZH,
        "video": VIDEO_PROMPT_ZH,
        "timeline": TIMELINE_PROMPT_ZH,
        "prompt": PROMPT_MODE_PROMPT_ZH,
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}


def guess_is_video(source: str) -> bool:
    """按扩展名/MIME 判断输入是视频还是图片（URL 取路径部分解析）。"""
    path = source.split("?")[0].split("#")[0]
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return True
    if ext in IMAGE_EXTS:
        return False
    mtype, _ = mimetypes.guess_type(path)
    if mtype and mtype.startswith("video"):
        return True
    if mtype and mtype.startswith("image"):
        return False
    raise SystemExit(f"[caprl] 无法识别输入类型（扩展名/MIME 均未知）: {source}")


def resolve_source(source: str, workdir: str) -> str:
    """URL 下载到临时目录，返回本地路径；本地路径原样返回。"""
    if source.startswith(("http://", "https://")):
        filename = os.path.basename(source.split("?")[0].split("#")[0]) or "input.bin"
        local = os.path.join(workdir, filename)
        # 流式下载，避免大文件整读进内存
        with urllib.request.urlopen(source, timeout=60) as resp, open(local, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return local
    return source


def load_model(model_dir: str):
    """加载 CapRL-Video-4B（Qwen3-VL 架构，transformers>=4.57 原生支持）。"""
    import torch  # noqa: F401 —— 触发 system-site-packages 的容器 torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        model_dir,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,  # 与官方 Qwen3-VL 推理示例一致的像素预算，控显存
    )
    return model, processor


def build_messages(source: str, is_video: bool, prompt: str, fps: float) -> list:
    """构造 Qwen3-VL chat messages（video 走 fps 采样，image 走原生分支）。"""
    media = (
        {"type": "video", "video": source, "fps": fps}
        if is_video
        else {"type": "image", "image": source}
    )
    return [{"role": "user", "content": [media, {"type": "text", "text": prompt}]}]


def run_caption(model, processor, messages: list, max_new_tokens: int) -> str:
    """官方 Qwen3-VL 推理接口：apply_chat_template(return_videos) → generate。"""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 确定性输出，串行链路可复现
        temperature=None,
        top_p=None,
        top_k=None,
    )
    # 仅解码新增 token（去除 prompt 部分）
    text = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def fill_template(template_path: str, out_path: str, placeholder: str, caption: str) -> None:
    """把反推文本填进下游 ComfyUI 工作流模板的 {{占位符}}。

    与 precondition_processor._replace_comfyui_placeholders_in_json 同款字符串级替换：
    保持 JSON 键序与节点 ID 不变，仅替换占位符；先校验替换后 JSON 仍合法。
    """
    with open(template_path, "r", encoding="utf-8") as f:
        workflow = f.read()
    token = "{{" + placeholder + "}}"
    if token not in workflow:
        raise SystemExit(f"[caprl] 模板中未找到占位符 {token}")
    # JSON 字符串内的合法转义（与云端 PHP replaceComfyuiPlaceholdersInJson 行为一致）
    escaped = caption.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "").replace("\t", "\\t")
    filled = workflow.replace(token, escaped)
    json.loads(filled)  # 合法性校验，坏 JSON 直接抛错终止
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(filled)


def main() -> None:
    parser = argparse.ArgumentParser(description="CapRL-Video-4B 通用反推（图片/视频 → 文字剧本）")
    parser.add_argument("--model", default="/data/ComfyUI/shared_models/LLM/CapRL-Video-4B",
                        help="模型目录（默认 4090 /data 池路径）")
    parser.add_argument("--input", required=True, help="本地路径或 http(s) URL（图片/视频）")
    parser.add_argument("--mode", choices=["image", "video", "timeline", "prompt", "auto"], default="auto",
                        help="image/video/timeline/prompt(生成端全局提示词)；auto=按输入类型选（视频→timeline，图片→image）")
    parser.add_argument("--fps", type=float, default=2.0, help="视频采样帧率（Qwen3-VL video.fps）")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="生成上限")
    parser.add_argument("--language", choices=["en", "zh"], default="en",
                        help="内置模板的输出语言：en=英文（Wan 等英文 prompt 域推荐）；zh=中文（H3/中文场景）")
    parser.add_argument("--prompt", default=None, help="自定义 prompt（覆盖 mode 自带模板）")
    parser.add_argument("--out-text", help="纯文本输出路径")
    parser.add_argument("--out-json", help="JSON 输出路径")
    parser.add_argument("--template", help="下游工作流模板（含 {{占位符}} 的 ComfyUI JSON）")
    parser.add_argument("--placeholder-name", default="caprl_caption", help="模板占位符名")
    parser.add_argument("--output-template", help="填完占位符的下游工作流输出路径")
    args = parser.parse_args()

    if not (args.out_text or args.out_json or (args.template and args.output_template)):
        parser.error("至少指定 --out-text / --out-json / --template+--output-template 之一")

    is_video = guess_is_video(args.input)
    mode = args.mode
    if mode == "auto":
        mode = "timeline" if is_video else "image"
    if mode in ("video", "timeline") and not is_video:
        print("[caprl] 警告: 视频模式收到图片输入，已按图片接口处理", file=sys.stderr)
    prompt = args.prompt or PROMPTS[args.language][mode]

    with tempfile.TemporaryDirectory(prefix="caprl_") as workdir:
        local = resolve_source(args.input, workdir)
        print(f"[caprl] 加载模型: {args.model}", file=sys.stderr)
        model, processor = load_model(args.model)
        print(f"[caprl] 推理: type={'video' if is_video else 'image'}, mode={mode}", file=sys.stderr)
        caption = run_caption(model, processor, build_messages(local, is_video, prompt, args.fps),
                               args.max_new_tokens)

    result = {
        "caption": caption,
        "type": "video" if is_video else "image",
        "mode": mode,
        "source": args.input,
        "placeholder_name": args.placeholder_name,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.out_text:
        with open(args.out_text, "w", encoding="utf-8") as f:
            f.write(caption)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    if args.template and args.output_template:
        fill_template(args.template, args.output_template, args.placeholder_name, caption)

    print("[caprl] 完成")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
