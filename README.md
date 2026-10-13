# ComfyUI-CapRL

ComfyUI custom nodes for [CapRL-Video-4B](https://huggingface.co/internlm/CapRL-Video-4B)
(InternLM / Shanghai AI Laboratory, Apache-2.0) — a Qwen3-VL-4B based dense video
captioner trained with RL + verifiable rewards.

Turn an image or video into text, inside your ComfyUI graph:

- **timeline** — strictly chronological, timestamped script (`[00:00 - 00:05] ...`),
  designed for video indexing / shot-listing / dataset labeling.
- **video** — structured detailed description (setting / subject / action).
- **image** — detailed image description.
- **prompt** — a single generation-ready global prompt: subject + clothing +
  environment + lighting + style, **no action words, no timestamps** — directly
  usable as `positive_prompt` for Wan / H3 and other generators that cannot
  consume temporal text.

## Nodes

| Node | Inputs | Outputs | Purpose |
|---|---|---|---|
| `CapRL Caption` | `media` (file in `input/`) or `image` socket, `mode`, `fps`, `max_tokens`, `model`, `python_bin`, `custom_prompt`, `save_txt` | `caption` STRING, `caption_file` STRING | Runs captioning; optionally writes `caprl_caption_*.txt` into `output/` |
| `CapRL Load Caption` | `source` (http URL / local path / pasted text) | `caption` STRING | Feeds a caption produced by a previous task into prompt nodes |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-org/ComfyUI-CapRL.git
```

### Inference environment (required)

The nodes ship **no heavyweight dependencies**. Captioning runs in one of two ways:

**Recommended — isolated subprocess env** (keeps ComfyUI's env untouched,
VRAM fully released after each call):

```bash
python3 -m venv --system-site-packages /opt/caprl_venv
/opt/caprl_venv/bin/pip install -r ComfyUI-CapRL/requirements.txt
```

Then set `python_bin` on the node to `/opt/caprl_venv/bin/python`
(or export `CAPRL_PYTHON=/opt/caprl_venv/bin/python` before launching ComfyUI).

**Alternative — in-process**: leave `python_bin` empty; requires ComfyUI's own
environment to satisfy `requirements.txt` (notably `transformers>=4.57`). The
model is cached in-process between calls.

### Model

Default `model` value is the HF id `internlm/CapRL-Video-4B` (~9.65 GB,
bf16, two safetensors shards). Point it at a local directory to avoid
downloads, or export `CAPRL_MODEL=/path/to/CapRL-Video-4B`.

ModelScope mirror: `Shanghai_AI_Laboratory/CapRL-Video-4B`.

## Example

See `workflows/`:

- `caprl_video_timeline.json` — video → timestamped script
- `caprl_image_caption.json` — image → detailed description
- `caprl_chain_wan_animate.json` — caption → inject into a Wan-Animate style
  prompt via `CapRL Load Caption` (serial-task pattern: the upstream task's
  caption URL fills the `source` input)

## Notes

- A 4B bf16 model wants ~10 GB VRAM; keep it off the GPU while large diffusion
  jobs run (subprocess mode exits and frees everything automatically).
- `timeline` output can be long — raise `max_tokens` (2048+) for videos over a
  few seconds.
- `prompt` mode is for *global* scene/style description. Temporal control of
  generation still belongs to pose/depth/control signals — CapRL is the eyes,
  not the hands.

## Queue/proxy integration (optional)

If your deployment uses a queue daemon that downloads `http(s)` media inputs
before submitting to ComfyUI, whitelist the node's `media` input so uploaded
URLs land in `input/` automatically:

```python
media_inputs = {
    ...
    'CapRLCaption': ['media'],
}
```

`save_txt` writes `caprl_caption_*.txt` to `output/` and declares it as a file
output (`images` key) so collectors upload it — the resulting URL feeds
`CapRLLoadCaption.source` in a downstream workflow (serial-task pattern).

## License

Apache-2.0 (same as the upstream CapRL-Video weights).
