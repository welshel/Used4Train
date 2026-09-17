"""TripoSplat Gradio demo with Spark.js in-browser viewer.
Usage: python run_gradio.py
"""
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr
import torch

from triposplat import TripoSplatPipeline


# ----------------------------------------------------------------------------
# Pipeline (loaded once at startup)
# ----------------------------------------------------------------------------

PIPE = TripoSplatPipeline(
    ckpt_path              = "ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path           = "ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path            = "ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path = "ckpts/vae/flux2-vae.safetensors",
    rmbg_path              = "ckpts/background_removal/birefnet.safetensors",
    device                 = "cuda",
)

OUT_ROOT     = Path("gradio_outputs").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
VIEWER_HTML  = Path("static/viewer/viewer.html").resolve()
EXAMPLES_DIR = Path("static/example_inputs").resolve()
EXAMPLES = [
    str(EXAMPLES_DIR / "creature_butterfly.webp"),
    str(EXAMPLES_DIR / "building_stone_house.webp"),
    str(EXAMPLES_DIR / "vehicle_pirate_ship.webp"),
    str(EXAMPLES_DIR / "plant_water_lily.webp"),
]

PLACEHOLDER_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;height:520px;"
    "color:#94a3b8;font:16px system-ui;background:#111318;border-radius:12px'>"
    "3D viewer will appear here after generation</div>"
)


def _gr_file(path: Path) -> str:
    """Gradio serves any file under `allowed_paths` at `/gradio_api/file=<abspath>`."""
    return f"/gradio_api/file={path.as_posix()}"


def _viewer_iframe(ply_path: Path) -> str:
    ts = time.time()  # cache-bust so the iframe reloads each generation
    src = f"{_gr_file(VIEWER_HTML)}?ply={_gr_file(ply_path)}&ts={ts}"
    return (
        f"<iframe src='{src}' "
        "style='width:100%;height:520px;border:0;border-radius:12px;background:#0a0b0e'></iframe>"
    )


# ----------------------------------------------------------------------------
# Event handlers
# ----------------------------------------------------------------------------

def generate(image, seed: int, steps: int, guidance_scale: float,
             num_gaussians: int, output_format: str,
             progress=gr.Progress(track_tqdm=True)):
    """Run the full pipeline (preprocess + encode + sample + decode)."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    progress(0, desc="Generating...")
    t0 = time.time()
    prepared = PIPE.preprocess_image(image)
    gen = torch.Generator(device=PIPE._device).manual_seed(int(seed))
    cond = PIPE.encode_image(prepared, generator=gen)
    out  = PIPE.sample_latent(cond, steps=int(steps),
                              guidance_scale=float(guidance_scale),
                              generator=gen, show_progress=True)
    gaussian = PIPE.decode_latent(out["latent"], num_gaussians=int(num_gaussians))
    gen_dt = time.time() - t0

    out_dir = OUT_ROOT / uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "splat.ply"
    gaussian.save_ply(str(ply_path))

    fmt = output_format.lower()
    if fmt == "ply":
        download_path = ply_path
    elif fmt == "splat":
        download_path = out_dir / "splat.splat"
        gaussian.save_splat(str(download_path))
    else:
        raise gr.Error(f"Unknown output format: {output_format}")

    info = (f"{gaussian.get_xyz.shape[0]:,} gaussians  ·  "
            f"generation: {gen_dt:.1f}s  ·  saved: {download_path.name}")
    return prepared, _viewer_iframe(ply_path), gr.update(value=str(download_path), interactive=True), info


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------

with gr.Blocks(title="TripoSplat") as demo:
    gr.Markdown("# TripoSplat")
    gr.Markdown(
        "TripoSplat converts a single 2D image into high-quality and variable number of 3D Gaussians developed by [TripoAI](https://www.tripo3d.ai/). "
        "It can serve as a powerful pipeline tool for asset creation, AR/VR, game development, simulation environments, and beyond.\n\n"
        "[Read Paper](https://arxiv.org/abs/2605.16355) | [Research Blog](https://www.tripo3d.ai/research/triposplat)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Input image", type="pil", image_mode="RGBA",
                                height=320)

            gr.Examples(
                examples=[[p] for p in EXAMPLES],
                inputs=[image_in],
                label="Examples (click to load)",
                examples_per_page=10,
                cache_examples=False,
            )

            with gr.Accordion("Sampling settings", open=False):
                seed_in = gr.Number(label="Seed", value=42, precision=0)
                steps_in = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=20)
                cfg_in = gr.Slider(label="Guidance scale", minimum=1.0, maximum=10.0, step=0.5, value=3.0)
                num_g_in = gr.Dropdown(
                    label="Number of gaussians",
                    choices=["32768", "65536", "131072", "262144"],
                    value="262144",
                )
                fmt_in = gr.Dropdown(label="Download format", choices=["ply", "splat"], value="ply")

            run_btn = gr.Button("Generate", variant="primary")
            prepared_out = gr.Image(label="Preprocessed input", interactive=False, height=240)
            info_out = gr.Markdown()

        with gr.Column(scale=2):
            viewer_out = gr.HTML(value=PLACEHOLDER_HTML, label="Spark.js viewer")
            file_out = gr.DownloadButton(label="Download", value=None, interactive=False)

    run_btn.click(
        fn=generate,
        inputs=[image_in, seed_in, steps_in, cfg_in, num_g_in, fmt_in],
        outputs=[prepared_out, viewer_out, file_out, info_out],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[
            str(VIEWER_HTML.parent),
            str(OUT_ROOT),
            str(EXAMPLES_DIR),
        ],
    )
