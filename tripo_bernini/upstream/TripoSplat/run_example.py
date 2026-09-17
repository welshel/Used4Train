"""TripoSplat minimal examples.
Usage: python run_example.py
"""
from triposplat import TripoSplatPipeline


pipe = TripoSplatPipeline(
    ckpt_path              = "ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path           = "ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path            = "ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path = "ckpts/vae/flux2-vae.safetensors",
    rmbg_path              = "ckpts/background_removal/birefnet.safetensors",
    device                 = "cuda",
)

INPUT = "static/example_inputs/building_stone_house.webp"


# ---------------------------------------------------------------------------
# Example 1 — one image → PLY + SPLAT
# ---------------------------------------------------------------------------

gaussian, prepared = pipe.run(INPUT, num_gaussians=262144, show_progress=True)

prepared.save("preprocessed_image.webp")
gaussian.save_ply("output.ply")
gaussian.save_splat("output.splat")


# ---------------------------------------------------------------------------
# Example 2 — one image → Gaussians at several densities (denoiser runs once,
# decoder is replayed per count) → one PLY per count
# ---------------------------------------------------------------------------

counts = [32768, 65536, 131072, 262144]
gaussians, _ = pipe.run(INPUT, num_gaussians=counts, show_progress=True)

for n, g in zip(counts, gaussians):
    g.save_ply(f"output_{n}.ply")
