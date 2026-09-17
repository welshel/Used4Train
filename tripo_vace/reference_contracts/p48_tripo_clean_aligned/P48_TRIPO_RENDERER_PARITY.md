# P48 renderer parity

Exact path calls gsplat `rasterization` directly with the same means/quats/scales/opacities/colors and OpenCV view/K tensors as the verified P47 renderer. No cinematic path, orbit, smoothing, DA3, or per-frame transform is used.
