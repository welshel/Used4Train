# No Target Leakage

The manifest uses Clean RGB only in the video target field.
The VACE control fields vace_video and vace_reference_image contain only P48.3 condition PNGs.
No Clean image, mask, depth, normal, geometry, or QA artifact is supplied as a condition input.
