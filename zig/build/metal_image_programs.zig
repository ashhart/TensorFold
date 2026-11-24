//! The image families' Metal programs (tf-qwen-image, tf-moss-image*, tf-idol, tf-uocr-*) for metal_targets.zig.

/// One engine program: its executable name, root file, step description and optional C source.
pub const Program = struct { name: []const u8, path: []const u8, about: []const u8, c_source: ?[]const u8 = null };

/// Their rows, in their place between tf-qwen-vision and tf-glm-attn-probe.
pub const rows = [_]Program{
    .{ .name = "tf-qwen-image", .path = "zig/tests/qwen_image.zig", .about = "Qwen-Image-2.1's transformer on a reference case: each step compared, then the schedule timed" },
    .{ .name = "tf-moss-image", .path = "zig/tests/moss_image.zig", .about = "MOSS-Image on a reference case: each step's two branches compared, then the guided schedule timed and decoded" },
    .{ .name = "tf-moss-image-encoder", .path = "zig/tests/moss_image_encoder.zig", .about = "MOSS-Image's prompt encoder on a reference case: the ids checked, each step gated against float32, then timed" },
    .{ .name = "tf-moss-image-decoder", .path = "zig/tests/moss_image_decoder.zig", .about = "MOSS-Image's image decoder on latents, gated against float64 beside torch float32, then timed" },
    .{ .name = "tf-moss-image-blocks", .path = "zig/tests/moss_image_blocks.zig", .about = "MOSS-Image's transformer on one step, each block alone gated against float32, repeated byte for byte" },
    .{ .name = "tf-moss-image-generate", .path = "zig/tests/moss_image_generate.zig", .about = "MOSS-Image from options alone: prompts, schedule and noise, then the decoder's picture" },
    .{ .name = "tf-idol", .path = "zig/tests/idol.zig", .about = "Idol's denoiser on a reference case: each step's two branches compared, then the guided schedule timed" },
    .{ .name = "tf-uocr-decode", .path = "zig/tests/uocr/decode.zig", .about = "Unlimited-OCR-2 end to end on a page against a reference dump: prompt, layers, logits, greedy reply" },
    .{ .name = "tf-uocr-vision", .path = "zig/tests/uocr/vision.zig", .about = "Unlimited-OCR-2's image encoder on a reference dump's pixels: SAM gated against float32, CLIP and the projector reported" },
    .{ .name = "tf-uocr-blocks", .path = "zig/tests/uocr/blocks.zig", .about = "Unlimited-OCR-2's image encoder step by step: each SAM block, the neck, CLIP's front, each CLIP layer and the projector alone against float32" },
    .{ .name = "tf-idol-decode", .path = "zig/tests/idol_decode.zig", .about = "Idol's octree and Gaussian decoders on a reference case: each level, the rows and the PLY gated, then timed" },
    .{ .name = "tf-idol-vae", .path = "zig/tests/idol_vae.zig", .about = "Idol's image encoder on a reference case: its condition rows gated against float64 beside the SDK's, then timed" },
    .{ .name = "tf-idol-dino", .path = "zig/tests/idol_dino.zig", .about = "Idol's DINOv3 image encoder on a reference case: feature1 and each block gated against torch's, then timed" },
    .{ .name = "tf-idol-generate", .path = "zig/tests/idol_generate.zig", .about = "Idol end to end: one preprocessed picture to a 3DGS PLY and camera through both encoders, the denoiser and the decoders, timed by stage" },
};
