//! The chat checkpoints the supported-models guide serves, with Hub sizes and the one-stream speeds our docs measured.
const std = @import("std");
const Backend = @import("sys").machine.Backend;

/// One stream's decode tok/s with drafts as shipped (or `plain`), on the chip it ran on, and the doc that records it.
pub const Speed = struct { backend: Backend, chip: []const u8, tok_s: f32, plain: bool = false, src: []const u8 };

/// TensorFold Web's quality suite on this checkpoint (tools/quality_suite.py): items passed of those graded.
pub const Quality = struct { passed: u16, items: u16, receipt: []const u8 };

/// The suite every quality figure here was measured with; a run under another digest is not comparable.
pub const quality_digest = "d21a62442423";

pub const Entry = struct {
    title: []const u8,
    format: []const u8,
    repo: []const u8,
    revision: ?[]const u8 = null,
    drafter: ?[]const u8 = null,
    drafter_revision: ?[]const u8 = null,
    /// Installed with `hf download` into ~/models/DIR as its recipe does, not with `tensorfold pull`.
    hf_dir: ?[]const u8 = null,
    /// Safetensors bytes on the Hub, drafter included.
    download: u64,
    /// What the engine keeps resident, where the model guide documents a figure other than the download.
    resident: ?u64 = null,
    metal: bool = false,
    cuda: bool = false,
    /// Machines its recipe needs; GLM-5.3-Flash is served by a pair.
    machines: u8 = 1,
    /// `drafts`: M1-M4 serve it without its drafter (M5 tensor units); `only`: M1-M4 untested.
    m5: enum { any, drafts, only } = .any,
    speeds: []const Speed = &.{},
    quality: ?Quality = null,
    recipe: []const u8,

    pub fn residentBytes(e: *const Entry) u64 {
        return e.resident orelse e.download;
    }
};

fn tenthsGib(t: u64) u64 {
    return t * (1 << 30) / 10;
}

const m = Backend.metal;
const c = Backend.cuda;

pub const entries = [_]Entry{
    .{ .title = "Qwen3.5-2B", .format = "MLX 4-bit", .repo = "mlx-community/Qwen3.5-2B-MLX-4bit", .revision = "93760be4f1f69842a46bc13dbdc0f19e291392a3", .download = 1_722_271_785, .metal = true, .cuda = true, .quality = .{ .passed = 142, .items = 228, .receipt = "docs/receipts/quality-suite/mlx-community--Qwen3.5-2B-MLX-4bit.m5-ultra.json" }, .recipe = "docs/recipes/qwen3.5-2b-native.md" },
    .{ .title = "Qwen3.8-27B", .format = "MLX 4-bit + DFlash2", .repo = "TensorFold/Qwen3.8-27B-MLX-4bit", .drafter = "z-lab/Qwen3.8-27B-DFlash2", .download = 16_054_541_349 + 3_848_817_896, .metal = true, .cuda = true, .recipe = "docs/recipes/qwen3.8-27b.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 126, .src = "CHANGELOG 1.1.0" },
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 77, .src = "CHANGELOG 1.2.6" },
        .{ .backend = m, .chip = "M5 Max", .tok_s = 64, .src = "CHANGELOG 1.0.3" },
        .{ .backend = m, .chip = "M5 Pro", .tok_s = 47.2, .src = "docs/recipes/qwen3.8-27b.md (1.2.1)" },
        .{ .backend = c, .chip = "GB10", .tok_s = 32.9, .src = "docs/receipts/qwen-cuda-first-token" },
    }, .quality = .{ .passed = 201, .items = 228, .receipt = "docs/receipts/quality-suite/TensorFold--Qwen3.8-27B-MLX-4bit.m5-ultra.json" } },
    .{ .title = "Nemotron 3.5 Lightning 30B-A3B", .format = "MLX 4-bit + MTP", .repo = "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", .download = 18_527_064_363, .metal = true, .cuda = true, .recipe = "docs/recipes/nemotron-3.5.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 348, .src = "docs/recipes/nemotron-3.5.md" },
        .{ .backend = m, .chip = "M5 Max", .tok_s = 262.8, .src = "RELEASE-NOTES-1.0.0.md" },
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 188, .src = "docs/recipes/nemotron-3.5.md" },
        .{ .backend = c, .chip = "GB10", .tok_s = 133, .src = "CHANGELOG 1.0.6" },
        .{ .backend = c, .chip = "RTX PRO 6000", .tok_s = 607, .src = "CHANGELOG 1.0.6" },
        .{ .backend = c, .chip = "RTX 3090", .tok_s = 252, .src = "CHANGELOG 1.0.3" },
    }, .quality = .{ .passed = 192, .items = 228, .receipt = "docs/receipts/quality-suite/TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit.m5-max.json" } },
    .{ .title = "Nemotron 3.5 Lightning 30B-A3B", .format = "oQ4", .repo = "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ4", .download = 18_563_590_359, .metal = true, .quality = .{ .passed = 192, .items = 228, .receipt = "docs/receipts/quality-suite/TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ4.m5-ultra.json" }, .recipe = "docs/recipes/nemotron-3.5.md" },
    .{ .title = "Nemotron 3.5 Lightning 30B-A3B", .format = "oQ6", .repo = "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ6", .download = 25_964_450_050, .metal = true, .recipe = "docs/recipes/nemotron-3.5.md" },
    .{ .title = "Nemotron 3.5 Lightning 30B-A3B", .format = "oQ8", .repo = "TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-oQ8", .download = 33_559_963_986, .metal = true, .recipe = "docs/recipes/nemotron-3.5.md" },
    .{ .title = "Nemotron 3.5 Lightning 30B-A3B", .format = "NVFP4 + MTP", .repo = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", .download = 21_561_882_284, .cuda = true, .recipe = "docs/recipes/nemotron-3.5.md", .speeds = &.{
        .{ .backend = c, .chip = "GB10", .tok_s = 105, .src = "CHANGELOG 1.0.6" },
    }, .quality = .{ .passed = 196, .items = 228, .receipt = "docs/receipts/quality-suite/nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gb10.json" } },
    .{ .title = "Nemotron 3.5 Super VL 120B-A12B", .format = "NVFP4", .repo = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", .revision = "82d93699acd3be9e4617c53360fe5b3f5dbeb1b3", .download = 75_010_188_240, .cuda = true, .recipe = "docs/recipes/nemotron-3.5-super-vl.md", .speeds = &.{
        .{ .backend = c, .chip = "GB10", .tok_s = 38.2, .src = "docs/recipes/nemotron-3.5-super-vl.md" },
    }, .quality = .{ .passed = 207, .items = 228, .receipt = "docs/receipts/quality-suite/nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gb10.json" } },
    .{ .title = "Nemotron 3.5 Bolt 74B-A9B", .format = "NVIDIA 24 GB", .repo = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", .download = 24_083_745_496, .cuda = true, .recipe = "docs/recipes/nemotron-omni-video.md" },
    .{ .title = "Qwen3.8 Flash Next", .format = "MLX 6-bit + MTP", .repo = "TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP", .download = 158_080_738_442, .metal = true, .recipe = "docs/recipes/qwen3.8-flash-next.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 150, .src = "docs/recipes/speed-up-mode.md" },
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 96, .src = "CHANGELOG 1.0.12" },
    } },
    .{ .title = "Qwen3.8 Flash Next", .format = "MLX 4-bit + MTP", .repo = "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP", .download = 113_209_682_735, .resident = tenthsGib(733), .metal = true, .cuda = true, .recipe = "docs/recipes/qwen3.8-flash-next.md", .speeds = &.{
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 137.3, .src = "CHANGELOG 1.2.7" },
        .{ .backend = c, .chip = "GB10", .tok_s = 65.8, .src = "docs/receipts/flashnext-lanes" },
    }, .quality = .{ .passed = 209, .items = 228, .receipt = "docs/receipts/quality-suite/TensorFold--Qwen3.8-Flash-Next-MLX-4bit-MTP.m5-ultra.json" } },
    .{ .title = "GLM-5.3-Flash", .format = "MLX 4-bit + MTP", .repo = "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP", .download = 181_709_451_790, .metal = true, .machines = 2, .recipe = "docs/recipes/glm-5.3-flash.md" },
    .{ .title = "GLM-5.3-Flash", .format = "NVFP4", .repo = "nvidia/GLM-5.3-Flash-NVFP4", .revision = "423acf37583782c51c142d145aef733d72943d93", .download = 204_439_103_396, .cuda = true, .machines = 2, .recipe = "docs/recipes/glm-5.3-flash-cuda.md" },
    .{ .title = "GLM-5.3-Flash", .format = "FP4-Local", .repo = "nvidia/GLM-5.3-Flash-FP4-Local", .revision = "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5", .download = 177_463_841_024, .cuda = true, .machines = 2, .recipe = "docs/recipes/glm-5.3-flash-cuda.md" },
    .{ .title = "LFM2.5-24B-A2B", .format = "MLX 4-bit", .repo = "LiquidAI/LFM2.5-24B-A2B-MLX-4bit", .revision = "667408d494bde191c2cb9dd4f4549827816b9275", .download = 13_567_294_335, .metal = true, .cuda = true, .recipe = "docs/recipes/lfm2.5-24b-a2b.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 266.5, .src = "docs/recipes/lfm2.5-24b-a2b.md" },
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 194, .src = "CHANGELOG 1.0.12" },
        .{ .backend = c, .chip = "GB10", .tok_s = 87, .src = "docs/recipes/lfm2.5-24b-a2b.md" },
    }, .quality = .{ .passed = 173, .items = 228, .receipt = "docs/receipts/quality-suite/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.m5-ultra.json" } },
    .{ .title = "LFM2.5-24B-A2B", .format = "MLX 6-bit", .repo = "LiquidAI/LFM2.5-24B-A2B-MLX-6bit", .revision = "1deec53eb49571e21cdcbf82fd5839248f07ed88", .download = 19_493_321_709, .metal = true, .recipe = "docs/recipes/lfm2.5-24b-a2b.md", .speeds = &.{
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 168, .src = "CHANGELOG 1.0.12" },
    } },
    .{ .title = "LFM2.5-24B-A2B", .format = "MLX 8-bit", .repo = "LiquidAI/LFM2.5-24B-A2B-MLX-8bit", .revision = "67ab417dd02d0c9545802a726df668620eae049d", .download = 25_484_884_997, .metal = true, .recipe = "docs/recipes/lfm2.5-24b-a2b.md", .speeds = &.{
        .{ .backend = m, .chip = "M3 Ultra", .tok_s = 160, .src = "CHANGELOG 1.0.12" },
    } },
    .{ .title = "Qwen3.6-35B-A3B", .format = "MLX 4-bit + MTP", .repo = "TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP", .revision = "81169a9bc511a27c1b4eedb77a2cd98ced431847", .download = 20_877_599_118, .cuda = true, .recipe = "docs/recipes/qwen3.6-moe.md", .speeds = &.{
        .{ .backend = c, .chip = "GB10", .tok_s = 115.7, .src = "docs/recipes/qwen3.6-moe.md" },
    } },
    .{ .title = "Qwen3.6-35B-A3B", .format = "NVFP4 + MTP", .repo = "nvidia/Qwen3.6-35B-A3B-NVFP4", .revision = "1355db6a052410cfd62085d94b58866fd0f2c3c5", .download = 23_424_338_320, .resident = tenthsGib(209), .cuda = true, .recipe = "docs/recipes/qwen3.6-moe.md", .speeds = &.{
        .{ .backend = c, .chip = "GB10", .tok_s = 111.8, .src = "docs/recipes/qwen3.6-moe.md" },
    } },
    .{ .title = "Gemma 4 31B", .format = "MLX 4-bit + DFlash", .repo = "mlx-community/gemma-4-31B-it-4bit", .revision = "696d436c404745a59f30e4939a658162b0a9e57f", .drafter = "z-lab/gemma-4-31B-it-DFlash", .drafter_revision = "eabd648301ce28583cc14757912e5e0f84e152e1", .download = 18_412_016_676 + 3_071_941_240, .metal = true, .cuda = true, .m5 = .drafts, .recipe = "docs/recipes/gemma-4-native.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 150, .src = "docs/recipes/gemma-4-native.md" },
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 51, .plain = true, .src = "docs/recipes/gemma-4-native.md" },
        .{ .backend = c, .chip = "GB10", .tok_s = 29.9, .src = "CHANGELOG 1.1.4" },
    } },
    .{ .title = "MiniMax-M3.1-Flash", .format = "3/4-bit, 32K window", .repo = "BobClawblaw/MiniMax-M3.1-Flash-MLX-mix34", .revision = "9524999057e0d277d26438606e033e2191fb465c", .hf_dir = "MiniMax-M3.1-Flash-MLX-mix34", .download = 218_737_136_413, .metal = true, .recipe = "docs/recipes/minimax-m3.1-flash.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 41.2, .src = "docs/recipes/minimax-m3.1-flash.md" },
    } },
    .{ .title = "MiniMax-M3.1-Flash", .format = "3/4-bit, 147K window", .repo = "BobClawblaw/MiniMax-M3.1-Flash-MLX-mix256k", .revision = "916e7ca1044f1c3e7566163096243c940327224d", .hf_dir = "MiniMax-M3.1-Flash-MLX-mix256k", .download = 203_109_159_689, .metal = true, .recipe = "docs/recipes/minimax-m3.1-flash.md", .speeds = &.{
        .{ .backend = m, .chip = "M5 Ultra", .tok_s = 44.0, .src = "docs/recipes/minimax-m3.1-flash.md" },
    } },
    .{ .title = "Recursive Flint 40B-A2.5B", .format = "MLX INT4", .repo = "RecursiveSI/flint-int4", .revision = "3b93f9053b530488405ae9fca51d4900d6ce21f0", .hf_dir = "flint-int4", .download = 22_568_912_600, .resident = tenthsGib(214), .metal = true, .m5 = .only, .recipe = "docs/recipes/flint.md" },
    .{ .title = "Kolibri-1", .format = "block FP8", .repo = "Aleph-Alpha/Kolibri-1", .revision = "e52eb4627d11516b0c01de49210ab5a4e4061444", .download = 78_841_467_208, .cuda = true, .recipe = "docs/recipes/kolibri-1.md", .speeds = &.{
        .{ .backend = c, .chip = "GB10", .tok_s = 49.8, .src = "docs/recipes/kolibri-1.md" },
    }, .quality = .{ .passed = 199, .items = 228, .receipt = "docs/receipts/quality-suite/Aleph-Alpha--Kolibri-1.gb10.json" } },
};

test "every entry is in the supported-models guide, its docs exist, and its speeds are on its backends" {
    const io = std.testing.io;
    const guide = try std.Io.Dir.cwd().readFileAlloc(io, "docs/supported-models.md", std.testing.allocator, .limited(4 << 20));
    defer std.testing.allocator.free(guide);
    for (entries) |e| {
        if (std.mem.indexOf(u8, guide, e.repo) == null) std.debug.print("{s} is not in docs/supported-models.md\n", .{e.repo});
        try std.testing.expect(std.mem.indexOf(u8, guide, e.repo) != null);
        try std.Io.Dir.cwd().access(io, e.recipe, .{});
        for (e.speeds) |s| if (std.mem.startsWith(u8, s.src, "docs/")) {
            const end = std.mem.indexOfScalar(u8, s.src, ' ') orelse s.src.len;
            try std.Io.Dir.cwd().access(io, s.src[0..end], .{});
        };
    }
}

test "every quality figure matches its receipt: the suite digest, the checkpoint, and the passed and graded items" {
    const io = std.testing.io;
    for (entries) |e| {
        const q = e.quality orelse continue;
        const text = try std.Io.Dir.cwd().readFileAlloc(io, q.receipt, std.testing.allocator, .limited(1 << 20));
        defer std.testing.allocator.free(text);
        var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        try std.testing.expectEqualStrings(quality_digest, root.get("suite").?.object.get("digest").?.string);
        const checkpoint = root.get("checkpoint").?.string;
        try std.testing.expectEqualStrings(e.repo, checkpoint[0 .. std.mem.indexOfScalar(u8, checkpoint, '@') orelse checkpoint.len]);
        const overall = root.get("quality").?.object.get("overall").?.object;
        try std.testing.expectEqual(@as(i64, q.passed), overall.get("passed").?.integer);
        try std.testing.expectEqual(@as(i64, q.items), overall.get("items").?.integer);
        try std.testing.expectEqual(@as(i64, 0), overall.get("unscored").?.integer);
    }
}

test "every entry has a size and a backend, and speeds on that backend" {
    for (entries) |e| {
        try std.testing.expect(e.download > 0 and (e.metal or e.cuda));
        try std.testing.expect(std.mem.startsWith(u8, e.recipe, "docs/recipes/"));
        try std.testing.expect(std.mem.indexOfScalar(u8, e.repo, '/') != null);
        if (e.revision) |r| try std.testing.expectEqual(@as(usize, 40), r.len);
        for (e.speeds) |s| {
            try std.testing.expect(s.tok_s > 0);
            try std.testing.expect(if (s.backend == .metal) e.metal else e.cuda);
        }
    }
}
