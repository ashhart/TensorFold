//! The two summary lines bare `tensorfold` and `--help` print, and `tensorfold device`'s report.
const std = @import("std");
const report = @import("report.zig");
const support = @import("support.zig");
const machine = @import("machine.zig");
const parse = @import("parse.zig");
const scan = @import("scan.zig");
const fx = @import("test_fixtures.zig");
const t = std.testing;

const metal: support.Offer = .{ .version = "test", .backends = &.{"metal"}, .families = &.{
    .{ .model_type = "nemotron_h", .formats = &.{ "mlx-q4g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "fish_omni", .formats = &.{ "fp8-block128", "nvfp4" } },
    .{ .model_type = "kimi_k3", .formats = &.{"mxfp4"} },
    .{ .model_type = "llama", .formats = &.{ "bf16", "compressed-tensors-int4-g32" } },
} };
const cuda: support.Offer = .{ .version = "test", .backends = &.{"cuda"}, .families = &.{
    .{ .model_type = "nemotron_h", .formats = &.{ "mlx-q4g64", "modelopt-nvfp4", "modelopt-fp8" } },
    .{ .model_type = "glm5_next", .formats = &.{ "exl3", "modelopt-mixed" } },
    .{ .model_type = "minimax_m3_vl", .formats = &.{"mlx-q3g64"} },
}, .kernel_sms = &support.release_sms };

fn summary(a: std.mem.Allocator, m: machine.Machine, o: support.Offer) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    try report.summary(&out.writer, m, try support.derive(a, m, o));
    return out.written();
}

fn driver(a: std.mem.Allocator, m: machine.Machine, g: machine.Gpu) !machine.Machine {
    var out = m;
    var d = g;
    d.source = .driver;
    out.gpus = try a.dupe(machine.Gpu, &.{d});
    return out;
}

test "device summary: Macs, M3 Ultra and M5 Ultra" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try t.expectEqualStrings(
        "This machine: Apple M3 Ultra, 256 GiB unified memory, macOS 27.2.1.\n" ++
            "Supported: Metal. Formats: MLX 4/6/8-bit, NVFP4, MXFP4, FP8. Details: tensorfold device\n",
        try summary(a, try parse.macos(a, "aarch64", fx.m3_ultra), metal),
    );
    try t.expectEqualStrings(
        "This machine: Apple M5 Ultra, 256 GiB unified memory, macOS 27.0.1.\n" ++
            "Supported: Metal. Formats: MLX 4/6/8-bit, NVFP4, MXFP4, FP8, int4 (compressed-tensors), BF16. Details: tensorfold device\n",
        try summary(a, try parse.macos(a, "aarch64", fx.m5_ultra), metal),
    );
    try t.expectEqualStrings(
        "This machine: Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz, 32 GiB RAM, no GPU listed, macOS 15.6.\n" ++
            "Not supported: this Mac's CPU is Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz: the Metal engines need Apple silicon (M1 or later). Details: tensorfold device\n",
        try summary(a, try parse.macos(a, "x86_64", fx.intel_mac), metal),
    );
}

test "device summary: RTX 5090, RTX 3090, GB10 and a refused Hopper card" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const win = try parse.windows(a, "x86_64", fx.rtx5090);
    try t.expectEqualStrings(
        "This machine: RTX 5090 32 GiB (sm_120), Windows 11 (build 26100).\n" ++
            "Supported: CUDA (likely). Formats: MLX 4-bit, NVFP4, FP8, EXL3. Details: tensorfold device\n",
        try summary(a, win, cuda),
    );
    try t.expectEqualStrings(
        "This machine: RTX 5090 32 GiB (sm_120), Windows 11 (build 26100).\n" ++
            "Supported: CUDA. Formats: MLX 4-bit, NVFP4, FP8, EXL3. Details: tensorfold device\n",
        try summary(a, try driver(a, win, win.gpus[0]), cuda),
    );
    const desk = try parse.linux(a, "x86_64", fx.rtx3090);
    try t.expectEqualStrings(
        "This machine: RTX 3090 24 GiB (sm_86), Ubuntu 22.04.4 LTS.\n" ++
            "Supported: CUDA. Formats: MLX 4-bit, NVFP4, FP8, EXL3. Details: tensorfold device\n",
        try summary(a, try driver(a, desk, desk.gpus[0]), cuda),
    );
    const spark = try parse.linux(a, "aarch64", fx.gb10);
    try t.expectEqualStrings(
        "This machine: NVIDIA GB10 (DGX Spark class), sm_121, 122 GiB unified memory, Ubuntu 24.04.5 LTS.\n" ++
            "Supported: CUDA. Formats: MLX 3/4-bit, NVFP4, FP8, EXL3. Details: tensorfold device\n",
        try summary(a, try driver(a, spark, spark.gpus[0]), cuda),
    );
    const hopper = try parse.linux(a, "x86_64", fx.h100);
    var h = hopper.gpus[0];
    h.memory = 80 << 30;
    try t.expectEqualStrings(
        "This machine: H100 80GB HBM3 80 GiB (sm_90), Ubuntu 22.04.4 LTS.\n" ++
            "Not supported: H100 80GB HBM3 is a Hopper GPU (sm_90): TensorFold has no Hopper kernels now. Details: tensorfold device\n",
        try summary(a, try driver(a, hopper, h), cuda),
    );
}

test "device summary: a machine with no GPU says so" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bare: machine.Machine = .{ .os = .linux, .arch = "x86_64", .cpu = "Intel(R) Xeon(R) Platinum 8480+", .ram = 64 << 30 };
    try t.expectEqualStrings(
        "This machine: Intel(R) Xeon(R) Platinum 8480+, 64 GiB RAM, no GPU listed, Linux.\n" ++
            "Not supported: this build's engines are CUDA's and no NVIDIA GPU is listed; this machine lists no GPU TensorFold runs on. Details: tensorfold device\n",
        try summary(a, bare, cuda),
    );
}

test "device summary: format names collapse MLX widths and name the rest" {
    var out: std.Io.Writer.Allocating = .init(t.allocator);
    defer out.deinit();
    try report.formatNames(&out.writer, &.{ "bf16", "exlr8", "gptq-int4-g128", "mlx-q2g128", "mlx-q4g32", "mlx-q4g64", "mlx-q8g64", "modelopt-ggml", "fp32", "something-new" });
    try t.expectEqualStrings("MLX 2/4/8-bit, EXLR8, GPTQ int4, GGML (ModelOpt), BF16, FP32, something-new", out.written());
    var none: std.Io.Writer.Allocating = .init(t.allocator);
    defer none.deinit();
    try report.formatNames(&none.writer, &.{});
    try t.expectEqualStrings("none", none.written());
}

test "device report: the machine, what runs, the memory for weights, models on disk and the profile" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const got = try scan.current(a, t.io, null, try parse.macos(a, "aarch64", fx.m3_ultra), .{ .version = "1.2.8", .offer = metal }, "2026-11-13T01:22:01Z", false, true);
    const fits = [_]report.Fit{
        .{ .repo = "Org/Small", .title = "Nemotron 3.5 Lightning", .model_type = "nemotron_h", .bytes = 17 << 30, .verdict = .fits },
        .{ .repo = "Org/Llama", .title = "Llama architecture", .model_type = "llama", .bytes = 60 << 30, .verdict = .not_here },
    };
    var out: std.Io.Writer.Allocating = .init(a);
    try report.full(&out.writer, got.doc, got.support, "1.2.8", &fits);
    try t.expectEqualStrings(
        \\This machine: Apple M3 Ultra, 256 GiB unified memory, macOS 27.2.1
        \\  CPU: Apple M3 Ultra, 32 cores (24 performance, 8 efficiency)
        \\  RAM: 256 GiB
        \\  GPU 0: Apple M3 Ultra, unified memory (from the OS)
        \\Supported: Metal, tensorfold-native 1.2.8
        \\  Formats: MLX 4/6/8-bit, NVFP4, MXFP4, FP8
        \\  Format names: fp8-block128, mlx-q4g64, mlx-q6g64, mlx-q8g64, mxfp4, nvfp4
        \\  Not here: llama: the native Llama engine needs Metal 4 tensor units (an M5 or later)
        \\  Memory for model weights: 177.2 GiB (the Metal load limit (70% of RAM, at most the GPU working set) less a 2 GiB reserve)
        \\Models on disk:
        \\  Org/Small: Nemotron 3.5 Lightning (nemotron_h), 17.0 GiB: fits
        \\  Org/Llama: Llama architecture (llama), 60.0 GiB: its family does not run on this machine
        \\Profile: ~/.tensorfold/device.json, full scan at 2026-11-13T01:22:01Z; rescanned when the hardware or this binary changes, or with --rescan
        \\
    , out.written());
}
