//! The support table: what each named machine runs with a Metal or a CUDA build, and why not when nothing does.
const std = @import("std");
const support = @import("support.zig");
const machine = @import("machine.zig");
const parse = @import("parse.zig");
const fx = @import("test_fixtures.zig");
const t = std.testing;
const Machine = machine.Machine;

const metal_families = [_]support.Family{
    .{ .model_type = "nemotron_h", .formats = &.{ "mlx-q4g64", "mlx-q6g64", "mlx-q8g64" } },
    .{ .model_type = "gemma4", .formats = &.{"mlx-q4g64"} },
    .{ .model_type = "fish_omni", .formats = &.{ "fp8-block128", "nvfp4" } },
    .{ .model_type = "kimi_k3", .formats = &.{"mxfp4"} },
    .{ .model_type = "llama", .formats = &.{ "bf16", "compressed-tensors-int4-g32" } },
};
const cuda_families = [_]support.Family{
    .{ .model_type = "nemotron_h", .formats = &.{ "mlx-q4g64", "modelopt-nvfp4", "modelopt-fp8" } },
    .{ .model_type = "glm5_next", .formats = &.{ "exl3", "modelopt-mixed" } },
    .{ .model_type = "minimax_m3_vl", .formats = &.{"mlx-q3g64"} },
};

/// cuda_memory.reserveBytes' defaults: unified a tenth (at least 4 GiB), discrete a 32nd (at least 1 GiB).
fn reserve(total: u64, unified: bool) u64 {
    return if (unified) @max(4 << 30, total / 10) else @max(1 << 30, total / 32);
}

const metal: support.Offer = .{ .version = "test", .backends = &.{"metal"}, .families = &metal_families };
const release: support.Offer = .{ .version = "test", .backends = &.{"cuda"}, .families = &cuda_families, .kernel_sms = &support.release_sms, .cuda_reserve = reserve };
const unrecorded: support.Offer = .{ .version = "test", .backends = &.{"cuda"}, .families = &cuda_families, .cuda_reserve = reserve };

fn driver(m: Machine, a: std.mem.Allocator, sm: u32, memory: u64, unified: bool, name: []const u8) !Machine {
    var out = m;
    out.gpus = try a.dupe(machine.Gpu, &.{.{ .vendor = .nvidia, .name = name, .sm = sm, .memory = memory, .unified = unified, .source = .driver }});
    return out;
}

fn hasNote(s: support.Support, needle: []const u8) bool {
    for (s.notes) |n| if (std.mem.indexOf(u8, n, needle) != null) return true;
    return false;
}

test "device support: an RTX 5090 runs CUDA, from its name until the driver says, without MiniMax's GB10-only path" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const quick = try parse.windows(a, "x86_64", fx.rtx5090);
    const s = try support.derive(a, quick, release);
    try t.expectEqualStrings("cuda", s.backend.?);
    try t.expectEqual(@as(?usize, 0), s.gpu);
    try t.expect(!s.certain);
    try t.expect(hasNote(s, "from the card's name"));
    try t.expectEqual(@as(usize, 1), s.unavailable.len);
    try t.expectEqualStrings("minimax_m3_vl", s.unavailable[0].family);
    try t.expect(!hasNote(s, "decoding to BF16"));
    try t.expectEqual(@as(u64, 34190917632) - (1 << 30), s.budget); // a 32nd is under the 1 GiB floor
    for (s.formats) |f| try t.expect(!std.mem.eql(u8, f, "mlx-q3g64"));
    const full = try support.derive(a, try driver(quick, a, 120, 34190917632, false, "NVIDIA GeForce RTX 5090"), release);
    try t.expect(full.certain);
    try t.expect(!hasNote(full, "from the card's name"));
    try t.expectEqualStrings("minimax_m3_vl does not run on this machine: MiniMax-M3.1-Flash on CUDA is served on GB10 (sm_121) only so far", (try support.refusal(a, full, "minimax_m3_vl")).?);
    try t.expect(try support.refusal(a, full, "nemotron_h") == null);
    try t.expect(try support.refusal(a, s, "minimax_m3_vl") == null); // a name-table verdict never refuses a serve
}

test "device support: an RTX 3090 (sm_86) runs CUDA, its NVFP4 and FP8 weights decoded to BF16" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try driver(try parse.linux(a, "x86_64", fx.rtx3090), a, 86, 24 << 30, false, "NVIDIA GeForce RTX 3090");
    const s = try support.derive(a, m, release);
    try t.expectEqualStrings("cuda", s.backend.?);
    try t.expect(s.certain);
    try t.expect(hasNote(s, "decoding to BF16"));
    try t.expect(hasNote(s, "sm_80 to sm_89"));
    try t.expect(hasNote(s, "--precision checkpoint"));
    try t.expect(hasNote(s, "--slide"));
    try t.expectEqual(@as(u64, 23) << 30, s.budget);
    var nvfp4 = false;
    for (s.formats) |f| nvfp4 = nvfp4 or std.mem.eql(u8, f, "modelopt-nvfp4");
    try t.expect(nvfp4); // the formats still serve: by decoding, not on FP4 tensor cores
    try t.expect(support.covers(&.{86}, 89) and !support.covers(&.{86}, 80) and !support.covers(&.{120}, 121) and !support.covers(&.{80}, 90));
}

test "device support: a GB10 runs CUDA on unified memory, MiniMax included" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const quick = try parse.linux(a, "aarch64", fx.gb10);
    const m = try driver(quick, a, 121, 0, true, "NVIDIA GB10");
    const s = try support.derive(a, m, release);
    try t.expectEqualStrings("cuda", s.backend.?);
    try t.expectEqual(@as(usize, 0), s.unavailable.len);
    try t.expect(!hasNote(s, "--slide"));
    try t.expectEqual(m.ram - m.ram / 10, s.budget);
    try t.expect(std.mem.indexOf(u8, s.budget_rule, "system RAM") != null);
    try t.expectEqualStrings("cuda", (try support.derive(a, quick, release)).backend.?); // the quick scan agrees
}

test "device support: a Hopper GPU is refused by name, and serve refuses only with a recorded kernel list" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try driver(try parse.linux(a, "x86_64", fx.h100), a, 90, 80 << 30, false, "NVIDIA H100 80GB HBM3");
    const s = try support.derive(a, m, release);
    try t.expect(s.backend == null);
    try t.expect(s.certain);
    try t.expectEqualStrings("H100 80GB HBM3 is a Hopper GPU (sm_90): TensorFold has no Hopper kernels now", s.reason);
    try t.expectEqualStrings("this machine cannot run this build's engines: H100 80GB HBM3 is a Hopper GPU (sm_90): TensorFold has no Hopper kernels now", (try support.refusal(a, s, "nemotron_h")).?);
    const assumed = try support.derive(a, m, unrecorded);
    try t.expect(assumed.backend == null and !assumed.certain);
    try t.expect(try support.refusal(a, assumed, "nemotron_h") == null);
    const quick = try support.derive(a, try parse.linux(a, "x86_64", fx.h100), release);
    try t.expect(quick.backend == null and !quick.certain); // from the name table: told, never refused
    const sm89: support.Offer = .{ .version = "test", .backends = &.{"cuda"}, .families = &cuda_families, .kernel_sms = &.{89} };
    const other = try support.derive(a, try driver(m, a, 86, 24 << 30, false, "NVIDIA GeForce RTX 3090"), sm89);
    try t.expectEqualStrings("RTX 3090 is sm_86: this build has kernels for sm_89, not for it", other.reason);
    const none: support.Offer = .{ .version = "test", .backends = &.{"cuda"}, .kernel_sms = &.{} };
    const bare = try support.derive(a, m, none);
    try t.expect(std.mem.indexOf(u8, bare.reason, "no CUDA kernels") != null and !bare.certain);
}

test "device support: an M3 Ultra runs Metal under the 70% load limit, Llama left to the M5" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parse.macos(a, "aarch64", fx.m3_ultra);
    const s = try support.derive(a, m, metal);
    try t.expectEqualStrings("metal", s.backend.?);
    try t.expect(s.certain);
    try t.expectEqual(@as(usize, 1), s.unavailable.len);
    try t.expectEqualStrings("llama", s.unavailable[0].family);
    try t.expect(hasNote(s, "Gemma 4"));
    try t.expectEqual(m.ram / 100 * 70 - (2 << 30), s.budget);
    for (s.formats) |f| try t.expect(!std.mem.eql(u8, f, "compressed-tensors-int4-g32"));
    try t.expect((try support.refusal(a, s, "llama")) != null);
    try t.expect(try support.refusal(a, s, "nemotron_h") == null);
    const m5 = try support.derive(a, try parse.macos(a, "aarch64", fx.m5_ultra), metal);
    try t.expectEqual(@as(usize, 0), m5.unavailable.len);
    try t.expectEqual(@as(usize, 0), m5.notes.len);
    try t.expectEqual(@as(usize, 8), m5.formats.len);
    const wired = try support.derive(a, try parse.macos(a, "aarch64", fx.m1_max_wired), metal);
    try t.expectEqual((@as(u64, 32) << 30) / 100 * 70 - (2 << 30), wired.budget); // 70% binds below the 28 GB wired limit
}

test "device support: no GPU, an Intel Mac, an AMD APU and an engine-less build each say why" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bare: Machine = .{ .os = .linux, .arch = "x86_64", .cpu = "Intel(R) Xeon(R) Platinum 8480+", .ram = 64 << 30 };
    const none = try support.derive(a, bare, release);
    try t.expect(none.backend == null and !none.certain);
    try t.expectEqualStrings("this build's engines are CUDA's and no NVIDIA GPU is listed; this machine lists no GPU TensorFold runs on", none.reason);
    try t.expect(try support.refusal(a, none, "nemotron_h") == null); // the OS's listing can miss a GPU: serve goes on
    const intel = try support.derive(a, try parse.macos(a, "x86_64", fx.intel_mac), metal);
    try t.expect(intel.backend == null and intel.certain);
    try t.expect(std.mem.indexOf(u8, intel.reason, "Apple silicon (M1 or later)") != null);
    try t.expect((try support.refusal(a, intel, "nemotron_h")) != null);
    const amd = try support.derive(a, try parse.linux(a, "x86_64", fx.strix_halo), release);
    try t.expect(std.mem.indexOf(u8, amd.reason, "AMD GPU (gfx1151): a HIP build (-Dhip=true) runs on it") != null);
    const hip: support.Offer = .{ .version = "test", .backends = &.{"hip"}, .families = &cuda_families };
    const on_hip = try support.derive(a, try parse.linux(a, "x86_64", fx.strix_halo), hip);
    try t.expectEqualStrings("hip", on_hip.backend.?);
    try t.expectEqual(@as(u64, 94) << 30, on_hip.budget);
    const wsl = try support.derive(a, try parse.linux(a, "x86_64", fx.wsl), release);
    try t.expect(std.mem.indexOf(u8, wsl.reason, "WSL 2") != null and !wsl.certain);
    const empty = try support.derive(a, bare, .{ .version = "test", .backends = &.{} });
    try t.expect(std.mem.indexOf(u8, empty.reason, "no engines") != null);
    try t.expect(try support.refusal(a, empty, "nemotron_h") == null); // the build's own refusal stands
}

test "device support: a checkpoint on disk fits, is too large, does not run here or is not served" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s = try support.derive(a, try parse.macos(a, "aarch64", fx.m3_ultra), metal);
    try t.expectEqual(support.Verdict.fits, support.fit(s, &metal_families, "nemotron_h", 17 << 30));
    try t.expectEqual(support.Verdict.too_large, support.fit(s, &metal_families, "nemotron_h", 200 << 30));
    try t.expectEqual(support.Verdict.not_here, support.fit(s, &metal_families, "llama", 17 << 30));
    try t.expectEqual(support.Verdict.not_served, support.fit(s, &metal_families, "deepseek_v4", 17 << 30));
}

test "device support: the assumed release kernel set is the one the release build validates" {
    const text = try std.Io.Dir.cwd().readFileAlloc(t.io, "zig/build/cuda.zig", t.allocator, .limited(1 << 20));
    defer t.allocator.free(text);
    var line: std.Io.Writer.Allocating = .init(t.allocator);
    defer line.deinit();
    try line.writer.writeAll("pub const release_sms = \"");
    for (support.release_sms, 0..) |sm, i| try line.writer.print("{s}{d}", .{ if (i > 0) "," else "", sm });
    try line.writer.writeAll("\";");
    try t.expect(std.mem.indexOf(u8, text, line.written()) != null);
}
