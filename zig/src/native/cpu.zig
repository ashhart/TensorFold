//! The CPU engine: Qwen3.5-2B on the host's cores over the unchanged lane core; no device, no kernel library.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"cpu"};
/// `--backend=cpu` makes the serve flag's native values {auto, cpu}, as the HIP build's `--backend=rocm` does.
pub const serves: []const []const u8 = &.{"--backend=cpu"};
pub const families: []const api.Family = &.{
    .{ .model_type = "qwen3_5", .formats = &.{"mlx-q4g64"} },
};

/// The host's processor, for `capabilities --json`.
pub fn chip(a: Allocator) ?[]const u8 {
    switch (builtin.os.tag) {
        .macos => {
            var buf: [256]u8 = undefined;
            var n: usize = buf.len;
            if (std.c.sysctlbyname("machdep.cpu.brand_string", &buf, &n, null, 0) != 0 or n == 0) return null;
            return a.dupe(u8, std.mem.trimEnd(u8, buf[0..n], "\x00\n ")) catch null;
        },
        .linux => {
            var buf: [1 << 16]u8 = undefined;
            const text = @import("cpu_admit.zig").readFile("/proc/cpuinfo", &buf) orelse return null;
            var lines = std.mem.splitScalar(u8, text, '\n');
            while (lines.next()) |line| if (std.mem.startsWith(u8, line, "model name")) {
                const colon = std.mem.indexOfScalar(u8, line, ':') orelse return null;
                return a.dupe(u8, std.mem.trim(u8, line[colon + 1 ..], " \t")) catch null;
            };
            return null;
        },
        .windows => {
            const value = @import("sys").getenv("PROCESSOR_IDENTIFIER") orelse return null;
            return a.dupe(u8, value) catch null;
        },
        else => return null,
    }
}

/// Serve flags this engine has no answer for, refused before any checkpoint is read.
pub fn refusal(a: Allocator, o: api.Open) !?[]const u8 {
    if (try o.slideRefusal(a, "the CPU")) |why| return why;
    if (o.fsrRefusal("cpu")) |why| return why;
    if (o.vision != null) return "--vision: the CPU engine reads text only";
    if (o.drafter != null) return "--drafter: the CPU engine runs plain decoding; serve without --drafter";
    if (o.speed_up != null) return "--speed-up: the CPU engine runs on one host";
    if (o.learn != null or o.spill != .off) return "--learn, --prompt-cache-disk and --spill-gib: the CPU engine keeps no states on disk yet; serve without them";
    if (o.recall) return "--recall: the CPU engine has no recall record yet; serve without --recall";
    if (o.kv8) return "--kv-cache fp8: the CPU engine keeps bf16 attention caches; serve without --kv-cache";
    if (o.prompt_int8) return "--prompt-precision int8: the CPU engine runs prompts in bf16; serve without --prompt-precision";
    if (o.rope_yarn != null) return "--rope-yarn: the CPU engine has no YaRN rope; serve without --rope-yarn";
    if (o.cfg_enabled or o.cfg_batched) return "--cfg-* guidance requires a Drama 3 speech model";
    if (o.tp != 1 or o.rank != 0) return "--tp and --rank: the CPU engine has no tensor-parallel ranks";
    if (o.device != null or o.segments != null) return "--device and --segments name CUDA devices; the CPU engine has none";
    if (o.precision != null) return "--precision applies to NVFP4 checkpoints on CUDA";
    if (o.mtp_drafts != null or o.mtp_confidence != null) return "--mtp-drafts and --mtp-confidence: this checkpoint has no MTP head";
    if (o.keep != null) return "--checkpoint-slots: the CPU engine keeps no prompt entries by slot";
    if (o.policy.len > 0) return "--policy: the CPU engine has no policy keys";
    if (o.loops != null) return "--loops applies to looped models (Dory)";
    return null;
}

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (try refusal(a, o)) |why| {
        problem.* = why;
        return null;
    }
    if (!std.mem.eql(u8, o.model_type, "qwen3_5")) {
        problem.* = try std.fmt.allocPrint(a, "the CPU engine serves qwen3_5 (Qwen3.5-2B, MLX 4-bit) checkpoints; {s} checkpoints need the Metal or CUDA build", .{o.model_type});
        return null;
    }
    if (@import("save_admission.zig").refused(io, o.dir, problem)) return null;
    return @import("qwen35_cpu.zig").open(a, gpa, io, o, problem);
}

test {
    _ = @import("cpu_admit.zig");
    _ = @import("qwen35_cpu.zig");
}

test "the CPU engine refuses device-only flags before reading a checkpoint" {
    const a = std.testing.allocator;
    const base: api.Open = .{ .dir = "", .model_type = "qwen3_5" };
    try std.testing.expectEqual(@as(?[]const u8, null), try refusal(a, base));
    var vision = base;
    vision.vision = .{};
    try std.testing.expect((try refusal(a, vision)) != null);
    var drafter = base;
    drafter.drafter = "z-lab/DFlash2";
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, drafter)).?, "--drafter"));
    var kv8 = base;
    kv8.kv8 = true;
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, kv8)).?, "--kv-cache"));
    var tp = base;
    tp.tp = 2;
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, tp)).?, "--tp"));
    var spill = base;
    spill.learn = "/tmp/states";
    spill.spill = .evicted;
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, spill)).?, "--learn"));
    var learn = base;
    learn.learn = "/tmp/states";
    try std.testing.expect(std.mem.startsWith(u8, (try refusal(a, learn)).?, "--learn"));
    var slide = base;
    slide.slide = true;
    const why = (try refusal(a, slide)).?;
    defer a.free(why);
    try std.testing.expect(std.mem.startsWith(u8, why, "--slide"));
}
