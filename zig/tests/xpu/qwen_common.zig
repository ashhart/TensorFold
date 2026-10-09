//! Shared helpers for the Qwen op tests: checkpoint arguments, device uploads and downloads, bf16 comparison.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

/// The MLX 4-bit checkpoint the op tests read: TF_QWEN_DIR, else the one under $HOME/models.
pub fn defaultCheckpoint() []const u8 {
    if (std.c.getenv("TF_QWEN_DIR")) |v| return std.mem.span(v);
    const home = std.mem.span(std.c.getenv("HOME") orelse "/root");
    return std.fmt.allocPrint(std.heap.page_allocator, "{s}/models/qwen3.8-27b-mlx4", .{home}) catch "/root/models/qwen3.8-27b-mlx4";
}
pub var gpa = std.heap.page_allocator;

/// First command-line argument as the checkpoint directory.
pub fn checkpoint(init: std.process.Init) ![]const u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    return if (args.len > 1) args[1] else defaultCheckpoint();
}

pub fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// Embedded bytes carry no alignment guarantee; copies them into u16 storage.
pub fn aligned(bytes: []const u8) ![]u16 {
    const out = try gpa.alloc(u16, bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

pub fn alignedF32(bytes: []const u8) ![]f32 {
    const out = try gpa.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

pub fn up(r: *rt.Runtime, bytes: []const u8) !rt.Buffer {
    const b = try r.alloc(bytes.len);
    try r.upload(b, bytes);
    try r.sync();
    return b;
}

pub fn fetch(r: *rt.Runtime, b: rt.Buffer, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    try r.download(std.mem.sliceAsBytes(out), b);
    try r.sync();
    return out;
}

pub fn fetchF32(r: *rt.Runtime, b: rt.Buffer, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    try r.download(std.mem.sliceAsBytes(out), b);
    try r.sync();
    return out;
}

/// Differing count, worst |got-want| as a fraction of max|want|, worst error in ulps of want (|want| > 5% of max).
pub const Stats = struct { differ: usize, rel_max: f32, ulps: f32 };

pub fn stats(got: []const u16, want: []const u16) Stats {
    var mx: f32 = 0;
    for (want) |w| mx = @max(mx, @abs(bf(w)));
    var s: Stats = .{ .differ = 0, .rel_max = 0, .ulps = 0 };
    for (got, want) |g, w| {
        if (g != w) s.differ += 1;
        const e = @abs(bf(g) - bf(w));
        s.rel_max = @max(s.rel_max, e / @max(mx, 1e-30));
        const a = @abs(bf(w));
        if (a > 0.05 * mx) {
            const ulp = std.math.pow(f32, 2, @floor(std.math.log2(a)) - 7);
            s.ulps = @max(s.ulps, e / ulp);
        }
    }
    return s;
}

/// Prints the stats and fails when the worst error exceeds max_rel of max|want| or max_ulps on the large values.
pub fn check(name: []const u8, got: []const u16, want: []const u16, max_rel: f32, max_ulps: f32) !void {
    const s = stats(got, want);
    std.debug.print("{s}: {d} values, {d} differ, worst {e:.2} of max|y|, worst {d:.2} ulp\n", .{ name, got.len, s.differ, s.rel_max, s.ulps });
    if (s.rel_max > max_rel or s.ulps > max_ulps) return error.TooInaccurate;
}
