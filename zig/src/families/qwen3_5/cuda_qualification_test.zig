//! The direct Qwen entry refuses experiments before reading a checkpoint or touching its CUDA context.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const sys = @import("sys");
const t = std.testing;
const names = [_][:0]const u8{ "TF_QWEN_FAST_PREFILL", "TF_QWEN_PREFILL_ROWS" };

const Saved = struct {
    values: [names.len]?[:0]u8 = @splat(null),
    fn take() !Saved {
        var saved: Saved = .{};
        errdefer for (saved.values) |value| if (value) |v| t.allocator.free(v);
        for (names, 0..) |name, i| {
            if (sys.getenv(name)) |value| saved.values[i] = try t.allocator.dupeSentinel(u8, value, 0);
        }
        return saved;
    }
    fn restore(saved: *Saved) void {
        for (names, saved.values) |name, value| if (value) |v| {
            _ = sys.env.setenv(name, v, 1);
            t.allocator.free(v);
        } else {
            _ = sys.env.unsetenv(name);
        };
    }
};

test "Qwen served qualification: direct open rejects an experiment before context or file access" {
    var saved = try Saved.take();
    defer saved.restore();
    try t.expectEqual(@as(c_int, 0), sys.env.setenv(names[0], "1", 1));
    try t.expectEqual(@as(c_int, 0), sys.env.setenv(names[1], "4096", 1));
    var ctx: cuda.Context = undefined;
    try t.expectError(error.UnqualifiedQwenFastPrefill, native.openLanes(t.allocator, t.io, &ctx, "missing-qualified-serving-checkpoint", null, .{ .context = 8192, .drafts = false }));
    const why = native.explain(null, error.UnqualifiedQwenFastPrefill) orelse return error.MissingNamedRefusal;
    try t.expect(std.mem.indexOf(u8, why, "TF_QWEN_FAST_PREFILL") != null);
}

test "Qwen served qualification: direct upstream profile reaches engine preflight instead of the guard" {
    var saved = try Saved.take();
    defer saved.restore();
    try t.expectEqual(@as(c_int, 0), sys.env.setenv(names[0], "0", 1));
    try t.expectEqual(@as(c_int, 0), sys.env.setenv(names[1], "4096", 1));
    var ctx: cuda.Context = undefined;
    const expected = if (cuda.kernels.available) error.FileNotFound else error.BuiltWithoutKernels;
    try t.expectError(expected, native.openLanes(t.allocator, t.io, &ctx, "missing-qualified-serving-checkpoint", null, .{ .context = 8192, .drafts = false }));
}
