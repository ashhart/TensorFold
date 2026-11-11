//! The engine's separate CUDA scratch allocation: mixer and dense MLP rows share one region.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const routed = @import("cuda_routed.zig");
const Round = @import("cuda_round.zig");
const DeviceBuffer = cuda.DeviceBuffer;
const head_dim = geometry.head_dim;
const max_rows = Round.max_rows;
const most_verify = Round.most_verify;
const max_chunks = 64;

pub const Mode = enum { measure, allocate };

pub fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

pub fn argmaxSlices(vocab: usize) usize {
    return (vocab + 4095) / 4096;
}

/// Measure or carve row-dependent forward scratch and fixed verify buffers.
pub fn scratch(e: anytype, rows: usize, mode: Mode) !usize {
    if (rows == 0 or rows > max_rows) return error.InvalidScratchRows;
    const Take = struct {
        e: @TypeOf(e),
        mode: Mode,
        total: usize = 0,

        const Mark = struct { total: usize };

        pub fn alloc(t: *@This(), bytes: usize) !u64 {
            const at = t.total;
            t.total += aligned(bytes);
            return if (t.mode == .measure) 0 else t.e.scratch_buf.?.ptr + at;
        }

        fn mark(t: *const @This()) Mark {
            return .{ .total = t.total };
        }

        fn rewind(t: *@This(), start: Mark) Mark {
            const end = t.mark();
            t.total = start.total;
            return end;
        }

        fn past(t: *@This(), ends: []const Mark) void {
            for (ends) |m| t.total = @max(t.total, m.total);
        }
    };
    var t: Take = .{ .e = e, .mode = mode };
    const g = e.g;
    const wide = @max(g.valueWidth(), g.attnWidth(), g.hidden);
    inline for (.{ "h", "x", "r" }) |name| @field(e, name) = try t.alloc(rows * g.hidden * 2);
    e.xs = try t.alloc(rows * (@max(g.intermediate, wide) / 64) * 4);
    // a layer's mixer, then its MLP, pass only h, x, xs and r (outside it): their rows share one start
    const shared = t.mark();
    inline for (.{ "y", "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(rows * wide * 2);
    inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(rows * g.k_heads * geometry.dk * 2);
    e.qkv = try t.alloc(rows * g.convWidth() * 2);
    inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(rows * g.v_heads * 2);
    e.g_ = try t.alloc(rows * g.v_heads * 4);
    const delta = t.rewind(shared);
    inline for (.{ "y", "mix" }) |_| _ = try t.alloc(rows * wide * 2); // the attention mixer's y and mix: the same rows
    e.qraw = try t.alloc(rows * 2 * g.attnWidth() * 2);
    e.qn = try t.alloc(rows * g.attnWidth() * 2);
    inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(rows * g.kv_heads * head_dim * 2);
    const attention = t.rewind(shared);
    inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(rows * g.intermediate * 2);
    t.past(&.{ delta, attention });
    // the decode partials: verify rows only (a prompt chunk's attention runs cuda_prompt_attention's tiles)
    const partials = most_verify * g.q_heads * max_chunks;
    e.pm = try t.alloc(partials * 4);
    e.pl = try t.alloc(partials * 4);
    e.po = try t.alloc(partials * head_dim * 4);
    // a stage without the head keeps no logits
    e.logits = if (g.heads) try t.alloc(most_verify * g.vocab * 2) else 0;
    if (g.moe()) e.routed = try routed.Scratch.carve(&t, g, rows);
    e.amax = if (g.heads) try t.alloc(most_verify * argmaxSlices(g.vocab) * 8) else 0;
    e.picks = if (g.heads) try t.alloc(most_verify * cuda.sample.Draws.row_scratch) else 0;
    return t.total;
}

/// Choose a reservation large enough for the requested rows, bounded by the configured ceiling.
pub fn growthCapacity(capacity: usize, rows: usize, ceiling: usize) usize {
    if (capacity >= rows) return capacity;
    if (capacity == 0) return rows;
    return @max(rows, @min(ceiling, max_rows));
}

/// The main scratch also serves verify forwards, independently of the prompt ceiling.
fn mainGrowthCeiling(prompt_rows: usize, verify_rows: usize) usize {
    return @min(max_rows, @max(prompt_rows, verify_rows));
}

/// Refuse a forward that exceeds the scratch capacity committed during startup.
pub fn requireCapacity(capacity: usize, rows: usize) !void {
    if (rows == 0 or rows > max_rows) return error.InvalidScratchRows;
    if (rows > capacity) return error.ScratchCapacityExceeded;
}

test "scratch startup capacity refuses rows beyond its reserved ceiling" {
    try requireCapacity(128, 128);
    try std.testing.expectError(error.ScratchCapacityExceeded, requireCapacity(128, 129));
    try std.testing.expectError(error.InvalidScratchRows, requireCapacity(128, 0));
    try std.testing.expectEqual(@as(usize, 128), mainGrowthCeiling(64, 128));
    try std.testing.expectEqual(@as(usize, 256), mainGrowthCeiling(256, 128));
    try std.testing.expectEqual(@as(usize, 40), growthCapacity(0, 40, 512));
    try std.testing.expectEqual(@as(usize, 512), growthCapacity(32, 40, 512));
    try std.testing.expectEqual(@as(usize, 64), growthCapacity(32, 40, 64));
}

/// Ensure scratch is allocated during startup, before any request forward uses it.
pub fn ensureScratch(e: anytype, rows: usize) !void {
    if (rows == 0 or rows > max_rows) return error.InvalidScratchRows;
    if (rows <= e.scratch_capacity) return;
    const capacity = growthCapacity(e.scratch_capacity, rows, mainGrowthCeiling(e.scratch_rows, e.verify_rows));
    var measured = e.*;
    const bytes = try scratch(&measured, capacity, .measure);
    var replacement = DeviceBuffer.alloc(e.d, bytes) catch {
        std.log.err("Qwen3.5 CUDA scratch startup allocation failed: {d} bytes for {d} rows", .{ bytes, capacity });
        return error.ScratchGrowFailed;
    };
    e.stream.synchronize() catch |err| {
        replacement.free();
        return err;
    };
    var it = e.graphs.valueIterator();
    while (it.next()) |graph| graph.deinit();
    e.graphs.clearRetainingCapacity();
    const previous = e.scratch_buf;
    const previous_capacity = e.scratch_capacity;
    e.scratch_buf = replacement;
    e.scratch_capacity = capacity;
    _ = scratch(e, capacity, .allocate) catch |err| {
        e.scratch_buf = previous;
        e.scratch_capacity = previous_capacity;
        if (previous_capacity > 0) _ = scratch(e, previous_capacity, .allocate) catch {};
        replacement.free();
        return err;
    };
    if (previous) |old_value| {
        var old = old_value;
        old.free();
    }
}
