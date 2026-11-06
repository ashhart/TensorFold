//! Qwen3.5 CUDA forward scratch sizing, ownership and growth.
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

fn aligned(n: usize) usize {
    return std.mem.alignForward(usize, n, 256);
}

fn argmaxSlices(vocab: usize) usize {
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
    const shared = t.mark();
    inline for (.{ "y", "mix", "z", "v" }) |name| @field(e, name) = try t.alloc(rows * wide * 2);
    inline for (.{ "q", "k" }) |name| @field(e, name) = try t.alloc(rows * g.k_heads * geometry.dk * 2);
    e.qkv = try t.alloc(rows * g.convWidth() * 2);
    inline for (.{ "a", "b", "beta" }) |name| @field(e, name) = try t.alloc(rows * g.v_heads * 2);
    e.g_ = try t.alloc(rows * g.v_heads * 4);
    const delta = t.rewind(shared);
    inline for (.{ "y", "mix" }) |_| _ = try t.alloc(rows * wide * 2);
    e.qraw = try t.alloc(rows * 2 * g.attnWidth() * 2);
    e.qn = try t.alloc(rows * g.attnWidth() * 2);
    inline for (.{ "kraw", "vraw", "kn" }) |name| @field(e, name) = try t.alloc(rows * g.kv_heads * head_dim * 2);
    const attention = t.rewind(shared);
    inline for (.{ "gate", "up", "act" }) |name| @field(e, name) = try t.alloc(rows * g.intermediate * 2);
    t.past(&.{ delta, attention });
    const partials = most_verify * g.q_heads * max_chunks;
    e.pm = try t.alloc(partials * 4);
    e.pl = try t.alloc(partials * 4);
    e.po = try t.alloc(partials * head_dim * 4);
    e.logits = try t.alloc(most_verify * g.vocab * 2);
    if (g.moe()) e.routed = try routed.Scratch.carve(&t, g, rows);
    e.amax = try t.alloc(most_verify * argmaxSlices(g.vocab) * 8);
    e.picks = try t.alloc(most_verify * cuda.sample.Draws.row_scratch);
    return t.total;
}

fn growthCapacity(capacity: usize, rows: usize) usize {
    if (capacity >= rows) return capacity;
    var target = if (capacity == 0) rows else capacity;
    while (target < rows) target = @min(max_rows, target * 2);
    return target;
}

fn pendingBytes(capacity: usize, rows: usize, replacement: usize) usize {
    return if (capacity < rows) replacement else 0;
}

/// Peak additional bytes when main scratch grows before the optional INT8 workspace.
fn pendingPeak(main_new: usize, main_old: usize, int8_new: usize) usize {
    if (main_new == 0) return int8_new;
    return @max(main_new, (main_new -| main_old) + int8_new);
}

/// Pending scratch bytes, including the replacement allocation needed for safe growth.
pub fn unallocatedScratchBytes(e: anytype) !usize {
    return unallocatedScratchBytesFor(e, e.int8.on);
}

/// Pending scratch bytes for a known prompt INT8 setting, including the replacement peak.
pub fn unallocatedScratchBytesFor(e: anytype, int8_on: bool) !usize {
    const rows = @min(e.scratch_rows, max_rows);
    // Admission must reserve the same geometric capacity ensureScratch/ensureRows will allocate.
    const main_capacity = growthCapacity(e.scratch_capacity, rows);
    var measured = e.*;
    const full = try scratch(&measured, main_capacity, .measure);
    const main_new = pendingBytes(e.scratch_capacity, main_capacity, full);
    var main_old: usize = 0;
    if (main_new > 0 and e.scratch_capacity > 0) {
        var old_measure = e.*;
        main_old = try scratch(&old_measure, e.scratch_capacity, .measure);
    }
    const int8_capacity = growthCapacity(e.int8.capacity, rows);
    const int8_new = if (int8_on and e.int8.capacity < int8_capacity) blk: {
        const g = e.g;
        const k = @max(g.intermediate, g.valueWidth(), g.attnWidth(), g.hidden);
        break :blk cuda.w4a8.Kernels.scratch(int8_capacity, k);
    } else 0;
    return pendingPeak(main_new, main_old, int8_new);
}

test "lazy scratch admission reserves a full replacement only while growth remains possible" {
    const mib: usize = 1 << 20;
    const at_32 = 195 * mib;
    const at_512 = 268 * mib;
    try std.testing.expectEqual(at_512, pendingBytes(32, 512, at_512));
    try std.testing.expectEqual(@as(usize, 0), pendingBytes(512, 512, at_512));
    try std.testing.expectEqual(@as(usize, 0), pendingBytes(512, 256, at_32));
    try std.testing.expectEqual(at_512, pendingPeak(at_512, at_32, 5 * mib));
    try std.testing.expectEqual(at_512 + 5 * mib, pendingPeak(at_512, 0, 5 * mib));
    try std.testing.expectEqual(5 * mib, pendingPeak(0, 0, 5 * mib));
    try std.testing.expectEqual(@as(usize, 64), growthCapacity(32, 40));
    try std.testing.expectEqual(@as(usize, 40), growthCapacity(0, 40));
    try std.testing.expectEqual(@as(usize, 512), growthCapacity(256, 300));
    try std.testing.expectEqual(@as(usize, 512), growthCapacity(512, 300));
}

/// Grow scratch without sacrificing the previous allocation on failure.
pub fn ensureScratch(e: anytype, rows: usize) !void {
    if (rows == 0 or rows > max_rows) return error.InvalidScratchRows;
    if (rows <= e.scratch_capacity) return;
    const capacity = growthCapacity(e.scratch_capacity, rows);
    var measured = e.*;
    const bytes = try scratch(&measured, capacity, .measure);
    var replacement = DeviceBuffer.alloc(e.d, bytes) catch return error.ScratchGrowFailed;
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
