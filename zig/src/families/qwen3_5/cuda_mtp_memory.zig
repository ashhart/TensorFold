//! Peak device-memory admission for loading and calibrating Qwen's CUDA MTP drafter.
const std = @import("std");
const cuda = @import("cuda");
const geometry = @import("cuda_geometry.zig");
const types = @import("cuda_mtp_types.zig");
const weights = @import("cuda_mtp_weights.zig");

const head_dim = types.head_dim;
const max_lanes = types.max_lanes;
const lp_words = types.lp_words;

pub const scratch_fields = .{
    "taps_in", "hin",   "hn",         "emb",    "en",     "h",   "r",   "x",
    "qraw",    "q",     "kraw",       "kb",     "vb",     "att", "xs",  "logits",
    "words",   "picks", "lane_state", "lane_k", "lane_v", "anc", "ids", "tok",
    "rules",   "meta",  "chain_tok",  "tab",
};

pub const ScratchLayout = struct { bytes: usize, offsets: [scratch_fields.len]usize };

pub const MemorySizes = struct { ring_bytes: usize, area_bytes: usize, stream_bytes: usize, scratch_bytes: usize };

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub const StartupPlan = struct { peak: usize, resident: usize };

fn arrayBytes(a: usize, b: usize, c: usize) !usize {
    return weights.checkedMul(try weights.checkedMul(a, b), c);
}

/// The MTP allocations for a target geometry and capacity, before any device memory is allocated.
pub fn sizes(g: geometry.Geometry, capacity: usize, batch_rows: usize, batch_row_bytes: usize) !MemorySizes {
    try weights.validateGeometry(g);
    const max_c_int: usize = @intCast(std.math.maxInt(c_int));
    if (capacity == 0 or capacity > max_c_int) return error.UnsupportedMtpGeometry;
    const hidden_bytes = try weights.checkedMul(g.hidden, 2);
    const kv_width = try weights.checkedMul(g.kv_heads, head_dim);
    const ring_bytes = try arrayBytes(try weights.checkedMul(2, capacity), kv_width, 2);
    const aligned_ring = try weights.checkedAligned(ring_bytes);
    const step_offset = try weights.checkedAligned(try weights.checkedAdd(aligned_ring, hidden_bytes));
    var area_bytes = try weights.checkedAdd(step_offset, hidden_bytes);
    area_bytes = try weights.checkedAdd(area_bytes, try weights.checkedMul(max_lanes, 4));
    var stream_bytes = try weights.checkedAdd(ring_bytes, try weights.checkedMul(2, hidden_bytes));
    stream_bytes = try weights.checkedAdd(stream_bytes, try weights.checkedMul(max_lanes, 4));
    return .{ .ring_bytes = ring_bytes, .area_bytes = area_bytes, .stream_bytes = stream_bytes, .scratch_bytes = (try scratchLayout(g, batch_rows, batch_row_bytes)).bytes };
}

/// The scratch rows, measured before allocation or assigned later from the scratch buffer.
pub fn scratchLayout(g: geometry.Geometry, batch_rows: usize, batch_row_bytes: usize) !ScratchLayout {
    const R = types.max_rows * 4;
    const L = max_lanes;
    const aw = g.attnWidth();
    const kw = try weights.checkedMul(g.kv_heads, head_dim);
    const anc_stride = 1 + max_lanes;
    const absorb_id_rows = @import("cuda_cache.zig").tap_window;
    const field_sizes = .{
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, g.hidden, 2),
        try arrayBytes(R, try weights.checkedMul(2, aw), 2),
        try arrayBytes(R, aw, 2),
        try arrayBytes(R, kw, 2),
        try arrayBytes(R, kw, 2),
        try arrayBytes(R, kw, 2),
        try arrayBytes(R, aw, 2),
        try arrayBytes(R, @max(g.hidden, aw, g.intermediate) / 64, 4),
        try arrayBytes(L, g.vocab, 2),
        try arrayBytes(L, lp_words, 4),
        try arrayBytes(L, 1, 4),
        try arrayBytes(L, g.hidden, 2),
        try arrayBytes(L, kw, 2),
        try arrayBytes(L, kw, 2),
        try arrayBytes(L, anc_stride, 4),
        try arrayBytes(absorb_id_rows, 1, 4),
        try arrayBytes(L, 1, 4),
        try arrayBytes(L, @sizeOf(cuda.sample.Rule), 1),
        try arrayBytes(L, 1, 4),
        try arrayBytes(L, 1, 4),
        try arrayBytes(batch_rows, batch_row_bytes, 1),
    };
    var offsets: [scratch_fields.len]usize = undefined;
    var total: usize = 0;
    inline for (field_sizes, 0..) |size, index| {
        const at = try weights.checkedAligned(total);
        offsets[index] = at;
        total = try weights.checkedAdd(at, size);
    }
    return .{ .bytes = total, .offsets = offsets };
}

/// Same startup accounting used by admission and by embedding residency planning.
pub fn startupPlan(gpa: std.mem.Allocator, io: std.Io, g: geometry.Geometry, dir: []const u8, scratch_bytes: usize, area_bytes: usize, calibration_cache_bytes: usize) !StartupPlan {
    const plan = try weights.allocationPlan(gpa, io, g, dir);
    const peak = try peakBytes(plan.resident, plan.staging, scratch_bytes, area_bytes, calibration_cache_bytes);
    const resident = try add(plan.resident, try add(scratch_bytes, area_bytes));
    return .{ .peak = peak, .resident = resident };
}

pub fn check(gpa: std.mem.Allocator, io: std.Io, g: geometry.Geometry, dir: []const u8, scratch_bytes: usize, area_bytes: usize, calibration_cache_bytes: usize, room: ?Room) !void {
    const plan = try startupPlan(gpa, io, g, dir, scratch_bytes, area_bytes, calibration_cache_bytes);
    if (room) |r| {
        const available = r.read(r.ctx) orelse return error.MtpBudgetUnavailable;
        const need = std.math.cast(u64, plan.peak) orelse return error.MtpMemorySizeOverflow;
        if (need > available) {
            std.log.err("qwen MTP startup needs {d:.2} GiB at its peak; {d:.2} GiB remains after the CUDA reserve", .{ @as(f64, @floatFromInt(need)) / (1 << 30), @as(f64, @floatFromInt(available)) / (1 << 30) });
            return error.MtpDoesNotFit;
        }
    }
}

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.MtpMemorySizeOverflow;
}

fn peakBytes(resident_weights: usize, staging_weights: usize, scratch: usize, area: usize, calibration_cache: usize) !usize {
    const resident_state = try add(scratch, area);
    const resident_peak = try add(resident_weights, resident_state);
    const loading_peak = try add(resident_weights, staging_weights);
    // Calibration follows warm-up release: target cache coexists with MTP weights and scratch, not warm-up context.
    const calibration_state = try add(scratch, calibration_cache);
    const calibration_peak = try add(resident_weights, calibration_state);
    return @max(@max(resident_peak, loading_peak), calibration_peak);
}

test "MTP startup peak counts resident state, conversion staging, and target calibration cache" {
    try std.testing.expectEqual(@as(usize, 158), try peakBytes(100, 40, 20, 38, 30));
    try std.testing.expectEqual(@as(usize, 300), try peakBytes(100, 200, 10, 18, 20));
    try std.testing.expectEqual(@as(usize, 230), try peakBytes(100, 20, 10, 18, 120));
}

test "MTP startup peak rejects allocation-size overflow" {
    try std.testing.expectError(error.MtpMemorySizeOverflow, peakBytes(std.math.maxInt(usize), 1, 0, 0, 0));
    try std.testing.expectError(error.MtpMemorySizeOverflow, peakBytes(0, 0, std.math.maxInt(usize), 0, 1));
    try std.testing.expectError(error.MtpMemorySizeOverflow, peakBytes(0, 0, 1, 0, std.math.maxInt(usize)));
}

test "MTP admission sizes share the exact ring, context-area, and scratch geometry" {
    const g: geometry.Geometry = .{ .hidden = 64, .intermediate = 128, .layers = 1, .vocab = 256, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 128, .linear = @splat(false) };
    const capacity = 32;
    const result = try sizes(g, capacity, 16, 32);
    const ring = 2 * capacity * g.kv_heads * head_dim * 2;
    const hidden_bytes = g.hidden * 2;
    const aligned_ring = try weights.checkedAligned(ring);
    const area = try weights.checkedAdd(try weights.checkedAligned(aligned_ring + hidden_bytes), hidden_bytes + max_lanes * 4);
    try std.testing.expectEqual(ring, result.ring_bytes);
    try std.testing.expectEqual(area, result.area_bytes);
    try std.testing.expectEqual(ring + 2 * hidden_bytes + max_lanes * 4, result.stream_bytes);
    try std.testing.expect(result.scratch_bytes > 0);
    try std.testing.expectEqual(scratch_fields.len, (try scratchLayout(g, 16, 32)).offsets.len);
    try std.testing.expectError(error.UnsupportedMtpGeometry, sizes(g, 0, 16, 32));
}
