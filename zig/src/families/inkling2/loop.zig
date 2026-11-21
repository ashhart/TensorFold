//! Inkling 2 Mini's loop host layout: 48 private visit caches, whole-state snapshots and one-rank admission.
const std = @import("std");
const config = @import("config.zig");
const plan_ = @import("plan.zig");

pub fn validateLoad(c: *const config.Config, rank: usize, images: bool) !void {
    if (rank != 1) return error.Inkling2LoopOneRankOnly;
    if (images) return error.Inkling2LoopTextOnly;
    const lp = c.text.loop orelse return error.Inkling2LoopRequired;
    if (lp.gate_rank != 64) return error.UnsupportedLoopGateRank;
}

pub const SnapshotSlot = struct {
    kv_offset: u64,
    kv_bytes: u64,
    conv_offset: u64,
    conv_bytes: u64,
};

/// A stream or a saved prompt owns the same complete layout: no visit shares KV or convolution state with another.
pub const CacheLayout = struct {
    slots: [config.max_visits]SnapshotSlot,
    n: u32,
    bytes: u64,

    pub fn init(c: *const config.Text, p: *const plan_.Plan, context: u32) !CacheLayout {
        if (context == 0 or context > c.max_context or p.n_visits != c.visitCount()) return error.BadLoopCache;
        var out: CacheLayout = undefined;
        out.n = p.n_visits;
        out.bytes = 0;
        const conv_values = 3 * (2 * @as(u64, c.local.kv * c.local.dim) + 2 * @as(u64, c.hidden));
        for (p.visits[0..p.n_visits], 0..) |visit, i| {
            const layer = p.layers[visit.layer];
            const rows: u32 = @intCast(@min(2048, context));
            const tokens = if (layer.kind == .global)
                std.mem.alignForward(u32, context, 64)
            else
                std.mem.alignForward(u32, c.window + rows - 1, 64);
            const kv = try std.math.mul(u64, tokens, 2 * @as(u64, layer.heads.kv * layer.heads.dim * 2));
            // The recurrence rounds each site then widens its complete state to FP32 before the FMAs.
            const conv = try std.math.mul(u64, conv_values, 4);
            out.slots[i] = .{ .kv_offset = out.bytes, .kv_bytes = kv, .conv_offset = 0, .conv_bytes = conv };
            out.bytes = try std.math.add(u64, out.bytes, kv);
        }
        var conv_offset = out.bytes;
        for (out.slots[0..out.n]) |*slot| {
            slot.conv_offset = conv_offset;
            conv_offset = try std.math.add(u64, conv_offset, slot.conv_bytes);
        }
        out.bytes = conv_offset;
        return out;
    }

    pub fn at(l: *const CacheLayout, visit: u32) SnapshotSlot {
        return l.slots[visit];
    }
};

/// The serial lane adapter's admission rule. One request occupies one complete 48-visit stream.
pub const Admission = struct {
    context: u32,

    pub fn fits(a: Admission, prompt: u32, max_tokens: u32) bool {
        return prompt <= a.context and max_tokens <= a.context - prompt;
    }
};

/// A prompt-cache save or restore copies every byte of this layout, including all four convolution recurrences.
pub fn snapshotBytes(l: *const CacheLayout) u64 {
    return l.bytes;
}

pub const Snapshot = struct {
    bytes: []u8,

    pub fn save(gpa: std.mem.Allocator, state: []const u8, l: *const CacheLayout) !Snapshot {
        if (state.len != l.bytes) return error.BadLoopSnapshot;
        return .{ .bytes = try gpa.dupe(u8, state) };
    }

    pub fn restore(s: *const Snapshot, state: []u8, l: *const CacheLayout) !void {
        if (s.bytes.len != l.bytes or state.len != l.bytes) return error.BadLoopSnapshot;
        @memcpy(state, s.bytes);
    }

    pub fn deinit(s: *Snapshot, gpa: std.mem.Allocator) void {
        gpa.free(s.bytes);
        s.* = undefined;
    }
};

test "a saved prompt restores every visit byte and owns its copy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    var prec: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &prec);
    const layout = try CacheLayout.init(&c.text, &p, 1);
    const state = try std.testing.allocator.alloc(u8, @intCast(layout.bytes));
    defer std.testing.allocator.free(state);
    for (state, 0..) |*v, i| v.* = @truncate(i * 29 + 7);
    var saved = try Snapshot.save(std.testing.allocator, state, &layout);
    defer saved.deinit(std.testing.allocator);
    @memset(state, 0xa5);
    try saved.restore(state, &layout);
    try std.testing.expectEqualSlices(u8, saved.bytes, state);
    try std.testing.expectError(error.BadLoopSnapshot, Snapshot.save(std.testing.allocator, state[0 .. state.len - 1], &layout));
}

test "Mini has 48 independent KV and convolution slots and a whole-state snapshot" {
    const config_test = @import("config_test.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.mini, &why);
    var prec: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &prec);
    const layout = try CacheLayout.init(&c.text, &p, 1_048_576);
    try std.testing.expectEqual(@as(u32, 48), layout.n);
    try std.testing.expectEqual(@as(u64, 3072 * 2 * 8 * 128 * 2), layout.at(0).kv_bytes);
    try std.testing.expectEqual(@as(u64, 1_048_576 * 2 * 8 * 128 * 2), layout.at(5).kv_bytes);
    try std.testing.expectEqual(@as(u64, 34_866_888_704), layout.bytes);
    try std.testing.expectEqual(layout.bytes, snapshotBytes(&layout));
    try std.testing.expect((Admission{ .context = 8192 }).fits(8000, 192));
    try std.testing.expectEqual(@as(u64, 79_872), layout.at(0).conv_bytes);
}

test "Mini loop load is one-rank text-only with rank-64 gated controllers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    try validateLoad(&c, 1, false);
    try std.testing.expectError(error.Inkling2LoopOneRankOnly, validateLoad(&c, 2, false));
    try std.testing.expectError(error.Inkling2LoopTextOnly, validateLoad(&c, 1, true));
}

test "pass two resets and only completed passes with a successor capture message banks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    var prec: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &prec);
    const lp = c.text.loop.?;
    var banks: u32 = 0;
    for (p.visits[0..p.n_visits]) |visit| {
        if (visit.pass == 2 and visit.layer == lp.core_start) try std.testing.expect(visit.hasMessages());
        if (visit.pass > 0 and visit.pass < lp.passes and visit.layer + 1 == lp.core_end) banks += 1;
    }
    try std.testing.expectEqual(lp.passes - 1, banks);
}
