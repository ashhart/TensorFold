//! Exact in-memory K3 prompt states include recurrent replay logs, convolution history, MLA keys and last logits.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("config.zig").Config;
const State = @import("state.zig");
const A = std.mem.Allocator;

pub const Snap = struct {
    owner: *anyopaque,
    position: State.Stream,
    data: []u8,
    tokens: []u32,
    logits: []u16,

    pub fn deinit(s: *Snap, a: A) void {
        a.free(s.data);
        a.free(s.tokens);
        a.free(s.logits);
        a.destroy(s);
    }
    pub fn charge(s: *const Snap) u64 {
        return @sizeOf(Snap) + s.data.len + s.tokens.len * 4 + s.logits.len * 2;
    }
};
const Part = struct { buffer: mtl.Buffer, offset: usize, len: usize };

pub fn bound(c: *const Config, at: u32, window: u32) u64 {
    const w = @as(u64, c.kda_heads) * c.kda_dim;
    const recurrent = w * c.kda_dim * 4 + @as(u64, c.conv - 1) * 3 * w * 2 + window * w * (3 * 4 + 3 * 2);
    const mla = @as(u64, at) * (c.kv_lora + c.rope) * 2;
    return @sizeOf(Snap) + @as(u64, at) * 4 + @as(u64, c.vocab) * 2 + recurrent * c.countKind(.kda) + mla * c.countKind(.mla);
}

fn parts(a: A, m: anytype, slot: u32, position: State.Stream) ![]Part {
    if (position.pos > m.state.limits.max_ctx or position.kept > m.state.limits.log_rows) return error.KimiSnapshotPosition;
    var out: std.ArrayList(Part) = .empty;
    errdefer out.deinit(a);
    const c = m.graph.c;
    const w: usize = @as(usize, c.kda_heads) * c.kda_dim;
    for (m.state.kda, m.state.mla) |maybe, mla| {
        if (maybe) |k| {
            const sizes = .{
                .{ "S", w * c.kda_dim * 4, w * c.kda_dim * 4 },
                .{ "conv", (c.conv - 1) * 3 * w * 2, (c.conv - 1) * 3 * w * 2 },
                .{ "log_a", m.state.limits.log_rows * w * 4, position.kept * w * 4 },
                .{ "log_k", m.state.limits.log_rows * w * 4, position.kept * w * 4 },
                .{ "log_u", m.state.limits.log_rows * w * 4, position.kept * w * 4 },
                .{ "log_x", m.state.limits.log_rows * 3 * w * 2, position.kept * 3 * w * 2 },
            };
            inline for (sizes) |entry| try out.append(a, .{ .buffer = @field(k, entry[0]), .offset = slot * entry[1], .len = entry[2] });
        } else if (mla) |buffer| {
            const row = (c.kv_lora + c.rope) * 2;
            try out.append(a, .{ .buffer = buffer, .offset = @as(usize, slot) * m.state.limits.max_ctx * row, .len = @as(usize, position.pos) * row });
        }
    }
    return out.toOwnedSlice(a);
}

/// Forward completion is synchronous; CPU copies read only shared buffers after their command has completed.
pub fn save(m: anytype, slot: u32, tokens: []const u32) !*Snap {
    const a = m.a;
    const position = m.state.streams[slot];
    if (!m.healthy.load(.acquire) or tokens.len != position.pos) return error.KimiSnapshotPosition;
    const list = try parts(a, m, slot, position);
    defer a.free(list);
    var size: usize = 0;
    for (list) |p| size += p.len;
    const s = try a.create(Snap);
    errdefer a.destroy(s);
    const data = try a.alloc(u8, size);
    errdefer a.free(data);
    const ids = try a.dupe(u32, tokens);
    errdefer a.free(ids);
    const logits = try a.dupe(u16, m.logits(slot));
    var at: usize = 0;
    for (list) |p| {
        @memcpy(data[at..][0..p.len], p.buffer.slice(u8, p.offset + p.len)[p.offset..]);
        at += p.len;
    }
    s.* = .{ .owner = m, .position = position, .data = data, .tokens = ids, .logits = logits };
    return s;
}

pub fn restore(m: anytype, slot: u32, s: *const Snap, prompt: []const u32, at: u32) !void {
    if (!m.healthy.load(.acquire)) return error.KimiEngineFailed;
    // A prompt-cache restore must obey max_ctx before any cache buffer or last-logit copy.
    if (at > m.state.limits.max_ctx or s.position.pos > m.state.limits.max_ctx) return error.KimiSnapshotCapacity;
    try m.graph.c.checkEnd(0, s.position.pos);
    if (s.owner != @as(*anyopaque, @ptrCast(m)) or at != s.position.pos or at > prompt.len or
        !std.mem.eql(u32, prompt[0..at], s.tokens) or s.logits.len != m.graph.c.vocab) return error.KimiSnapshotPrefix;
    const list = try parts(m.a, m, slot, s.position);
    defer m.a.free(list);
    var size: usize = 0;
    for (list) |p| size += p.len;
    if (size != s.data.len) return error.KimiSnapshotShape;
    var offset: usize = 0;
    for (list) |p| {
        @memcpy(p.buffer.slice(u8, p.offset + p.len)[p.offset..], s.data[offset..][0..p.len]);
        offset += p.len;
    }
    @memcpy(m.logits(slot), s.logits);
    m.state.streams[slot] = s.position;
}

test "Kimi prompt-cache restore checks max_ctx and the K3.1 ceiling before copying" {
    const Graph = struct { c: Config = .{} };
    const Model = struct {
        a: A = std.testing.allocator,
        healthy: std.atomic.Value(bool) = .init(true),
        graph: *Graph,
        state: State.State = undefined,
        pub fn logits(_: *@This(), _: u32) []u16 {
            unreachable;
        }
    };
    var graph: Graph = .{};
    var m: Model = .{ .graph = &graph };
    m.state.limits = .{ .slots = 1, .max_ctx = 32 };
    var saved: Snap = .{ .owner = &m, .position = .{ .pos = 33 }, .data = &.{}, .tokens = &.{}, .logits = &.{} };
    try std.testing.expectError(error.KimiSnapshotCapacity, restore(&m, 0, &saved, &.{}, 33));
    m.state.limits.max_ctx = 4096;
    graph.c.dsa = .{};
    saved.position.pos = 2048;
    try std.testing.expectError(error.K31ContextLimit, restore(&m, 0, &saved, &.{}, 2048));
}

test "Kimi snapshot bound includes recurrent logs even at the last prompt row" {
    const c: Config = .{};
    const base = bound(&c, 2047, 0);
    const logged = bound(&c, 2047, 16);
    const w: u64 = c.kdaWidth();
    try std.testing.expectEqual(@as(u64, c.countKind(.kda)) * 16 * w * 18, logged - base);
    try std.testing.expectEqual(@as(u64, c.countKind(.mla)) * (c.kv_lora + c.rope) * 2 + 4, bound(&c, 2047, 16) - bound(&c, 2046, 16));
}
