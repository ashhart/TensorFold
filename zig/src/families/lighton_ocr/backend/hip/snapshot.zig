//! Exact pinned-host copies of a HIP stream's positional KV prefix, independent of the stream's reusable slot.
const std = @import("std");
const hip = @import("hip");
const cfg = @import("../../config.zig");
const Allocator = std.mem.Allocator;

pub const Saved = struct { at: u32, layers: u32, kv_width: u32, host: hip.HostBuffer };

pub fn bytes(c: cfg.Config, at: u32) u64 {
    return @as(u64, c.layers) * 2 * at * c.kvWidth() * 2;
}

/// Release GPU dependencies only after the stream or device proves completion.
pub fn drain(s: hip.Stream) void {
    if (s.synchronize()) |_| return else |_| {}
    if (hip.runtime.check(s.r.api.hipDeviceSynchronize())) |_| return else |_| {}
    std.log.err("LightOnOCR HIP work did not drain; live resources cannot be freed", .{});
    std.c._exit(1);
}

pub fn save(gpa: Allocator, m: anytype, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context or slot >= m.slots) return error.NotAtMark;
    const kept = try gpa.create(Saved);
    errdefer gpa.destroy(kept);
    kept.* = .{ .at = at, .layers = m.c.layers, .kv_width = m.c.kvWidth(), .host = try hip.HostBuffer.alloc(m.s.r, @intCast(bytes(m.c, at))) };
    errdefer kept.host.free();
    errdefer drain(m.s);
    try each(m, slot, at, Copy{ .stream = m.s, .host = kept.host.bytes, .out = true });
    try m.s.synchronize();
    return kept;
}

pub fn restore(m: anytype, slot: usize, kept: *const Saved) !void {
    if (kept.at == 0 or kept.at > m.context or slot >= m.slots or kept.layers != m.c.layers or kept.kv_width != m.c.kvWidth() or
        kept.host.bytes.len != bytes(m.c, kept.at)) return error.BadReuse;
    errdefer drain(m.s);
    try each(m, slot, kept.at, Copy{ .stream = m.s, .host = kept.host.bytes, .out = false });
    try m.s.synchronize();
}

pub fn drop(gpa: Allocator, s: hip.Stream, kept: *Saved) void {
    drain(s);
    kept.host.free();
    gpa.destroy(kept);
}

/// Head-major caches keep each head's prefix apart; copying one contiguous at*kv_width span would include stale rows.
fn each(m: anytype, slot: usize, at: u32, sink: anytype) !void {
    if (at == 0 or at > m.context or slot >= m.slots) return error.NotAtMark;
    const count: usize = @as(usize, at) * m.c.head_dim * 2;
    var offset: usize = 0;
    for (0..m.c.layers) |layer| for ([_]u64{ m.key(slot, layer), m.value(slot, layer) }) |dev| {
        for (0..m.c.kv_heads) |head| {
            try sink.copy(dev + head * m.context * m.c.head_dim * 2, offset, count);
            offset += count;
        }
    };
}

const Copy = struct {
    stream: hip.Stream,
    host: []u8,
    out: bool,

    fn copy(c: Copy, device: u64, offset: usize, count: usize) !void {
        if (offset > c.host.len or count > c.host.len - offset) return error.BadReuse;
        if (c.out) {
            try hip.runtime.check(c.stream.r.api.hipMemcpyDtoHAsync(c.host.ptr + offset, @ptrFromInt(device), count, c.stream.handle));
        } else {
            try hip.runtime.check(c.stream.r.api.hipMemcpyHtoDAsync(@ptrFromInt(device), c.host.ptr + offset, count, c.stream.handle));
        }
    }
};

test "LightOnOCR HIP snapshot bytes include every layer key and value" {
    const c = try cfg.Config.parse(std.testing.allocator, cfg.sample);
    try std.testing.expectEqual(@as(u64, 28 * 2 * 128 * 1024 * 2), bytes(c, 128));
    try std.testing.expectEqual(@as(u64, 0), bytes(c, 0));
}

test "LightOnOCR HIP snapshots skip uncommitted rows between head prefixes" {
    const Model = struct {
        c: cfg.Config,
        context: usize = 8208,
        slots: usize = 2,

        fn key(m: @This(), slot: usize, layer: usize) u64 {
            return 4096 + (layer * 2 * m.slots + slot) * m.context * m.c.kvWidth() * 2;
        }

        fn value(m: @This(), slot: usize, layer: usize) u64 {
            return m.key(slot, layer) + m.slots * m.context * m.c.kvWidth() * 2;
        }
    };
    const Recorder = struct {
        runs: *std.ArrayList([3]u64),

        fn copy(c: @This(), device: u64, offset: usize, count: usize) !void {
            try c.runs.append(std.testing.allocator, .{ device, offset, count });
        }
    };
    const c = try cfg.Config.parse(std.testing.allocator, cfg.sample);
    const model: Model = .{ .c = c };
    var runs: std.ArrayList([3]u64) = .empty;
    defer runs.deinit(std.testing.allocator);
    try each(model, 1, 256, Recorder{ .runs = &runs });
    try std.testing.expectEqual(@as(usize, 28 * 2 * 8), runs.items.len);
    const head_bytes = 256 * 128 * 2;
    try std.testing.expectEqual([3]u64{ model.key(1, 0), 0, head_bytes }, runs.items[0]);
    try std.testing.expectEqual([3]u64{ model.key(1, 0) + 8208 * 128 * 2, head_bytes, head_bytes }, runs.items[1]);
    try std.testing.expectEqual([3]u64{ model.value(1, 0), 8 * head_bytes, head_bytes }, runs.items[8]);
    const last = runs.items[runs.items.len - 1];
    try std.testing.expectEqual(model.value(1, 27) + 7 * 8208 * 128 * 2, last[0]);
    try std.testing.expectEqual(bytes(c, 256), last[1] + last[2]);
    try std.testing.expectError(error.NotAtMark, each(model, 2, 256, Recorder{ .runs = &runs }));
}
