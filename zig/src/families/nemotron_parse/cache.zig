//! Owned exact KV/hidden snapshots for the shared prompt cache, keyed by the server's image-aware token identity.
const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const lanes = @import("lanes");
const c = @import("config.zig");
const Backend = @import("backend.zig").Backend;
const State = @import("decoder.zig").State;
const engine = @import("engine.zig");
const pc = api.prompt_cache;

pub fn snapshots(h: *Backend) pc.Snapshots {
    return .{ .ptr = h, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
}

fn of(ptr: *anyopaque) *Backend {
    return Backend.of(ptr);
}

fn bytes(_: *anyopaque, at: u32) u64 {
    return engine.stateBytes(at);
}

fn lane(h: *Backend, owner: ?*anyopaque) !*@import("backend.zig").Lane {
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoCacheOwner));
    return h.streams.get(s);
}

fn copy(h: *Backend, x: *const State, y: *State, n: usize) !void {
    if (n == 0 or n > x.len or n > y.total) return error.BadSnapshot;
    try hip.raw.copy(y.hidden, 0, x.hidden.base(), n * c.hidden * 2, h.e.stream.handle);
    for (x.layers, y.layers) |src, dst| for (0..c.kv_heads) |head| {
        const xo = head * x.total * c.head_dim * 2;
        const yo = head * y.total * c.head_dim * 2;
        try hip.raw.copy(dst.k, yo, src.k.base() + xo, n * c.head_dim * 2, h.e.stream.handle);
        try hip.raw.copy(dst.v, yo, src.v.base() + xo, n * c.head_dim * 2, h.e.stream.handle);
    };
    try h.e.stream.synchronize();
    y.len = n;
}

fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
    const h = of(ptr);
    const l = try lane(h, owner);
    if (at == 0 or at > l.state.len) return error.BadSnapshot;
    const pool = try hip.admission.budget(h.a, h.e.io, h.e.ctx);
    if (pool.room(hip.usage(false).device) < engine.stateBytes(at)) return error.LaneMemory;
    const saved = try h.a.create(State);
    errdefer h.a.destroy(saved);
    saved.* = try State.init(&h.e.r, at);
    errdefer saved.deinit();
    try copy(h, &l.state, saved, at);
    return saved;
}

pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
    const h = of(ptr);
    const x: *State = @ptrCast(@alignCast(saved));
    const l = try lane(h, owner);
    try copy(h, x, &l.state, x.len);
    l.fill = x.len;
}

fn drop(ptr: *anyopaque, saved: pc.Saved) void {
    const h = of(ptr);
    const x: *State = @ptrCast(@alignCast(saved));
    h.e.stream.synchronize() catch {};
    x.deinit();
    h.a.destroy(x);
}
