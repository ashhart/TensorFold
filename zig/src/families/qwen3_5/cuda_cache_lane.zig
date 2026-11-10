//! A pool lane's cache on pieces (cuda_state_pieces.zig): mapped at open, zeroed, rolled back and copied by piece.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const geometry = @import("cuda_geometry.zig");
const caches = @import("cuda_cache.zig");
const sp = @import("cuda_state_pieces.zig");
const States = @import("cuda_states.zig").States;

const Engine = engine.Engine;
const Cache = caches.Cache;
const DeviceBuffer = cuda.DeviceBuffer;

/// A lane's state pages: page i at `base` + ids[i] pages (its lease's state ids, fixed until it closes).
pub const Pieces = struct { base: u64, ids: []const u32, page: usize };

/// A lane's bytes outside the pool: its tap ring and a drafter's ring, each wider than a page (#2444, option a).
pub fn outsideBytes(e: *const Engine, ring: usize) usize {
    return std.mem.alignForward(usize, e.tapBytes(), 256) + ring;
}

/// The lane's state on its state pages `ids` (the state table written and uploaded), its rings in `outside`, zeroed.
pub fn map(self: *Cache, e: *const Engine, m: *const sp.Map, base: u64, ids: []const u32, page: usize, outside: ?DeviceBuffer, ring: usize) !void {
    const t = &(self.states orelse return error.MissingQwenStates);
    m.fill(t.host, e.g, base, ids);
    try t.dev.uploadAsync(0, std.mem.sliceAsBytes(t.host), e.stream.handle);
    t.from = null;
    t.version = States.next();
    self.pieces = .{ .base = base, .ids = ids, .page = page };
    self.on_table = true;
    self.pend = 0;
    const o = outside orelse return;
    self.tap = if (e.n_taps > 0) .{ .d = e.d, .ptr = o.ptr, .len = e.tapBytes(), .borrowed = true } else null;
    self.drafter_ring = if (ring > 0) o.ptr + std.mem.alignForward(usize, e.tapBytes(), 256) else null;
    try o.fill8(0, e.stream.handle);
}

/// A closed lane's pages dropped: they went back to the pool; its table is refused until the next map (version 0).
pub fn unmap(self: *Cache, e: *const Engine) void {
    _ = e;
    self.pieces = null;
    self.on_table = false;
    self.pend = 0;
    self.tap = null;
    self.drafter_ring = null;
    if (self.states) |*t| t.version = 0;
}

/// Every state page zeroed: the recurrences and tails, and the log and copy beside them (as a run's open did).
pub fn zero(x: Pieces, e: *const Engine) !void {
    for (x.ids) |id| {
        const v: DeviceBuffer = .{ .d = e.d, .ptr = x.base + @as(u64, id) * x.page, .len = x.page, .borrowed = true };
        try v.fill8(0, e.stream.handle);
    }
}

/// A part of the state: its device bytes and where they sit in a one-block state (recurrences, then tails).
pub const Part = struct { dev: u64, at: usize, len: usize };

/// Parts the state is in: one block, or on pieces a recurrence and a tail a linear layer.
pub fn parts(self: *const Cache, g: geometry.Geometry) usize {
    return if (self.pieces != null) 2 * g.deltas() else 1;
}

/// Part `i` of `parts`; `copy`: the rollback copy's (pieces only).
pub fn part(self: *const Cache, g: geometry.Geometry, i: usize, copy: bool) Part {
    const nl = g.deltas();
    if (self.pieces == null) return .{ .dev = self.state.ptr, .at = 0, .len = g.stateBytes() };
    const host = self.states.?.host;
    const dev = host[i + if (copy) 2 * nl else 0];
    return if (i < nl) .{ .dev = dev, .at = i * g.recBytes(), .len = g.recBytes() } else .{ .dev = dev, .at = nl * g.recBytes() + (i - nl) * g.convBytes(), .len = g.convBytes() };
}

/// A one-row window not kept: the state back from its copy, a piece at a time.
pub fn rollBack(self: *Cache, e: *const Engine) !void {
    for (0..parts(self, e.g)) |i| {
        const to = part(self, e.g, i, false);
        const v: DeviceBuffer = .{ .d = e.d, .ptr = to.dev, .len = to.len, .borrowed = true };
        try v.copyFrom(0, part(self, e.g, i, true).dev, to.len, e.stream.handle);
    }
}

/// The state to or from `host` in a one-block state's layout (stateBytes), on the engine's stream.
pub fn copyState(self: *const Cache, e: *const Engine, host: []u8, to_device: bool) !void {
    if (host.len < e.g.stateBytes()) return error.QwenStateCopyShort;
    for (0..parts(self, e.g)) |i| {
        const p = part(self, e.g, i, false);
        if (to_device)
            try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(p.dev, host.ptr + p.at, p.len, e.stream.handle), "cuMemcpyHtoDAsync")
        else
            try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(host.ptr + p.at, p.dev, p.len, e.stream.handle), "cuMemcpyDtoHAsync");
    }
}
