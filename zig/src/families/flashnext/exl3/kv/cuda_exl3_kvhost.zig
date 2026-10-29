//! The host tier of the Flash Next pool (TF_FLASHNEXT_KV_HOST=MiB): canonical K/V in pinned host RAM, cached in frames.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const pages = @import("../../cuda_pages.zig");

/// TF_FLASHNEXT_KV_HOST=n: n MiB of pinned host RAM hold the K/V plane of every logical page (unset or 0: no tier).
pub fn hostMiB() !usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_HOST") orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextKvHost;
}

/// Whether the host tier is on (the page layout then keeps two planes).
pub fn tiered() bool {
    return (hostMiB() catch 0) > 0;
}

/// TF_FLASHNEXT_KV_FRAMES=n: at most n device frames of the K/V plane (unset: the pages the plan gave).
pub fn framesCap() !?usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_FRAMES") orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextKvHost;
}

/// The tier of one pool: logical pages, frames, the pinned host store they are canonical in.
pub const Host = struct {
    tier: kv.tier.Tier,
    store: cuda.HostBuffer,
    kv_page: usize, // bytes of one page's K/V plane

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, logical: usize, frames: usize, kv_page: usize) !*Host {
        const h = try gpa.create(Host);
        errdefer gpa.destroy(h);
        var store = try cuda.HostBuffer.allocMapped(d, try std.math.mul(usize, logical, kv_page));
        errdefer store.free();
        h.* = .{ .tier = undefined, .store = store, .kv_page = kv_page };
        try h.tier.init(gpa, store.bytes, frames, kv_page);
        return h;
    }

    /// The ranks exchange the tier's shape, slot count and digest at open; one that differs refuses.
    pub fn lockstep(h: *const Host, e: anytype, slots: usize) !void {
        const c = e.collective orelse return;
        const both = try exchange(e, c, kv.tier.Lockstep.vector(&h.tier, slots));
        kv.tier.Lockstep.agree(both[0], both[1]) catch return error.FlashNextTierRanksDisagree;
    }

    pub fn destroy(h: *Host, gpa: std.mem.Allocator) void {
        h.tier.deinit();
        h.store.free();
        gpa.destroy(h);
    }

    /// Page `id`'s K/V plane from its frame (in `arena`) to the host store; sync the stream, then call `done`.
    pub fn flush(h: *Host, arena: cuda.DeviceBuffer, id: u32, stream: cuda.Stream) !void {
        const tk = h.tier.resident(id) orelse return;
        try arena.downloadAsync(@as(usize, tk.frame) * h.kv_page, h.tier.host_page(id), stream.handle);
    }

    pub fn done(h: *Host, id: u32) void {
        h.tier.sync(id);
    }
};

/// Each rank's `mine`, side by side, from one sum across the ranks.
fn exchange(e: anytype, c: anytype, mine: [kv.tier.Lockstep.len]f32) ![2][kv.tier.Lockstep.len]f32 {
    const n = kv.tier.Lockstep.len;
    const st = e.stream.*;
    var b = try cuda.DeviceBuffer.alloc(e.driver, 256);
    defer b.free();
    var v: [2][n]f32 = .{ @splat(0), @splat(0) };
    v[e.layers[0].exl3.?.tp.?.rank] = mine;
    try st.synchronize();
    try b.upload(0, std.mem.asBytes(&v));
    try c.sum(st, b.ptr, 2 * n);
    try c.finish(st);
    try b.download(0, std.mem.asBytes(&v));
    return v;
}

/// The pages and frames both ranks can hold: the smaller of each (rank 1 has no MTP head, so its pages are smaller).
pub fn agreeShape(e: anytype, logical: usize, frames: usize) !kv.tier.Shape {
    const mine: kv.tier.Shape = .{ .host_pages = logical, .frames = frames };
    const c = e.collective orelse return mine;
    var m: [kv.tier.Lockstep.len]f32 = @splat(0);
    m[0] = @floatFromInt(logical);
    m[1] = @floatFromInt(frames);
    const both = try exchange(e, c, m);
    const a: kv.tier.Shape = .{ .host_pages = @intFromFloat(both[0][0]), .frames = @intFromFloat(both[0][1]) };
    const b: kv.tier.Shape = .{ .host_pages = @intFromFloat(both[1][0]), .frames = @intFromFloat(both[1][1]) };
    return a.agree(b);
}
