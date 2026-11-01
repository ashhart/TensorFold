//! Initial Nemotron ownership and later allocator callbacks share the same nested buffer inventory.
const std = @import("std");
const mtl = @import("metal");
const fields = mtl.resident.fields;

pub fn openCache(b: anytype, stream: *@import("lanes").Stream, capacity: usize) !*@import("state.zig").Cache {
    const state = @import("state.zig");
    const cache = try b.gpa.create(state.Cache);
    errdefer b.gpa.destroy(cache);
    const lease = try leasePages(b, capacity);
    errdefer if (lease) |l| l.pages.finish(l.lane);
    cache.* = try state.Cache.initPaged(b.m.device, b.m.config, capacity, b.head != null, lease);
    cache.setResident(b.residency);
    errdefer cache.deinit(&b.pool);
    cache.rid = try b.pool.takeRid();
    cache.decide = if (stream.decide != null and @import("layers.zig").firstAdapter(&b.m.weights) != null)
        stream.decide.?
    else
        @import("living").lesson_route.none;
    cache.untaught = stream.untaught;
    try b.caches.put(b.gpa, stream, cache);
    return cache;
}

/// TF_NEMOTRON_METAL_KV_PAGES: a free lane promised and holding the pages of `capacity` rows; LaneMemory: none now.
fn leasePages(b: anytype, capacity: usize) !?@import("state.zig").Lease {
    const pages = b.kv_pages orelse return null;
    const lane = for (pages.leases, 0..) |l, i| {
        if (l.promised == 0 and l.ids.items.len == 0) break i;
    } else return error.LaneMemory;
    try pages.admit(lane, pages.layout.pages(capacity));
    errdefer pages.finish(lane);
    try pages.cover(lane, capacity); // every row's page now: a GPU round writes rows the host does not see first
    return .{ .pages = pages, .lane = lane };
}

fn apply(b: anytype, hook: mtl.ResidencySet.Follow, joining: bool) void {
    fields.apply(b, hook, joining);
    fields.apply(&b.scratch, hook, joining);
    if (b.wide) |*w| {
        fields.apply(w, hook, joining);
        fields.apply(&w.s, hook, joining);
    }
    if (b.head) |*h| {
        fields.apply(h, hook, joining);
        fields.apply(&h.scratch, hook, joining);
    }
    if (b.masks) |*m| fields.apply(m, hook, joining);
    if (b.o.block) |*block| fields.apply(block, hook, joining);
}

pub fn set(b: anytype, hook: ?mtl.ResidencySet.Follow) void {
    if (b.residency) |f| apply(b, f, false);
    b.pool.setResident(hook);
    var it = b.caches.valueIterator();
    while (it.next()) |cache| cache.*.setResident(hook);
    if (b.living) |*s| s.setResident(hook);
    b.residency = hook;
    if (hook) |f| apply(b, f, true);
}

pub fn append(b: anytype, gpa: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
    try fields.append(gpa, list, b);
    try fields.append(gpa, list, &b.scratch);
    try b.pool.appendBuffers(gpa, list);
    if (b.wide) |*w| {
        try fields.append(gpa, list, w);
        try fields.append(gpa, list, &w.s);
    }
    if (b.head) |*h| {
        try fields.append(gpa, list, h);
        try fields.append(gpa, list, &h.scratch);
    }
    if (b.masks) |*m| try fields.append(gpa, list, m);
    if (b.o.block) |*block| try fields.append(gpa, list, block);
    if (b.living) |*s| try s.appendBuffers(gpa, list);
    var it = b.caches.valueIterator();
    while (it.next()) |cache| try cache.*.appendBuffers(gpa, list);
}

pub fn hold(h: anytype, io: std.Io) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    for (h.m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    try list.appendSlice(h.gpa, h.m.weights.owned.items);
    try append(h.metal, h.gpa, &list);
    h.resident = try mtl.resident.Resident.hold(h.gpa, io, h.m.device, h.m.queue, list.items, "nemotron");
    if (h.resident) |*r| set(h.metal, r.follow());
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = h.m.queue };
}

pub fn drop(h: anytype) void {
    set(h.metal, null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.m.queue };
}

pub fn trainer(t: anytype, hook: ?mtl.ResidencySet.Follow) void {
    if (t.residency) |f| {
        fields.apply(t, f, false);
        fields.apply(&t.tmp, f, false);
        fields.apply(&t.bufs, f, false);
    }
    t.sites.setResident(hook);
    t.cache.setResident(hook);
    t.prefixes.setResident(hook);
    t.residency = hook;
    if (hook) |f| {
        fields.apply(t, f, true);
        fields.apply(&t.tmp, f, true);
        fields.apply(&t.bufs, f, true);
    }
}
