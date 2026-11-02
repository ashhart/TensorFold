//! Gemma's paged leases and prompt store; snapshots hold sliding tails in pages of the same pool.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const fwd = @import("cuda_forward.zig");
const snap = @import("cuda_snap.zig");
const Stream = lanes.Stream;
const max_window = @import("cuda_backend.zig").max_window;

pub fn pagesFor(prompt: usize, max_new: usize) usize {
    return pagesOf(prompt + max_new + max_window);
}

fn pagesOf(n: usize) usize {
    return (n + fwd.pages.tokens - 1) / fwd.pages.tokens;
}

pub fn attach(x: anytype, extra: usize, count: usize) !void {
    try x.s.synchronize();
    if (x.relocated) |r| r.call(r.ctx);
    try fwd.pages.attach(x.m, x.s, extra, count);
    x.pool = try core.kv_pool.Pool.init(x.gpa, count, fwd.pages.pageBytes(x.m));
}

pub fn admits(x: anytype, prompt: usize, max_new: usize) bool {
    if (prompt + max_new + max_window > x.m.context) return true;
    if (std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true) == null) return false;
    return if (x.pool) |*p| p.fits(pagesFor(prompt, max_new)) else true;
}

pub fn reach(x: anytype, l: anytype, end: usize) !void {
    const p = if (x.pool) |*p| p else return;
    const n = pagesOf(end);
    if (n <= l.lease.ids.items.len) return;
    const Map = struct {
        x: @TypeOf(x),
        slot: usize,
        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const m: *@This() = @ptrCast(@alignCast(ctx));
            try fwd.pages.map(m.x.m, m.x.s, m.slot, ids[from..], from);
        }
    };
    var map: Map = .{ .x = x, .slot = l.slot };
    try p.cover(&l.lease, n, .{ .ctx = &map, .write = Map.write });
}

/// Restore rings while begin protects its found state, before a promise can reclaim the snapshot.
pub fn restore(x: anytype, k: *const snap.Saved) !void {
    const slot = x.restore_slot orelse return error.NoOwner;
    if (k.from != k.at or k.at == 0) return error.KeptStateMismatch;
    try snap.restore(x.m, x.s, slot, k);
    x.restored_at = k.at;
}

pub fn save(x: anytype, slot: usize, at: u32) !*snap.Saved {
    const p = if (x.pool) |*p| p else return error.NoPool;
    const ids = try x.gpa.alloc(u32, snap.blobPages(x.m, at, at));
    if (!p.hold(ids)) {
        x.gpa.free(ids);
        return error.NoCachePages;
    }
    return snap.saveShared(x.gpa, x.m, x.s, slot, at, at, ids) catch |err| {
        cuda.completion.drain(x.d, x.s, "Gemma shared snapshot save failed before pool pages could be returned");
        for (ids) |id| p.release(id);
        x.gpa.free(ids);
        return err;
    };
}

pub fn drop(x: anytype, k: *snap.Saved) void {
    if (k.blob.len > 0) {
        cuda.completion.drain(x.d, x.s, "Gemma shared snapshot drop did not drain before releasing its pool pages");
        for (k.blob) |id| x.pool.?.release(id);
        x.gpa.free(k.blob);
        k.blob = &.{};
    }
    snap.drop(x.gpa, x.s, k);
}

pub fn opening(comptime Lane: type, x: anytype, s: *Stream) !*Lane {
    const prompt = s.prompt();
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len + s.max_new + max_window > x.m.context) return error.PastContext;
    if (x.lanes.contains(s)) return error.StreamAlreadyOpen;
    const slot = std.mem.indexOfScalar(bool, x.free[0..x.m.slots], true) orelse return error.NoFreeSlot;
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(x.gpa);
    x.restore_slot = slot;
    defer x.restore_slot = null;
    x.restored_at = 0;
    var adopted = false;
    errdefer if (!adopted) for (adopt.items) |id| x.pool.?.release(id);
    const pr = if (s.images.len == 0) x.prompts else null;
    const plan: core.kv_prompts.Plan = if (pr) |p| try p.begin(prompt, s.history_len, s.shared_prefixes, s, &adopt) else .{};
    var l: Lane = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr, .marks = plan.marks, .shared = pr != null };
    errdefer {
        x.m.caps[slot] = 0;
        x.pool.?.finish(&l.lease);
        x.gpa.free(l.marks);
        if (l.images) |*b| {
            x.s.synchronize() catch {};
            b.free();
        }
    }
    try x.pool.?.promise(&l.lease, pagesFor(prompt.len, s.max_new) - adopt.items.len);
    adopted = true; // adopt releases its input references itself on failure
    try x.pool.?.adopt(&l.lease, adopt.items);
    if (adopt.items.len > 0) try fwd.pages.map(x.m, x.s, slot, adopt.items, 0);
    try reach(x, &l, prompt.len);
    s.cached = 0;
    if (pr != null) {
        if (plan.from != x.restored_at) return error.KeptStateMismatch;
        s.cached = plan.from;
        l.fill = plan.from;
    } else l.fill = x.resumeAt(s, &l, prompt.len);
    if (s.sampling) |sp| if (sp.temperature > 0) {
        const rule = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
        try x.upload(l.rule, std.mem.asBytes(&rule));
        l.sampled = true;
    };
    if (s.images.len > 0) try x.encodeImages(s, &l);
    try x.lanes.put(x.gpa, s, l);
    x.free[slot] = false;
    s.fill_at.store(@intCast(l.fill.?), .monotonic);
    return x.lanes.getPtr(s).?;
}

pub fn marked(x: anytype, s: *Stream, l: anytype, at: usize) !void {
    const pr = x.prompts orelse return;
    if (!l.shared) return;
    if (std.mem.indexOfScalar(u32, l.marks, @intCast(at)) != null) {
        const n = at / fwd.pages.tokens;
        const path = try x.gpa.alloc(u32, n);
        defer x.gpa.free(path);
        const mine = try x.gpa.dupe(u32, l.lease.ids.items[0..n]);
        defer x.gpa.free(mine);
        const shared = pr.keep(s.prompt(), @intCast(at), s, mine, path);
        if (shared == n) l.kept += 1;
        if (!std.mem.eql(u32, mine[0..shared], path[0..shared])) {
            try fwd.pages.map(x.m, x.s, l.slot, path[0..shared], 0);
            for (path[0..shared], 0..) |id, i| x.pool.?.share(&l.lease, i, id);
            // Remapping a prefix must not shorten the already covered writable suffix.
            x.m.caps[l.slot] = l.lease.ids.items.len * fwd.pages.tokens;
        }
    }
    if (at == s.prompt().len) pr.report(at, s.cached, l.kept);
}

pub fn legacyOpening(comptime Lane: type, x: anytype, s: *Stream) !*Lane {
    const p = s.prompt();
    if (p.len == 0) return error.EmptyPrompt;
    if (p.len + @as(usize, s.max_new) + max_window > x.m.context) return error.PastContext;
    const got = try x.lanes.getOrPut(x.gpa, s);
    if (!got.found_existing) {
        const slot = std.mem.indexOfScalar(bool, x.free, true) orelse {
            x.lanes.removeByPtr(got.key_ptr);
            return error.NoFreeSlot;
        };
        x.free[slot] = false;
        got.value_ptr.* = .{ .slot = slot, .last = x.slot_bufs[2 * slot].ptr, .rule = x.slot_bufs[2 * slot + 1].ptr };
    }
    const l = got.value_ptr;
    l.sampled = false;
    l.mark = null;
    if (s.sampling) |sp| if (sp.temperature > 0) {
        const r = cuda.sample.Rule.from(lanes.gpu_rule.rule(sp));
        try x.upload(l.rule, std.mem.asBytes(&r));
        l.sampled = true;
    };
    s.cached = 0;
    if (s.images.len > 0) try x.encodeImages(s, l);
    l.fill = x.resumeAt(s, l, p.len);
    s.fill_at.store(@intCast(l.fill.?), .monotonic);
    return l;
}
