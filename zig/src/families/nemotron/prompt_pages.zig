//! TF_KV_SHARE on Metal: a prompt pass resumes on the store's kept pages and keeps its own pages at the store's marks.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const kvp = @import("../../core/metal_kv_pages.zig");
const Metal = @import("backend.zig").Metal;

/// One prompt pass's plan from the store: the pages it adopts, where they end, and the marks it keeps at.
pub const Pass = struct {
    on: bool = false,
    adopt: std.ArrayList(u32) = .empty, // held once more for the stream until its lane adopts them
    from: u32 = 0,
    marks: []const u32 = &.{}, // the store's allocation, freed by `end`
    kept: u64 = 0,

    /// The store's deepest kept state of `s`'s prompt (none without TF_KV_SHARE): its pages held, its state pending.
    pub fn begin(b: *Metal, s: *lanes.Stream) !Pass {
        const pr = b.prompts orelse return .{};
        b.pending = null;
        var p: Pass = .{ .on = true };
        errdefer p.adopt.deinit(b.gpa);
        const got = try pr.begin(s.prompt(), s.history_len, s.shared_prefixes, @ptrCast(s), &p.adopt);
        p.from = got.from;
        p.marks = got.marks;
        return p;
    }

    /// Pages the lane holds from the store: its promise is that many fewer.
    pub fn held(p: *const Pass) usize {
        return p.adopt.items.len;
    }

    /// The lane's table starts with the adopted pages and the pending Mamba states go in; the pass resumes there.
    pub fn resume_(p: *Pass, b: *Metal, s: *lanes.Stream, c: *st.Cache) !usize {
        if (!p.on) return 0;
        if (p.adopt.items.len > 0) {
            const pg = c.kv[0].paged orelse return error.NotPaged;
            const ids = p.adopt.items;
            p.adopt.items.len = 0; // the lane's now: adopt releases them if it fails
            try pg.lease.pages.adopt(pg.lease.lane, ids);
        }
        if (p.from == 0) return 0;
        const k = b.pending orelse return error.KeptStateMismatch;
        b.pending = null;
        if (k.at != p.from or !k.pages) return error.KeptStateMismatch;
        try snapshot.restore(b, c, k);
        s.cached = p.from;
        return p.from;
    }

    /// Where a chunk from `at` ends: `stop`, or the store's first mark between.
    pub fn cut(p: *const Pass, at: usize, stop: usize) usize {
        for (p.marks) |m| if (m > at and m < stop) return m;
        return stop;
    }

    /// At one of the store's marks the store takes the pages it lacks and the state; pages it held replace the lane's.
    pub fn mark(p: *Pass, b: *Metal, s: *lanes.Stream, c: *st.Cache, at: usize) void {
        if (!p.on or std.mem.indexOfScalar(u32, p.marks, @intCast(at)) == null) return;
        p.keep(b, s, c, at) catch |e| std.log.warn("nemotron: prompt pages not kept: {s}", .{@errorName(e)});
    }

    fn keep(p: *Pass, b: *Metal, s: *lanes.Stream, c: *st.Cache, at: usize) !void {
        const pg = c.kv[0].paged orelse return error.NotPaged;
        const n = at / kvp.page_tokens;
        const mine = try b.gpa.dupe(u32, pg.ids()[0..n]);
        defer b.gpa.free(mine);
        const path = try b.gpa.alloc(u32, n);
        defer b.gpa.free(path);
        const shared = b.prompts.?.keep(s.prompt(), @intCast(at), @ptrCast(s), mine, path);
        if (shared == n) p.kept += 1;
        if (std.mem.eql(u32, mine[0..shared], path[0..shared])) return;
        try b.drain(); // the chunk in flight writes the lane's pages: none goes back while it may
        for (path[0..shared], mine[0..shared], 0..) |id, own, i| if (id != own) pg.lease.pages.share(pg.lease.lane, i, id);
    }

    /// The pass's line, and the plan's memory back.
    pub fn end(p: *Pass, b: *Metal, s: *lanes.Stream, ok: bool) void {
        if (!p.on) return;
        if (ok) b.prompts.?.report(s.prompt().len, s.cached, p.kept);
        for (p.adopt.items) |id| b.kv_pages.?.pool.release(id); // pages a failed pass never adopted
        p.adopt.deinit(b.gpa);
        b.prompts.?.free(p.marks);
        p.* = .{};
    }
};
