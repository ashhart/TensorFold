//! A verify round's inputs laid out as tree_forward or multi_tree_forward lays them (Triton variants follow alignment).

const std = @import("std");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const st = @import("state.zig");
const tree = @import("tree.zig");

const bf = 2;

/// One stream's window: its tokens from its committed position, and each row's parent (null: a chain).
pub const Part = struct { seq: *st.Seq, tokens: []const u32, parents: ?[]const i32 = null };

/// Where a round's inputs live on the device, and what the layer loop launches with.
pub const Round = struct {
    rows: usize,
    streams: usize,
    multi: bool,
    ids: u64,
    pos: u64,
    windows: u64,
    sids: u64, // multi: each row's stream; else the windows again (gdn_pre passes WIN as SID)
    plan: u64, // GDN schedule entries (rows, 3)
    starts: u64, // stream row starts (streams + 1)
    slots: usize, // the GDN schedule's live states (0: every window a chain)
    most: usize, // the most rows a stream's window holds
    tables: u64, // multi: (DeltaNet layers, streams) state pointers, a layer's row at its DeltaNet index
    attn: tri.Tri.Plan,
    offs: u64, // (attention layers, streams, 2) int64 cache offsets; single: indexed by layer
    starts_host: [st.max_streams + 1]usize,

    /// The conv rows gdn_pre reads for layer `i` (its DeltaNet index `j`): the stream's own, or the stacked copy.
    pub fn conv(r: *const Round, parts: []const Part, s: *const st.Scratch, g: c.Geometry, i: usize, j: usize) u64 {
        if (!r.multi) return parts[0].seq.conv[i];
        return s.conv_cat + j * st.max_streams * (c.conv_taps - 1) * g.convDim() * bf;
    }

    /// Layer `i`'s attention cache offsets: tree_forward's (one stream, by layer) or the packed (layer, streams) rows.
    pub fn offsets(r: *const Round, i: usize, a: usize) u64 {
        return if (r.multi) r.offs + a * r.streams * 16 else r.offs + i * 16;
    }

    /// The DeltaNet state table of DeltaNet layer `j` (multi only).
    pub fn table(r: *const Round, j: usize) u64 {
        return r.tables + j * r.streams * 8;
    }
};

/// A part's parents: its own tree's, or a chain's.
pub fn parentsOf(p: Part, buf: []i32) []const i32 {
    if (p.parents) |x| return x;
    for (buf[0..p.tokens.len], 0..) |*x, r| x.* = @as(i32, @intCast(r)) - 1;
    return buf[0..p.tokens.len];
}

/// attention.plan_host: rows, (first, rows, keys, slots) per stream, items, then window parents (global rows).
fn attnPlan(gpa: std.mem.Allocator, out: *std.ArrayList(i32), parts: []const Part, group: usize) !usize {
    for (parts, 0..) |p, s| for (p.tokens) |_| try out.append(gpa, @intCast(s));
    var start: usize = 0;
    for (parts) |p| {
        const w = p.tokens.len;
        try out.appendSlice(gpa, &.{ @intCast(start), @intCast(w), @intCast(p.seq.pos), @intCast(tri.slots(p.seq.pos, w)) });
        start += w;
    }
    var n_items: usize = 0;
    for (parts, 0..) |p, s| {
        const w = p.tokens.len;
        const folded = tri.groups(p.seq.pos, w);
        const loose = p.seq.pos / tri.chunk - folded * tri.group_chunks;
        for (0..folded + loose) |j| {
            const code: i32 = if (j < folded) @intCast(j) else -1 - @as(i32, @intCast(j - folded));
            var first: usize = 0;
            while (first < w * group) : (first += 16) {
                try out.appendSlice(gpa, &.{ @intCast(s), @intCast(first), code });
                n_items += 1;
            }
        }
    }
    start = 0;
    var buf: [tree.max_nodes]i32 = undefined;
    for (parts) |p| {
        for (parentsOf(p, &buf)) |x| try out.append(gpa, if (x < 0) -1 else @intCast(start + @as(usize, @intCast(x))));
        start += p.tokens.len;
    }
    return n_items;
}

/// Each stream's key and value caches as bf16 element offsets from the scratch's origin.
fn offsetOf(s: *const st.Scratch, ptr: u64) i64 {
    return @divExact(@as(i64, @bitCast(ptr -% s.origin)), bf);
}

/// Every stream's host plans: positions (pos + depth), the GDN schedule (nodes offset by its start), conv windows.
const Host = struct { pos: std.ArrayList(i32) = .empty, entries: std.ArrayList(i32) = .empty, windows: std.ArrayList(i32) = .empty, slots: usize = 0, most: usize = 0 };

fn plans(gpa: std.mem.Allocator, parts: []const Part, starts: []const usize, multi: bool, h: *Host) !void {
    const keep = c.conv_taps - 1;
    var buf: [tree.max_nodes]i32 = undefined;
    var dep: [tree.max_nodes]usize = undefined;
    var win: [tree.max_nodes * c.conv_taps]i32 = undefined;
    for (parts, 0..) |p, k| {
        const parents = parentsOf(p, &buf);
        const w = parents.len;
        try tree.depths(parents, dep[0..w]);
        for (dep[0..w]) |d| try h.pos.append(gpa, @intCast(p.seq.pos + d));
        const s = try tree.schedule(gpa, parents);
        defer gpa.free(s.entries);
        for (s.entries, 0..) |x, j| try h.entries.append(gpa, if (j % 3 == 0) x + @as(i32, @intCast(starts[k])) else x);
        h.slots = @max(h.slots, s.slots);
        h.most = @max(h.most, w);
        // one stream's windows index [its state; its rows]; several streams' index [their states; all rows]
        tree.convWindows(parents, keep, if (multi) starts[k] else 0, win[0 .. w * (keep + 1)]);
        try h.windows.appendSlice(gpa, win[0 .. w * (keep + 1)]);
    }
}

/// Uploads the round's inputs and runs attention's _paths; the caller's layer loop reads the returned Round.
pub fn stage(gpa: std.mem.Allocator, ops: kern.Ops, t: tri.Tri, s: *st.Scratch, g: c.Geometry, parts: []const Part) !Round {
    if (parts.len == 0 or parts.len > st.max_streams) return error.InvalidQwenWindows;
    var W: usize = 0;
    var r: Round = undefined;
    r.starts_host[0] = 0;
    for (parts, 0..) |p, k| {
        if (p.tokens.len == 0 or p.tokens.len > tree.max_nodes) return error.InvalidQwenWindows;
        if (p.parents) |x| if (x.len != p.tokens.len) return error.InvalidQwenWindows;
        if (p.seq.pos + p.tokens.len > p.seq.capacity) return error.PromptTooLong;
        W += p.tokens.len;
        r.starts_host[k + 1] = W;
    }
    if (W > st.round_rows) return error.WindowTooWide;
    r.rows = W;
    r.streams = parts.len;
    r.multi = parts.len > 1;
    const S = parts.len;
    const group = g.query_heads / g.kv_heads;
    var h: Host = .{};
    defer {
        h.pos.deinit(gpa);
        h.entries.deinit(gpa);
        h.windows.deinit(gpa);
    }
    try plans(gpa, parts, r.starts_host[0 .. S + 1], r.multi, &h);
    r.slots = h.slots;
    r.most = h.most;
    var host: std.ArrayList(i32) = .empty;
    defer host.deinit(gpa);
    var offs: std.ArrayList(i64) = .empty;
    defer offs.deinit(gpa);
    if (!r.multi) {
        // tree_forward: tokens, positions, plan (entries then starts), windows and attention's plan apart
        const p = parts[0];
        for (p.tokens) |tok| try host.append(gpa, @intCast(tok));
        try ops.upload(s.ids, std.mem.sliceAsBytes(host.items));
        try ops.upload(s.pos, std.mem.sliceAsBytes(h.pos.items));
        try ops.upload(s.windows, std.mem.sliceAsBytes(h.windows.items));
        try ops.s.synchronize();
        host.clearRetainingCapacity();
        try host.appendSlice(gpa, h.entries.items);
        try host.appendSlice(gpa, &.{ 0, @intCast(W) });
        try ops.upload(s.plan, std.mem.sliceAsBytes(host.items));
        try ops.s.synchronize();
        host.clearRetainingCapacity();
        const n_items = try attnPlan(gpa, &host, parts, group);
        try ops.upload(s.attn_plan, std.mem.sliceAsBytes(host.items));
        try ops.s.synchronize();
        for (0..g.layers) |i| try offs.appendSlice(gpa, if (c.linear(i)) &.{ 0, 0 } else &.{ offsetOf(s, p.seq.keys[i]), offsetOf(s, p.seq.values[i]) });
        try ops.upload(s.offs, std.mem.sliceAsBytes(offs.items));
        try ops.s.synchronize();
        r.ids = s.ids;
        r.pos = s.pos;
        r.windows = s.windows;
        r.sids = s.windows;
        r.plan = s.plan;
        r.starts = s.plan + 3 * W * 4;
        r.tables = 0;
        r.offs = s.offs;
        r.attn = .{ .rows = s.attn_plan, .streams = s.attn_plan + W * 4, .items = s.attn_plan + (W + 4) * 4, .n_items = n_items, .paths = s.paths, .depths = s.depths, .width = W };
        try t.paths(s.attn_plan + (W + 4 + 3 * n_items) * 4, s.paths, s.depths, W);
        return r;
    }
    // multi_tree_forward: positions, stream ids, plan entries, starts, ids, windows, attention's plan in one copy
    try host.appendSlice(gpa, h.pos.items);
    for (parts, 0..) |p, k| for (p.tokens) |_| try host.append(gpa, @intCast(k));
    try host.appendSlice(gpa, h.entries.items);
    for (r.starts_host[0 .. S + 1]) |x| try host.append(gpa, @intCast(x));
    for (parts) |p| for (p.tokens) |tok| try host.append(gpa, @intCast(tok));
    try host.appendSlice(gpa, h.windows.items);
    const at_attn = host.items.len;
    const n_items = try attnPlan(gpa, &host, parts, group);
    try ops.upload(s.round, std.mem.sliceAsBytes(host.items));
    try ops.s.synchronize();
    const base = s.round;
    r.pos = base;
    r.sids = base + W * 4;
    r.plan = base + 2 * W * 4;
    r.starts = base + 5 * W * 4;
    r.ids = base + (5 * W + S + 1) * 4;
    r.windows = base + (6 * W + S + 1) * 4;
    const ab = base + at_attn * 4;
    r.attn = .{ .rows = ab, .streams = ab + W * 4, .items = ab + (W + 4 * S) * 4, .n_items = n_items, .paths = s.paths, .depths = s.depths, .width = W };
    // the state pointer table and cache offsets, one copy each; every stream's conv rows stacked by layer
    var ptrs: std.ArrayList(u64) = .empty;
    defer ptrs.deinit(gpa);
    var stack: std.ArrayList(kern.Copy) = .empty;
    defer stack.deinit(gpa);
    const keep = c.conv_taps - 1;
    var j: usize = 0;
    for (0..g.layers) |i| {
        if (!c.linear(i)) {
            for (parts) |p| try offs.appendSlice(gpa, &.{ offsetOf(s, p.seq.keys[i]), offsetOf(s, p.seq.values[i]) });
            continue;
        }
        const cat = s.conv_cat + j * st.max_streams * keep * g.convDim() * bf;
        for (parts, 0..) |p, k| {
            try ptrs.append(gpa, p.seq.rec[i]);
            try stack.append(gpa, .{ .dst = cat + k * keep * g.convDim() * bf, .src = p.seq.conv[i], .bytes = keep * g.convDim() * bf });
        }
        j += 1;
    }
    try ops.upload(s.tables, std.mem.sliceAsBytes(ptrs.items));
    try ops.upload(s.offs, std.mem.sliceAsBytes(offs.items));
    try ops.upload(s.copy_items, std.mem.sliceAsBytes(stack.items));
    try ops.copies(s.copy_items, stack.items.len);
    try ops.s.synchronize();
    r.tables = s.tables;
    r.offs = s.offs;
    try t.paths(ab + (W + 4 * S + 3 * n_items) * 4, s.paths, s.depths, W);
    return r;
}
