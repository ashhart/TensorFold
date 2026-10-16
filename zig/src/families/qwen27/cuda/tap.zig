//! Prompts in a conversation read through the 27B's own batched prompt pass (as serving runs them, at any length).
const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const st = @import("state.zig");
const kern = @import("kernels.zig");
const engine = @import("engine.zig");
const fwd = @import("forward.zig");

/// Rows one piece of the pass taps at most.
pub const tap_rows = 256;
/// Questions after one conversation readMany runs side by side at most (each a stream of the tap's capacity).
pub const max_many = 8;

pub const Tap = struct {
    e: *engine.Engine,
    seq: st.Seq,
    after: st.Seq, // the state after the last conversation run
    talk: u64 = 0, // that conversation's tokens as one number (0: none kept)
    layers: []const usize = &.{}, // the layers a call taps, in the order `out` holds them
    table: cuda.DeviceBuffer, // kern.Copy [layers, tap_rows]: a piece's gathers, layer by layer
    normed: cuda.DeviceBuffer, // bf16 [layers, tap_rows, hidden]: the tapped MLP inputs
    host: []u16,
    items: []kern.Copy,
    m: usize = 0, // rows the running piece taps
    many: []st.Seq = &.{}, // readMany's questions' streams, made the first time
    runs: usize = 0, // conversations run
    reused: usize = 0, // questions that found theirs kept

    pub fn init(gpa: std.mem.Allocator, e: *engine.Engine, cap: usize) !Tap {
        const g = e.w.g;
        var seq = try e.sequence(cap);
        errdefer seq.deinit();
        var after = try e.sequence(cap);
        errdefer after.deinit();
        var table = try cuda.DeviceBuffer.alloc(e.ctx.d, g.layers * tap_rows * @sizeOf(kern.Copy));
        errdefer table.free();
        var normed = try cuda.DeviceBuffer.alloc(e.ctx.d, g.layers * tap_rows * g.hidden * 2);
        errdefer normed.free();
        const host = try gpa.alloc(u16, g.layers * tap_rows * g.hidden);
        errdefer gpa.free(host);
        const items = try gpa.alloc(kern.Copy, g.layers * tap_rows);
        std.log.info("slide: prompts in a conversation are read through the engine's pass, up to {d} tokens", .{cap - 1});
        return .{ .e = e, .seq = seq, .after = after, .table = table, .normed = normed, .host = host, .items = items };
    }

    pub fn deinit(tp: *Tap, gpa: std.mem.Allocator) void {
        tp.e.s.synchronize() catch {};
        for (tp.many) |*q| q.deinit();
        gpa.free(tp.many);
        tp.seq.deinit();
        tp.after.deinit();
        tp.table.free();
        tp.normed.free();
        gpa.free(tp.host);
        gpa.free(tp.items);
    }

    /// The MLP inputs of prompt `ids` at `rows` (ascending) at each of `layers`, into `out` [rows, layers.
    pub fn read(tp: *Tap, ids: []const u32, context: usize, rows: []const u32, layers: []const usize, out: []u16) !void {
        const e = tp.e;
        const g = e.w.g;
        if (ids.len == 0 or context > ids.len or ids.len >= tp.seq.capacity or out.len != rows.len * layers.len * g.hidden) return error.BadTap;
        for (rows, 0..) |r, i| if (r >= ids.len or (i > 0 and r <= rows[i - 1])) return error.BadTap;
        for (layers) |l| if (l >= g.layers) return error.BadTap;
        tp.layers = layers;
        const talk = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids[0..context]));
        var r: usize = 0;
        if (context > 0 and tp.talk == talk and tp.after.pos == context and (rows.len == 0 or rows[0] >= context)) {
            try tp.seq.copyFrom(e.ops(), g, &tp.after);
            tp.reused += 1;
        } else {
            try tp.seq.reset(e.ops(), g);
            // the conversation, its rows read on the way, then its state kept for the questions after it
            try tp.run(ids, 0, context, rows, &r, out);
            tp.talk = 0;
            if (context > 0) {
                try tp.after.copyFrom(e.ops(), g, &tp.seq);
                tp.talk = talk;
                tp.runs += 1;
            }
        }
        try tp.run(ids, context, ids.len, rows, &r, out);
        if (r != rows.len) return error.BadTap;
    }

    /// One question after a conversation, as `read` takes it: its tokens, the conversation's rows.
    pub const Item = struct { ids: []const u32, context: usize, rows: []const u32, out: []u16 };

    /// Questions sharing one conversation (each item's first `context` tokens the same) read as `read` reads each.
    pub fn readMany(tp: *Tap, gpa: std.mem.Allocator, items: []const Item, layers: []const usize) !void {
        const e = tp.e;
        const g = e.w.g;
        const D = g.hidden;
        if (items.len == 0) return;
        const context = items[0].context;
        var rows_n: usize = 0;
        var taps: usize = 0;
        var fits = items.len <= max_many and context > 0;
        for (items) |it| {
            if (it.context != context or !std.mem.eql(u32, it.ids[0..context], items[0].ids[0..context])) fits = false;
            if (it.ids.len + 1 > tp.seq.capacity) fits = false;
            rows_n += it.ids.len - context;
            for (it.rows) |r| taps += @intFromBool(r >= context);
        }
        if (!fits or rows_n > engine.prompt_rows or taps > tap_rows) {
            for (items) |it| try tp.read(it.ids, it.context, it.rows, layers, it.out);
            return;
        }
        // the conversation and its own tapped rows (the first item's), as `read` runs it
        {
            const it = items[0];
            var k: usize = 0;
            while (k < it.rows.len and it.rows[k] < context) k += 1;
            try tp.read(it.ids[0..context], context, it.rows[0..k], layers, it.out[0 .. k * layers.len * D]);
        }
        if (tp.many.len == 0) {
            tp.many = try gpa.alloc(st.Seq, max_many);
            var made: usize = 0;
            errdefer {
                for (tp.many[0..made]) |*q| q.deinit();
                gpa.free(tp.many);
                tp.many = &.{};
            }
            while (made < max_many) : (made += 1) tp.many[made] = try e.sequence(tp.seq.capacity);
        }
        if (tp.many[0].capacity < tp.seq.capacity) return error.BadTap;
        var pieces: [max_many]fwd.Forward.Piece = undefined;
        var starts: [max_many + 1]usize = undefined;
        starts[0] = 0;
        var m: usize = 0;
        tp.layers = layers;
        for (items, 0..) |it, k| {
            try tp.many[k].copyFrom(e.ops(), g, &tp.after);
            pieces[k] = .{ .ids = it.ids[context..], .seq = &tp.many[k], .last = false };
            starts[k + 1] = starts[k] + it.ids.len - context;
        }
        // each tapped layer's gathers: item k's row r of the pass's MLP input into slot (j, i)
        var i: usize = 0;
        for (items, 0..) |it, k| for (it.rows) |r| {
            if (r < context) continue;
            for (layers, 0..) |_, j| tp.items[j * taps + i] = .{ .dst = tp.normed.ptr + (j * taps + i) * D * 2, .src = e.scratch.y + (starts[k] + r - context) * D * 2, .bytes = D * 2 };
            i += 1;
        };
        m = taps;
        if (m > 0) {
            try e.ops().upload(tp.table.ptr, std.mem.sliceAsBytes(tp.items[0 .. layers.len * m]));
            try e.s.synchronize();
        }
        tp.m = m;
        var f = e.forward();
        f.mlp = .{ .ctx = tp, .f = tapAdd };
        try f.chunks(pieces[0..items.len]);
        tp.m = 0;
        if (m == 0) return;
        try e.s.synchronize();
        try tp.normed.download(0, std.mem.sliceAsBytes(tp.host[0 .. layers.len * m * D]));
        i = 0;
        for (items) |it| {
            var at: usize = 0;
            for (it.rows) |r| {
                if (r < context) {
                    at += 1;
                    continue;
                }
                for (0..layers.len) |j| @memcpy(it.out[(at * layers.len + j) * D ..][0..D], tp.host[(j * m + i) * D ..][0..D]);
                at += 1;
                i += 1;
            }
        }
    }

    /// ids[from..to] on through the pass in pieces of at most a prompt chunk and `tap_rows` tapped rows.
    fn run(tp: *Tap, ids: []const u32, from: usize, to: usize, rows: []const u32, r: *usize, out: []u16) !void {
        const e = tp.e;
        const g = e.w.g;
        const D = g.hidden;
        const L = tp.layers.len;
        var a = from;
        while (a < to) {
            var b = @min(to, a + engine.prompt_rows);
            var k = r.*;
            while (k < rows.len and rows[k] < b) : (k += 1) if (k - r.* == tap_rows) {
                b = rows[k];
                break;
            };
            const m = k - r.*;
            // each tapped layer's gathers: row rows[i] - a of the pass's MLP input into slot (j, i)
            for (tp.layers, 0..) |_, j| for (0..m) |i| {
                tp.items[j * m + i] = .{ .dst = tp.normed.ptr + (j * m + i) * D * 2, .src = e.scratch.y + (rows[r.* + i] - a) * D * 2, .bytes = D * 2 };
            };
            if (m > 0) {
                try e.ops().upload(tp.table.ptr, std.mem.sliceAsBytes(tp.items[0 .. L * m]));
                try e.s.synchronize(); // the host table must outlive the async upload
            }
            tp.m = m;
            var f = e.forward();
            f.mlp = .{ .ctx = tp, .f = tapAdd };
            try f.chunks(&.{.{ .ids = ids[a..b], .seq = &tp.seq, .last = false }});
            tp.m = 0;
            if (m > 0) {
                try e.s.synchronize();
                try tp.normed.download(0, std.mem.sliceAsBytes(tp.host[0 .. L * m * D]));
                for (0..m) |i| for (0..L) |j| {
                    @memcpy(out[((r.* + i) * L + j) * D ..][0..D], tp.host[(j * m + i) * D ..][0..D]);
                };
            }
            r.* = k;
            a = b;
        }
    }

    /// The pass's MLP hook: the layer's pocket as serving adds it, then its tapped rows' MLP inputs gathered.
    fn tapAdd(ctx: *anyopaque, layer: usize, y: u64, out: u64, rows: usize, taught: u64) anyerror!void {
        const tp: *Tap = @ptrCast(@alignCast(ctx));
        const e = tp.e;
        try e.pockets.add(e.s, layer, y, out, rows, taught);
        if (tp.m == 0) return;
        const j = std.mem.indexOfScalar(usize, tp.layers, layer) orelse return;
        if (y != e.scratch.y) return error.BadTap; // the gathers read the prompt pass's MLP input rows
        try e.ops().copies(tp.table.ptr + j * tp.m * @sizeOf(kern.Copy), tp.m);
    }
};
