//! Ling-3.1's MTP layer (model.layers.72) as the family's drafter: enorm | hnorm through eh_proj, MLA + MoE, its norm.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("weights.zig");
const fw = @import("forward.zig");

const glue = @import("graph.zig").glue;

/// The most drafts one chain holds.
pub const max_drafts = 3;

/// The MTP layer's graph over a view of the weights (its layer alone, its final norm as `norm`) and its scratch.
pub const Head = struct {
    view: weights.Weights,
    mtp: weights.Mtp,
    g: fw.Graph,
    bm: fw.Buffers, // chained draft rows and a verify's kept rows
    cat: cuda.DeviceBuffer, // bf16 [rows][2 hidden]: enorm(embed(next)) | hnorm(hidden)

    /// In place (the graph points at `view`); `w` holds the MTP layer last; cat rows for `rows` (the prompt chunk).
    pub fn init(h: *Head, gpa: std.mem.Allocator, d: *const cuda.Driver, m: *const fw.Modules, s: cuda.Stream, w: *const weights.Weights, rows: usize, capacity: usize) !void {
        const c = &w.config;
        const mtp = w.mtp orelse return error.NoMtpLayer;
        const last = w.layers[w.layers.len - 1];
        if (last.index != c.layers or last.mixer != .mla or last.ffn != .moe) return error.NoMtpLayer;
        h.mtp = mtp;
        h.view = w.*;
        h.view.layers = w.layers[w.layers.len - 1 ..];
        h.view.norm = mtp.norm; // final_layernorm before the shared lm_head
        h.g = try fw.Graph.init(gpa, d, m, s, &h.view);
        errdefer h.g.deinit();
        h.bm = try fw.Buffers.init(d, m, w, fw.window_rows, false, capacity);
        errdefer h.bm.deinit();
        h.cat = try cuda.DeviceBuffer.alloc(d, @max(rows, fw.window_rows) * 2 * @as(usize, c.hidden) * 2);
    }

    pub fn deinit(h: *Head) void {
        h.cat.free();
        h.bm.deinit();
        h.g.deinit();
        h.* = undefined;
    }

    pub fn forward(h: *const Head, m: *const fw.Modules, s: cuda.Stream, b: *const fw.Buffers) fw.Forward {
        return .{ .m = m, .g = &h.g, .s = s, .b = b };
    }

    /// The layer on b.x's `rows` embedding rows and `hidden`'s rows (bf16 [rows][hidden]), committed on `hd`.
    pub fn step(h: *const Head, f: fw.Forward, hd: *fw.State, rows: usize, hidden: u64, picks: ?u64) !void {
        const b = f.b;
        const c = &h.view.config;
        const D: usize = c.hidden;
        const o = f.ops();
        if (rows == 0 or rows > b.rows or hd.pending != 0) return error.BadRows;
        try glue.rmsnorm(o.set(), o.s, b.x, D, h.mtp.enorm, h.cat.ptr, 2 * D, null, c.eps, rows, D);
        try glue.rmsnorm(o.set(), o.s, hidden, D, h.mtp.hnorm, h.cat.ptr + D * 2, 2 * D, null, c.eps, rows, D);
        try o.mm(h.mtp.eh, h.cat.ptr, 2 * D, rows, b.x);
        try f.layer(&.{.{ .st = hd, .row0 = 0, .rows = rows }}, 0, rows, picks);
        hd.pending = rows;
        hd.prompt = b.prompt;
        try f.commit(hd, rows);
    }

    /// `next` (the token after each row) with `hidden`'s rows through the layer on `b`, committed on `hd`.
    pub fn absorb(h: *const Head, m: *const fw.Modules, s: cuda.Stream, b: *const fw.Buffers, hd: *fw.State, next: []const i32, hidden: u64) !void {
        const f = h.forward(m, s, b);
        try f.embed(next);
        try h.step(f, hd, next.len, hidden, null);
    }
};
