//! TF_GLM_GRAPHS=1: verify windows as CUDA graphs, keyed on every host value their launches take (cuda.keyed_graph).
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const dsa_block = @import("cuda_dsa_block.zig");

const kg = cuda.keyed_graph;

/// Graphs a cache holds at most: four streams' windows of 1 to 8 rows on either KDA half fit.
pub const max_graphs = 128;

/// The engine's caches: one stream's windows (Engine.verify) and several streams' in one pass (Engine.verifyShared).
pub const Graphs = struct {
    one: kg.Cache(Key, max_graphs) = .{},
    shared: kg.Cache(SharedKey, max_graphs) = .{},

    pub fn clear(g: *Graphs, gpa: std.mem.Allocator) void {
        g.one.clear(gpa);
        g.shared.clear(gpa);
    }
};

/// A verify window's pass on one rank: `rows` rows of the stream in `slot`, then the hashes of everything else.
pub const Key = struct {
    rows: u32,
    slot: u32,
    shape: u64,
    moving: u64,

    pub fn init(f: *const fw.Forward, slot: u32, rows: usize) Key {
        const r = f.ranks[0];
        var h = pass(f);
        sequence(&h, r, r.st, rows);
        return .{ .rows = @intCast(rows), .slot = slot, .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// Several streams' windows in one pass (Engine.verifyShared): each segment's slot, place and rows, then its sequence.
pub const SharedKey = struct {
    lanes: u32,
    rows: u32,
    shape: u64,
    moving: u64,

    pub fn init(f: *const fw.Forward, slots: []const u32, rows: usize) SharedKey {
        const r = f.ranks[0];
        var h = pass(f);
        for (r.segs, slots) |g, slot| {
            h.fixed(slot);
            h.fixed(g.row0);
            h.fixed(g.rows);
            sequence(&h, r, g.st, g.rows);
        }
        return .{ .lanes = @intCast(slots.len), .rows = @intCast(rows), .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

/// The pass's own host values: its settings, its exchange and its buffers.
fn pass(f: *const fw.Forward) kg.Hash {
    var h: kg.Hash = .{};
    h.fixed(@intFromBool(f.kda_wide));
    h.fixed(@intFromBool(f.split));
    h.fixed(@intFromBool(f.ring));
    h.fixed(@backingInt(std.meta.activeTag(f.exchange)));
    h.values(&[_]u64{@intFromPtr(f.ranks[0].b)});
    return h;
}

/// A sequence's host values for `rows` rows: buffers, KDA state halves, 512-key chunks, each latent layer's branches.
fn sequence(h: *kg.Hash, r: fw.Rank, st: anytype, rows: usize) void {
    h.values(&[_]u64{ @intFromPtr(st), st.pos_dev, st.conv, st.rec, st.proj, st.save_k, st.save_v, st.save_g, st.save_b, st.table, st.base, st.capacity });
    for (st.cur) |half| h.fixed(half);
    h.values(&[_]u64{@min(latent.chunksFor(st.pos + rows), r.b.nch)});
    var di: usize = 0;
    for (r.w.layers) |l| switch (l.mixer) {
        .kda => {},
        .mla => |a| {
            const c = st.caches[di];
            di += 1;
            h.values(&[_]u64{ c.lc, c.ls, c.ik, c.ig, c.pk, c.pk_rows, c.pt, c.ps, c.pks });
            h.fixed(@intFromBool(st.pos < a.dense_limit));
            const scored = dsa_block.sparseRows(a, st.pos, rows);
            h.fixed(@intFromBool(scored));
            if (!scored) continue;
            const np_max = sparse.poolBucket(st.pos, rows, c.pk_rows - 2);
            h.fixed(np_max);
            h.values(&[_]u64{(sparse.scoreColumns(st.pos, rows, np_max) + 63) / 64});
        },
    };
}

/// Engine.verify's forward: a graph with TF_GLM_GRAPHS=1 (none beside TF_GLM_PROF, image rows or DFlash2's taps).
pub fn window(e: anytype, f: *const fw.Forward, ids: []const i32) !void {
    const g = graphed(e, f) orelse return f.run(ids);
    try f.stage(ids);
    try g.one.run(e.gpa, e.stream, Key.init(f, e.slot, ids.len), Body{ .f = f, .rows = ids.len });
}

/// Engine.verifyShared's forward, as `window`: `slots` the parts' streams in segment order.
pub fn windows(e: anytype, f: *const fw.Forward, ids: []const i32, slots: []const u32) !void {
    const g = graphed(e, f) orelse return f.run(ids);
    try f.stage(ids);
    try g.shared.run(e.gpa, e.stream, SharedKey.init(f, slots, ids.len), Body{ .f = f, .rows = ids.len });
}

fn graphed(e: anytype, f: *const fw.Forward) ?*Graphs {
    const g = if (e.graphs) |*x| x else return null;
    return if (f.prof != null or f.images != null or f.taps != null or f.ranks.len != 1) null else g;
}

/// Forward.kernels over a staged window: the body a Cache records and replays.
pub const Body = struct {
    f: *const fw.Forward,
    rows: usize,

    pub fn run(b: Body) !void {
        return b.f.kernels(b.rows);
    }
};
