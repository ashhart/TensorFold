//! K3.1's experimental indexer launch contract; the native 2047-position ceiling remains unchanged.
const mtl = @import("metal");
const Ref = @import("kernels.zig").Ref;
const MlaArgs = @import("kernels.zig").MlaArgs;

pub const heads = 32;
pub const dim = 128;
pub const topk = 2048;
pub const Row = extern struct { slot: u32, pos: u32, list: u32, nkeys: u32 = 0 };
pub const Args = extern struct {
    rows: u32,
    slot_keys: u32,
    q_stride: u32 = heads * dim,
    k_stride: u32 = dim,
    w_stride: u32 = heads,
    score_stride: u32,
    eps: f32 = 1e-5,
    key_scale: f32 = 1.0 / @sqrt(@as(f32, dim)),
    head_scale: f32 = 1.0 / @sqrt(@as(f32, heads)),
};

pub const Pipelines = struct {
    key: mtl.Pipeline,
    scores: mtl.Pipeline,
    select: mtl.Pipeline,
    attend: mtl.Pipeline,
    merge: mtl.Pipeline,

    pub fn init(device: mtl.Device, library: mtl.Library) !Pipelines {
        const key = try mtl.Pipeline.init(device, library, "k3_index_key", false);
        errdefer key.deinit();
        const scores = try mtl.Pipeline.init(device, library, "k3_index_scores", false);
        errdefer scores.deinit();
        const select = try mtl.Pipeline.init(device, library, "k3_index_topk", false);
        errdefer select.deinit();
        const attend = try mtl.Pipeline.init(device, library, "k3_index_attend", false);
        errdefer attend.deinit();
        return .{ .key = key, .scores = scores, .select = select, .attend = attend, .merge = try mtl.Pipeline.init(device, library, "k3_index_merge", false) };
    }

    pub fn deinit(p: *Pipelines) void {
        p.merge.deinit();
        p.attend.deinit();
        p.select.deinit();
        p.scores.deinit();
        p.key.deinit();
    }

    pub fn store(p: *const Pipelines, e: mtl.ComputeEncoder, key_proj: Ref, norm: Ref, bias: Ref, rows: Ref, cache: Ref, a: Args) void {
        e.setPipeline(p.key);
        bind(e, .{ key_proj, norm, bias, rows });
        e.setValue(a, 4);
        e.setBuffer(cache.buf, cache.off, 5);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(32, 1, 1));
    }

    pub fn score(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, head_weights: Ref, cache: Ref, rows: Ref, scores: Ref, a: Args, visible: u32) void {
        e.setPipeline(p.scores);
        bind(e, .{ q, head_weights, cache, rows });
        e.setValue(a, 4);
        e.setBuffer(scores.buf, scores.off, 5);
        e.dispatchGroups(mtl.Size.of((visible + 127) / 128, a.rows, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn top(p: *const Pipelines, e: mtl.ComputeEncoder, scores: Ref, rows: Ref, list: Ref, a: Args) void {
        e.setPipeline(p.select);
        bind(e, .{ scores, rows });
        e.setValue(a, 2);
        e.setBuffer(list.buf, list.off, 3);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }

    pub fn attention(p: *const Pipelines, e: mtl.ComputeEncoder, q: Ref, qlat: Ref, rows: Ref, list: Ref, partial: Ref, olat: Ref, a: MlaArgs) void {
        e.setPipeline(p.attend);
        bind(e, .{ q, qlat, rows, partial });
        e.setValue(a, 4);
        e.setBuffer(list.buf, list.off, 5);
        const fit: u32 = @intCast(@min(32, p.attend.maxThreads() / 32));
        const groups = (a.heads + fit - 1) / fit;
        const per = (a.heads + groups - 1) / groups;
        e.dispatchGroups(mtl.Size.of(groups, 16, a.rows), mtl.Size.of(32 * per, 1, 1));
        e.barrier();
        e.setPipeline(p.merge);
        bind(e, .{ partial, rows, olat });
        e.setValue(a, 3);
        e.dispatchGroups(mtl.Size.of(a.heads, a.rows, 1), mtl.Size.of(256, 1, 1));
    }
};

fn bind(e: mtl.ComputeEncoder, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}
