//! The layer program preserves Python's pending residual and only exposes post-block native tap rows.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const wts = @import("gpu_weights.zig");
const Config = @import("config.zig").Config;
/// A tensor in a frame buffer: the `Reduce` callbacks pass refs, so the type the seam API exposes here.
pub const Ref = @import("projection.zig").Ref;
const Frame = @import("gpu_frame.zig").Frame;
const ane_mlp = @import("ane_mlp.zig");

pub const Mixer = struct {
    ptr: *anyopaque,
    gdn: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Gdn, *Frame, u32) anyerror!void,
    attention: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Attention, *Frame, u32) anyerror!void,
    tap: ?*const fn (*anyopaque, mtl.ComputeEncoder, usize, Ref, u32) anyerror!void = null,
};

/// One exchange: `x` its index, `ref` this rank's `width` columns of the seam in the window's slot for x.
pub const Seam = struct { x: u32, ref: Ref };

/// A rank's per-layer all-gather behind one interface: transport is a model flag, not a fork in the layer program.
pub const Reduce = struct {
    ptr: *anyopaque,
    /// Pure, and taken BEFORE the producer kernel writes `ref`: this seam's index and where this rank's part goes.
    part: *const fn (*anyopaque) Seam,
    /// This rank's own columns of the seam's input rows into x's slot: a bf16 copy, so it carries the frame's bits.
    stage: *const fn (*anyopaque, *mtl.ComputeEncoder, u32, Ref, u32, u32, u32) void,
    /// Relays x, waits for the peer's columns, and reopens the encoder: the caller uses the reopened copy.
    gather: *const fn (*anyopaque, *mtl.ComputeEncoder, u32, Ref, Ref, u32, u32, u32) void,
    /// Out of band: a seam cannot fail while encoding, so `failed()` is checked after each seam and command buffer.
    failed: *const fn (*anyopaque) bool,
    /// Head seam: this rank's argmax over `lo..lo+n` (stride `stride`) meets the peer's in `picks`, one u32 a row.
    argmax: ?*const fn (*anyopaque, *mtl.ComputeEncoder, u32, Ref, u32, u32, u32, u32, Ref) void = null,
    pub fn next(r: Reduce) Seam {
        return r.part(r.ptr);
    }
    /// The next exchange's index, for a seam whose part the kernel writes itself (`argmax` writes the picks).
    pub fn nextX(r: Reduce) u32 {
        return r.part(r.ptr).x;
    }
    /// A seam's input rows: stage this rank's columns of `src`, then take both halves of the frame into `out`.
    pub fn rowsIn(r: Reduce, e: *mtl.ComputeEncoder, se: Seam, src: Ref, rows: u32, width: u32, stride: u32, out: Ref, out_stride: u32) void {
        r.stage(r.ptr, e, se.x, src, rows, width, stride);
        r.gather(r.ptr, e, se.x, se.ref, out, rows, width, out_stride);
    }
    /// A seam's output rows: the producer wrote this rank's rows to `se.ref`; both ranks' rows land in `out`.
    pub fn rowsOut(r: Reduce, e: *mtl.ComputeEncoder, se: Seam, out: Ref, rows: u32, width: u32, out_stride: u32) void {
        r.gather(r.ptr, e, se.x, se.ref, out, rows, width, out_stride);
    }
    pub fn pick(r: Reduce, e: *mtl.ComputeEncoder, x: u32, logits: Ref, stride: u32, lo: u32, n: u32, rows: u32, picks: Ref) void {
        r.argmax.?(r.ptr, e, x, logits, stride, lo, n, rows, picks);
    }
    /// Whether the peer link failed out of band: after a seam, before the next seam's kernel is encoded.
    pub fn broken(r: Reduce) bool {
        return r.failed(r.ptr);
    }
};

/// Serial decode and prompt segments own separate frames and order shared caches at the mixer.
pub const Head = enum { all, last, none };
/// A verified head's dependent readback work stays in its final command buffer.
pub const Tail = struct { ptr: *anyopaque, encode: *const fn (*anyopaque, mtl.ComputeEncoder) anyerror!void };
pub const Stages = struct {
    model: *Model,
    rows: u32,
    hooks: Mixer,
    pending: ?Ref = null,
    hidden: Ref = undefined,
    residual: Ref = undefined,
    head: Head = .all,
    tail: ?Tail = null,
    /// An image prompt's rows: the frame's `takes` rows take theirs over their placeholders' embeddings.
    images: ?Ref = null,
    /// A tensor-parallel rank's all-reduce: null on one Mac, and then every seam is compiled out of the program.
    reduce: ?Reduce = null,
    /// Head seam picks (one u32 a row), required with `Reduce.argmax`: the ranks must pick the same token.
    picks: ?Ref = null,
    pub fn begin(s: *Stages, lane: *@import("core").segments.Lane) !void {
        const m = s.model;
        if (s.rows == 0 or s.rows > m.frame.capacity or lane.k != 0) return error.BadFrameRows;
        s.hidden = m.frame.get(.hidden);
        s.residual = m.frame.get(.residual);
        try m.glue.embedding(lane.enc, m.config, m.weights.embed, m.frame.get(.ids), s.hidden, s.rows);
        lane.enc.barrier();
        if (s.images) |rows| {
            m.glue.imageRows(lane.enc, s.hidden, rows, m.frame.get(.takes), @intCast(m.config.hidden), s.rows);
            lane.enc.barrier();
        }
    }
    pub fn wait(_: *Stages, _: usize) @import("core").segments.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Stages, _: *@import("core").segments.Lane, _: usize) !void {}
    pub fn pre(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        try m.kernels.norm(e, s.hidden, s.pending, wts.ref(layer.norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        if (s.pending != null) {
            std.mem.swap(Ref, &s.hidden, &s.residual);
            try s.steer(lane, index - 1);
            if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, index - 1, s.hidden, s.rows);
        }
    }
    pub fn mixer(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        switch (m.weights.layers[index].mixer) {
            .linear => |g| try s.hooks.gdn(s.hooks.ptr, lane.enc, index, g, &m.frame, s.rows),
            .attention => |a| try s.hooks.attention(s.hooks.ptr, lane.enc, index, a, &m.frame, s.rows),
        }
        lane.enc.barrier();
    }
    pub fn post(s: *Stages, lane: *@import("core").segments.Lane, index: usize) !void {
        const m = s.model;
        const f = &m.frame;
        var e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        const out = switch (layer.mixer) {
            .linear => |g| g.out,
            .attention => |a| a.out,
        };
        // Seam A: a rank owns output rows, not a K half: gather the mixer's columns before o_proj, its rows after.
        const split_attention = s.reduce != null and m.shard.splitsAttention();
        var out_in = f.get(.mixer_out);
        if (split_attention) {
            std.debug.assert(m.shard.ranks == 2); // two Macs: gather's other half is the peer's half
            const cols: u32 = @intCast(@divExact(out.k, 2)); // this rank's mixer columns, rows contiguous in `.mixer_out`
            const first = s.reduce.?.next();
            s.reduce.?.rowsIn(&e, first, out_in, s.rows, cols, cols, f.get(.mixed), out.k);
            lane.enc = e; // rowsIn reopened the encoder; the lane carries it to the next layer
            if (s.reduce.?.broken()) return error.PairFailed;
            e.barrier();
            out_in = f.get(.mixed);
        }
        const product = if (split_attention) s.reduce.?.next() else null;
        if (product) |sm| try m.kernels.quant(lane.enc, out, out_in, null, f.get(.dims), sm.ref, s.rows) else try m.kernels.quant(lane.enc, out, out_in, null, f.get(.dims), f.get(.output), s.rows);
        lane.enc.barrier();
        if (product) |sm| {
            s.reduce.?.rowsOut(&e, sm, f.get(.output), s.rows, out.n, h);
            lane.enc = e;
            if (s.reduce.?.broken()) return error.PairFailed;
            e.barrier();
        }
        try m.kernels.norm(e, s.hidden, f.get(.output), wts.ref(layer.post_norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        std.mem.swap(Ref, &s.hidden, &s.residual);
        var up = f.get(.mixed);
        up.offset += @as(usize, m.config.intermediate) * 2;
        const width: u32 = @intCast(m.config.intermediate);
        // No ANE share on a pair (a rank's MLP columns are the GPU's); the assert keeps the branches from disagreeing.
        std.debug.assert(s.reduce == null or m.ane == null);
        const shared = if (m.ane) |sh| m.kernels.prompt and s.rows <= sh.capacity() else false;
        if (if (shared) m.ane else null) |share| {
            // ANE slices run the shared columns; the GPU adds its remaining columns.
            const used: u32 = (s.rows + ane_mlp.width - 1) / ane_mlp.width;
            const v = share.next(index, used);
            const planes = Ref{ .buffer = share.outputs[0].buffer };
            const own = share.columns / 64;
            const all = width / 64;
            for (0..used) |slice| try m.kernels.aneInput(e, f.get(.input), f.get(.dims), .{ .buffer = share.inputs[slice].buffer }, h, width, share.columns, @intCast(slice * ane_mlp.width));
            try share.signal(lane, v[0]);
            try m.kernels.quantSkip(lane.enc, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows, share.columns / 32);
            lane.enc.barrier();
            try m.kernels.mlpAne(lane.enc, f.get(.mixed), up, f.get(.dims), f.get(.activated), f.get(.act_sums), planes, h, width, share.columns, s.rows, own, all - own);
            lane.enc.barrier();
            try m.kernels.quantCut(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), f.get(.dims), .{ .buffer = share.partial }, s.rows, .{ own, all - own });
            share.wait(lane, v[1]);
            for (0..used) |slice| {
                const base: u32 = @intCast(slice * ane_mlp.width);
                try m.kernels.addAne(lane.enc, .{ .buffer = share.partial }, .{ .buffer = share.outputs[slice].buffer }, f.get(.output), h, width, share.columns, @min(ane_mlp.width, s.rows - base), base);
            }
            lane.enc.barrier();
            s.pending = f.get(.output);
            return;
        } else {
            try m.kernels.quant(e, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows);
            e.barrier();
            try m.kernels.mlp(e, f.get(.mixed), up, f.get(.dims), f.get(.activated), f.get(.act_sums), width, s.rows);
        }
        lane.enc.barrier();
        // Seam B: gate|up gives each rank its columns, so down_proj's full K needs the peer's; `.mlp` keeps it whole.
        var down_in = f.get(.activated);
        const acts = if (s.reduce) |r| r.next() else null;
        if (acts) |sm| {
            s.reduce.?.rowsIn(&e, sm, down_in, s.rows, width, width, f.get(.mixed), layer.down.k);
            lane.enc = e;
            if (s.reduce.?.broken()) return error.PairFailed;
            e.barrier();
            down_in = f.get(.mixed);
        }
        // The MLP is column-split on every pair (`.mlp` mode splits it too), so its seam is unconditional there.
        const split_down = layer.down.n != h;
        const seam_b = if (s.reduce != null and split_down) s.reduce.?.next() else null;
        // A gathered frame's group sums are the peer's to write, so the kernel derives them where there is no peer.
        const down_sums: ?@import("quant_gpu.zig").Sums = if (s.reduce == null) sums(f, .act_sums, s.rows) else null;
        if (seam_b) |sm| try m.kernels.quant(lane.enc, layer.down, down_in, down_sums, f.get(.dims), sm.ref, s.rows) else try m.kernels.quant(lane.enc, layer.down, down_in, down_sums, f.get(.dims), f.get(.output), s.rows);
        lane.enc.barrier();
        if (seam_b) |sm| {
            s.reduce.?.rowsOut(&e, sm, f.get(.output), s.rows, layer.down.n, h);
            lane.enc = e;
            if (s.reduce.?.broken()) return error.PairFailed;
            e.barrier();
        }
        s.pending = f.get(.output);
    }
    pub fn finish(s: *Stages, lane: *@import("core").segments.Lane) !void {
        const m = s.model;
        const f = &m.frame;
        // The head seam reopens the encoder, so it goes back in the lane; rank 0's half-vocab argmax is not a token.
        var e = lane.enc;
        try m.kernels.norm(e, s.hidden, s.pending, wts.ref(m.weights.norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), @intCast(m.config.hidden), s.rows);
        e.barrier();
        std.mem.swap(Ref, &s.hidden, &s.residual);
        try s.steer(lane, m.config.layers - 1);
        if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, m.config.layers - 1, s.hidden, s.rows);
        if (s.head == .all and s.rows > @import("gpu_frame.zig").logit_rows) return error.BadFrameRows;
        if (s.head != .none) {
            var input = f.get(.input);
            var head_sums = sums(f, .sums, s.rows);
            if (s.head == .last) {
                input.offset += @as(usize, s.rows - 1) * m.config.hidden * 2;
                head_sums.ref.offset += @as(usize, s.rows - 1) * 4;
            }
            try m.kernels.quant(e, m.weights.head, input, head_sums, f.get(.dims), f.get(.logits), if (s.head == .last) 1 else s.rows);
        }
        e.barrier();
        if (s.reduce) |r| {
            if (s.head != .none) {
                const picks = s.picks orelse return error.HeadSeamMissing;
                const seam = r.argmax orelse return error.HeadSeamMissing;
                const span = m.shard.vocabRows();
                const local_rows: u32 = if (s.head == .last) 1 else s.rows;
                seam(r.ptr, &e, r.nextX(), f.get(.logits), @intCast(m.config.vocab), @intCast(span.from), @intCast(span.len), local_rows, picks);
                lane.enc = e; // the head seam reopened the encoder; the caller commits this one
                if (r.broken()) return error.PairFailed;
            }
        }
        if (s.tail) |tail| try tail.encode(tail.ptr, e);
    }

    /// The control vector's direction for the layer whose residual just landed; none in force means nothing done.
    fn steer(s: *Stages, lane: *@import("core").segments.Lane, layer: usize) !void {
        const plan = s.model.steer orelse return;
        const dir = plan.dir(@intCast(layer)) orelse return;
        try s.model.kernels.steer(lane.enc, s.hidden, dir, plan.alpha, @intFromEnum(plan.mode), plan.hidden, s.rows);
        lane.enc.barrier();
    }
};

/// `e` is the caller's encoder, but seams reopen it: the final one is written back through the pointer, so commit it.
pub fn encode(m: *Model, e: *mtl.ComputeEncoder, rows: u32, mixer: Mixer) !void {
    var stages = Stages{ .model = m, .rows = rows, .hooks = mixer };
    try run(m, e, &stages);
}

/// The tensor-parallel entry point: the one-Mac program with each layer's all-reduces inserted behind `reduce`.
pub fn encodeTp(m: *Model, e: *mtl.ComputeEncoder, rows: u32, mixer: Mixer, reduce: Reduce) !void {
    var stages = Stages{ .model = m, .rows = rows, .hooks = mixer, .reduce = reduce };
    try run(m, e, &stages);
}

fn run(m: *Model, e: *mtl.ComputeEncoder, stages: *Stages) !void {
    // `cb`/`fence` stay undefined: a seam is the only closer, and it orders its writes through the transport's fence.
    var lane = @import("core").segments.Lane{ .k = 0, .cb = undefined, .enc = e.*, .fence = undefined };
    defer e.* = lane.enc;
    try stages.begin(&lane);
    for (0..m.config.layers) |index| {
        try stages.pre(&lane, index);
        try stages.mixer(&lane, index);
        try stages.post(&lane, index);
    }
    try stages.finish(&lane);
}

/// A norm's or the MLP gate's group sums for the frame's padded rows.
pub fn sums(f: *Frame, field: @import("gpu_frame.zig").Field, rows: u32) @import("quant_gpu.zig").Sums {
    return .{ .ref = f.get(field), .stride = @import("quant_gpu.zig").padded(rows) };
}
