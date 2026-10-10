//! The layer program preserves Python's pending residual and only exposes post-block native tap rows.
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const Model = q.model.Model;
const wts = q.gpu_weights;
const Config = q.config.Config;
const Ref = q.projection.Ref;
const Frame = q.gpu_frame.Frame;

pub const Mixer = struct {
    ptr: *anyopaque,
    gdn: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Gdn, *Frame, u32) anyerror!void,
    attention: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Attention, *Frame, u32) anyerror!void,
    tap: ?*const fn (*anyopaque, mtl.ComputeEncoder, usize, Ref, u32) anyerror!void = null,
};

/// Each stage owns its frame and orders shared mixer caches.
pub const Head = enum { all, last, none };
pub const Stages = struct {
    model: *Model,
    rows: u32,
    hooks: Mixer,
    pending: ?Ref = null,
    head: Head = .all,
    pub fn begin(s: *Stages, lane: *@import("tensorfold").segments.Lane) !void {
        const m = s.model;
        if (s.rows == 0 or s.rows > m.frame.capacity or lane.k != 0) return error.BadFrameRows;
        try m.glue.embedding(lane.enc, m.config, m.weights.embed, m.frame.get(.ids), m.frame.get(.hidden), s.rows);
        lane.enc.barrier();
    }
    pub fn wait(_: *Stages, _: usize) @import("tensorfold").segments.Wait {
        return .mixer;
    }
    pub fn handoff(_: *Stages, _: *@import("tensorfold").segments.Lane, _: usize) !void {}
    pub fn pre(s: *Stages, lane: *@import("tensorfold").segments.Lane, index: usize) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        try m.kernels.norm(e, f.get(.hidden), s.pending, wts.ref(layer.norm), f.get(.eps), f.get(.dims), f.get(.residual), f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        if (s.pending != null) {
            try m.glue.unstack(e, f.get(.residual), f.get(.hidden), .{ .width = h, .stride = h, .offset = 0, .rows = s.rows });
            e.barrier();
            if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, index - 1, f.get(.hidden), s.rows);
        }
        if (layer.mixer == .linear) {
            try m.kernels.quant(e, layer.mixer.linear.qkv, f.get(.input), q.forward.sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows);
            e.barrier();
        }
    }
    pub fn mixer(s: *Stages, lane: *@import("tensorfold").segments.Lane, index: usize) !void {
        const m = s.model;
        switch (m.weights.layers[index].mixer) {
            .linear => |g| try s.hooks.gdn(s.hooks.ptr, lane.enc, index, g, &m.frame, s.rows),
            .attention => |a| try s.hooks.attention(s.hooks.ptr, lane.enc, index, a, &m.frame, s.rows),
        }
        lane.enc.barrier();
    }
    pub fn post(s: *Stages, lane: *@import("tensorfold").segments.Lane, index: usize) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        const out = switch (layer.mixer) {
            .linear => |g| g.out,
            .attention => |a| a.out,
        };
        try m.kernels.sum(e, f.get(.mixer_out), f.get(.dims), f.get(.sums), out.k, s.rows);
        e.barrier();
        try m.kernels.quant(e, out, f.get(.mixer_out), q.forward.sums(f, .sums, s.rows), f.get(.dims), f.get(.output), s.rows);
        e.barrier();
        try m.kernels.norm(e, f.get(.hidden), f.get(.output), wts.ref(layer.post_norm), f.get(.eps), f.get(.dims), f.get(.residual), f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        try m.kernels.quant(e, layer.gu, f.get(.input), q.forward.sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows);
        e.barrier();
        try splitGu(m, e, m.config, s.rows);
        e.barrier();
        try m.kernels.mlp(e, f.get(.gate), f.get(.up), f.get(.dims), f.get(.activated), f.get(.act_sums), @intCast(m.config.intermediate), s.rows);
        e.barrier();
        try m.kernels.quant(e, layer.down, f.get(.activated), q.forward.sums(f, .act_sums, s.rows), f.get(.dims), f.get(.output), s.rows);
        e.barrier();
        try m.glue.unstack(e, f.get(.residual), f.get(.hidden), .{ .width = h, .stride = h, .offset = 0, .rows = s.rows });
        e.barrier();
        s.pending = f.get(.output);
    }
    pub fn finish(s: *Stages, lane: *@import("tensorfold").segments.Lane) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        try m.kernels.norm(e, f.get(.hidden), s.pending, wts.ref(m.weights.norm), f.get(.eps), f.get(.dims), f.get(.residual), f.get(.input), f.get(.sums), @intCast(m.config.hidden), s.rows);
        e.barrier();
        if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, m.config.layers - 1, f.get(.residual), s.rows);
        if (s.head != .none) {
            var input = f.get(.input);
            var head_sums = q.forward.sums(f, .sums, s.rows);
            if (s.head == .last) {
                input.offset += @as(usize, s.rows - 1) * m.config.hidden * 2;
                head_sums.ref.offset += @as(usize, s.rows - 1) * 4;
            }
            try m.kernels.quant(e, m.weights.head, input, head_sums, f.get(.dims), f.get(.logits), if (s.head == .last) 1 else s.rows);
        }
        e.barrier();
    }
};

pub fn encode(m: *Model, e: mtl.ComputeEncoder, rows: u32, mixer: Mixer) !void {
    var stages = Stages{ .model = m, .rows = rows, .hooks = mixer };
    var lane = @import("tensorfold").segments.Lane{ .k = 0, .cb = undefined, .enc = e, .fence = undefined };
    try stages.begin(&lane);
    for (0..m.config.layers) |index| {
        try stages.pre(&lane, index);
        try stages.mixer(&lane, index);
        try stages.post(&lane, index);
    }
    try stages.finish(&lane);
}

fn splitGu(m: *Model, e: mtl.ComputeEncoder, c: Config, rows: u32) !void {
    const n: u32 = @intCast(c.intermediate);
    try m.glue.unstack(e, m.frame.get(.mixed), m.frame.get(.gate), .{ .rows = rows, .width = n, .stride = 2 * n, .offset = 0 });
    try m.glue.unstack(e, m.frame.get(.mixed), m.frame.get(.up), .{ .rows = rows, .width = n, .stride = 2 * n, .offset = n });
}
