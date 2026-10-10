//! The layer program preserves Python's pending residual and only exposes post-block native tap rows.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const wts = @import("gpu_weights.zig");
const Config = @import("config.zig").Config;
const Ref = @import("projection.zig").Ref;
const Frame = @import("gpu_frame.zig").Frame;
const ane_mlp = @import("ane_mlp.zig");

pub const Mixer = struct {
    ptr: *anyopaque,
    gdn: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Gdn, *Frame, u32) anyerror!void,
    attention: *const fn (*anyopaque, mtl.ComputeEncoder, usize, wts.Attention, *Frame, u32) anyerror!void,
    tap: ?*const fn (*anyopaque, mtl.ComputeEncoder, usize, Ref, u32) anyerror!void = null,
};

/// Serial decode and prompt segments own separate frames and order shared caches at the mixer.
pub const Head = enum { all, last, none };
pub const Stages = struct {
    model: *Model,
    rows: u32,
    hooks: Mixer,
    pending: ?Ref = null,
    hidden: Ref = undefined,
    residual: Ref = undefined,
    head: Head = .all,
    pub fn begin(s: *Stages, lane: *@import("core").segments.Lane) !void {
        const m = s.model;
        if (s.rows == 0 or s.rows > m.frame.capacity or lane.k != 0) return error.BadFrameRows;
        s.hidden = m.frame.get(.hidden);
        s.residual = m.frame.get(.residual);
        try m.glue.embedding(lane.enc, m.config, m.weights.embed, m.frame.get(.ids), s.hidden, s.rows);
        lane.enc.barrier();
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
        const e = lane.enc;
        const layer = m.weights.layers[index];
        const h: u32 = @intCast(m.config.hidden);
        const out = switch (layer.mixer) {
            .linear => |g| g.out,
            .attention => |a| a.out,
        };
        try m.kernels.quant(e, out, f.get(.mixer_out), null, f.get(.dims), f.get(.output), s.rows);
        e.barrier();
        try m.kernels.norm(e, s.hidden, f.get(.output), wts.ref(layer.post_norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), h, s.rows);
        e.barrier();
        std.mem.swap(Ref, &s.hidden, &s.residual);
        var up = f.get(.mixed);
        up.offset += @as(usize, m.config.intermediate) * 2;
        const width: u32 = @intCast(m.config.intermediate);
        if (if (m.kernels.prompt) m.ane else null) |share| {
            // the Neural Engine runs MLP columns below share.columns; the GPU runs the rest, then adds its partial
            const v = share.next(index);
            const planes = Ref{ .buffer = share.output.buffer };
            const own = share.columns / 64;
            const all = width / 64;
            try m.kernels.aneInput(e, f.get(.input), f.get(.dims), .{ .buffer = share.input.buffer }, h, width, share.columns);
            try share.signal(lane, v[0]);
            try m.kernels.quantSkip(lane.enc, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows, share.columns / 32);
            lane.enc.barrier();
            try m.kernels.mlpAne(lane.enc, f.get(.mixed), up, f.get(.dims), f.get(.activated), f.get(.act_sums), planes, h, width, share.columns, s.rows, own, all - own);
            lane.enc.barrier();
            try m.kernels.quantCut(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), f.get(.dims), .{ .buffer = share.partial }, s.rows, .{ own, all - own });
            share.wait(lane, v[1]);
            try m.kernels.addAne(lane.enc, .{ .buffer = share.partial }, planes, f.get(.output), h, width, share.columns, s.rows);
            lane.enc.barrier();
            s.pending = f.get(.output);
            return;
        } else {
            try m.kernels.quant(e, layer.gu, f.get(.input), sums(f, .sums, s.rows), f.get(.dims), f.get(.mixed), s.rows);
            e.barrier();
            try m.kernels.mlp(e, f.get(.mixed), up, f.get(.dims), f.get(.activated), f.get(.act_sums), width, s.rows);
        }
        lane.enc.barrier();
        try m.kernels.quant(lane.enc, layer.down, f.get(.activated), sums(f, .act_sums, s.rows), f.get(.dims), f.get(.output), s.rows);
        lane.enc.barrier();
        s.pending = f.get(.output);
    }
    pub fn finish(s: *Stages, lane: *@import("core").segments.Lane) !void {
        const m = s.model;
        const f = &m.frame;
        const e = lane.enc;
        try m.kernels.norm(e, s.hidden, s.pending, wts.ref(m.weights.norm), f.get(.eps), f.get(.dims), s.residual, f.get(.input), f.get(.sums), @intCast(m.config.hidden), s.rows);
        e.barrier();
        std.mem.swap(Ref, &s.hidden, &s.residual);
        if (s.hooks.tap) |tap| try tap(s.hooks.ptr, e, m.config.layers - 1, s.hidden, s.rows);
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
    }
};

pub fn encode(m: *Model, e: mtl.ComputeEncoder, rows: u32, mixer: Mixer) !void {
    var stages = Stages{ .model = m, .rows = rows, .hooks = mixer };
    var lane = @import("core").segments.Lane{ .k = 0, .cb = undefined, .enc = e, .fence = undefined };
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
