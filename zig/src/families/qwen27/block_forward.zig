//! A block-bidirectional block: full-attention rows see the whole block (and the cache before it); GDN stays causal.
const std = @import("std");
const mtl = @import("metal");
const forward = @import("forward.zig");
const decode_round = @import("decode_round.zig");
const Runner = decode_round.Runner;
const Round = @import("round_plan.zig").Round;
const GdnPlan = @import("gdn_plan.zig").Plan;
const attn = @import("attention.zig");
const Ref = @import("projection.zig").Ref;

/// Layer-major storage: each chunk's residual stream and pending output between layers (grown on demand).
pub const Store = struct {
    hidden: ?mtl.Buffer = null,
    pending: ?mtl.Buffer = null,
    rows: usize = 0,
    pub fn deinit(s: *Store) void {
        if (s.hidden) |b| b.deinit();
        if (s.pending) |b| b.deinit();
        s.* = .{};
    }
    fn ensure(s: *Store, device: mtl.Device, rows: usize, width: usize) !void {
        if (rows <= s.rows) return;
        s.deinit();
        s.hidden = try device.buffer(rows * width * 2, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        s.pending = try device.buffer(rows * width * 2, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
        s.rows = rows;
    }
};

/// `ids` into `slot` as one block (layer-major past `chunk` rows; causal if not `bidirectional`): last pass's rows.
pub fn run(r: *Runner, store: *Store, slot: u32, ids: []const u32, chunk: u32, bidirectional: bool) !u32 {
    const start = r.offsets[slot];
    if (ids.len == 0 or chunk == 0 or chunk > r.promptRows(chunk) or ids.len > r.capacity - start) return error.BadPrompt;
    if (r.model.ane != null or r.reduce != null or r.tail != null or r.capture_taps) return error.BlockUnsupported;
    const end: u32 = start + @as(u32, @intCast(ids.len));
    r.model.kernels.prompt = true;
    defer r.model.kernels.prompt = false;
    if (!bidirectional or ids.len <= chunk) {
        r.block_end = if (bidirectional) end else 0;
        defer r.block_end = 0;
        var first: usize = 0;
        while (true) {
            const rows = @min(chunk, ids.len - first);
            var round = try Round.init(r.allocator, &.{.{ .slot = slot, .start = start + @as(u32, @intCast(first)), .capacity = r.capacity, .ids = ids[first..][0..rows] }}, @intCast(r.model.config.conv_kernel), r.slots, @intCast(r.model.shard.global.vocab));
            defer round.deinit();
            try r.advance(&round, .none);
            first += rows;
            if (first == ids.len) return @intCast(rows);
        }
    }
    return layerMajor(r, store, slot, ids, chunk, start, end);
}

/// One chunk's plans, built once and replayed at every layer.
const Piece = struct {
    round: Round,
    gdn: GdnPlan,
    plan: attn.Plan,
    first: usize,
    rows: u32,
    fn deinit(p: *Piece) void {
        p.plan.deinit();
        p.gdn.deinit();
        p.round.deinit();
    }
};

fn layerMajor(r: *Runner, store: *Store, slot: u32, ids: []const u32, chunk: u32, start: u32, end: u32) !u32 {
    const a = r.allocator;
    const m = r.model;
    if (r.failed or r.active != null) return error.RoundNotReady;
    const h = m.config.hidden;
    try store.ensure(m.device, ids.len, h);
    const count = (ids.len + chunk - 1) / chunk;
    const pieces = try a.alloc(Piece, count);
    defer a.free(pieces);
    var made: usize = 0;
    defer for (pieces[0..made]) |*p| p.deinit();
    var accepted: [@import("gpu_frame.zig").max_rows]u32 = undefined;
    for (&accepted, 0..) |*v, i| v.* = @intCast(i);
    for (pieces, 0..) |*p, i| {
        const first = i * chunk;
        const rows: u32 = @intCast(@min(chunk, ids.len - first));
        p.first = first;
        p.rows = rows;
        p.round = try Round.init(a, &.{.{ .slot = slot, .start = start + @as(u32, @intCast(first)), .capacity = r.capacity, .ids = ids[first..][0..rows] }}, @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
        errdefer p.round.deinit();
        p.gdn = try p.round.recurrent(try r.params(rows), &.{accepted[0..rows]});
        errdefer p.gdn.deinit();
        p.plan = try r.attentionPlan(&p.round);
        made += 1;
    }
    r.accepted_chain = true;
    defer r.accepted_chain = false;
    defer {
        r.current_round = null;
        r.current_gdn = null;
        r.current_attention = null;
        r.current_placed = .{};
    }
    for (pieces) |*p| try pass(r, store, p, .embed, 0, end);
    for (m.weights.layers, 0..) |layer, index| switch (layer.mixer) {
        .linear => for (pieces) |*p| try pass(r, store, p, .layer, index, end),
        .attention => {
            for (pieces) |*p| try pass(r, store, p, .keys, index, end); // every chunk's keys before any row reads them
            for (pieces) |*p| try pass(r, store, p, .layer, index, end);
        },
    };
    try pass(r, store, &pieces[count - 1], .finish, 0, end);
    r.offsets[slot] = end;
    return pieces[count - 1].rows;
}

const Step = enum { embed, keys, layer, finish };

/// One chunk through one step in its own waited command buffer: the frame holds one chunk at a time.
fn pass(r: *Runner, store: *Store, p: *Piece, step: Step, index: usize, end: u32) !void {
    const m = r.model;
    const f = &m.frame;
    const h: u32 = @intCast(m.config.hidden);
    try r.gdn.prepare(m.device, p.gdn, true);
    try f.setRows(p.round.ids, p.round.feed_positions);
    r.current_placed = @import("image_rows.zig").place(f, &r.rope, null, m.config.mrope_section, &p.round);
    try r.scratch.upload(p.plan);
    r.current_round = &p.round;
    r.current_gdn = &p.gdn;
    r.current_attention = &p.plan;
    r.block_end = if (step == .keys) 0 else end;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.concurrent);
    var lane = @import("core").segments.Lane{ .k = 0, .cb = cb, .enc = e, .fence = undefined };
    const at = p.first * @as(usize, h) * 2;
    const saved_hidden = Ref{ .buffer = store.hidden.?, .offset = at };
    const saved_pending = Ref{ .buffer = store.pending.?, .offset = at };
    const copy = @import("glue_gpu.zig").Slice{ .rows = p.rows, .width = h, .stride = h, .offset = 0 };
    var stages = forward.Stages{ .model = m, .rows = p.rows, .hooks = .{ .ptr = r, .gdn = Runner.gdnForward, .attention = Runner.attentionForward }, .head = .none };
    if (step == .embed) {
        try stages.begin(&lane);
        try m.glue.unstack(lane.enc, stages.hidden, saved_hidden, copy);
    } else {
        try m.glue.unstack(lane.enc, saved_hidden, f.get(.hidden), copy);
        try m.glue.unstack(lane.enc, saved_pending, f.get(.output), copy);
        lane.enc.barrier();
        stages.hidden = f.get(.hidden);
        stages.residual = f.get(.residual);
        stages.pending = if (index == 0 and step != .finish) null else f.get(.output);
        switch (step) {
            .keys => {
                try stages.pre(&lane, index);
                try stages.mixer(&lane, index);
            },
            .layer => {
                try stages.pre(&lane, index);
                try stages.mixer(&lane, index);
                try stages.post(&lane, index);
                try m.glue.unstack(lane.enc, stages.hidden, saved_hidden, copy);
                try m.glue.unstack(lane.enc, stages.pending.?, saved_pending, copy);
            },
            .finish => try stages.finish(&lane),
            .embed => unreachable,
        }
    }
    lane.enc.barrier();
    lane.enc.end();
    cb.commit();
    cb.wait();
    if (decode_round.gpuFailed(cb, "a block-bidirectional pass")) {
        r.failed = true;
        return error.RoundGpuFailure;
    }
}

/// The last row of the frame's `.input` (rows `rows`) as float32: the final norm's output the readout reads.
pub fn lastHidden(r: *const Runner, rows: u32, out: []f32) !void {
    const h = r.model.config.hidden;
    if (rows == 0 or out.len != h) return error.BadReadout;
    const words = r.model.frame.get(.input).buffer.slice(u16, @as(usize, rows) * h);
    for (out, words[(rows - 1) * h ..][0..h]) |*o, bits| o.* = @bitCast(@as(u32, bits) << 16);
}
