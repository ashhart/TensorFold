//! The 27B's MTP draft head: the target's hidden row and the next token in, one block, its norm, the shared head out.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const forward = @import("forward.zig");
const wts = @import("gpu_weights.zig");
const Ref = @import("projection.zig").Ref;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// The head's own buffers: the normed planes `fc` reads side by side, the gather's identity rows, its logits rows.
pub const Scratch = struct {
    planes: mtl.Buffer,
    rows: mtl.Buffer,
    logits: mtl.Buffer,
    capacity: u32,
    logit_rows: u32,

    pub fn init(device: mtl.Device, capacity: u32, hidden: usize, vocab: usize, logit_rows: u32) !Scratch {
        const planes = try device.buffer(2 * @as(usize, capacity) * hidden * 2, opts);
        errdefer planes.deinit();
        const rows = try device.buffer(@as(usize, capacity) * 4, opts);
        errdefer rows.deinit();
        const logits = try device.buffer(@as(usize, logit_rows) * vocab * 2, opts);
        const ids: [*]u32 = @ptrCast(@alignCast(rows.contents()));
        for (0..capacity) |i| ids[i] = @intCast(i);
        return .{ .planes = planes, .rows = rows, .logits = logits, .capacity = capacity, .logit_rows = logit_rows };
    }
    pub fn deinit(s: *Scratch) void {
        s.planes.deinit();
        s.rows.deinit();
        s.logits.deinit();
    }
};

/// A logits row's pick and its probability under the head's softmax.
pub const Pick = struct { token: u32, chance: f32 };
pub fn pick(values: []const u16) !Pick {
    const token = try @import("session.zig").argmax(values);
    const top: f32 = @bitCast(@as(u32, values[token]) << 16);
    var sum: f64 = 0;
    for (values) |v| sum += @exp(@as(f64, @as(f32, @bitCast(@as(u32, v) << 16)) - top));
    return .{ .token = token, .chance = @floatCast(1 / sum) };
}

/// One head pass over `rows` frame rows (ids and positions set): `hidden` row i pairs with token i; returns its rows.
pub fn encode(m: *Model, sc: *const Scratch, e: *mtl.ComputeEncoder, hooks: forward.Mixer, hidden: Ref, rows: u32, head: forward.Head) !Ref {
    const w = m.weights.mtp orelse return error.NoMtpHead;
    if (rows == 0 or rows > sc.capacity or rows > m.frame.capacity or (if (head == .all) rows else 1) > sc.logit_rows) return error.BadFrameRows;
    const f = &m.frame;
    const h: u32 = @intCast(m.config.hidden);
    const plane = @as(usize, sc.capacity) * h * 2;
    try m.glue.embedding(e.*, m.config, m.weights.embed, f.get(.ids), f.get(.output), rows);
    e.barrier();
    try m.glue.headNorm(e.*, f.get(.output), wts.ref(w.norm_embed), .{ .buffer = sc.planes }, h, rows, m.config.eps);
    try m.glue.headNorm(e.*, hidden, wts.ref(w.norm_hidden), .{ .buffer = sc.planes, .offset = plane }, h, rows, m.config.eps);
    e.barrier();
    try m.glue.tapRows(e.*, .{ .buffer = sc.planes }, .{ .buffer = sc.rows }, f.get(.mixed), .{ .rows = rows, .width = h, .capacity = sc.capacity, .planes = 2 });
    e.barrier();
    try m.kernels.quant(e.*, w.fc, f.get(.mixed), null, f.get(.dims), f.get(.hidden), rows);
    e.barrier();
    var lane = @import("core").segments.Lane{ .k = 0, .cb = undefined, .enc = e.*, .fence = undefined };
    defer e.* = lane.enc;
    var s = forward.Stages{ .model = m, .rows = rows, .hooks = hooks, .head = head, .norm = w.norm, .logits = .{ .buffer = sc.logits }, .hidden = f.get(.hidden), .residual = f.get(.residual) };
    const block = m.config.layers;
    try s.pre(&lane, block);
    try s.mixer(&lane, block);
    try s.post(&lane, block);
    try s.finish(&lane);
    return s.hidden;
}
