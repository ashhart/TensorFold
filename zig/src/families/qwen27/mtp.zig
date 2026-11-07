//! The 27B's MTP draft head: the target's hidden row and the next token in, one block, its norm, the shared head out.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const forward = @import("forward.zig");
const wts = @import("gpu_weights.zig");
const Ref = @import("projection.zig").Ref;

/// The head's own buffers: the two normed planes `fc` reads side by side, and the gather's identity rows.
pub const Scratch = struct {
    planes: mtl.Buffer,
    rows: mtl.Buffer,
    capacity: u32,

    pub fn init(device: mtl.Device, capacity: u32, hidden: usize) !Scratch {
        const planes = try device.buffer(2 * @as(usize, capacity) * hidden * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer planes.deinit();
        const rows = try device.buffer(@as(usize, capacity) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        const ids: [*]u32 = @ptrCast(@alignCast(rows.contents()));
        for (0..capacity) |i| ids[i] = @intCast(i);
        return .{ .planes = planes, .rows = rows, .capacity = capacity };
    }
    pub fn deinit(s: *Scratch) void {
        s.planes.deinit();
        s.rows.deinit();
    }
};

/// One head pass over `rows` frame rows (ids and positions set): `hidden` row i pairs with token i; returns its rows.
pub fn encode(m: *Model, sc: *const Scratch, e: *mtl.ComputeEncoder, hooks: forward.Mixer, hidden: Ref, rows: u32, head: forward.Head) !Ref {
    const w = m.weights.mtp orelse return error.NoMtpHead;
    if (rows == 0 or rows > sc.capacity or rows > m.frame.capacity) return error.BadFrameRows;
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
    var s = forward.Stages{ .model = m, .rows = rows, .hooks = hooks, .head = head, .norm = w.norm, .hidden = f.get(.hidden), .residual = f.get(.residual) };
    const block = m.config.layers;
    try s.pre(&lane, block);
    try s.mixer(&lane, block);
    try s.post(&lane, block);
    try s.finish(&lane);
    return s.hidden;
}
