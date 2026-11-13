//! The shared pass's MoE write-back: the slots' fp32 sum, the adds and the kept branch.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const LayerKeep = batch.LayerKeep;
const dims = wt.dims;
const slots = wt.slots;

/// After a layer's experts, when something adds to or keeps its output: the slots' fp32 sum, adds, kept branch.
pub fn moeAdds(self: anytype, layer: usize, keep: LayerKeep) !void {
    const canon = self.canon_f32;
    self.canon_f32 = false;
    const add = self.addAt(layer);
    if (add == null and keep.moe_branch == 0) return;
    const s = self.s;
    const n = self.rows;
    // moe_f32 may already hold the sum: the EXL3 canonical branch, or a prompt pass's rank exchange
    if (!canon and !self.pending_f32) try sumSlots(self, 0, n);
    if (add) |m| try m.add(m.ctx, layer, s.mixed, s.picks, s.wts, s.moe_f32, n);
    if (keep.moe_branch != 0) try self.sumSlices(s.moe_f32, keep.moe_branch, true, n * dims, 1);
    self.pending_f32 = true;
}

/// Canonical subtree rounded once from f64 into [rows, dims] f32.
pub fn leafTree(self: anytype, out: u64, start: usize, log: usize) !void {
    try leaf(self, self.k.leaf_tree, out, start, log);
}

/// The same subtree's f64 value, unrounded, into [rows, dims] f64: a rank's partial for the finish.
pub fn leafPartial(self: anytype, out: u64, start: usize, log: usize) !void {
    try leaf(self, self.k.leaf_partial, out, start, log);
}

fn leaf(self: anytype, kernel: cuda.Function, out: u64, start: usize, log: usize) !void {
    const s = self.s;
    var a: cuda.Args = .{};
    a.add(s.y);
    a.add(s.wts);
    a.add(s.picks);
    a.add(out);
    for ([_]usize{ dims, slots, start, log }) |v| a.add(batch.cint(v));
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = @intCast(self.rows), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
}

/// The pieces in `g` (piece i at i * stride floats) merged by `meta` into `out`.
pub fn leafFinish(self: anytype, g: u64, out: u64, meta: u64, pieces: usize, stride: usize) !void {
    var a: cuda.Args = .{};
    a.add(g);
    a.add(out);
    a.add(meta);
    a.add(batch.cint(pieces));
    a.add(batch.cint(dims));
    a.add(@as(i64, @intCast(stride)));
    try cuda.launch.launch(self.k.leaf_finish, .{ .grid = .{ .x = @intCast(self.rows), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
}

/// moe_f32 holds the canonical branch: the write-back reads it and moeAdds keeps it.
pub fn kept(self: anytype) void {
    self.canon_f32 = true;
    self.pending_f32 = true;
}

/// The pending MoE output (or none: mode 0, the streams' sums alone) written back into rows [r0, r0 + n).
pub fn writeback(self: anytype, r0: usize, n: usize) !void {
    const s = self.s;
    const dummy = s.dummy;
    const h = s.h + r0 * wt.wide * 2;
    const pss = s.pss + r0 * batch.pss_n * 4;
    const inj = s.inj + r0 * wt.streams * 2;
    const rs = self.rows * dims;
    if (self.pending == null) {
        try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", dummy, dummy, "*bf16", dummy, "*bf16", rs, n, dims, wt.streams, 0, 1, 1, 1);
        return;
    }
    self.pending = null;
    if (self.pending_f32) {
        // the fp32 branch rounded once, as MODE 2 rounds its sum (a slice of one)
        self.pending_f32 = false;
        if (self.pending_f32_weights) |weights| {
            self.pending_f32_weights = null;
            try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", inj, s.moe_f32 + r0 * dims * 4, "*fp32", weights + r0 * 4, "*fp32", rs, n, dims, wt.streams, 2, 0, 1, 1);
            return;
        }
        try self.tri.hcWriteback(h, h, pss, s.moe_f32 + r0 * dims * 4, "*fp32", inj, dummy, "*bf16", dummy, "*bf16", rs, n, dims, wt.streams, 4, 1, 1, 1);
        return;
    }
    const y = s.y + r0 * slots * dims * 4;
    const wts = s.wts + r0 * slots * 4;
    try self.tri.hcWriteback(h, h, pss, dummy, "*bf16", inj, y, "*fp32", wts, "*fp32", rs, n, dims, wt.streams, 2, 10, 11, 1);
}

/// moe_f32's rows [r0, r0 + rows): each row's slots of y times its weights, summed in slot order (fn_moe_sum).
pub fn sumSlots(self: anytype, r0: usize, rows: usize) !void {
    const s = self.s;
    var a: cuda.Args = .{};
    a.add(s.y + r0 * slots * dims * 4);
    a.add(s.wts + r0 * slots * 4);
    a.add(s.moe_f32 + r0 * dims * 4);
    a.add(batch.cint(dims));
    a.add(batch.cint(slots));
    try cuda.launch.launch(self.k.moe_sum, .{ .grid = .{ .x = @intCast(rows), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
}
