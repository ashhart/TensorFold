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
    var a: cuda.Args = .{};
    a.add(s.y);
    a.add(s.wts);
    a.add(s.moe_f32);
    a.add(batch.cint(dims));
    a.add(batch.cint(slots));
    // The EXL3 branch already holds the canonical reduction in moe_f32.
    if (!canon) try cuda.launch.launch(self.k.moe_sum, .{ .grid = .{ .x = @intCast(n), .y = dims / 256 }, .block = .{ .x = 256 } }, self.stream, &a);
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

/// Keep the canonical fp32 branch for writeback and any branch consumer.
pub fn kept(self: anytype) void {
    self.canon_f32 = true;
    self.pending_f32 = true;
}
