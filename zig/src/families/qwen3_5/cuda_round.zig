//! Shared lane trees staged for CUDA: full-width ancestor bits, separate chain depth and recycled register slots.
const std = @import("std");
const slots = @import("core").tree_slots;
const Seg = @import("cuda_cache.zig").Seg;
const Rule = @import("cuda").sample.Rule;
pub const Path = @import("cuda_cache.zig").Path;
pub const max_rows = 128;
pub const max_segs = 128;
pub const most_verify = 128;
pub const most_window = 64;
pub const Ancestor = extern struct { bits: u64, depth: u32, chain: u32 };
pub const ids_at = 0;
pub const segs_at = max_rows * 4;
pub const rowseg_at = segs_at + max_segs * @sizeOf(Seg);
pub const rowpar_at = rowseg_at + max_rows * 4;
pub const rowanc_at = rowpar_at + max_rows * 4;
pub const steps_at = rowanc_at + max_rows * @sizeOf(Ancestor);
pub const rules_at = steps_at + max_rows * @sizeOf(slots.Step);
pub const meta_at = rules_at + most_verify * @sizeOf(Rule);
pub const tokens_at = meta_at + most_verify * 4;
pub const params_at = tokens_at + most_verify * 4;
pub const src_at = params_at + 8 * 4;
pub const mpos_at = src_at + max_rows * 4;
pub const bytes = mpos_at + 3 * max_rows * 4;
pub fn ancestors(st: []u8) []Ancestor {
    return @alignCast(std.mem.bytesAsSlice(Ancestor, st[rowanc_at..][0 .. max_rows * @sizeOf(Ancestor)]));
}
pub fn steps(st: []u8) []slots.Step {
    return @alignCast(std.mem.bytesAsSlice(slots.Step, st[steps_at..][0 .. max_rows * @sizeOf(slots.Step)]));
}
pub fn variant(peak: usize) usize {
    return if (peak <= 1) 1 else if (peak <= 2) 2 else if (peak <= 4) 4 else if (peak <= 8) 8 else 16;
}
pub fn build(st: []u8, parts: []const @import("cuda_engine.zig").Part, record: bool) !usize {
    if (st.len < bytes or @intFromPtr(st.ptr) % @alignOf(Ancestor) != 0 or parts.len == 0 or parts.len > max_segs) return error.InvalidQwenSegments;
    const rowseg = std.mem.bytesAsSlice(i32, st[rowseg_at..][0 .. max_rows * 4]);
    const rowpar = std.mem.bytesAsSlice(i32, st[rowpar_at..][0 .. max_rows * 4]);
    const anc = ancestors(st);
    const schedule = steps(st);
    var size_buffer: [max_rows * @sizeOf(usize)]u8 align(@alignOf(usize)) = undefined;
    var sizes = std.heap.FixedBufferAllocator.init(&size_buffer);
    var first: usize = 0;
    var peak: usize = 1;
    for (parts, 0..) |p, segment| {
        if (p.rows == 0 or p.rows > max_rows - first or (record and p.rows > most_window)) return error.InvalidQwenSegments;
        if (p.parents) |ps| if (!record or ps.len != p.rows or ps[0] != -1) return error.InvalidQwenParents;
        var local: [max_rows]i32 = undefined;
        for (0..p.rows) |row| {
            const parent: i32 = if (p.parents) |ps| ps[row] else if (row == 0) -1 else @intCast(row - 1);
            if (parent < -1 or parent >= @as(i32, @intCast(row))) return error.InvalidQwenParents;
            local[row] = parent;
            rowseg[first + row] = @intCast(segment);
            rowpar[first + row] = if (parent < 0) -1 else @intCast(first + @as(usize, @intCast(parent)));
            const prior: Ancestor = if (parent < 0) .{ .bits = 0, .depth = 0, .chain = 0 } else anc[first + @as(usize, @intCast(parent))];
            anc[first + row] = .{ .bits = if (p.parents == null) 0 else prior.bits | (@as(u64, 1) << @intCast(row)), .depth = if (parent < 0) 0 else prior.depth + 1, .chain = @intFromBool(p.parents == null) };
        }
        if (record) {
            sizes.reset();
            peak = @max(peak, try slots.build(sizes.allocator(), local[0..p.rows], schedule[first..][0..p.rows], @intCast(first), slots.max_slots));
        }
        first += p.rows;
    }
    if (record and first > most_verify) return error.InvalidQwenSegments;
    return peak;
}
