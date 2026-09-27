//! Alternating capacity buffers. Forward writes may consume a spare but never
//! change the committed prefix. Write records borrow Scope handles; Buffer owns
//! its handles. Commit retains the previous current buffer as the next spare.
const std = @import("std");
const mx = @import("mlx.zig");
pub var enabled = true;
pub var track_reuse = false;
pub var attempted: usize = 0;
pub var reused: usize = 0;
// Pointer ABI only: Zig cannot translate Clang's arm __bf16 element typedef.
extern "mlxc" fn mlx_array_data_bfloat16(mx.Array) [*c]const u16;
pub fn address(array: mx.Array) usize {
    return @intFromPtr(mlx_array_data_bfloat16(array));
}
pub fn observe(write: Write) !void {
    if (!track_reuse or write.donor == 0) return;
    try mx.eval(write.capacity);
    attempted += 1;
    if (address(write.capacity) == write.donor) reused += 1;
}
pub const Write = struct {
    capacity: mx.Array = mx.empty,
    added: mx.Array = mx.empty,
    view: mx.Array = mx.empty,
    start: i32 = 0,
    rows: i32 = 0,
    axis: usize = 2,
    donor: usize = 0,
};
pub const Buffer = struct {
    current: mx.Array = mx.empty,
    spare: mx.Array = mx.empty,
    recent: mx.Array = mx.empty,
    offset: i32 = 0,
    spare_end: i32 = 0,
    axis: usize = 2,
    pub fn deinit(b: *Buffer) void {
        mx.free(b.current);
        mx.free(b.spare);
        mx.free(b.recent);
        b.* = .{};
    }
    pub fn clone(b: Buffer) !Buffer {
        var out = Buffer{ .offset = b.offset, .spare_end = b.spare_end, .axis = b.axis };
        errdefer out.deinit();
        inline for (.{ "current", "spare", "recent" }) |field| {
            const value = @field(b, field);
            if (value.ctx != null) @field(out, field) = try mx.retain(value);
        }
        return out;
    }
    /// Returned handles belong to s, including capacity handles that are not sliced.
    pub fn prefix(b: Buffer, s: *mx.Scope, end: i32) !Buffer {
        if (b.current.ctx == null) return .{};
        if (end < 0 or end > b.offset) return error.InvalidCachePrefix;
        var out = Buffer{ .offset = end, .axis = b.axis, .spare_end = @min(b.spare_end, end) };
        out.current = try s.own(try mx.retain(b.current));
        if (b.spare.ctx != null and end >= b.spare_end) {
            out.spare = try s.own(try mx.retain(b.spare));
            if (end > b.spare_end) out.recent = try s.slice(b.recent, b.axis, 0, end - b.spare_end);
        }
        return out;
    }
    fn grown(source: mx.Array, valid: i32, end: i32, axis: usize, shape: []const i32, dtype: mx.c.mlx_dtype) !mx.Array {
        var s = mx.Scope{};
        defer s.deinit();
        var dims: [8]i32 = undefined;
        @memcpy(dims[0..shape.len], shape);
        dims[axis] = std.mem.alignForward(i32, end, 2048) - valid;
        const zeros = try s.zeros(dims[0..shape.len], dtype);
        const out = if (valid == 0) zeros else try s.cat(&.{ try s.slice(source, axis, 0, valid), zeros }, @intCast(axis));
        return mx.retain(out);
    }
    pub fn append(b: *Buffer, s: *mx.Scope, base: mx.Array, added: mx.Array, axis: usize) !Write {
        const shape = mx.shape(added);
        if (shape.len > 8 or axis >= shape.len or shape[axis] <= 0) return error.InvalidCacheShape;
        const start = if (base.ctx == null) 0 else mx.dim(base, @intCast(axis));
        if (b.current.ctx != null and (b.offset != start or b.axis != axis)) return error.InvalidCacheOffset;
        const rows = shape[axis];
        const end = try std.math.add(i32, start, rows);
        if (end > std.math.maxInt(i32) - 2047) return error.InvalidCacheShape;
        var at = start;
        var fill = added;
        var target = mx.empty;
        var donor: usize = 0;
        defer mx.free(target);
        if (b.spare.ctx != null) {
            at = b.spare_end;
            if (at < 0 or at > start) return error.InvalidCacheOffset;
            if (at < start) {
                if (b.recent.ctx == null or mx.dim(b.recent, @intCast(axis)) != start - at) return error.InvalidCacheOffset;
                fill = try s.cat(&.{ b.recent, added }, @intCast(axis));
            }
            target = b.spare;
            b.spare = mx.empty;
            if (track_reuse and mx.dim(target, @intCast(axis)) >= end) {
                try mx.eval(target);
                donor = address(target);
                if (donor == 0) return error.InvalidBufferAddress;
            }
        } else if (b.current.ctx != null) {
            target = try mx.retain(b.current);
        } else {
            target = try grown(base, start, end, axis, shape, mx.dtype(added));
        }
        if (mx.dim(target, @intCast(axis)) < end) {
            const next = try grown(target, at, end, axis, shape, mx.dtype(added));
            mx.free(target);
            target = next;
        }
        const index = try s.ints(&.{at});
        var output = mx.c.mlx_array_new();
        const ax: i32 = @intCast(axis);
        const rc = mx.c.mlx_slice_update_dynamic(&output, target, fill, index, &ax, 1, mx.stream);
        const capacity = try s.result(rc, output);
        // target is released before evaluation, allowing MLX to donate its storage.
        return .{ .capacity = capacity, .added = added, .view = try s.slice(capacity, axis, 0, end), .start = start, .rows = rows, .axis = axis, .donor = donor };
    }
    /// Owned replacement state. The Write never retains the old current capacity:
    /// doing so would keep next round's donor alive in a pipelined forward scope.
    pub fn finish(b: Buffer, s: *mx.Scope, write: Write, keep: i32) !Buffer {
        if (write.capacity.ctx == null) return .{};
        if (keep < 0 or keep > write.rows or (b.current.ctx != null and b.offset != write.start)) return error.InvalidCachePrefix;
        var out = Buffer{ .offset = write.start + keep, .spare_end = write.start, .axis = write.axis };
        errdefer out.deinit();
        out.current = try mx.retain(write.capacity);
        if (b.current.ctx != null) out.spare = try mx.retain(b.current);
        if (keep > 0) out.recent = try mx.retain(try s.slice(write.added, write.axis, 0, keep));
        return out;
    }
};
