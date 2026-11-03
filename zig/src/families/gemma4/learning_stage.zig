//! Each analytic projection reads the checkpoint's original affine rows, never a backend's packed serving layout.
const std = @import("std");
const core = @import("core");
const matrix = @import("living").dense_matrix;
pub fn matrixBack(t: anytype, stem: []const u8, n: usize, k: usize, dy: anytype, dx: @TypeOf(dy), add: bool) !void {
    var arena: std.heap.ArenaAllocator = .init(t.a);
    defer arena.deinit();
    var views: [3]@TypeOf(dy) = undefined;
    var offset: usize = 0;
    const role = matrix.role(stem);
    inline for (.{ "weight", "scales", "biases" }, 0..) |leaf, i| {
        var name: [224]u8 = undefined;
        const tensor = try t.reader.get(try std.fmt.bufPrint(&name, "{s}.{s}", .{ stem, leaf }));
        const view = matrix.View.of(tensor);
        const bytes = if (role) |r| try matrix.base(arena.allocator(), t.shape.hidden, t.shape.intermediate, t.e.learned, r, @fromBackingInt(@intCast(i)), view) else view.bytes;
        const columns = k / (if (i == 0) @as(usize, 8) else 64);
        if (!view.valid or view.rank != 2 or view.words != (i == 0) or ((role == null or t.e.learned == 0) and (view.shape[0] != n or view.shape[1] != columns)) or bytes.len != n * columns * (if (i == 0) @as(usize, 4) else 2)) return error.GemmaTrainingWeight;
        views[i] = try t.buf(.stage).plus(offset, bytes.len);
        try t.write(views[i], bytes);
        offset = (offset + bytes.len + 15) & ~@as(usize, 15);
    }
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(n);
    p.u[2] = @intCast(k);
    p.u[3] = 1;
    p.u[4] = 64;
    p.u[5] = @intFromBool(add);
    try t.ops().row(.linear, p, &.{ .{ 0, dy }, .{ 1, views[0] }, .{ 2, views[1] }, .{ 3, views[2] }, .{ 8, dx } });
}
pub fn norm(t: anytype, stem: []const u8, input: anytype, dy: @TypeOf(input), out: @TypeOf(input), width: usize, groups: usize, weighted: bool) !void {
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(width);
    p.u[2] = @intCast(groups);
    p.u[3] = @intFromBool(weighted);
    p.f[0] = 1e-6;
    const w = try t.buf(.stage).plus(0, width * 2);
    if (weighted) {
        const tensor = try t.reader.get(stem);
        if (tensor.dtype != .bf16 or tensor.bytes.len != width * 2) return error.GemmaTrainingNorm;
        try t.write(w, tensor.bytes);
    }
    try t.ops().gemma(.rms, p, &.{ .{ 0, input }, .{ 1, w }, .{ 2, dy }, .{ 8, out } });
}
pub fn named(t: anytype, l: usize, suffix: []const u8, buffer: *[224]u8) ![]const u8 {
    _ = t;
    return std.fmt.bufPrint(buffer, "language_model.model.layers.{d}.{s}", .{ l, suffix });
}
