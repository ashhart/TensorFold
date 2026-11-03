//! Read-only first-layer projection proof against the independent GGUF decoder; no device or model serving.
const std = @import("std");
const core = @import("core");
const preparation = @import("preparation");
const adapter = preparation.adapter;
fn float(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedGgufPath;
    var file = try core.gguf.File.open(init.gpa, init.io, args[1], .{});
    defer file.close(init.io);
    const doc = &file.document;
    var problem: adapter.Problem = .{};
    preparation.admit(doc, &problem) catch |err| {
        std.debug.print("REFUSED {s}\n", .{problem.message()});
        return err;
    };
    const c = try adapter.metadata.parse(doc, &problem);
    const plan = try adapter.names.plan(init.arena.allocator(), c);
    var matrices: usize = 0;
    var elements: usize = 0;
    for (plan) |name| {
        if (name.role != .matrix or !std.mem.startsWith(u8, name.gguf, "blk.0.")) continue;
        var arena = std.heap.ArenaAllocator.init(init.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const tensor = doc.getTensor(name.gguf).?;
        const repack = try core.gguf.affine.repack(a, tensor.kind, tensor.bytes);
        const t: adapter.Tensor = .{ .name = name, .source = tensor, .data = .{ .affine = .{ .shape = repack.shape, .words = repack.words, .scales = repack.scales, .biases = repack.biases } } };
        const matrix = try preparation.prepare(a, c, t, &problem);
        const geometry = core.gguf_repack_ref.geometry(tensor.kind);
        const n = name.dims[0];
        const k = name.dims[1];
        var different_weights: usize = 0;
        var different_rows: usize = 0;
        var worst: f64 = 0;
        for (0..n) |row| {
            var got: f64 = 0;
            var want: f64 = 0;
            for (0..k) |column| {
                const at = row * k + column;
                const order = preparation.layout.order(name.native);
                const source_at = preparation.layout.row(c, order, row) * k + preparation.layout.column(c, order, column);
                const block = tensor.bytes[source_at / geometry.values * geometry.bytes ..][0..geometry.bytes];
                const original: f32 = @floatCast(try core.gguf_repack_ref.value(tensor.kind, block, source_at % geometry.values));
                const code: f32 = @floatFromInt(if (matrix.params.format == .mlx4) (matrix.words[at / 2] >> @as(u3, @intCast(at % 2 * 4))) & 15 else matrix.words[at]);
                const group = at / @as(usize, @intCast(matrix.params.group));
                const value: f32 = @mulAdd(f32, float(matrix.scales, group), code, float(matrix.biases, group));
                different_weights += @intFromBool(@as(u32, @bitCast(value)) != @as(u32, @bitCast(original)));
                const x: f64 = @as(f64, @floatFromInt(@as(i32, @intCast(column % 23)) - 11)) / 32;
                got += @as(f64, value) * x;
                want += @as(f64, original) * x;
            }
            different_rows += @intFromBool(@as(u64, @bitCast(got)) != @as(u64, @bitCast(want)));
            worst = @max(worst, @abs(got - want));
        }
        std.debug.print("PROJECTION\t{s}\t{d}\t{d}\tweights_differ={d}\trows_differ={d}\tmax_abs={d}\n", .{ name.gguf, n, k, different_weights, different_rows, worst });
        if (different_weights != 0 or different_rows != 0) return error.GgufProjectionMismatch;
        matrices += 1;
        elements += n * k;
    }
    if (matrices != 8) return error.MissingFirstLayerProjection;
    std.debug.print("SUMMARY matrices={d} weights={d} bit_equal=true fp64_rows_equal=true\n", .{ matrices, elements });
}
