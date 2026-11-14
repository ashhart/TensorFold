const std = @import("std");
const st = @import("safetensors");
const ops = @import("ops.zig");
const Store = @import("checkpoint.zig").Store;

const Stage = struct {
    name: []const u8,
    values: usize,
    unequal_values: usize,
    native_maximum: f64,
    native_mean: f64,
    reference_maximum: f64,
    reference_mean: f64,
    native_columns: []f64,
    reference_columns: []f64,
    worse_columns: usize,
    worst_index: usize = 0,
    worst_native: f64 = 0,
    worst_reference: f64 = 0,
    worst_truth: f64 = 0,
};

const Capture = struct {
    allocator: std.mem.Allocator,
    reference: *st.File,
    stages: std.ArrayList(Stage) = .empty,

    fn observe(context: *anyopaque, name: []const u8, tensor: ops.Tensor, shape: []const usize) !void {
        const self: *Capture = @ptrCast(@alignCast(context));
        const a = self.allocator;
        const fp32 = self.reference.get(try std.fmt.allocPrint(a, "fp32.{s}", .{name})) orelse return error.MissingTextReference;
        const fp64 = self.reference.get(try std.fmt.allocPrint(a, "fp64.{s}", .{name})) orelse return error.MissingTextOracle;
        if (!fp32.is(.f32, shape) or !fp64.is(.f64, shape)) return error.TextReferenceShape;
        try tensor.checkFinite(name);
        const columns = shape[shape.len - 1];
        var report = Stage{
            .name = try a.dupe(u8, name),
            .values = tensor.count,
            .unequal_values = 0,
            .native_maximum = 0,
            .native_mean = 0,
            .reference_maximum = 0,
            .reference_mean = 0,
            .native_columns = try a.alloc(f64, columns),
            .reference_columns = try a.alloc(f64, columns),
            .worse_columns = 0,
        };
        @memset(report.native_columns, 0);
        @memset(report.reference_columns, 0);
        for (0..tensor.count) |index| {
            const truth: f64 = @bitCast(std.mem.readInt(u64, fp64.bytes[index * 8 ..][0..8], .little));
            const expected: f32 = @bitCast(std.mem.readInt(u32, fp32.bytes[index * 4 ..][0..4], .little));
            const native = tensor.get(index);
            if (!std.math.isFinite(truth) or !std.math.isFinite(expected)) return error.NonFiniteTextReference;
            const native_error = @abs(native - truth);
            const reference_error = @abs(expected - truth);
            if (native_error > report.native_maximum) {
                report.worst_index = index;
                report.worst_native = native;
                report.worst_reference = expected;
                report.worst_truth = truth;
            }
            report.unequal_values += @intFromBool(@as(u32, @bitCast(native)) != @as(u32, @bitCast(expected)));
            report.native_maximum = @max(report.native_maximum, native_error);
            report.reference_maximum = @max(report.reference_maximum, reference_error);
            report.native_mean += native_error;
            report.reference_mean += reference_error;
            const column = index % columns;
            report.native_columns[column] = @max(report.native_columns[column], native_error);
            report.reference_columns[column] = @max(report.reference_columns[column], reference_error);
        }
        report.native_mean /= @floatFromInt(tensor.count);
        report.reference_mean /= @floatFromInt(tensor.count);
        for (report.native_columns, report.reference_columns) |native, reference| report.worse_columns += @intFromBool(native > reference);
        std.debug.print("{s}: FP64 max native {e:.6}, reference {e:.6}; {d}/{d} columns worse\n", .{ name, report.native_maximum, report.reference_maximum, report.worse_columns, columns });
        try self.stages.append(a, report);
    }
};

pub fn run(init: std.process.Init, g: *@import("tensor_metal").Gpu, args: []const [:0]const u8) !void {
    if (args.len != 3 and args.len != 4) return error.ExpectedCheckpointReferenceAndReceipt;
    const compensated = args.len == 4 and std.mem.eql(u8, args[3], "--compensated-mlp");
    if (args.len == 4 and !compensated) return error.UnexpectedTextProjectionBackend;
    const a = init.arena.allocator();
    var store = try Store.open(init.gpa, init.io, args[0]);
    defer store.deinit();
    var reference = try st.File.open(init.gpa, init.io, args[1]);
    defer reference.close(init.io);
    const tokens = reference.get("tokens") orelse return error.MissingTextTokens;
    if (tokens.dtype != .i64 or tokens.rank != 2 or tokens.dim(0) != 1 or tokens.dim(1) == 0 or tokens.dim(1) > 32) return error.TextReferenceShape;
    const ids = try a.alloc(u32, tokens.dim(1));
    for (ids, 0..) |*id, index| {
        const value: i64 = @bitCast(std.mem.readInt(u64, tokens.bytes[index * 8 ..][0..8], .little));
        id.* = std.math.cast(u32, value) orelse return error.SamTokenOutOfRange;
    }
    var capture = Capture{ .allocator = a, .reference = &reference };
    var runtime = ops.Runtime{
        .g = g,
        .store = &store,
        .a = a,
        .config = try store.config(),
        .precision = .fp32,
        .observer = .{ .context = &capture, .write = Capture.observe },
    };
    defer runtime.deinit();
    _ = try @import("text_encoder.zig").encodeWithMlp(&runtime, ids, compensated);
    var passed = capture.stages.items.len == 33;
    for (capture.stages.items) |stage| passed = passed and stage.worse_columns == 0 and stage.native_mean <= stage.reference_mean;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(reference.map.memory, &digest, .{});
    const source_hash = std.fmt.bytesToHex(digest, .lower);
    const report = try std.json.Stringify.valueAlloc(a, .{ .scope = "diagnostic text-only replay", .reference = "external Meta CPU FP32 and CPU FP64 on identical token IDs", .capture_sha256 = source_hash[0..], .compensated_mlp = compensated, .passed = passed, .stages = capture.stages.items }, .{ .whitespace = .indent_2 });
    const file = try std.Io.Dir.cwd().createFile(init.io, args[2], .{ .exclusive = true });
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, report);
    try file.writeStreamingAll(init.io, "\n");
    if (!passed) return error.TextReferenceGateFailed;
}
