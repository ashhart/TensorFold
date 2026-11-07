//! The joint schema loader on generated checkpoints: the tiny head loads, each malformed one is refused by name.
const std = @import("std");
const load = @import("load.zig");
const tiny = @import("tiny.zig");
const config = @import("config.zig");

fn tmpPath(a: std.mem.Allocator, tmp: std.testing.TmpDir) ![]const u8 {
    return std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

test "joint schema load: the generated tiny head loads with every tensor checked" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tiny.write(a, io, tmp.dir, tiny.cfg, .none);
    var p: config.Problem = .{};
    var h = try load.open(std.testing.allocator, io, try tmpPath(a, tmp), &p);
    defer h.close(io);
    try std.testing.expect(h.weights.cfg.eql(tiny.cfg));
    try std.testing.expectEqual(@as(usize, 122), h.file.names.count());
    try std.testing.expectEqual(@as(usize, 4), h.weights.layers.len);
    try std.testing.expectEqual(@as(usize, 3 * 32), h.weights.evidence[1].attention.in_weight.dim(0));
}

test "joint schema load: missing, extra, non-BF16 and misshaped tensors and bad configs are refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const cases = [_]struct { tiny.Edit, anyerror, []const u8 }{
        .{ .{ .drop = "layers.2.norm3.bias" }, error.MissingTensor, "missing tensor layers.2.norm3.bias" },
        .{ .{ .drop = "residual_gate" }, error.MissingTensor, "missing tensor residual_gate" },
        .{ .{ .extra = "lm_head.weight" }, error.ExtraTensor, "unexpected tensor lm_head.weight" },
        .{ .{ .dtype = "memory_projection.weight" }, error.BadTensor, "memory_projection.weight is f32" },
        .{ .{ .shape = "evidence_layers.1.attention.in_proj_weight" }, error.BadTensor, "evidence_layers.1.attention.in_proj_weight is bf16" },
        .{ .{ .shape = "residual_gate" }, error.BadTensor, "residual_gate is bf16" },
        .{ .{ .shape = "type_embedding.weight" }, error.BadTensor, "type_embedding.weight" },
        .{ .{ .config = "{\"hidden_size\":128,\"width\":32,\"routing_layers\":2,\"layers\":4,\"heads\":5,\"feedforward\":128}" }, error.BadConfig, "not divisible by heads 5" },
        .{ .{ .config = "{\"hidden_size\":128,\"width\":32,\"routing_layers\":2,\"layers\":4,\"heads\":4}" }, error.BadConfig, "missing \"feedforward\"" },
        .{ .{ .config = "{\"hidden_size\":128,\"width\":32,\"routing_layers\":2,\"layers\":4,\"heads\":4,\"feedforward\":128,\"x\":1}" }, error.BadConfig, "\"x\"" },
    };
    for (cases) |c| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try tiny.write(a, io, tmp.dir, tiny.cfg, c[0]);
        var p: config.Problem = .{};
        try std.testing.expectError(c[1], load.open(std.testing.allocator, io, try tmpPath(a, tmp), &p));
        if (std.mem.indexOf(u8, p.text, c[2]) == null) {
            std.debug.print("refusal \"{s}\" lacks \"{s}\"\n", .{ p.text, c[2] });
            return error.TestUnexpectedResult;
        }
    }
    // a config whose layer counts differ from the file: the first absent layer is named
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var more = tiny.cfg;
    more.layers = 5;
    try tiny.write(a, io, tmp.dir, tiny.cfg, .{ .config = try std.fmt.allocPrint(a, "{{\"hidden_size\":128,\"width\":32,\"routing_layers\":2,\"layers\":{d},\"heads\":4,\"feedforward\":128}}", .{more.layers}) });
    var p: config.Problem = .{};
    try std.testing.expectError(error.MissingTensor, load.open(std.testing.allocator, io, try tmpPath(a, tmp), &p));
    try std.testing.expect(std.mem.indexOf(u8, p.text, "layers.4.norm1.weight") != null);
}

test "joint schema load: the release's joint_head.safetensors loads as Clef-Flash's head (TF_CLEF_DIR)" {
    const dir = std.c.getenv("TF_CLEF_DIR") orelse {
        std.debug.print("joint schema load: skipped, TF_CLEF_DIR names no clef-flash checkpoint\n", .{});
        return error.SkipZigTest;
    };
    const io = std.testing.io;
    var p: config.Problem = .{};
    var h = load.open(std.testing.allocator, io, std.mem.span(dir), &p) catch |e| {
        std.debug.print("joint schema load: {s}\n", .{p.text});
        return e;
    };
    defer h.close(io);
    try std.testing.expect(h.weights.cfg.eql(config.Config.clef_flash));
    try std.testing.expectEqual(@as(usize, 122), h.file.names.count());
    std.debug.print("joint schema load: {d} BF16 tensors of Clef-Flash's head checked\n", .{h.file.names.count()});
}
