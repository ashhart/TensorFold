const std = @import("std");
const st = @import("safetensors");
const diff = @import("sam3_tensor_diff.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print("tf-sam3-tensors --list CAPTURE.safetensors\ntf-sam3-tensors --values CAPTURE.safetensors KEY FIRST COUNT\ntf-sam3-tensors ACTUAL.safetensors EXPECTED.safetensors ACTUAL_KEY EXPECTED_KEY OUTPUT.json [options]\nReports every column; passes only finite, byte-exact tensors with identical dtype and shape.\n--leading-rows: the first rows of a taller expected tensor; --axis AXIS INDEX: one index of an expected axis\n--actual-axis AXIS INDEX: the same on the actual tensor; --flat: expected values in the actual shape (same count)\n", .{});
        return;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--list")) {
        var file = try st.File.open(init.gpa, init.io, args[2]);
        defer file.close(init.io);
        for (file.names.keys()) |name| {
            const tensor = file.get(name).?;
            std.debug.print("{s} {s} {any}\n", .{ name, @tagName(tensor.dtype), tensor.shape[0..tensor.rank] });
        }
        return;
    }
    if (args.len == 6 and std.mem.eql(u8, args[1], "--values")) {
        var file = try st.File.open(init.gpa, init.io, args[2]);
        defer file.close(init.io);
        const tensor = file.get(args[3]) orelse return error.MissingTensor;
        const first = try std.fmt.parseInt(usize, args[4], 10);
        const count = @min(try std.fmt.parseInt(usize, args[5], 10), tensor.numel() -| first);
        for (first..first + count) |i| std.debug.print("{d}: {e}\n", .{ i, diff.element(tensor, i) });
        const summary = diff.summarize(tensor);
        std.debug.print("min {e}, max {e}, positive {d}, zero {d}, negative {d}, nonzero strictly inside (min, max) {d} of {d}\n", .{ summary.minimum, summary.maximum, summary.positive, summary.zero, summary.negative, summary.interior, tensor.numel() });
        return;
    }
    if (args.len < 6) return error.ExpectedTwoFilesTwoKeysAndOutput;
    var actual = try st.File.open(init.gpa, init.io, args[1]);
    defer actual.close(init.io);
    var expected = try st.File.open(init.gpa, init.io, args[2]);
    defer expected.close(init.io);
    var x = actual.get(args[3]) orelse return error.MissingActualTensor;
    var y = expected.get(args[4]) orelse return error.MissingExpectedTensor;
    var at: usize = 6;
    while (at < args.len) : (at += 1) {
        if (std.mem.eql(u8, args[at], "--leading-rows")) {
            y = try diff.leadingRows(y, x);
        } else if (std.mem.eql(u8, args[at], "--axis") and at + 2 < args.len) {
            y = try diff.alongAxis(a, y, try std.fmt.parseInt(usize, args[at + 1], 10), try std.fmt.parseInt(usize, args[at + 2], 10));
            at += 2;
        } else if (std.mem.eql(u8, args[at], "--actual-axis") and at + 2 < args.len) {
            x = try diff.alongAxis(a, x, try std.fmt.parseInt(usize, args[at + 1], 10), try std.fmt.parseInt(usize, args[at + 2], 10));
            at += 2;
        } else if (std.mem.eql(u8, args[at], "--flat")) {
            y = try diff.reshapeLike(y, x);
        } else return error.UnknownComparisonOption;
    }
    const difference = try diff.analyze(a, x, y);
    const json = try std.json.Stringify.valueAlloc(a, .{
        .actual_key = args[3],
        .expected_key = args[4],
        .actual_dtype = @tagName(x.dtype),
        .expected_dtype = @tagName(y.dtype),
        .shape = x.shape[0..x.rank],
        .passed = difference.passed(),
        .difference = difference,
    }, .{ .whitespace = .indent_2 });
    const file = try std.Io.Dir.cwd().createFile(init.io, args[5], .{ .exclusive = true });
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, json);
    try file.writeStreamingAll(init.io, "\n");
    std.debug.print("{s}: {d}/{d} unequal values, {d} positivity changes, {d} nonfinite, maximum absolute {e:.6} in column {d}; byte-exact {}\n", .{
        args[3],                     difference.unequal_values, difference.values,      difference.sign_changes, difference.nonfinite_values,
        difference.maximum_absolute, difference.worst_column,   difference.exact_bytes,
    });
    if (!difference.passed()) return error.TensorComparisonFailed;
}

test {
    _ = diff;
}
