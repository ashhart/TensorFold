const std = @import("std");
const st = @import("safetensors");
const diff = @import("sam3_tensor_diff.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print("tf-sam3-tensors ACTUAL.safetensors EXPECTED.safetensors ACTUAL_KEY EXPECTED_KEY OUTPUT.json\nReports every column; passes only finite, byte-exact tensors with identical dtype and shape.\n", .{});
        return;
    }
    if (args.len != 6) return error.ExpectedTwoFilesTwoKeysAndOutput;
    var actual = try st.File.open(init.gpa, init.io, args[1]);
    defer actual.close(init.io);
    var expected = try st.File.open(init.gpa, init.io, args[2]);
    defer expected.close(init.io);
    const x = actual.get(args[3]) orelse return error.MissingActualTensor;
    const y = expected.get(args[4]) orelse return error.MissingExpectedTensor;
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
    std.debug.print("{s}: {d}/{d} unequal values, {d} nonfinite, maximum absolute {e:.6} in column {d}; byte-exact {}\n", .{
        args[3],                     difference.unequal_values, difference.values,      difference.nonfinite_values,
        difference.maximum_absolute, difference.worst_column,   difference.exact_bytes,
    });
    if (!difference.passed()) return error.TensorComparisonFailed;
}

test {
    _ = diff;
}
