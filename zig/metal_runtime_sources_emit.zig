//! Emit exactly the source gate's libraries without acquiring a Metal device.
const std = @import("std");
const inventory = @import("inventory").metal_runtime_sources;
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedOutputDirectoryAndTensorUnitsZeroOrOne;
    const units = if (std.mem.eql(u8, args[2], "0")) false else if (std.mem.eql(u8, args[2], "1")) true else return error.InvalidTensorUnits;
    const a = init.arena.allocator();
    const set = try inventory.generate(a, units, if (units) 17 else 15);
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(init.io, args[1]);
    var manifest: std.Io.Writer.Allocating = .init(a);
    for (set.entries.items, 0..) |e, i| {
        const name = try std.fmt.allocPrint(a, "{d}.metal", .{i});
        try cwd.writeFile(init.io, .{ .sub_path = try std.fs.path.join(a, &.{ args[1], name }), .data = e.text });
        try manifest.writer.print("{s}\t{s}\t{s}\t{t}\t{t}\n", .{ name, e.family, e.name, e.options.math_mode, e.options.math_functions });
    }
    try cwd.writeFile(init.io, .{ .sub_path = try std.fs.path.join(a, &.{ args[1], "manifest.tsv" }), .data = manifest.written() });
}
