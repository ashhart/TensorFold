const std = @import("std");
const cuda = @import("cuda");
pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 4) return error.ExpectedChunkedAndSerialFatbins;
    const image = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], init.arena.allocator(), .limited(1 << 28));
    const serial = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.arena.allocator(), .limited(1 << 28));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    const tile = if (args.len == 4) try std.fmt.parseInt(usize, args[3], 10) else 0;
    try @import("contract_gdn_chunked.zig").runTensor(.{ .d = &driver, .ctx = &context, .gpa = init.gpa, .io = init.io }, image, serial, tile);
    return 0;
}
