//! x3ld's arbitrary PDL forwarding requires a global-entry wait, not its later nested wait.
const std = @import("std");
const kernel = @import("pdl_kernel_scan");

test "EXL3 x3ld's arbitrary PDL forwarding requires its scanned global entry wait" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const host = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/src/cuda/exl3_block_exl3.zig", a, .limited(2 << 20));
    try std.testing.expect(std.mem.indexOf(u8, host, ".pdl = pdl") != null);
    try std.testing.expect(std.mem.indexOf(u8, host, "128, 0, pdl, &a") != null);
    const src = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/cuda/exl3_block/x3ld.cu", a, .limited(2 << 20));
    var definitions: std.ArrayList(kernel.Definition) = .empty;
    try kernel.definitions(a, src, &definitions);
    var selected: usize = 0;
    var missing: usize = 0;
    for (definitions.items) |d| {
        if (!std.mem.eql(u8, d.name, "ld_kernel")) continue;
        selected += 1;
        if (std.mem.startsWith(u8, d.opening, "pdl_wait();")) continue;
        std.debug.print("dependent x3ld definition of ld_kernel has no scanned global pdl_wait prologue\n", .{});
        missing += 1;
    }
    try std.testing.expect(selected >= 2);
    try std.testing.expectEqual(@as(usize, 0), missing);
}
