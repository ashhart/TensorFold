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
    var table: std.ArrayList([]const u8) = .empty;
    try kernel.prologues(a, src, "pdl_wait();", &table);
    if (!kernel.contains(table.items, "ld_kernel")) {
        std.debug.print("dependent x3ld launch of ld_kernel has no scanned global pdl_wait prologue\n", .{});
        try std.testing.expect(false);
    }
}
