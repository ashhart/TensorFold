const std = @import("std");
// A command-line gate must not free stack-owned drivers after a failed completion wait.
pub fn finish(owner: anytype) void {
    owner.close() catch |err| {
        std.log.err("CUDA close failed ({t}); retaining owners until process exit", .{err});
        std.process.exit(1);
    };
}
