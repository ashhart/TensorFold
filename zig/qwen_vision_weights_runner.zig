//! A small named runner that enforces both allocation balance and the fixture's explicit checkpoint diagnostics.
const std = @import("std");
const builtin = @import("builtin");
const diagnostics = @import("qwen_weights_diagnostics");
pub const std_options: std.Options = .{ .logFn = diagnostics.log };

pub fn main(init: std.process.Init.Minimal) void {
    var failed: usize = 0;
    var skipped: usize = 0;
    for (builtin.test_functions, 1..) |test_fn, i| {
        std.testing.allocator_instance = .init(std.heap.page_allocator, .{ .check_write_after_free = true });
        std.testing.io_instance = .init(std.testing.allocator, .{ .argv0 = .init(init.args), .environ = init.environ });
        std.testing.environ = init.environ;
        std.debug.print("{d}/{d} {s}...", .{ i, builtin.test_functions.len, test_fn.name });
        var bad = false;
        var skip = false;
        test_fn.func() catch |err| {
            if (err == error.SkipZigTest) {
                skip = true;
            } else {
                bad = true;
                std.debug.print("FAIL ({t})\n", .{err});
                if (@errorReturnTrace()) |trace| std.debug.dumpErrorReturnTrace(trace);
            }
        };
        diagnostics.expect(0) catch {
            bad = true;
        };
        std.testing.io_instance.deinit();
        if (std.testing.allocator_instance.deinit() != 0) bad = true;
        if (bad) {
            failed += 1;
            std.debug.print("FAIL\n", .{});
        } else if (skip) {
            skipped += 1;
            std.debug.print("SKIP\n", .{});
        } else {
            std.debug.print("OK\n", .{});
        }
    }
    std.debug.print("{d} passed; {d} skipped; {d} failed.\n", .{ builtin.test_functions.len - failed - skipped, skipped, failed });
    if (failed != 0) std.process.exit(1);
}
