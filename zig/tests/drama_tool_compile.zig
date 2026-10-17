//! Compiles the actual standalone CLI, its reference checks and replay controls without running them.
const std = @import("std");
export fn dramaToolCompile(init: *const std.process.Init) void {
    _ = @import("drama_generate").main(init.*) catch return;
}
