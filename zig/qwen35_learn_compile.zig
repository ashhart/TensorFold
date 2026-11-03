//! Analyze the actual diagnostic entry point without loading a checkpoint or opening CUDA.
const std = @import("std");
export fn compileQwenLearn(p: *anyopaque) void {
    const init: *std.process.Init = @ptrCast(@alignCast(p));
    _ = @import("tool").main(init.*) catch 1;
}
