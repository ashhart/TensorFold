//! MiMo-V2(.6-Flash) on the CUDA engine: native kernels on the shared EXL3 operators, after Jay Leaton's DeepSeek-V4.1.

pub const config = @import("config.zig");
pub const load = @import("load.zig");
pub const ops = @import("ops.zig");
pub const exl3 = @import("exl3.zig");
pub const forward = @import("forward.zig");
pub const dflash = @import("dflash.zig");
pub const prompt = @import("prompt.zig");
pub const engine = @import("engine.zig");
pub const kept = @import("kept.zig");
pub const native = @import("native.zig");
pub const Config = config.Config;

test {
    _ = config;
    _ = load;
    _ = ops;
    _ = exl3;
    _ = forward;
    _ = dflash;
    _ = prompt;
    _ = engine;
    _ = kept;
    _ = native;
    if (comptime @import("builtin").os.tag != .windows) _ = @import("native_test.zig"); // open reaches the TP sockets
}
