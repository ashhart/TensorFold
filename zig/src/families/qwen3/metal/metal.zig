//! Qwen3 embedding-only Metal extension of the dense GQA forward and the shared prompt lane.
pub const engine = @import("engine.zig");
pub const Engine = engine.Engine;
pub const config = @import("config.zig");
pub fn Prompt(comptime api: type) type {
    return @import("../lane.zig").PromptWithApi(Engine, engine.max_inputs, api);
}
pub fn Folder(comptime api: type) type {
    return @import("../head.zig").WithApi(api);
}

test {
    _ = engine;
    _ = config;
}
