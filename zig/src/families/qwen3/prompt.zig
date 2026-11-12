//! The lane backend of a prompt-only Qwen3 engine on HIP: lane.zig's backend over the HIP engine.
const lane = @import("lane.zig");
const engine = @import("engine.zig");

pub const Refusal = lane.Refusal;
pub const Prompt = lane.Prompt(engine.Engine, engine.max_inputs);
