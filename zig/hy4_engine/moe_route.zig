//! The engine's moe_route for the Hy4-flash family files, which import it by name (the speed probe builds it alone).
const m = @import("tensorfold").moe_route;
pub const Shape = m.Shape;
pub const source = m.source;
pub const names = m.names;
pub const logits = m.logits;
pub const Args = m.Args;
pub const Outputs = m.Outputs;
pub const select = m.select;
pub const pack = m.pack;
