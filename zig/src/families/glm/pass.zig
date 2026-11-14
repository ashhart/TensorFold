//! A GLM command buffer and its current compute encoder, which changes at an exchange.
const mtl = @import("metal");

pub const Pass = struct {
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
};
