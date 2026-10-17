//! Compile addresses force actual startup and prompt callers; this object is never executed.
const tf = @import("tensorfold");
export fn compiledOpen() usize {
    return @intFromPtr(&@import("host").open);
}
export fn compiledGlm() usize {
    return @intFromPtr(&tf.glm.prompt.backbone);
}
export fn compiledLfm() usize {
    return @intFromPtr(&tf.lfm2_moe.forward.run);
}
