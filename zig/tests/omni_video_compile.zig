//! Taking the actual native open address emits the model and video offer bodies; this object is never executed.
export fn compileOpen() usize {
    return @intFromPtr(&@import("native").open);
}
