//! Compile the served Flint opener and its drafter vtable without loading Metal or weights.
export fn compileFlintDraftContext() usize {
    return @intFromPtr(&@import("flint").open);
}
