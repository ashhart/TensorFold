//! Semantic compile of served callbacks; this object is never linked or executed.
const q = @import("qwen35");
export fn targetOpen() usize {
    return @intFromPtr(&q.native.open);
}
export fn drafterOpen() usize {
    return @intFromPtr(&q.native.openDrafter);
}
export fn coreRound() usize {
    return @intFromPtr(&@import("lanes").Engine.step);
}
