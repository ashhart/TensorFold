//! Compile the real Linux GLM startup body without executing it or compiling GPU images.
const glm = @import("glm");
export fn analyzeGlmOpen() usize {
    return @intFromPtr(&glm.native.open);
}
