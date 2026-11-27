//! Compile actual prompt, paired prompt and close operations with inert GPU images; never execute them.
const Engine = @import("glm").engine.Engine;
export fn analyzeGlmPrefillLedger() usize {
    return @intFromPtr(&Engine.prefillChunk) ^ @intFromPtr(&Engine.verify) ^ @intFromPtr(&Engine.deinit);
}
