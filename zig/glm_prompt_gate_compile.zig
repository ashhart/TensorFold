//! Compile the real gate on a runtime device bundle, with no device execution.
const gate = @import("gate");
export fn compileGlmPromptGate(gpu: *const gate.Gpu) void {
    gate.run(gpu.*) catch {};
}
