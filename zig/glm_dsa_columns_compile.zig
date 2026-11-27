//! Compile the real DSA output-column contract on a runtime device bundle, with no device execution.
const gate = @import("gate");
export fn compileGlmDsaColumns(gpu: *const gate.Gpu) void {
    gate.run(gpu.*) catch {};
}
