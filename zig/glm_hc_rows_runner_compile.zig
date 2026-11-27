//! Analyze the actual isolated numerical and embedded-resource bodies with a runtime image argument.
const run = @import("runner");
export fn analyzeHcRowsRunner() usize {
    return @intFromPtr(&run.numerical) ^ @intFromPtr(&run.resources) ^ @intFromPtr(&run.dispatch);
}
