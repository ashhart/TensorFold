//! Taking the served open function's address compiles its body; this artifact is never executed.
const host = @import("batch_host");
export fn compiledOpen() usize {
    return @intFromPtr(&host.open);
}
