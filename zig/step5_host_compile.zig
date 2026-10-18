//! Compile the served open function and its family; never execute the emitted object.
const host = @import("host");
export fn compiledOpen() usize {
    return @intFromPtr(&host.open);
}
