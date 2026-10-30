//! Compile the actual Qwen2B served open function and snapshot callbacks; never execute this object.
const host = @import("host");
export fn compiledOpen() usize {
    return @intFromPtr(&host.open);
}
