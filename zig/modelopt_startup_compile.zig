//! Force actual native startup and its returned close body into an object; never run it here.
export fn compiledOpen() usize {
    return @intFromPtr(&@import("native").open);
}
