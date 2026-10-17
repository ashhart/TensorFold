//! Force the actual standard Metal contract runner into an object without linking or executing GPU calls.
export fn compiledContract() usize {
    return @intFromPtr(&@import("runner").main);
}
