//! Settle pending decode windows before reusing projections; prompt IDs have their own shared input buffer.
pub fn prepare(io: anytype, prompt: []const u32, at: u32, n: u32) !void {
    if (n == 0 or at > prompt.len or n > prompt.len - at) return error.ChunkOutOfStep;
    io.settle();
    try io.flush();
    const take = @min(@as(usize, n) + 1, prompt.len - at); // the MTP lookahead, except after the last prompt row
    io.stage(at, prompt[at..][0..take]);
}
