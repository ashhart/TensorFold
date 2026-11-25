//! Inkling 2 Mini's message and rank-64 gate shaders, then the split gates the contract compares bit for bit.
pub const loop = @embedFile("loop.metal") ++ "\n" ++ @embedFile("loop_split.metal");
