//! Kimi Metal kernel sources, for tests that compile a kernel from source (the engine loads the metallib).
pub const common = @embedFile("kimi_common.h");
pub const kda = @embedFile("kda.metal");
