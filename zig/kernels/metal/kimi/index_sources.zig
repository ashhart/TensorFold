//! Experimental K3.1 operator sources shared by the native family and its contract probes.
pub const index = @embedFile("index.metal");
pub const attention = @embedFile("index_attn.metal");
pub const dense = @embedFile("mla.metal");
pub const common = @embedFile("kimi_common.h");
