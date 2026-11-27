//! DeepSeek-V4.1 prefill/decode hand-off: CUDA state in, a neutral bf16/f32 part plan out, transport-independent.
pub const geometry = @import("geometry.zig");
pub const codec = @import("codec.zig");
pub const fp4 = @import("fp4.zig");
pub const plan = @import("plan.zig");
pub const export_cuda = @import("export.zig");
