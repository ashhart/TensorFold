//! The CUDA SMs `zig build -Dgpu` builds a server for (zig/build/product.zig); a refusal names only these.

/// 10 * major + minor, as `Context.capability` reports it.
pub const sms = [_]u32{ 80, 86, 89, 120, 121 };
