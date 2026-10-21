//! Share the core-owned NumPy parser with fixture programs that import it as a standalone module.
const npy = @import("core").npy;
pub const Array = npy.Array;
pub const parse = npy.parse;
pub const parseHeader = npy.parseHeader;
