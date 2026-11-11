//! DeepSeek-V4.1 prefill/decode hand-off: CUDA state in, Metal state out, transport-independent.
pub const geometry = @import("geometry.zig");
pub const codec = @import("codec.zig");
pub const fp4 = @import("fp4.zig");
pub const plan = @import("plan.zig");
pub const export_cuda = @import("export.zig");
pub const metal = @import("metal.zig");
pub const import_metal = @import("import.zig");
pub const export_metal = @import("export_metal.zig");
pub const blob = @import("blob.zig");
pub const landing = @import("landing.zig");
/// The hand-off as a byte stream (a file, or an HTTP body) and the decoder's fetch over HTTP: one reader checks both.
pub const file = @import("../handoff_file.zig");
pub const http = @import("../handoff_http.zig");
