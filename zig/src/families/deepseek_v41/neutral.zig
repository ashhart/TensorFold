//! The family's backend-neutral host files as one module, for a backend package built beside them (metal/).
pub const config = @import("config.zig");
pub const pack = @import("pack.zig");
pub const projection = @import("projection.zig");
pub const names = @import("names.zig");
pub const engram_host = @import("engram_host.zig");
pub const engram_rows = @import("engram_rows.zig");
pub const engram_pack = @import("engram_pack.zig");
