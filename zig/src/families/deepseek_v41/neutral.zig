//! The family's backend-neutral host files as one module, for a backend package built beside them (metal/).
pub const config = @import("config.zig");
pub const pack = @import("pack.zig");
pub const projection = @import("projection.zig");
pub const names = @import("names.zig");
pub const engram_host = @import("engram_host.zig");
pub const engram_rows = @import("engram_rows.zig");
pub const engram_pack = @import("engram_pack.zig");
pub const engram_manifest = @import("engram_manifest.zig");
pub const ced = @import("ced.zig");
/// The drafter's host half: Params, the Markov heads, the candidate lists and the chain.
pub const dspark = @import("draft/dspark.zig");
/// The lanes' contract with a backend (Target, Pass) and the served engine over one (serve_lanes.Model).
pub const iface = @import("draft/iface.zig");
pub const serve_lanes = @import("serve_lanes.zig");
pub const calib_env = @import("draft/calib_env.zig");
pub const calib_lanes = @import("calib_lanes.zig");
pub const lanes = @import("lanes");
pub const api = @import("engine_api");
