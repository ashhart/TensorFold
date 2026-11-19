//! Hy4-flash's launches: a records file for each group of tests/contract/metal_hy4_*.zig.
pub const metal = @import("hy4_index_records.zig").records ++
    @import("hy4_project_records.zig").records ++
    @import("hy4_attend_records.zig").records ++
    @import("hy4_nax_records.zig").records ++
    @import("hy4_ihc_records.zig").records ++
    @import("hy4_route_records.zig").records ++
    @import("hy4_experts_records.zig").records ++
    @import("hy4_rows_records.zig").records ++
    @import("hy4_lane_records.zig").records;
