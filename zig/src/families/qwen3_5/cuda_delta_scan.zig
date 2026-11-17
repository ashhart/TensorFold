//! Delta scan selection shared by decode and recorded windows.
const Round = @import("cuda_round.zig");
const geometry = @import("cuda_geometry.zig");

pub fn deltaScan(e: anytype, record: bool, slot_count: u8, segments: usize, layer: c_int) !void {
    const scan = if (!record) e.fns.gdn_chain else switch (slot_count) {
        1 => e.fns.gdn_slots_1,
        2 => e.fns.gdn_slots_2,
        4 => e.fns.gdn_slots_4,
        8 => e.fns.gdn_slots_8,
        16 => e.fns.gdn_slots_16,
        else => return error.InvalidGdnSlots,
    };
    const input = e.tables.ptr + @as(usize, if (record) Round.steps_at else Round.rowpar_at);
    // qwen35_slots.cuh: up to four slots scan two value rows a warp, so a block covers eight
    const rows_a_warp: usize = if (record and slot_count <= 4) 2 else 1;
    try e.run(scan, .{ 1, geometry.dk / (4 * rows_a_warp), e.g.v_heads * segments }, .{ 32, 4, 1 }, .{ e.geo, e.q, e.k, e.v, e.g_, e.beta, e.tables.ptr + Round.segs_at, input, layer, e.y, e.tables.ptr + Round.params_at });
}
