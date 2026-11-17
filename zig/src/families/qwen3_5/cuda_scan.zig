const geometry = @import("cuda_geometry.zig");
const round = @import("cuda_round.zig");

pub fn deltaScan(e: anytype, record: bool, slot_count: u8, segments: usize, layer: c_int) !void {
    const scan = if (!record) e.fns.gdn_chain else switch (slot_count) {
        1 => e.fns.gdn_slots_1,
        2 => e.fns.gdn_slots_2,
        4 => e.fns.gdn_slots_4,
        8 => e.fns.gdn_slots_8,
        16 => e.fns.gdn_slots_16,
        else => return error.InvalidGdnSlots,
    };
    const input = e.tables.ptr + @as(usize, if (record) round.steps_at else round.rowpar_at);
    const seg_table = e.tables.ptr + round.segs_at;
    const rows_a_warp: usize = if (record and slot_count <= 4) 2 else 1;
    try e.run(scan, .{ 1, geometry.dk / (4 * rows_a_warp), e.g.v_heads * segments }, .{ 32, 4, 1 }, .{ e.geo, e.q, e.k, e.v, e.g_, e.beta, seg_table, input, layer, e.y, e.tables.ptr + round.params_at });
}
