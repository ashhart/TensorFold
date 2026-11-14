//! The decode MoE block's epilogue fusions: which of them a call takes, and the scratch rows they place.
const std = @import("std");
const exl3 = @import("core").exl3_projection;
const work = @import("workspace.zig");
const weights = @import("weights.zig");

/// Bytes of one rotated down-projection input row (2304 halves).
const rotated_row_bytes: usize = 2304 * 2;

/// What one decode call fuses.
pub const Fusion = struct {
    /// The gate+up call leaves its f32 sums; one pass applies its rotate_out, both SwiGLUs and the down rotate_in.
    gateup: bool = false,
    /// The shared expert's down runs inside the routed down call, its rows after the routed rows.
    shared_down: bool = false,
    /// The down call stops at its sums and one pass applies its rotate_out and the combine.
    down_combine: bool = false,
    /// The first rotated-row index of the shared expert's down input.
    shared_row: usize = 0,
};

/// The call the fusions apply to.
pub const Call = struct {
    rows: u32,
    picks: u32,
    /// The gate+up call carries the shared pair and the routed picks in one dispatch.
    merged: bool,
    shared_split: bool,
    swiglu_rot: bool,
    shared_rot: bool,
    skip_experts: bool,
    skip_shared: bool,
    mlp_passes: u32,
    fuse_up_picks: u32,
};

/// Whether the gate+up call can stop at its sums and hand the routed and shared rows to the fused pass.
pub fn plan(ws: *const work.Workspace, w: *const weights.Moe, c: Call) !Fusion {
    var out = Fusion{};
    if (std.c.getenv("TF_DSV41_NO_FUSE_GATEUP_SWIGLU") != null) return out;
    if (!c.merged or c.shared_split or !c.swiglu_rot or !c.shared_rot or c.skip_experts or c.mlp_passes != 1) return out;
    if (c.fuse_up_picks != c.picks or w.shared_down == null) return out;
    const r: usize = c.rows;
    const routed: usize = r * c.picks;
    const gate_up_rows = routed * 2 + 2 * r;
    const room = ws.view(.grouping).offset - ws.view(.projection).offset;
    // The shared down joins the routed one: its input rows follow the routed rows in the same scratch.
    if (std.c.getenv("TF_DSV41_NO_FUSE_SHARED_DOWN") == null and !c.skip_shared and w.down_shared != null and
        try exl3.scratchBytes(2304, 5120, routed + r) <= room)
    {
        const combine = std.c.getenv("TF_DSV41_NO_FUSE_DOWN_COMBINE") == null and
            std.c.getenv("TF_DSV41_SKIP_COMBINE") == null and std.c.getenv("TF_DSV41_DOUBLE_COMBINE") == null and c.picks <= 7;
        return .{ .gateup = true, .shared_down = true, .down_combine = combine, .shared_row = routed };
    }
    // The shared rotated rows stay clear of the routed down call's scratch and of the gate+up sums they are read from.
    const row = (try exl3.scratchBytes(2304, 5120, routed) + rotated_row_bytes - 1) / rotated_row_bytes;
    const start = row * rotated_row_bytes;
    const sums_end = exl3.sumsView(.{ .buffer = ws.buffer }, gate_up_rows, 5120).offset + gate_up_rows * 2304 * 4;
    if (start < sums_end or start + try exl3.scratchBytes(2304, 5120, r) > room) return out;
    out.gateup = true;
    out.shared_row = row;
    return out;
}

/// The projection scratch viewed from the shared expert's rotated rows on, past the routed down call's scratch.
pub fn sharedScratch(ws: *const work.Workspace, f: Fusion) exl3.View {
    const base = ws.projection();
    return .{ .buffer = base.buffer, .offset = base.offset + f.shared_row * rotated_row_bytes };
}
