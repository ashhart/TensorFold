//! Pending DeltaNet rows (TF_FLASHNEXT_KEEP_LATE): a window's rows into its stream's log, its chain after them.
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const parts = @import("cuda_batch_state.zig");
const Layer = @import("cuda_prompt_load.zig").Layer;

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// fn_gdn_front_log: q as fn_gdn_front's, the rows' k, v and gates into `keep`'s log after its pending rows.
pub fn front(k: anytype, stream: cuda.Stream, lw: *const Layer, proj: u64, conv: u64, rows: usize, gq: u64, keep: parts.LayerKeep) !void {
    var f: cuda.Args = .{};
    f.add(proj);
    f.add(conv);
    f.add(lw.cw.ptr);
    f.add(lw.a_log.ptr);
    f.add(lw.dt.ptr);
    f.add(gq);
    f.add(keep.log);
    f.add(keep.log_ctl);
    f.add(cint(parts.late_rows));
    try cuda.launch.launch(k.front_log, .{ .grid = .{ .x = @intCast(rows), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, stream, &f);
}

/// fn_gdn_rows_late: the pending rows on `state`, then the window's rows into ys, the end to keep.state_out.
pub fn chain(k: anytype, stream: cuda.Stream, gq: u64, keep: parts.LayerKeep, state: u64, rows: usize, ys: u64) !void {
    if (keep.state_out == 0) return error.UnexpectedTensor;
    var a: cuda.Args = .{};
    a.add(gq);
    a.add(keep.log);
    a.add(keep.log_ctl);
    a.add(cint(parts.late_rows));
    a.add(cint(parts.late_store));
    a.add(state);
    a.add(keep.state_out);
    a.add(cint(rows));
    a.add(ys);
    try cuda.launch.launch(k.chain_late, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, stream, &a);
}
