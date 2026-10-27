//! The batched pass's DeltaNet front and recurrence launches, over every head or a rank's half of them.
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const gdn_split = @import("exl3/model/cuda_exl3_gdn_split.zig");
const parts = @import("cuda_batch_state.zig");

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// fn_gdn_front over `rows` projection rows after the conv window `conv`.
pub fn front(self: anytype, lw: anytype, proj: u64, conv: u64, rows: usize, gq: u64, gk: u64, gv: u64, ggb: u64, shard: ?*const gdn_split.Views) !void {
    var f: cuda.Args = .{};
    f.add(proj);
    f.add(conv);
    f.add(lw.cw.ptr);
    f.add(lw.a_log.ptr);
    f.add(lw.dt.ptr);
    f.add(gq);
    f.add(gk);
    f.add(gv);
    f.add(ggb);
    if (shard) |h| {
        f.add(cint(gdn_split.rank_nk));
        f.add(@as(c_int, @intCast(h.hk0)));
        f.add(@as(c_int, @intCast(h.hv0)));
        return cuda.launch.launch(self.k.front_h, .{ .grid = .{ .x = @intCast(rows), .y = gdn_split.rank_nk + gdn_split.rank_nv }, .block = .{ .x = 256 } }, self.stream, &f);
    }
    try cuda.launch.launch(self.k.front, .{ .grid = .{ .x = @intCast(rows), .y = gdn.nk + gdn.nv }, .block = .{ .x = 256 } }, self.stream, &f);
}

/// fn_gdn_rows over `rows` rows from `state`, in place or (state_out) to fn_gdn_rows_to's; readouts into ys.
pub fn chain(self: anytype, gq: u64, gk: u64, gv: u64, ggb: u64, state: u64, state_out: u64, rows: usize, ys: u64, shard: ?*const gdn_split.Views) !void {
    var a: cuda.Args = .{};
    a.add(gq);
    a.add(gk);
    a.add(gv);
    a.add(ggb);
    a.add(state);
    if (rows >= 64) {
        // two threads a state row (fn_gdn_chain_tpr), fn_gdn_rows' bits; in place without a rollback copy
        a.add(if (state_out != 0) state_out else state);
        a.add(cint(rows));
        a.add(ys);
        return cuda.launch.launch(self.k.chain_tpr, .{ .grid = .{ .x = gdn.nv }, .block = .{ .x = 256 }, .shared = parts.chain_tpr_smem }, self.stream, &a);
    }
    if (state_out != 0) a.add(state_out);
    a.add(cint(rows));
    a.add(ys);
    if (shard) |h| {
        a.add(@as(c_int, @intCast(h.hv0)));
        return cuda.launch.launch(if (state_out != 0) self.k.chain_to_h else self.k.chain_h, .{ .grid = .{ .x = gdn_split.rank_nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, self.stream, &a);
    }
    try cuda.launch.launch(if (state_out != 0) self.k.chain_to else self.k.chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, self.stream, &a);
}
