//! A rolled-back lane with pending DeltaNet rows: its conv windows and n-gram tail moved on in one launch.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");
const prompt = @import("cuda_prompt.zig");
const gdn = @import("cuda_gdn.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// fn_late_roll: each DeltaNet layer's conv window and the n-gram tail from their saved copies, `keep` rows on.
pub fn lateRoll(o: anytype, seq: *prompt.Seq, d: *Draft, keep: usize) !void {
    var t: parts.RollJobs = .{};
    var n: usize = 0;
    for (o.e.layers[0..o.e.n_layers], 0..) |*lw, layer| {
        if (!lw.linear) continue;
        const lk = d.keep.layers[layer];
        t.j[n] = .{ .win = seq.conv[n].ptr, .hist = lk.hist, .rows = lk.proj, .stride = @intCast(gdn.proj_width), .taps = @intCast(gdn.taps - 1), .c = @intCast(gdn.conv_dim) };
        n += 1;
    }
    t.j[n] = .{ .win = seq.tail.ptr, .hist = d.saved.ptr + Draft.tail_at, .rows = d.saved.ptr + Draft.nrow_at, .stride = @intCast(wt.wide), .taps = @intCast(wt.ple_tail_n), .c = @intCast(wt.wide) };
    n += 1;
    t.n = @intCast(n);
    var a: cuda.Args = .{};
    a.add(t);
    a.add(cint(keep));
    const widest = @max(gdn.conv_dim, wt.wide);
    try cuda.launch.launch(o.e.batch_k.late_roll, .{ .grid = .{ .x = @intCast((widest + 255) / 256), .y = @intCast(n) }, .block = .{ .x = 256 } }, o.e.stream.*, &a);
}
