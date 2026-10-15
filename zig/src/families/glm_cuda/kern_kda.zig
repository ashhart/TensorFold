//! GLM-5.3-Flash on CUDA: the KDA layers' launches (glm/kda.cu), re-exported by kern.zig.

const cuda = @import("cuda");
const kern = @import("kern.zig");
const K = kern.K;
const LL = kern.LL;
const EPS = kern.EPS;
const LOWER = kern.LOWER;

/// A batched step's KDA layer ``li``: every stream's window rows through its own states (segs: [n] device KdaSeg).
pub fn kdaChainSeg(k: *K, segs: u64, nseg: i64, li: usize, p: u64, p_stride: i64, b_off: i64, a_: u64, a_stride: i64, g: u64, g_stride: i64, cw: u64, a_log: u64, dt_bias: u64, norm_w: u64, out: u64, max_rows: i64, proj_w: i64) !void {
    var a: cuda.Args = .{};
    a.add(segs);
    a.add(@as(i32, @intCast(li)));
    a.add(@as(i32, @intCast(LL)));
    a.add(p);
    a.add(@as(i32, @intCast(p_stride)));
    a.add(@as(i32, @intCast(b_off)));
    a.add(a_);
    a.add(@as(i32, @intCast(a_stride)));
    a.add(g);
    a.add(@as(i32, @intCast(g_stride)));
    a.add(cw);
    a.add(a_log);
    a.add(dt_bias);
    a.add(norm_w);
    a.add(EPS);
    a.add(LOWER);
    a.add(out);
    a.add(@as(i32, @intCast(max_rows)));
    a.add(@as(i32, @intCast(proj_w)));
    try k.go(try k.ext(.kda, "glm_kda_chain_seg"), .{ .x = @intCast(LL), .y = @intCast(nseg) }, 1024, 0, &a);
}

// -- kda.py: the KDA extension (glm/kda.cu) --------------------------------------------------------------------------
pub const KdaScratch = struct { out: u64, k: u64, v: u64, g: u64, b: u64 };

/// kda.chain: projection, gate rows, conv window and state -> out, state_out, replay inputs (64+ rows: 3 kernels).
pub fn kdaChain(k: *K, p: u64, p_stride: i64, b_off: i64, a: u64, a_stride: i64, g: u64, g_stride: i64, cs: u64, cw: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, rows: i64, sc: KdaScratch, state_out: u64, q_tmp: u64, y_tmp: u64) !void {
    const H: i32 = @intCast(LL);
    if (rows >= 64) {
        var pa: cuda.Args = .{};
        pa.add(H);
        pa.add(p);
        pa.add(@as(i32, @intCast(p_stride)));
        pa.add(@as(i32, @intCast(b_off)));
        pa.add(a);
        pa.add(@as(i32, @intCast(a_stride)));
        pa.add(cs);
        pa.add(cw);
        pa.add(a_log);
        pa.add(dt_bias);
        pa.add(LOWER);
        pa.add(q_tmp);
        pa.add(sc.k);
        pa.add(sc.v);
        pa.add(sc.g);
        pa.add(sc.b);
        try k.go(try k.ext(.kda, "11prep_kernel"), .{ .x = @intCast(rows), .y = @intCast(H) }, 512, 0, &pa);
        var sa: cuda.Args = .{};
        sa.add(H);
        sa.add(state_in);
        sa.add(q_tmp);
        sa.add(sc.k);
        sa.add(sc.v);
        sa.add(sc.g);
        sa.add(sc.b);
        sa.add(@as(i32, @intCast(rows)));
        sa.add(y_tmp);
        sa.add(state_out);
        try k.go(try k.ext(.kda, "11step_kernelILi4ELi16E"), .{ .x = @intCast(H), .y = 128 / 4 / 4 }, 4 * 32, 0, &sa);
        var oa: cuda.Args = .{};
        oa.add(H);
        oa.add(y_tmp);
        oa.add(g);
        oa.add(@as(i32, @intCast(g_stride)));
        oa.add(norm_w);
        oa.add(EPS);
        oa.add(sc.out);
        try k.go(try k.ext(.kda, "10out_kernel"), .{ .x = @intCast(rows), .y = @intCast(H) }, 128, 0, &oa);
        return;
    }
    var ca: cuda.Args = .{};
    ca.add(H);
    ca.add(p);
    ca.add(@as(i32, @intCast(p_stride)));
    ca.add(@as(i32, @intCast(b_off)));
    ca.add(a);
    ca.add(@as(i32, @intCast(a_stride)));
    ca.add(g);
    ca.add(@as(i32, @intCast(g_stride)));
    ca.add(cs);
    ca.add(cw);
    ca.add(state_in);
    ca.add(a_log);
    ca.add(dt_bias);
    ca.add(norm_w);
    ca.add(EPS);
    ca.add(LOWER);
    ca.add(@as(i32, @intCast(rows)));
    ca.add(sc.out);
    ca.add(state_out);
    ca.add(sc.k);
    ca.add(sc.v);
    ca.add(sc.g);
    ca.add(sc.b);
    try k.go(try k.ext(.kda, "12chain_kernel"), .{ .x = @intCast(H) }, 1024, 0, &ca);
}

/// kda.replay_layers: every layer's state after its first ``rows`` rows (states [layers, H, 128, 128]).
pub fn kdaReplayLayers(k: *K, state_in: u64, sk: u64, sv: u64, sg: u64, sb: u64, layers: i64, rows_cap: i64, rows: i64, state_out: u64) !void {
    const H: i32 = @intCast(LL);
    var a: cuda.Args = .{};
    a.add(H);
    a.add(state_in);
    a.add(@as(u64, @intCast(LL * 128 * 128)));
    a.add(sk);
    a.add(sv);
    a.add(sg);
    a.add(sb);
    a.add(@as(u64, @intCast(rows_cap * LL * 128)));
    a.add(@as(u64, @intCast(rows_cap * LL)));
    a.add(@as(i32, @intCast(rows)));
    a.add(state_out);
    try k.go(try k.ext(.kda, "20replay_layers_kernel"), .{ .x = @intCast(layers * LL) }, 1024, 0, &a);
}
