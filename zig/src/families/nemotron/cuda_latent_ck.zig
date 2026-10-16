//! A latent MoE in the checkpoint's own math (--precision checkpoint): NVFP4 rows into NVFP4 experts on the FP4 mma.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");

const Layer = cuda.ck.Layer;
const Precision = @import("config.zig").Precision;

/// Whether the checkpoint's math runs: as asked, else by default for a latent ModelOpt checkpoint on Blackwell.
pub fn wanted(ctx: *const cuda.Context, c: Config, asked: ?Precision) !bool {
    const built = c.format == .modelopt and c.latent > 0;
    const major: u32 = @intCast(try ctx.attribute(.compute_capability_major));
    const p = asked orelse return built and major >= kern.ck_major;
    if (p == .full) return false;
    if (!built) return error.PrecisionNotBuilt;
    if (major < kern.ck_major) return error.CheckpointMathNeedsBlackwell;
    return true;
}

/// A layer's expert tables in the FP4 mma's tiles: routed (latent width) and shared (hidden width).
pub const Tables = struct { routed: Layer, shared: Layer };

/// A pass's NVFP4 rows: latent rows, hidden rows, routed activations, shared activations (codes, then block scales).
pub const Rows = struct { lat: u64, hid: u64, act: u64, sact: u64 };

/// Bytes of each section of Rows at `rows` rows, for this rank's shares, each 256-byte aligned.
fn sections(c: Config, rows: usize) [4]usize {
    const own = c.ownShares();
    const sw = c.expert_width / c.shares;
    const ssw = c.shared_width / c.shares;
    const b = [4]usize{ cuda.ck.rowsBytes(rows, c.latent), cuda.ck.rowsBytes(rows, c.hidden), cuda.ck.rowsBytes(rows * c.top_k * own, sw), cuda.ck.rowsBytes(rows * own, ssw) };
    var out: [4]usize = undefined;
    for (b, &out) |n, *o| o.* = std.mem.alignForward(usize, n, 256);
    return out;
}

/// Bytes a pass of `rows` rows needs for its NVFP4 rows.
pub fn rowsBytes(c: Config, rows: usize) usize {
    var total: usize = 0;
    for (sections(c, rows)) |n| total += n;
    return total;
}

/// Rows laid out from `base` (rowsBytes(c, rows) bytes).
pub fn rowsAt(c: Config, rows: usize, base: u64) Rows {
    const s = sections(c, rows);
    return .{ .lat = base, .hid = base + s[0], .act = base + s[0] + s[1], .sact = base + s[0] + s[1] + s[2] };
}

/// One share set's expert rows into `out` (fp32), as latent.partials lays them out: routed pairs, then shared rows.
pub fn partials(o: kern.Ops, t: Tables, plan: kern.Plan, act: u64, sact: u64, sids: u64, ck4: u64, picks: u64, lat: u64, hidden: u64, rows: usize, own: usize, c: Config, out: u64) !void {
    const q = if (o.k.ck) |*x| x else return error.CheckpointMathNotLoaded;
    if (ck4 == 0) return error.NoCheckpointRows;
    const at = rowsAt(c, @import("cuda_state.zig").pass_rows, ck4);
    const pairs = rows * c.top_k * own;
    const r = t.routed;
    try q.quant4(o.s, lat, c.latent, rows, c.latent, r.up_act, at.lat);
    try o.plan(picks, pairs, r.experts, 16, plan);
    const items = kern.maxItems(pairs, r.experts, 16);
    try q.up(o.s, r, at.lat, rows, c.top_k * own, plan, act, items);
    try q.quant4(o.s, act, r.width, pairs, r.width, r.down_act, at.act);
    try q.down(o.s, r, at.act, pairs, plan, out, items);
    const h = t.shared;
    try q.quant4(o.s, hidden, c.hidden, rows, c.hidden, h.up_act, at.hid);
    try o.plan(sids, rows * own, own, 16, plan);
    const sh_items = kern.maxItems(rows * own, own, 16);
    try q.up(o.s, h, at.hid, rows, own, plan, sact, sh_items);
    try q.quant4(o.s, sact, h.width, rows * own, h.width, h.down_act, at.sact);
    try q.down(o.s, h, at.sact, rows * own, plan, out + @as(u64, pairs) * c.latent * 4, sh_items);
}

test "a pass's NVFP4 rows" {
    var c: Config = undefined;
    c.top_k = 22;
    c.latent = 1024;
    c.hidden = 4096;
    c.expert_width = 2688;
    c.shared_width = 5376;
    c.shares = 2;
    c.world = 1;
    const s = sections(c, 1);
    try std.testing.expectEqual(@as(usize, 768), s[0]); // 512 + 64, aligned
    try std.testing.expectEqual(std.mem.alignForward(usize, 44 * 672 + 44 * 84, 256), s[2]);
    const r = rowsAt(c, 1, 0);
    try std.testing.expectEqual(@as(u64, 768), r.hid);
}
