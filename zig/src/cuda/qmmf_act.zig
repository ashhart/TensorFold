//! TF_Q35_MLP_ACT_FUSED: the NVFP4 up projection at decode rows writing the MLP act (qmmf.cu tf_qmmf_fp4_up_act16).
const std = @import("std");
const qmmf = @import("qmmf.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Function = module.Function;
const Stream = stream_.Stream;
const Weight = qmmf.Weight;

pub const symbol = "tf_qmmf_fp4_up_act16";
const bn = 64; // the kernel's column tile, qmmf.zig's bn
const threads = 128; // WM 1 x WN 4 warps, qmmf.zig's threads

/// The kernel when the module has it (a module built before it does not): its dynamic shared bytes allowed.
pub fn load(mod: module.Module) !?Function {
    const f = mod.function(symbol) catch return null;
    try f.allowDynamicShared(qmmf.sharedBytes(.fp4, 16));
    return f;
}

/// Whether today's up launch for `m` rows is the instance the kernel copies: FP4, row tile 16, one K slice, no reduce.
pub fn eligible(l: qmmf.Lane, m: usize, w: Weight) bool {
    if (w.mode != .fp4 or m == 0 or m > 16 or l.usesPrompt(m, w)) return false;
    const p = qmmf.planFor(m, w, l.clusters);
    return p.bm == 16 and p.sk == 1 and !p.fused and !p.cluster and !p.reduce and (w.n & 1) == 0;
}

/// The up projection of `m` rows of `x` into `act`, each value tf_q35_mlp_act's of `gate`'s row and this one's.
pub fn launch(f: Function, l: qmmf.Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, gate: u64, act: u64, params: u64) !void {
    if (!eligible(l, m, w)) return error.NotFusable;
    var args: launch_.Args = .{};
    args.add(x);
    args.add(w.codes);
    args.add(w.scales);
    args.add(w.scale);
    args.add(act);
    args.add(@as(u64, 0));
    args.add(@as(i32, @intCast(m)));
    args.add(@as(i32, @intCast(w.n)));
    args.add(@as(i32, @intCast(w.k)));
    args.add(@as(i32, 1));
    args.add(@as(i32, @intCast(w.npad)));
    args.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
    args.add(@as(i32, 1)); // matmulOut's band for one row tile
    args.add(gate);
    args.add(params);
    try launch_.launch(f, .{
        .grid = .{ .x = @intCast((w.n + bn - 1) / bn), .y = 1, .z = 1 },
        .block = .{ .x = threads, .y = 1, .z = 1 },
        .shared = qmmf.sharedBytes(.fp4, 16),
        .cluster = null,
        .pdl = l.pdl,
    }, s, &args);
}

test "the fused up launch copies today's plan only: FP4, row tile 16, one K slice, no cluster, reduce or prompt" {
    var lane: qmmf.Lane = undefined;
    lane.clusters = true;
    lane.prompt = null;
    lane.prompt_k32 = null;
    const up: Weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 17408, .k = 5120, .npad = 17408 };
    try std.testing.expectEqual(@as(u32, 1), qmmf.planFor(1, up, true).sk); // the 27B's up: one slice, as the map says
    for (1..17) |m| try std.testing.expect(eligible(lane, m, up));
    try std.testing.expect(!eligible(lane, 0, up));
    try std.testing.expect(!eligible(lane, 17, up));
    var fp8 = up;
    fp8.mode = .fp8;
    try std.testing.expect(!eligible(lane, 4, fp8));
    const down: Weight = .{ .mode = .fp4, .codes = 0, .scales = 0, .n = 5120, .k = 17408, .npad = 5120 };
    try std.testing.expect(qmmf.planFor(4, down, true).sk > 1);
    try std.testing.expect(!eligible(lane, 4, down)); // split K: the reduce or the cluster would change the plan
    var odd = up;
    odd.n = 17407;
    odd.npad = 17407;
    try std.testing.expect(!eligible(lane, 4, odd));
}
