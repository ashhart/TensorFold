//! Semantic-only CFG on CUDA: an independent backbone state over the conditional engine's weights.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("../config.zig");
const Engine = @import("engine.zig").Engine;
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const H = cfg.hidden;
pub const Guide = @import("../guidance.zig").Guide(Engine);

/// Paired: each stream waits on the other's event with no host round trip; separate: the host waits on each side.
pub fn frame(g: *Guide, conditional: *Engine, sampling: @import("../common.zig").Sampling) !void {
    if (!g.state.active) return conditional.frameStep(sampling, true, null);
    const u = g.e;
    const s = u.stream;
    try u.k.gemv(s, u.w.cand_head, u.hlast, H, u.w.n_cand, 0, u.cand_logits);
    try handoff(g, s, conditional.stream);
    const guidance: ?fwd.Guidance = if (g.state.mixing()) |gamma| .{ .unconditioned = u.cand_logits, .gamma = gamma } else null;
    try fwd.enqueueFrameGuided(conditional, sampling, null, null, guidance);
    try handoff(g, conditional.stream, s);
    // Feed the actual sampled conditional frame to the unconditional backbone, never its codec tower.
    try u.k.nextInput(s, conditional.frame, u.bb_input, u.pos, u.pos_bb);
    try u.k.embedBackbone(s, u.bb_input, u.w.embed, u.w.cb_emb, u.x, 1, H, u.c.ids.sem_begin, u.c.ids.sem_end, cfg.codebook_size);
    for (0..cfg.layers) |li| try fwd.backboneLayer(u, fwd.laneBB(u, u.pos_bb), li, 1, 0);
    try u.k.rmsnorm(s, u.x, H, u.w.final_norm, u.hlast, H, 1, H, cfg.eps, true);
    try u.k.copyI32(s, u.pos_bb, u.pos, 1);
}

fn handoff(g: *Guide, from: cuda.Stream, to: cuda.Stream) !void {
    if (!g.batched) return from.synchronize();
    try g.e.ev_fork.record(from);
    try to.wait(g.e.ev_fork);
}

test "drama guidance: the mix launch passes conditional, unconditional, count and gamma in the kernel's order" {
    const Mock = struct {
        var grid: c_uint = 0;
        var block: c_uint = 0;
        var args: struct { c: u64, u: u64, n: c_int, gamma: f32 } = undefined;
        fn launch(_: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            grid = gx;
            block = bx;
            args = .{ .c = read(u64, p, 0), .u = read(u64, p, 1), .n = read(c_int, p, 2), .gamma = read(f32, p, 3) };
            return 0;
        }
        fn read(comptime T: type, p: ?[*]?*anyopaque, i: usize) T {
            return @as(*const T, @ptrCast(@alignCast(p.?[i].?))).*;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    var k: kern.Kernels = undefined;
    k.fns[@backingInt(kern.F.cfg_mix)] = .{ .d = &d, .handle = @ptrFromInt(1) };
    const candidates = cfg.codebook_size + 1;
    try k.cfgMix(.borrowed(&d, null), 0x1000, 0x2000, candidates, 2.5);
    try std.testing.expectEqual((candidates + 255) / 256, Mock.grid);
    try std.testing.expectEqual(256, Mock.block);
    try std.testing.expectEqual(0x1000, Mock.args.c);
    try std.testing.expectEqual(0x2000, Mock.args.u);
    try std.testing.expectEqual(candidates, Mock.args.n);
    try std.testing.expectEqual(2.5, Mock.args.gamma);
}
