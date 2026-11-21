//! Score and bound DSpark prefixes before the target verifies them.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const wts = @import("weights.zig");
const Enc = @import("forward.zig").Enc;

pub const Args = extern struct {
    step: u32,
    cap: u32,
    prune: u32,
    markov_groups: u32,
    row_base: u32,
    cost_ms: [c.draft_block + 1]f32,
    margin: u32,
};

pub const State = extern struct {
    survival: f32,
    expected: f32,
    best_rate: f32,
    chosen: u32,
    stop: u32,
    score_step: u32,
    full_passes: u32,
    examined: u32,
    pending: f32,
};

pub const Config = struct {
    prune: bool,
    margin: bool = false,
    costs: [c.draft_block + 1]f32,
};

pub const pinned_costs: [c.draft_block + 1]f32 = .{ 18.16, 23.17, 27.08, 30.99, 34.39, 37.79 };

pub const carry_discount_ms: f32 = 2.45;
const fresh_rate_per_ms: f64 = 0.071;

fn fromBF16(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

/// Only a scheduling estimate. The vocabulary head and target use their usual Metal math.
pub fn carryWins(hidden: f32, projection: wts.Tensor, embedding: wts.Tensor, token: u32, first_cost: f32) bool {
    const w = projection.host(u16)[c.hidden..];
    const e = embedding.host(u16)[@as(usize, token) * c.markov_rank ..][0..c.markov_rank];
    var score: f64 = hidden;
    for (w, e) |a, b| score += @as(f64, fromBF16(a)) * @as(f64, fromBF16(b));
    const chance: f64 = 1 / (1 + @exp(-std.math.clamp(score, -32, 32)));
    return (1 + chance) / first_cost > fresh_rate_per_ms;
}

pub const Kernels = struct {
    hidden_pipe: mtl.Pipeline,
    step_pipe: mtl.Pipeline,
    pick_pipe: mtl.Pipeline,

    pub fn load(device: mtl.Device) !?Kernels {
        const lib = try mtl.Library.fromSource(device, @embedFile("confidence.metal"), mtl.CompileOptions.mlx());
        defer lib.deinit();
        const h = try mtl.Pipeline.init(device, lib, "tf_confidence_hidden", false);
        errdefer h.deinit();
        const s = try mtl.Pipeline.init(device, lib, "tf_confidence_step", false);
        errdefer s.deinit();
        return .{ .hidden_pipe = h, .step_pipe = s, .pick_pipe = try mtl.Pipeline.init(device, lib, "tf_confidence_pick", false) };
    }

    pub fn deinit(p: Kernels) void {
        p.hidden_pipe.deinit();
        p.step_pipe.deinit();
        p.pick_pipe.deinit();
    }

    pub fn hidden(p: Kernels, x: Enc, projection: wts.Tensor, cap: usize) void {
        comptime std.debug.assert(c.hidden == 5120 and c.markov_rank == 256 and c.hc == 4 and c.draft_block == 5);
        x.fence();
        const enc = x.ep.*;
        enc.setPipeline(p.hidden_pipe);
        enc.setBuffer(x.s.x, 0, 0);
        enc.setBuffer(x.s.pre[0], 0, 1);
        enc.setBuffer(projection.buffer, projection.offset, 2);
        enc.setBuffer(x.s.conf_hidden, 0, 3);
        enc.dispatchGroups(mtl.Size.of(cap, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn step(p: Kernels, x: Enc, projection: wts.Tensor, embedding: wts.Tensor, a: Args) void {
        x.fence();
        const enc = x.ep.*;
        enc.setPipeline(p.step_pipe);
        enc.setBuffer(x.s.conf_hidden, 0, 0);
        enc.setBuffer(projection.buffer, projection.offset, 1);
        enc.setBuffer(embedding.buffer, embedding.offset, 2);
        enc.setBuffer(x.s.draft_ids, 0, 3);
        enc.setBuffer(x.s.conf_state, 0, 4);
        enc.setBuffer(x.s.conf_dispatch, 0, 5);
        enc.setBytes(std.mem.asBytes(&a), 6);
        enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn pick(p: Kernels, x: Enc, a: Args) void {
        x.fence();
        const enc = x.ep.*;
        enc.setPipeline(p.pick_pipe);
        enc.setBuffer(x.s.markov_part, 0, 0);
        enc.setBuffer(x.s.draft_ids, 0, 1);
        enc.setBuffer(x.s.conf_state, 0, 2);
        const dims = [2]u32{ a.markov_groups, a.row_base + a.step };
        enc.setBytes(std.mem.asBytes(&dims), 3);
        enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    }
};
