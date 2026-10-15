//! GLM-5.3-Flash's routed NVFP4 experts on CUDA (the Python family's experts): grouped.Router, experts.Grouped.

const std = @import("std");
const cuda = @import("cuda");

/// Pairs an item holds on a decode plan (tensorfold.cuda.experts.TILE).
pub const tile = 16;
/// Pairs an item holds on a prompt plan (experts.py STAGED_TILE, experts.promptTile).
pub const staged_tile = 64;
pub const maxItems = cuda.grouped.maxItems;

pub const Which = enum(u2) { gate_up, down_f32, down };

/// Upstream's NVFP4 experts and plan kernels.
pub const Kernels = struct {
    nvfp4: cuda.experts.Nvfp4Experts,
    router: cuda.grouped.Router,

    /// `plans`: the loaded experts module (its plan kernels).
    pub fn load(d: *const cuda.Driver, plans: cuda.Module, sms: u32) !Kernels {
        var nvfp4 = try cuda.experts.Nvfp4Experts.load(d, sms);
        errdefer nvfp4.unload();
        return .{ .nvfp4 = nvfp4, .router = try cuda.grouped.Router.resolve(plans) };
    }

    pub fn unload(k: *Kernels) void {
        k.nvfp4.unload();
    }
};

/// A grouped plan (tensorfold.cuda.experts.Plan): the router's arrays, the slots a row holds, the experts planned.
pub const Plan = struct { route: cuda.grouped.Plan, slots: usize, experts: usize, prompt: bool };

/// experts.route: `pairs` picks (i32) grouped by expert into the plan's items, 16 pairs a decode item, 64 a prompt one.
pub fn route(ks: *const Kernels, stream: cuda.Stream, picks: u64, pairs: usize, p: Plan) !void {
    return ks.router.route(stream, picks, pairs, p.experts, if (p.prompt) staged_tile else tile, p.route);
}

/// nvx.gate_up / down (prompt_* on a prompt plan): `x` rows -> `out` bf16 SwiGLU, fp32 or bf16; `skip` untouched.
pub fn run(ks: *const Kernels, stream: cuda.Stream, which: Which, l: cuda.experts.Layer, plan: Plan, x: u64, x_stride: usize, rows: usize, out: u64, skip: i32) !void {
    const g: cuda.experts.Grouped = .{ .nvfp4 = &ks.nvfp4 };
    const epi: cuda.experts.Epi = switch (which) {
        .gate_up => .up,
        .down_f32 => .down_f32,
        .down => .down_bf16,
    };
    const in: cuda.experts.Rows = .{ .x = x, .stride = x_stride, .slots = plan.slots };
    const pairs = rows * plan.slots;
    if (plan.prompt) return g.prompt(stream, epi, in, l, plan.route, out, maxItems(pairs, plan.experts, staged_tile), skip);
    return g.decode(stream, epi, in, l, plan.route, out, maxItems(pairs, plan.experts, tile), skip);
}

test "plan sizes as the Python family's" {
    try std.testing.expectEqual(@as(usize, 1 + 0), maxItems(1, 257, tile));
    try std.testing.expectEqual(@as(usize, 9 + 2), maxItems(45, 9, tile)); // 9 experts used, 45 pairs: two more items of 16
    try std.testing.expectEqual(@as(usize, staged_tile), cuda.experts.promptTile(.nvfp4));
}
