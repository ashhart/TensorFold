//! GLM-5.3-Flash's routed NVFP4 experts on CUDA (the Python family's experts): grouped.Router, experts.Grouped.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

/// Pairs an item holds on a decode plan (tensorfold.cuda.experts.TILE).
pub const tile = 16;
/// Pairs an item holds on a prompt plan (experts.py STAGED_TILE, experts.promptTile).
pub const staged_tile = 64;
pub const maxItems = cuda.grouped.maxItems;

pub const Which = enum(u2) { gate_up, down_f32, down };

/// Upstream's NVFP4 experts and plan kernels, and the block 32 experts of a csf0_v1 checkpoint's routed layers.
pub const Kernels = struct {
    nvfp4: cuda.experts.Nvfp4Experts,
    b32: ?cuda.experts32.Nvfp4b32Experts, // loaded for a csf0_v1 checkpoint alone (wantsBlock32)
    router: cuda.grouped.Router,
    fact: u64 = 0, // bit r - 1: a decode window of r rows on NVFP4's factored tiles (the same bits); load timing picks them by width (cuda_lane_costs)

    /// Whether a decode window of `rows` runs on the factored tiles; past 64 rows, as 64 do.
    pub fn factored(ks: *const Kernels, rows: usize) bool {
        const r = @min(rows, 64);
        return r > 0 and (ks.fact >> @intCast(r - 1)) & 1 != 0;
    }

    /// `plans`: the loaded experts module (its plan kernels); `block32`: block 32's image too.
    pub fn load(d: *const cuda.Driver, plans: cuda.Module, sms: u32, block32: bool) !Kernels {
        var nvfp4 = try cuda.experts.Nvfp4Experts.load(d, sms);
        errdefer nvfp4.unload();
        var b32: ?cuda.experts32.Nvfp4b32Experts = if (block32) try cuda.experts32.Nvfp4b32Experts.load(d, sms) else null;
        errdefer if (b32) |*x| x.unload();
        return .{ .nvfp4 = nvfp4, .b32 = b32, .router = try cuda.grouped.Router.resolve(plans) };
    }

    pub fn unload(k: *Kernels) void {
        k.nvfp4.unload();
        if (k.b32) |*x| x.unload();
    }
};

/// Whether config.json declares a csf0_v1 module: block 32's kernels load for such a checkpoint alone.
pub fn wantsBlock32(gpa: std.mem.Allocator, config_json: []const u8) !bool {
    var why: core.modelopt.Why = .{};
    var recipes = try core.modelopt.admitRecipes(gpa, config_json, &why);
    defer recipes.deinit();
    return recipes.has(.nvfp4_csf0);
}

/// A grouped plan: the router's arrays, the slots a row holds, the experts; `fact`: decode rows on prompt-sized items.
pub const Plan = struct { route: cuda.grouped.Plan, slots: usize, experts: usize, prompt: bool, fact: bool = false };

/// experts.route: `pairs` picks (i32) grouped by expert into the plan's items, 16 pairs a decode item, 64 a prompt one.
pub fn route(ks: *const Kernels, stream: cuda.Stream, picks: u64, pairs: usize, p: Plan) !void {
    return ks.router.route(stream, picks, pairs, p.experts, if (p.prompt or p.fact) staged_tile else tile, p.route);
}

/// nvx.gate_up / down (prompt_* on a prompt plan): `x` rows -> `out` bf16 SwiGLU, fp32 or bf16; `skip` untouched.
pub fn run(ks: *const Kernels, stream: cuda.Stream, which: Which, l: cuda.experts.Layer, plan: Plan, x: u64, x_stride: usize, rows: usize, out: u64, skip: i32) !void {
    const g: cuda.experts.Grouped = .{ .nvfp4 = &ks.nvfp4, .nvfp4_csf0 = if (ks.b32) |*b| b else null };
    const epi: cuda.experts.Epi = switch (which) {
        .gate_up => .up,
        .down_f32 => .down_f32,
        .down => .down_bf16,
    };
    const in: cuda.experts.Rows = .{ .x = x, .stride = x_stride, .slots = plan.slots };
    const pairs = rows * plan.slots;
    if (plan.prompt) return g.prompt(stream, epi, in, l, plan.route, out, maxItems(pairs, plan.experts, staged_tile), pairs, staged_tile, skip);
    if (plan.fact) {
        // gate-up reads two matrices a K tile and gains from the factored tiles; down is short and keeps the waves
        if (epi == .up and !l.relu2) return cuda.experts.Nvfp4Experts.runFact(&ks.nvfp4, stream, epi, in, l, plan.route, out, maxItems(pairs, plan.experts, staged_tile), skip);
        return cuda.experts_waves.run(&ks.nvfp4, stream, epi, in, l, plan.route, out, maxItems(pairs, plan.experts, tile), skip);
    }
    return g.decode(stream, epi, in, l, plan.route, out, maxItems(pairs, plan.experts, tile), pairs, tile, skip);
}

test "block 32's kernels are wanted for a csf0_v1 module alone, not for block 16's recipes or a single-format NVFP4" {
    const gpa = std.testing.allocator;
    try std.testing.expect(try wantsBlock32(gpa,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"m.3.experts": {"quant_algo": "NVFP4", "group_size": 32, "weight_scale_encoding": "csf0_v1"},
        \\  "m.45.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}}}
    ));
    try std.testing.expect(!try wantsBlock32(gpa,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"m.3.experts": {"quant_algo": "NVFP4", "group_size": 16}, "m.45.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}}}
    ));
    try std.testing.expect(!try wantsBlock32(gpa,
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}}
    ));
}

test "plan sizes as the Python family's" {
    try std.testing.expectEqual(@as(usize, 1 + 0), maxItems(1, 257, tile));
    try std.testing.expectEqual(@as(usize, 9 + 2), maxItems(45, 9, tile)); // 9 experts used, 45 pairs: two more items of 16
    try std.testing.expectEqual(@as(usize, staged_tile), cuda.experts.promptTile(.nvfp4));
}
