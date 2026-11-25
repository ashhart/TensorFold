//! CUDA's latent routed branch. The enclosing layer reduces the rank's partial before projecting it back.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const ranks = @import("rank_weights.zig");
const affine = cuda.affine_experts;

pub const Shape = struct {
    hidden: u32,
    route: cuda.moe_glue.Route,
    experts: affine.Shape,

    pub fn init(text: config.Text, rank: ranks.Range, rows: u32) !Shape {
        const first = std.math.cast(u32, rank.first) orelse return error.BadHy4Rank;
        const count = std.math.cast(u32, rank.count) orelse return error.BadHy4Rank;
        if (@as(u64, first) + count > text.n_routed_experts or text.num_experts_per_tok == 0 or
            text.n_routed_experts > 512 or text.num_experts_per_tok > @min(32, text.n_routed_experts)) return error.BadHy4Route;
        const experts: affine.Shape = .{
            .width = text.routed_expert_hidden_size,
            .inner = text.moe_intermediate_size,
            .first = first,
            .experts = count,
            .pairs = try std.math.mul(u32, rows, text.num_experts_per_tok),
            .rows = rows,
            .swiglu_limit = text.swiglu_limit,
        };
        try experts.check();
        return .{
            .hidden = text.hidden_size,
            .route = .{
                .experts = text.n_routed_experts,
                .topk = text.num_experts_per_tok,
                .shared = false,
                .normalize = true,
                .scale = text.routed_scaling_factor,
                .score = true,
            },
            .experts = experts,
        };
    }
};

pub const Weights = struct {
    router: u64,
    correction_bias: u64,
    gate_up: affine.Weights,
    down: affine.Weights,
    precision: affine.Precision,
};

pub const Buffers = struct {
    hidden: u64, // BF16 [rows][hidden]
    latent: u64, // BF16 [rows][routed width] from the replicated down projection
    expert_rows: u64, // i32 [rows][top-k], each entry names the owning input row
    logits: u64, // FP32 [rows][global experts]
    ids: u64, // int32 [rows][top-k]
    probabilities: u64, // FP32 [rows][top-k]
    activated: u64, // BF16 [rows * top-k][inner]
    contributions: u64, // FP32 [rows * top-k][routed width]
    rank_output: u64, // FP32 [rows][routed width], summed over this rank's experts
};

pub const Kernels = struct {
    router: cuda.route_rows.RouteRows,
    glue: cuda.moe_glue.MoeGlue,
    experts: affine.Kernels,

    pub fn load(driver: *const cuda.Driver) !Kernels {
        var router = try cuda.route_rows.RouteRows.load(driver);
        errdefer router.unload();
        var glue = try cuda.moe_glue.MoeGlue.load(driver);
        errdefer glue.unload();
        return .{ .router = router, .glue = glue, .experts = try affine.Kernels.load(driver) };
    }

    pub fn deinit(self: *Kernels) void {
        self.experts.deinit();
        self.glue.unload();
        self.router.unload();
        self.* = undefined;
    }

    pub fn forward(self: *const Kernels, s: cuda.Stream, w: Weights, b: Buffers, shape: Shape) !void {
        const a = shape.experts;
        try a.check();
        if (b.expert_rows == 0 or !shape.route.score or !shape.route.normalize or shape.route.shared or
            @as(usize, a.rows) * shape.route.topk != a.pairs) return error.BadHy4Route;
        try self.router.logits(s, b.hidden, shape.hidden, w.router, b.logits, a.rows, shape.hidden, shape.route.experts);
        try self.glue.topk(s, b.logits, w.correction_bias, shape.route, b.ids, b.probabilities, a.rows);
        try self.experts.run(s, w.precision, .up, .{
            .input = b.latent,
            .ids = b.ids,
            .src_rows = b.expert_rows,
            .weights = w.gate_up,
            .output = b.activated,
        }, a);
        try self.experts.run(s, w.precision, .down, .{
            .input = b.activated,
            .ids = b.ids,
            .weights = w.down,
            .output = b.contributions,
        }, a);
        try self.glue.combine(s, b.contributions, .f32, b.probabilities, b.rank_output, .f32, a.rows, shape.route.topk, a.width);
    }
};
