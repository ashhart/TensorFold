//! Unlimited-OCR-2's decoder kernels (uocr_dec_*.cuh) under the contract, each with its planted controls.
const std = @import("std");
const core = @import("core");
const uocr = @import("contract_uocr.zig");
const rows = @import("contract_uocr_decoder_rows.zig");
const glue = @import("contract_uocr_decoder_glue.zig");
const route = @import("contract_uocr_decoder_route.zig");
const attend = @import("contract_uocr_decoder_attend.zig");
const experts_ = @import("contract_uocr_decoder_experts.zig");
const mv = @import("contract_uocr_decoder_mv.zig");

pub const c = core.contract.spec;
pub const g = core.contract.gates;
/// The float64 helpers the CUDA contract's row tests share: ranges, ends, bf16 rounding, planted values.
pub const mi = @import("contract_moss_image_ref.zig");
pub const kernels = uocr.kernels;
pub const Kernels = kernels.Kernels;

pub const H = uocr.hidden;
pub const D = uocr.router_dim;
pub const heads = uocr.heads;
pub const experts = uocr.experts;
pub const topk = uocr.topk;
pub const expert_ffn = uocr.expert_ffn;
pub const shared_ffn = uocr.shared_ffn;
pub const dense_ffn = uocr.dense_ffn;
pub const vocab = uocr.vocab;
pub const eps: f32 = 1e-6;

/// The release's clamp at +-1e4: NaN stays NaN.
pub fn clamp(x: f64) f64 {
    return if (std.math.isNan(x)) x else @min(@max(x, -1e4), 1e4);
}

/// 1 / sqrt(128) and 128 / sqrt(128) as the kernels hold them (fp32).
pub const inv_sqrt_hd: f32 = 0.0883883476483184405;
pub const router_scale: f32 = 11.3137084989847604;

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    norms: [5]rows.Norm = undefined,
    qks: [4]rows.Qk = undefined,
    keys: [3]glue.BlockKeys = undefined,
    routes: [3]route.Route = undefined,
    attends: [4]attend.Attend = undefined,
    gates: [3]route.Gate = undefined,
    silus: [4]glue.Silu = undefined,
    gus: [3]experts_.Experts = undefined,
    downs: [3]experts_.Experts = undefined,
    combines: [3]glue.Combine = undefined,
    matvecs: [3]mv.Matvec = undefined,
    gathers: [2]mv.Gather = undefined,

    const fields = .{ "norms", "qks", "keys", "routes", "attends", "gates", "silus", "gus", "downs", "combines", "matvecs", "gathers" };

    fn init(ctx: uocr.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const gpa = ctx.gpu.gpa;
        const set = try gpa.create(Set);
        errdefer gpa.destroy(set);
        set.* = .{ .gpa = gpa };
        const k = ctx.k.*;
        const ends: mi.Ends = .{ .a = gpa };
        set.norms = .{
            .{ .rig = ctx.rig(), .k = k, .add = true, .name = "cuda uocr tf_uocr_norm residual add", .ends = ends },
            .{ .rig = ctx.rig(), .k = k, .add = false, .name = "cuda uocr tf_uocr_norm", .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("norm", "tf_ctl_uocr_norm"), .add = true, .name = "control: uocr tf_uocr_norm drops bf16(r inv)", .control = .shapes, .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("norm", "tf_ctl_uocr_norm_mean"), .add = false, .name = "control: uocr tf_uocr_norm, the mean of squares divides by the threads", .control = .shapes, .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("norm", "tf_ctl_uocr_norm_sign"), .add = true, .name = "control: uocr tf_uocr_norm, the added rows are subtracted", .control = .shapes, .ends = ends },
        };
        set.qks = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_qk rotary, caches and router rows", .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("qk", "tf_ctl_uocr_qk"), .name = "control: uocr tf_uocr_qk drops bf16(x cos)", .control = .shapes, .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("qk", "tf_ctl_uocr_qk_turn"), .name = "control: uocr tf_uocr_qk, the rotation turns the other way", .control = .shapes, .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("qk", "tf_ctl_uocr_qk_clamp"), .name = "control: uocr tf_uocr_qk, the router's clamp turns NaN into -1e4", .control = .nonfinite, .ends = ends },
        };
        set.keys = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_block_keys" },
            .{ .rig = ctx.rig(), .k = try ctx.with("block_keys", "tf_ctl_uocr_block_keys"), .name = "control: uocr tf_uocr_block_keys drops 1 / sqrt(128)", .control = .shapes },
            .{ .rig = ctx.rig(), .k = try ctx.with("block_keys", "tf_ctl_uocr_block_keys_sum"), .name = "control: uocr tf_uocr_block_keys, the weights are not divided by their sum", .control = .shapes },
        };
        set.routes = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_route (remote blocks, entmax biases)" },
            .{ .rig = ctx.rig(), .k = try ctx.with("route", "tf_ctl_uocr_route"), .name = "control: uocr tf_uocr_route ties to the higher block", .control = .shapes },
            .{ .rig = ctx.rig(), .k = try ctx.with("route", "tf_ctl_uocr_route_mean"), .name = "control: uocr tf_uocr_route, the log-weights keep their mean", .control = .shapes },
        };
        set.attends = .{
            .{ .rig = ctx.rig(), .k = k, .prompt = true, .name = "cuda uocr tf_uocr_attend prompt rows (bf16 biases)" },
            .{ .rig = ctx.rig(), .k = k, .prompt = false, .name = "cuda uocr tf_uocr_attend decode step" },
            .{ .rig = ctx.rig(), .k = try ctx.with("attend", "tf_ctl_uocr_attend"), .prompt = true, .name = "control: uocr tf_uocr_attend reads 129 local keys", .control = .shapes },
            .{ .rig = ctx.rig(), .k = try ctx.with("attend", "tf_ctl_uocr_attend_bias"), .prompt = true, .name = "control: uocr tf_uocr_attend, the read blocks' biases are dropped", .control = .shapes },
        };
        set.gates = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_moe_gate" },
            .{ .rig = ctx.rig(), .k = try ctx.with("moe_gate", "tf_ctl_uocr_moe_gate"), .name = "control: uocr tf_uocr_moe_gate ties to the higher expert", .control = .shapes },
            .{ .rig = ctx.rig(), .k = try ctx.with("moe_gate", "tf_ctl_uocr_moe_gate_mass"), .name = "control: uocr tf_uocr_moe_gate, the scores are not divided by their sum", .control = .shapes },
        };
        set.silus = .{
            .{ .rig = ctx.rig(), .k = k, .halves = true, .name = "cuda uocr tf_uocr_silu_mul gate | up rows", .ends = ends },
            .{ .rig = ctx.rig(), .k = k, .halves = false, .name = "cuda uocr tf_uocr_silu_mul two inputs", .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("silu_mul", "tf_ctl_uocr_silu_mul"), .halves = true, .name = "control: uocr tf_uocr_silu_mul drops bf16(silu(g))", .control = .shapes, .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("silu_mul", "tf_ctl_uocr_silu_mul_swap"), .halves = true, .name = "control: uocr tf_uocr_silu_mul, the gate is read from the up rows", .control = .shapes, .ends = ends },
        };
        set.gus = .{
            .{ .rig = ctx.rig(), .k = k, .gu = true, .name = "cuda uocr tf_uocr_expert_gu" },
            .{ .rig = ctx.rig(), .k = try ctx.with("expert_gu", "tf_ctl_uocr_expert_gu"), .gu = true, .name = "control: uocr tf_uocr_expert_gu sums in bf16", .control = .sums },
            .{ .rig = ctx.rig(), .k = try ctx.with("expert_gu", "tf_ctl_uocr_expert_gu_ids"), .gu = true, .name = "control: uocr tf_uocr_expert_gu, pick j reads expert j", .control = .shapes },
        };
        set.downs = .{
            .{ .rig = ctx.rig(), .k = k, .gu = false, .name = "cuda uocr tf_uocr_expert_down" },
            .{ .rig = ctx.rig(), .k = try ctx.with("expert_down", "tf_ctl_uocr_expert_down"), .gu = false, .name = "control: uocr tf_uocr_expert_down skips 8 columns", .control = .shapes },
            .{ .rig = ctx.rig(), .k = try ctx.with("expert_down", "tf_ctl_uocr_expert_down_ids"), .gu = false, .name = "control: uocr tf_uocr_expert_down, pick j reads expert j", .control = .shapes },
        };
        set.combines = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_combine", .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("combine", "tf_ctl_uocr_combine"), .name = "control: uocr tf_uocr_combine drops bf16(sum)", .control = .shapes, .ends = ends },
            .{ .rig = ctx.rig(), .k = try ctx.with("combine", "tf_ctl_uocr_combine_shared"), .name = "control: uocr tf_uocr_combine, the shared expert is not added", .control = .shapes, .ends = ends },
        };
        set.matvecs = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_matvec" },
            .{ .rig = ctx.rig(), .k = try ctx.with("matvec", "tf_ctl_uocr_matvec"), .name = "control: uocr tf_uocr_matvec, each row's last 8 columns are dropped", .control = .shapes },
            .{ .rig = ctx.rig(), .k = try ctx.with("matvec", "tf_ctl_uocr_matvec_half"), .name = "control: uocr tf_uocr_matvec, the dot products sum in half", .control = .sums },
        };
        set.gathers = .{
            .{ .rig = ctx.rig(), .k = k, .name = "cuda uocr tf_uocr_gather rows" },
            .{ .rig = ctx.rig(), .k = try ctx.with("gather", "tf_ctl_uocr_gather"), .name = "control: uocr tf_uocr_gather, row i reads rows[i + 1]", .control = .shapes },
        };
        inline for (fields) |f| for (&@field(set, f)) |*x| try list.append(a, x.kernel());
        return set;
    }

    fn deinit(set: *Set) void {
        inline for (fields) |f| for (&@field(set, f)) |*x| x.deinit();
        set.gpa.destroy(set);
    }

    fn release(ptr: *anyopaque) void {
        deinit(@ptrCast(@alignCast(ptr)));
    }
};

/// The decoder group's registrations appended to `list`; the returned Held frees them once the run is done.
pub fn register(ctx: uocr.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !uocr.Held {
    const set = try Set.init(ctx, a, list);
    return .{ .ptr = set, .free = Set.release };
}
