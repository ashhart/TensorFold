//! Combined CUDA contract set for all 26 `hy4_route`, `hy4_rows`, and `hy4_dfly2_glue` kernels + 15 controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hr = @import("hy4_route_rows_cuda");
const options = @import("contract_options");
const check = @import("check.zig");
const route_k = @import("contract_hy4_route.zig");
const rows_k = @import("contract_hy4_rows.zig");
const argmax_k = @import("contract_hy4_argmax.zig");
const sample_k = @import("contract_hy4_sample_rows.zig");
const copy_k = @import("contract_hy4_rows_copy.zig");
const ops_k = @import("contract_hy4_rows_ops.zig");
const exact_k = @import("contract_hy4_rows_exact.zig");
const dfly_k = @import("contract_hy4_dfly2.zig");

const g = core.contract.gates;

const ctl_img = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

pub const Set = struct {
    gpa: std.mem.Allocator,
    rows: hr.Rows,
    route: hr.route.Route,
    dfly: hr.dfly2.Kernels,
    dense: cuda.dense_rows.DenseRows,
    moe: cuda.moe_glue.MoeGlue,
    mla: exact_k.MlaRef,
    ctl_mod: ?cuda.Module = null,
    rows_cases: [10]rows_k.Case = undefined,
    rows_n: usize = 0,
    route_cases: [4]route_k.Case = undefined,
    route_n: usize = 0,
    argmax_cases: [3]argmax_k.Case = undefined,
    argmax_n: usize = 0,
    sample_cases: [4]sample_k.Case = undefined,
    sample_n: usize = 0,
    copy_cases: [6]copy_k.CopyCase = undefined,
    copy_n: usize = 0,
    num_cases: [6]ops_k.NumCase = undefined,
    num_n: usize = 0,
    kv_paged_cases: [2]ops_k.KvWritePagedCase = undefined,
    kv_paged_n: usize = 0,
    dfly_norm_cases: [3]dfly_k.NormCase = undefined,
    dfly_norm_n: usize = 0,
    dfly_exact_cases: [3]dfly_k.ExactCase = undefined,
    dfly_exact_n: usize = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var rows = try hr.Rows.load(gpu.d);
        errdefer rows.unload();
        var route = try hr.route.Route.load(gpu.d);
        errdefer route.unload();
        var dfly = try hr.dfly2.Kernels.load(gpu.d);
        errdefer dfly.unload();
        var dense = try cuda.dense_rows.DenseRows.load(gpu.d);
        errdefer dense.unload();
        var moe = try cuda.moe_glue.MoeGlue.load(gpu.d);
        errdefer moe.unload();
        var mla = try exact_k.MlaRef.load(gpu.d);
        errdefer mla.unload();
        const ctl_mod = if (options.controls) try cuda.Module.load(gpu.d, &ctl_img.bytes) else null;
        set.* = .{ .gpa = gpu.gpa, .rows = rows, .route = route, .dfly = dfly, .dense = dense, .moe = moe, .mla = mla, .ctl_mod = ctl_mod };
        errdefer if (set.ctl_mod) |*m| m.unload();

        set.rows_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = set.rows.wide_fn, .dense = &set.dense, .tile = .wide, .ctl = null, .name = "cuda hy4_rows hy4_rows_wide" };
        set.rows_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = set.rows.narrow_fn, .dense = &set.dense, .tile = .narrow, .ctl = null, .name = "cuda hy4_rows hy4_rows_narrow" };
        set.rows_n = 2;

        set.route_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.route.m4, .moe = &set.moe, .fault = .none, .name = "cuda hy4_route tf_hy4_route_m4" };
        set.route_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.route.m4_plan1, .moe = &set.moe, .fault = .none, .plan1 = true, .name = "cuda hy4_route tf_hy4_route_m4_plan1" };
        set.route_n = 2;

        set.argmax_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.rows.argmax_fn, .fault = .none, .name = "cuda hy4_rows hy4_rows_argmax" };
        set.argmax_n = 1;

        set.sample_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.rows.f32_to_bf16_fn, .mode = .f32_to_bf16, .fault = .none, .name = "cuda hy4_rows hy4_rows_f32_to_bf16" };
        set.sample_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = set.rows.interleave_bf16_fn, .mode = .interleave_bf16, .fault = .none, .name = "cuda hy4_rows hy4_rows_interleave_bf16" };
        set.sample_n = 2;

        const copy_kinds = [_]copy_k.CopyKind{ .slice_bf16, .interleave2_bf16, .slice_f32, .splice_streams, .vit_merge_gather, .vit_append_newline };
        for (copy_kinds, 0..) |kind, i| set.copy_cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = &set.rows, .kind = kind };
        set.copy_n = copy_kinds.len;

        const num_kinds = [_]ops_k.NumKind{ .comb_swiglu, .vit_qkv, .q_rope, .q_rope_mrope, .kv_write, .kv_write_ex };
        for (num_kinds, 0..) |kind, i| set.num_cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = &set.rows, .mla = &set.mla, .kind = kind };
        set.num_n = num_kinds.len;

        set.kv_paged_cases[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = &set.rows, .wrong_page = false };
        set.kv_paged_n = 1;

        const dfly_norms = [_]dfly_k.NormKind{ .dual, .fusion, .boundary };
        for (dfly_norms, 0..) |kind, i| set.dfly_norm_cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.dfly, .kind = kind };
        set.dfly_norm_n = dfly_norms.len;

        const dfly_exacts = [_]dfly_k.ExactKind{ .extend_topk, .pick_argmax, .gather_hidden };
        for (dfly_exacts, 0..) |kind, i| set.dfly_exact_cases[i] = .{ .rig = .{ .gpu = gpu, .s = s }, .k = &set.dfly, .kind = kind };
        set.dfly_exact_n = dfly_exacts.len;

        if (set.ctl_mod) |m| {
            const row_ctls = [_]struct { tile: hr.Tile, sym: [:0]const u8, gate: g.Gate, name: []const u8 }{
                .{ .tile = .wide, .sym = "hy4_rows_wide_bad_nan", .gate = .nonfinite, .name = "control: hy4_rows hy4_rows_wide drops NaN" },
                .{ .tile = .wide, .sym = "hy4_rows_wide_bad_repeat", .gate = .repeat, .name = "control: hy4_rows hy4_rows_wide drifts on repeat" },
                .{ .tile = .wide, .sym = "hy4_rows_wide_bad_tail", .gate = .shapes, .name = "control: hy4_rows hy4_rows_wide skips tail" },
                .{ .tile = .wide, .sym = "hy4_rows_wide_bad_sums", .gate = .sums, .name = "control: hy4_rows hy4_rows_wide sums in fp16" },
                .{ .tile = .narrow, .sym = "hy4_rows_narrow_bad_nan", .gate = .nonfinite, .name = "control: hy4_rows hy4_rows_narrow drops NaN" },
                .{ .tile = .narrow, .sym = "hy4_rows_narrow_bad_repeat", .gate = .repeat, .name = "control: hy4_rows hy4_rows_narrow drifts on repeat" },
                .{ .tile = .narrow, .sym = "hy4_rows_narrow_bad_tail", .gate = .shapes, .name = "control: hy4_rows hy4_rows_narrow skips tail" },
                .{ .tile = .narrow, .sym = "hy4_rows_narrow_bad_sums", .gate = .sums, .name = "control: hy4_rows hy4_rows_narrow sums in fp16" },
            };
            for (row_ctls) |ctl| {
                set.rows_cases[set.rows_n] = .{ .rig = .{ .gpu = gpu, .s = s }, .fn_handle = try m.function(ctl.sym), .dense = &set.dense, .tile = ctl.tile, .ctl = ctl.gate, .name = ctl.name };
                set.rows_n += 1;
            }
            set.route_cases[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_route_high_ties"), .moe = &set.moe, .fault = .high_ties, .name = "control: hy4_route tf_hy4_route_m4 high ties" };
            set.route_cases[3] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_route_distinct"), .moe = &set.moe, .fault = .distinct, .name = "control: hy4_route tf_hy4_route_m4 distinct past -inf" };
            set.route_n = 4;
            set.argmax_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_argmax_bad_nan"), .fault = .bad_nan, .name = "control: hy4_rows hy4_rows_argmax drops NaN" };
            set.argmax_cases[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_argmax_bad_tie"), .fault = .bad_tie, .name = "control: hy4_rows hy4_rows_argmax breaks ties high" };
            set.argmax_n = 3;
            set.sample_cases[2] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_f32_to_bf16_bad_nan"), .mode = .f32_to_bf16, .fault = .bad_nan, .name = "control: hy4_rows hy4_rows_f32_to_bf16 drops NaN" };
            set.sample_cases[3] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try m.function("tf_ctl_hy4_interleave_bad_nan"), .mode = .interleave_bf16, .fault = .bad_nan, .name = "control: hy4_rows hy4_rows_interleave_bf16 drops NaN" };
            set.sample_n = 4;
            set.kv_paged_cases[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = &set.rows, .wrong_page = true };
            set.kv_paged_n = 2;
        }
        for (set.rows_cases[0..set.rows_n]) |*x| try list.append(a, x.kernel());
        for (set.route_cases[0..set.route_n]) |*x| try list.append(a, x.kernel());
        for (set.argmax_cases[0..set.argmax_n]) |*x| try list.append(a, x.kernel());
        for (set.sample_cases[0..set.sample_n]) |*x| try list.append(a, x.kernel());
        for (set.copy_cases[0..set.copy_n]) |*x| try list.append(a, x.kernel());
        for (set.num_cases[0..set.num_n]) |*x| try list.append(a, x.kernel());
        for (set.kv_paged_cases[0..set.kv_paged_n]) |*x| try list.append(a, x.kernel());
        for (set.dfly_norm_cases[0..set.dfly_norm_n]) |*x| try list.append(a, x.kernel());
        for (set.dfly_exact_cases[0..set.dfly_exact_n]) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (set.dfly_exact_cases[0..set.dfly_exact_n]) |*x| x.deinit();
        for (set.dfly_norm_cases[0..set.dfly_norm_n]) |*x| x.deinit();
        for (set.kv_paged_cases[0..set.kv_paged_n]) |*x| x.deinit();
        for (set.num_cases[0..set.num_n]) |*x| x.deinit();
        for (set.copy_cases[0..set.copy_n]) |*x| x.deinit();
        for (set.sample_cases[0..set.sample_n]) |*x| x.deinit();
        for (set.argmax_cases[0..set.argmax_n]) |*x| x.deinit();
        for (set.route_cases[0..set.route_n]) |*x| x.deinit();
        for (set.rows_cases[0..set.rows_n]) |*x| x.deinit();
        if (set.ctl_mod) |*m| m.unload();
        set.mla.unload();
        set.moe.unload();
        set.dense.unload();
        set.dfly.unload();
        set.route.unload();
        set.rows.unload();
        set.gpa.destroy(set);
    }
};
