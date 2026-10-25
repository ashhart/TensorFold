//! Format admission and production launch selection across MLX, ModelOpt and EXL3, with an inert CUDA ABI.
const std = @import("std");
const cuda = @import("cuda");
const geo = @import("src/families/qwen3_5/cuda_geometry.zig");
const Engine = @import("src/families/qwen3_5/cuda_engine.zig").Engine;
const Kit = @import("src/families/qwen3_5/cuda_exl3_weights.zig").Kit;
const modelopt = @import("src/families/qwen3_5/cuda_modelopt.zig");
const Proj = @import("src/families/qwen3_5/cuda_layers.zig").Proj;
const t = std.testing;
const mlx = "\"quantization\":{\"bits\":4,\"group_size\":64,\"mode\":\"affine\"}";
const fixture =
    \\{"model_type":"qwen3_5_moe","quantization":{"bits":4,"group_size":64,"mode":"affine"},"tie_word_embeddings":false,
    \\ "text_config":{"model_type":"qwen3_5_moe_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
    \\ "head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "hidden_size":2048,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_experts":256,
    \\ "num_experts_per_tok":8,"num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,
    \\ "linear_num_value_heads":32,"num_attention_heads":16,"num_key_value_heads":2,"rms_norm_eps":1e-06,
    \\ "max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;
fn config(quant: []const u8, moe: bool, pocket: bool) ![]u8 {
    var s = try std.mem.replaceOwned(u8, t.allocator, fixture, mlx, quant);
    errdefer t.allocator.free(s);
    if (!moe) {
        const n = try std.mem.replaceOwned(u8, t.allocator, s, "qwen3_5_moe", "qwen3_5");
        t.allocator.free(s);
        s = n;
        const d = try std.mem.replaceOwned(u8, t.allocator, s, "moe_intermediate_size", "intermediate_size");
        t.allocator.free(s);
        s = d;
    }
    if (pocket) {
        const n = try std.mem.replaceOwned(u8, t.allocator, s, "\"hidden_size\":2048", "\"hidden_size\":2048,\"tensorfold_learned_neurons\":64");
        t.allocator.free(s);
        s = n;
        const d = try std.mem.replaceOwned(u8, t.allocator, s, "\"shared_expert_intermediate_size\":512", "\"shared_expert_intermediate_size\":448");
        t.allocator.free(s);
        s = d;
    }
    return s;
}
const exl3 = "\"quantization_config\":{\"quant_method\":\"exl3\"}";
const ct = "\"quantization_config\":{\"quant_method\":\"compressed-tensors\"}";

test "format coexistence: EXL3 stays MoE only and ModelOpt keeps its admission and refusals" {
    const x = try config(exl3, true, false);
    defer t.allocator.free(x);
    const gx = try geo.parse(t.allocator, x);
    try t.expect(gx.exl3);
    try t.expect(!gx.kinds.modelopt);
    const m = try config(ct, false, false);
    defer t.allocator.free(m);
    const gm = try geo.parse(t.allocator, m);
    try t.expect(gm.kinds.modelopt and gm.kinds.ct and !gm.exl3);
    try t.expectEqual(@as(u8, 16), gm.bits.embed);
    for ([_]struct { q: []const u8, moe: bool, pocket: bool, err: anyerror }{
        .{ .q = exl3, .moe = false, .pocket = false, .err = error.UnsupportedQwenQuantization },
        .{ .q = exl3, .moe = true, .pocket = true, .err = error.UnsupportedQwenQuantization },
        .{ .q = ct, .moe = true, .pocket = false, .err = error.UnsupportedQwenQuantization },
        .{ .q = ct, .moe = false, .pocket = true, .err = error.UnsupportedLearningFormat },
        .{ .q = "\"quantization_config\":{\"quant_method\":\"gptq\"}", .moe = true, .pocket = false, .err = error.UnsupportedQwenQuantization },
    }) |bad| {
        const c = try config(bad.q, bad.moe, bad.pocket);
        defer t.allocator.free(c);
        try t.expectError(bad.err, geo.parse(t.allocator, c));
    }
}

test "format coexistence: an MLX table retains main precedence and all parsed fields" {
    const ordinary = try config(mlx, true, false);
    defer t.allocator.free(ordinary);
    const want = try geo.parse(t.allocator, ordinary);
    for ([_][]const u8{ exl3, ct, "\"quantization_config\":[]", "\"quantization_config\":false" }) |extra| {
        const q = try std.fmt.allocPrint(t.allocator, "{s},{s}", .{ mlx, extra });
        defer t.allocator.free(q);
        const c = try config(q, true, false);
        defer t.allocator.free(c);
        try t.expectEqualDeep(want, try geo.parse(t.allocator, c));
    }
    const bad_q = try std.mem.replaceOwned(u8, t.allocator, mlx, "\"group_size\":64", "\"group_size\":32");
    defer t.allocator.free(bad_q);
    const joined = try std.fmt.allocPrint(t.allocator, "{s},{s}", .{ bad_q, exl3 });
    defer t.allocator.free(joined);
    const bad = try config(joined, true, false);
    defer t.allocator.free(bad);
    try t.expectError(error.UnsupportedQwenQuantization, geo.parse(t.allocator, bad));
}

const Recorded = struct {
    var count: usize = 0;
    var id: usize = 0;
    var first: [3]u64 = undefined;
    var params: u64 = 0;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        count += 1;
        id = @intFromPtr(f);
        for (&first, 0..) |*p, i| p.* = @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
        if (id == 3) params = @as(*const u64, @ptrCast(@alignCast(args.?[4].?))).*;
        return 0;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
fn fullKernels(d: *cuda.Driver) modelopt.Kernels {
    var out: modelopt.Kernels = undefined;
    out.part = null;
    out.lane.clusters = false;
    for (&out.lane.fns) |*set| for (set) |*f| {
        f.* = function(d, 5);
    };
    out.prompt.fp4 = function(d, 6);
    out.prompt.fp8 = function(d, 6);
    out.embed.bf16_copies_fn = function(d, 2);
    return out;
}

test "format coexistence: real embedding dispatch retains each owner's BF16 or MLX path" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Recorded.launch;
    var e: Engine = undefined;
    e.stream = .{ .d = &d, .handle = null };
    e.g.hidden = 256;
    e.embed_w = 0x1000;
    e.embed_s = 0x2000;
    e.embed_b = 0x3000;
    e.fns.embed = function(&d, 10);
    e.fns.embed8 = function(&d, 11);
    var kit: Kit = undefined;
    kit.b16.embed_fn = function(&d, 3);
    for ([_]u8{ 4, 8, 16 }) |bits| for ([_]bool{ false, true }) |ex| {
        if (ex and bits != 16) continue;
        e.kit = if (ex) &kit else null;
        e.mo = if (!ex and bits == 16) fullKernels(&d) else null;
        e.embed_bits = bits;
        for ([_]usize{ 1, 16, 17, 512 }) |rows| {
            Recorded.count = 0;
            try e.embed(0x4000, 0x5000, rows, 0x6000);
            try t.expectEqual(@as(usize, 1), Recorded.count);
            try t.expectEqual(@as(usize, if (ex) 3 else if (bits == 16) 2 else if (bits == 8) 11 else 10), Recorded.id);
            try t.expectEqual(@as(u64, 0x4000), Recorded.first[0]);
            try t.expectEqual(@as(u64, 0x1000), Recorded.first[1]);
            if (ex or bits == 16) try t.expectEqual(@as(u64, 0x5000), Recorded.first[2]);
            if (ex) try t.expectEqual(@as(u64, 0x6000), Recorded.params);
        }
    };
}

test "format coexistence: all projection tags retain their dimensions and BF16 input dispatch" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Recorded.launch;
    var e: Engine = undefined;
    e.stream = .{ .d = &d, .handle = null };
    e.mo = fullKernels(&d);
    var kit: Kit = undefined;
    kit.b16.dense_fn = function(&d, 4);
    e.kit = &kit;
    const q: cuda.qmmf.Weight = .{ .mode = .bf16, .codes = 0x7000, .scales = 0, .n = 64, .k = 64, .npad = 128 };
    const bf: Proj = .{ .bf = .{ .w = 0x7000, .n = 64, .k = 64 } };
    const ql: Proj = .{ .ql = .{ .bf16 = q } };
    const projections = [_]Proj{
        .{ .a4 = .{ .w = 0, .s = 0, .b = 0, .n = 64, .k = 64, .npad = 128 } },
        .{ .an = .{ .w = 0, .s = 0, .b = 0, .n = 64, .k = 64, .npad = 128, .bits = 8 } },
        ql,
        bf,
        .{ .x3 = .{ .trellis = 0, .suh = 0, .svh = 0, .n = 64, .k = 64, .halves = 8, .codebook = .mul1, .stride_k = 32, .stride_nb = 128 } },
    };
    for (projections) |p| {
        try t.expectEqual(@as(usize, 64), p.inputs());
        try t.expectEqual(@as(usize, 64), p.outputs());
    }
    for ([_]Proj{ ql, bf }) |p| for (1..18) |rows| for ([_]bool{ false, true }) |prompt| {
        Recorded.count = 0;
        try e.project(p, 0x8000, 0x9000, 0xa000, rows, prompt, 0xb000, true);
        try t.expectEqual(@as(usize, 1), Recorded.count);
        try t.expectEqual(@as(usize, if (p == .ql) 5 else 4), Recorded.id);
        try t.expectEqual(@as(u64, 0x8000), Recorded.first[0]);
    };
}

const PromptTrace = struct {
    const Call = struct { grid: [3]u32, block: [3]u32, shared: u32, stream: usize, ptrs: [3]u64, n: i32, k: i32 };
    var calls: [64]Call = undefined;
    var count: usize = 0;
    fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        std.debug.assert(@intFromPtr(f) == 4 and count < calls.len);
        var ptrs: [3]u64 = undefined;
        for (&ptrs, 0..) |*p, i| p.* = @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
        calls[count] = .{ .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .shared = shared, .stream = @intFromPtr(s), .ptrs = ptrs, .n = @as(*const i32, @ptrCast(@alignCast(args.?[3].?))).*, .k = @as(*const i32, @ptrCast(@alignCast(args.?[4].?))).* };
        count += 1;
        return 0;
    }
};

fn bf16Prompt(rows: usize) !void {
    const c = try config(exl3, true, false);
    defer t.allocator.free(c);
    const g = try geo.parse(t.allocator, c);
    try t.expect(g.exl3 and g.moe());
    try t.expectEqual(@as(usize, 2048), g.hidden);
    try t.expectEqual(@as(usize, 32), g.v_heads);
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = PromptTrace.launch;
    var kit: Kit = undefined;
    kit.b16.dense_fn = function(&d, 4);
    var e: Engine = undefined;
    e.stream = .{ .d = &d, .handle = @ptrFromInt(0x1800) };
    e.g = g;
    e.kit = &kit;
    e.affine.split = null;
    e.xs = 0;
    e.tables.ptr = 0x100000;
    const x = 0x10000;
    const outs = [_]u64{ 0x2000000, 0x3000000 };
    const weights = [_]u64{ 0x4000000, 0x5000000 };
    // The loader stores both linear_attn.in_proj_a and in_proj_b as BF16 [value_heads, hidden].
    const ab = [_]Proj{
        .{ .bf = .{ .w = weights[0], .n = g.v_heads, .k = g.hidden } },
        .{ .bf = .{ .w = weights[1], .n = g.v_heads, .k = g.hidden } },
    };
    PromptTrace.count = 0;
    try e.projMany(&ab, x, &outs, rows, true);
    var at: usize = 0;
    for (outs, weights) |out, weight| {
        var covered: usize = 0;
        while (covered < rows) {
            try t.expect(at < PromptTrace.count);
            const call = PromptTrace.calls[at];
            try t.expectEqual([3]u32{ @intCast(g.v_heads), @intCast(@min(rows - covered, 128)), 1 }, call.grid);
            try t.expectEqual([3]u32{ 256, 1, 1 }, call.block);
            try t.expectEqual(@as(u32, 0), call.shared);
            try t.expectEqual(@as(usize, 0x1800), call.stream);
            try t.expectEqual([3]u64{ x + covered * g.hidden * 2, weight, out + covered * g.v_heads * 2 }, call.ptrs);
            try t.expectEqual(@as(i32, @intCast(g.v_heads)), call.n);
            try t.expectEqual(@as(i32, @intCast(g.hidden)), call.k);
            covered += call.grid[1];
            at += 1;
        }
        try t.expectEqual(rows, covered);
    }
    try t.expectEqual(at, PromptTrace.count);
}

test "format coexistence: EXL3 BF16 prompt: 2048 rows reach both GDN scalar projections without RowShape" {
    try bf16Prompt(2048);
}

test "format coexistence: EXL3 BF16 prompt: 4096 rows reach both GDN scalar projections without RowShape" {
    try bf16Prompt(4096);
}

test "format coexistence: EXL3 BF16 prompt: existing rows and partial final slices keep every argument and row" {
    for ([_]usize{ 0, 1, 16, 17, 127, 128, 129, 2049, 4095 }) |rows| try bf16Prompt(rows);
}
