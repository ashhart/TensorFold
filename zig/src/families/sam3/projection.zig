const std = @import("std");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const dense_device = @import("dense_device.zig");
const dense_expansion = @import("dense_expansion.zig");
const dense_planes = @import("dense_planes.zig");
const dense_pair = @import("dense_pair.zig");

const Runtime = runtime.Runtime;
const Scope = runtime.Scope;
const Tensor = runtime.Tensor;

pub const Part = enum { query, key, value };
pub const Activation = enum { relu, gelu };
pub const TrunkRole = enum { qkv, attention_out, expand, contract };

/// y = x Wᵀ + b with `leaf.weight` `[out, in]` and `leaf.bias` `[out]`.
pub fn affine(s: Scope, leaf: []const u8, x: Tensor) !Tensor {
    return affineAt(try s.sub(leaf), x);
}

fn affineAt(p: Scope, x: Tensor) !Tensor {
    const shape = try p.info("weight");
    if (shape.rank != 2) return error.SamProjectionShape;
    const out = shape.dim(0);
    const in = shape.dim(1);
    if (in == 0 or out == 0 or x.count % in != 0) return error.SamProjectionShape;
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    if (weight.count != out * in or bias.count != out) return error.SamProjectionShape;
    const rows = x.count / in;
    if (p.rt.routes.exactDevice() and x.dtype == .f32 and weight.dtype == .f32 and bias.dtype == .f32 and
        out <= 4736 and in <= 4736)
    {
        const flags = try wideFlags(p, weight, bias, out, in);
        return p.rt.hold(try dense_device.runCached(p.rt.gpu, x, weight, bias, flags, rows, out, in, .wide));
    }
    return p.rt.hold(try p.rt.gpu.linear(x, weight, bias, rows, out, in));
}

/// One third of the packed `in_proj_weight` `[3W, W]` and `in_proj_bias` `[3W]` under `s`.
pub fn packedPart(s: Scope, x: Tensor, which: Part) !Tensor {
    const bias = try s.param("in_proj_bias");
    const weight = try s.param("in_proj_weight");
    if (bias.count == 0 or bias.count % 3 != 0) return error.SamProjectionShape;
    const width = bias.count / 3;
    if (weight.count != 3 * width * width or x.count % width != 0) return error.SamProjectionShape;
    const index: usize = @backingInt(which);
    const w = runtime.part(weight, index * width * width, width * width);
    const b = runtime.part(bias, index * width, width);
    const rows = x.count / width;
    if (s.rt.routes.exactDevice() and x.dtype == .f32 and w.dtype == .f32 and b.dtype == .f32 and width <= 4736) {
        const name = try s.rt.print("wide-operands.{s}.{s}", .{ try s.key("in_proj_weight"), @tagName(which) });
        const flags = try wideFlagsNamed(s.rt, name, w, b, width, width);
        return s.rt.hold(try dense_device.runCached(s.rt.gpu, x, w, b, flags, rows, width, width, .wide));
    }
    return s.rt.hold(try s.rt.gpu.linear(x, w, b, rows, width, width));
}

/// `leaf.layers.0 … layers.(depth−1)` with ReLU between layers and none after the last.
pub fn reluChain(s: Scope, leaf: []const u8, x: Tensor, depth: usize) !Tensor {
    if (depth == 0) return error.SamProjectionShape;
    const chain = try s.sub(leaf);
    var y = x;
    for (0..depth) |i| {
        y = try affineAt(try chain.item("layers", i), y);
        if (i + 1 < depth) y = try elementwise.relu(s.rt, y);
    }
    return y;
}

/// `linear1`, the activation, then `linear2` under `s`.
pub fn expandContract(s: Scope, x: Tensor, activation: Activation) !Tensor {
    const expanded = try affineAt(try s.sub("linear1"), x);
    const activated = switch (activation) {
        .relu => try elementwise.relu(s.rt, expanded),
        .gelu => try elementwise.gelu(s.rt, expanded),
    };
    return affineAt(try s.sub("linear2"), activated);
}

const TrunkKernel = enum { affine, affine_gelu, dense, dense_half, dense_gelu, dense_compensated, exact_expansion };

fn trunkKernel(exact_device: bool, weight_f16: bool, operand_exact: bool, role: TrunkRole) TrunkKernel {
    if (!exact_device) return if (role == .expand) .affine_gelu else .affine;
    if (!weight_f16) return .dense;
    return switch (role) {
        .qkv, .attention_out => .dense_half,
        .expand => if (operand_exact) .exact_expansion else .dense_gelu,
        .contract => .dense_compensated,
    };
}

fn wideFlagsNamed(rt: *Runtime, name: []const u8, weight: Tensor, bias: Tensor, cols: usize, depth: usize) !Tensor {
    if (rt.constant(name)) |flags| return flags;
    const flags = try dense_device.operandFlags(rt.gpu, weight, bias, cols, depth);
    errdefer flags.deinit();
    return rt.adoptConstant(name, flags);
}

/// The weight and bias domain flags of the wide layouts for the weight under `p`, computed once and kept.
pub fn wideFlags(p: Scope, weight: Tensor, bias: Tensor, cols: usize, depth: usize) !Tensor {
    const name = try p.rt.print("wide-operands.{s}", .{try p.key("weight")});
    return wideFlagsNamed(p.rt, name, weight, bias, cols, depth);
}

/// The device layout of a trunk projection whose checkpoint weight is not FP16.
pub fn fp32WeightLayout(role: TrunkRole) dense_device.Layout {
    return switch (role) {
        .qkv, .attention_out => .wide,
        .expand => .wide_gelu,
        .contract => .wide_compensated,
    };
}

pub const WideOut = struct { tensor: Tensor, flag: ?Tensor = null };

/// True when the trunk projection `leaf` under `s` uses the FP32 wide device route.
pub fn takesWide(s: Scope, leaf: []const u8) !bool {
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    return s.rt.routes.exactDevice() and shape.dtype != .f16;
}

/// True when the trunk projection `leaf` under `s` uses the 3-pass FP16-plane NAX route for FP32 weights.
pub fn takesPair(s: Scope, leaf: []const u8) !bool {
    if (!s.rt.routes.exactDevice() or s.rt.observer != null) return false;
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    return shape.rank == 2 and shape.dtype != .f16 and shape.dim(0) <= 4736 and shape.dim(1) <= 4736;
}

fn pairWeight(p: Scope, weight: Tensor, bias: Tensor, cols: usize, depth: usize) !Tensor {
    const name = try p.rt.print("pair-weight.{s}", .{try p.key("weight")});
    if (p.rt.constant(name)) |cached| return cached;
    const pair = try dense_pair.prepareWeight(p.rt.gpu, weight, bias, cols, depth);
    errdefer pair.deinit();
    return p.rt.adoptConstant(name, pair);
}

/// 3-pass FP16-plane trunk projection on FP32 checkpoint weights (accepts FP32 or pair input).
pub fn trunkPair(s: Scope, leaf: []const u8, x: Tensor, role: TrunkRole, residual: ?Tensor, tiled: bool) !Tensor {
    const rt = s.rt;
    if (!rt.routes.exactDevice()) return error.SamProjectionRoute;
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 2 or shape.dtype == .f16) return error.SamProjectionShape;
    const cols = shape.dim(0);
    const depth = shape.dim(1);
    const rows = if (x.dtype == .f32) x.count / depth else (x.count - 16) / (4 * depth);
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    const w_pair = try pairWeight(p, weight, bias, cols, depth);
    const epilogue: dense_pair.Epilogue = if (role == .expand)
        .gelu_pair
    else if (residual) |r|
        (if (tiled) .{ .tile_residual = r } else .{ .residual = r })
    else
        .none;
    return rt.hold(try dense_pair.run(rt.gpu, x, w_pair, weight, bias, rows, cols, depth, epilogue));
}

/// Fused FP32 wide trunk projection with optional precomputed input flag, residual/tile epilogue, and output flag.
pub fn trunkWide(s: Scope, leaf: []const u8, x: Tensor, role: TrunkRole, input_flag: ?Tensor, residual: ?Tensor, tiled: bool, emit_flag: bool) !WideOut {
    const rt = s.rt;
    if (!rt.routes.exactDevice()) return error.SamProjectionRoute;
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 2 or shape.dtype == .f16) return error.SamProjectionShape;
    const cols = shape.dim(0);
    const depth = shape.dim(1);
    if (cols == 0 or depth == 0 or x.count % depth != 0) return error.SamProjectionShape;
    const rows = x.count / depth;
    const weight = try p.param("weight");
    const bias = try p.param("bias");
    const wflags = try wideFlags(p, weight, bias, cols, depth);
    const epilogue: dense_device.WideEpilogue = if (residual) |r|
        (if (tiled) .{ .tile_residual = r } else .{ .residual = r })
    else
        .none;
    const res = try dense_device.runWide(rt.gpu, x, weight, bias, wflags, input_flag, rows, cols, depth, fp32WeightLayout(role), epilogue, emit_flag);
    const out = rt.hold(res.out) catch |err| {
        if (res.flag) |f| f.deinit();
        return err;
    };
    const flag = if (res.flag) |f| try rt.hold(f) else null;
    return .{ .tensor = out, .flag = flag };
}

/// The FP32 trunk projection routes; `expand` returns the GELU-activated expansion.
pub fn trunkFp32(s: Scope, leaf: []const u8, x: Tensor, role: TrunkRole) !Tensor {
    const rt = s.rt;
    if (rt.routes.precision != .fp32) return error.SamProjectionRoute;
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 2) return error.SamProjectionShape;
    const cols = shape.dim(0);
    const depth = shape.dim(1);
    if (cols == 0 or depth == 0 or x.count % depth != 0) return error.SamProjectionShape;
    const half = shape.dtype == .f16;
    const device = rt.routes.exactDevice();
    const full: ?Tensor = if (device and half and role == .expand) try p.param("weight") else null;
    const kernel = trunkKernel(device, half, if (full) |w| w.tensor_operand_exact else false, role);
    switch (kernel) {
        .affine => return affineAt(p, x),
        .affine_gelu => return elementwise.gelu(rt, try affineAt(p, x)),
        else => {},
    }
    const rows = x.count / depth;
    const bias = try p.param("bias");
    const weight = full orelse if (kernel == .dense_half) try rt.paramAs(try p.key("weight"), .f16) else try p.param("weight");
    const g = rt.gpu;
    return rt.hold(try switch (kernel) {
        .dense => dense_device.runCached(g, x, weight, bias, try wideFlags(p, weight, bias, cols, depth), rows, cols, depth, fp32WeightLayout(role)),
        .dense_half => dense_device.run(g, x, weight, bias, rows, cols, depth, .nk_half),
        .dense_gelu => dense_device.run(g, x, weight, bias, rows, cols, depth, .nk_gelu),
        .dense_compensated => dense_device.run(g, x, weight, bias, rows, cols, depth, .nk_compensated),
        .exact_expansion => dense_expansion.run(g, x, weight, bias, rows, cols, depth, true),
        .affine, .affine_gelu => unreachable,
    });
}

fn planesRoute(routes: runtime.Routes, weight_f16: bool, role: TrunkRole, key: []const u8) bool {
    const gelu = role == .expand;
    return routes.exactDevice() and weight_f16 and gelu == std.mem.endsWith(u8, key, ".mlp.fc1");
}

/// Route P-planes holds for the trunk projection `leaf` under `s`: it runs on the FP16 planes kernels.
pub fn takesPlanes(s: Scope, leaf: []const u8, role: TrunkRole) !bool {
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    return planesRoute(s.rt.routes, shape.dtype == .f16, role, p.path);
}

/// Route P-planes on FP32 rows or planes; residual roles add `residual` (by grid row if `tiled`); expand gives planes.
pub fn trunkPlanes(s: Scope, leaf: []const u8, x: Tensor, role: TrunkRole, residual: ?Tensor, tiled: bool) !Tensor {
    const rt = s.rt;
    const p = try s.sub(leaf);
    const shape = try p.info("weight");
    if (shape.rank != 2) return error.SamProjectionShape;
    if (!planesRoute(rt.routes, shape.dtype == .f16, role, p.path)) return error.SamProjectionRoute;
    const cols = shape.dim(0);
    const depth = shape.dim(1);
    const values = if (x.dtype == .f32) x.count else dense_planes.values(x.count) orelse return error.SamProjectionShape;
    if (cols == 0 or depth == 0 or values % depth != 0) return error.SamProjectionShape;
    const adds_residual = role == .attention_out or role == .contract;
    if (adds_residual != (residual != null) or (tiled and role != .attention_out)) return error.SamProjectionShape;
    const accumulation: dense_planes.Accumulation = switch (role) {
        .qkv, .attention_out => .plain,
        .expand, .contract => .compensated,
    };
    const epilogue: dense_planes.Epilogue = switch (role) {
        .qkv => .none,
        .expand => .gelu_planes,
        .attention_out => if (tiled) .{ .tile_residual = residual.? } else .{ .residual = residual.? },
        .contract => .{ .residual = residual.? },
    };
    const weight = try rt.paramAs(try p.key("weight"), .f16);
    const bias = try p.param("bias");
    return rt.hold(try dense_planes.run(rt.gpu, x, weight, bias, values / depth, cols, depth, accumulation, epilogue));
}

/// Core FP32 GEMM y = x Wᵀ with x `[rows, depth]` and w `[cols, depth]`.
pub fn matmul(rt: *Runtime, x: Tensor, w: Tensor, rows: usize, cols: usize, depth: usize) !Tensor {
    return rt.hold(try rt.gpu.matmul(x, w, rows, cols, depth));
}

test "SAM trunk projection route follows the exact device route, weight dtype and operand exactness" {
    const expect = std.testing.expectEqual;
    for ([_]TrunkRole{ .qkv, .attention_out, .expand, .contract }) |role| {
        for ([_]bool{ false, true }) |half| for ([_]bool{ false, true }) |exact| {
            const plain: TrunkKernel = if (role == .expand) .affine_gelu else .affine;
            try expect(plain, trunkKernel(false, half, exact, role));
        };
    }
    for ([_]bool{ false, true }) |exact| {
        try expect(TrunkKernel.dense_half, trunkKernel(true, true, exact, .qkv));
        try expect(TrunkKernel.dense_half, trunkKernel(true, true, exact, .attention_out));
        try expect(TrunkKernel.dense_compensated, trunkKernel(true, true, exact, .contract));
        for ([_]TrunkRole{ .qkv, .attention_out, .expand, .contract }) |role| try expect(TrunkKernel.dense, trunkKernel(true, false, exact, role));
    }
    try expect(dense_device.Layout.wide, fp32WeightLayout(.qkv));
    try expect(dense_device.Layout.wide, fp32WeightLayout(.attention_out));
    try expect(dense_device.Layout.wide_gelu, fp32WeightLayout(.expand));
    try expect(dense_device.Layout.wide_compensated, fp32WeightLayout(.contract));
    try expect(TrunkKernel.exact_expansion, trunkKernel(true, true, true, .expand));
    try expect(TrunkKernel.dense_gelu, trunkKernel(true, true, false, .expand));
}

test "SAM trunk planes route needs FP32 with tensor units, an FP16 weight and GELU exactly on mlp.fc1" {
    const on = runtime.Routes{ .precision = .fp32, .tensor_units = true };
    const block = "detector.backbone.vision_backbone.trunk.blocks.3";
    try std.testing.expect(planesRoute(on, true, .qkv, block ++ ".attn.qkv"));
    try std.testing.expect(planesRoute(on, true, .attention_out, block ++ ".attn.proj"));
    try std.testing.expect(planesRoute(on, true, .expand, block ++ ".mlp.fc1"));
    try std.testing.expect(planesRoute(on, true, .contract, block ++ ".mlp.fc2"));
    try std.testing.expect(!planesRoute(on, true, .expand, block ++ ".mlp.fc2"));
    try std.testing.expect(!planesRoute(on, true, .contract, block ++ ".mlp.fc1"));
    try std.testing.expect(!planesRoute(on, false, .qkv, block ++ ".attn.qkv"));
    try std.testing.expect(!planesRoute(.{ .precision = .fp32 }, true, .qkv, block ++ ".attn.qkv"));
    try std.testing.expect(!planesRoute(.{ .precision = .mixed, .tensor_units = true }, true, .qkv, block ++ ".attn.qkv"));
}
