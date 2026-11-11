//! One ModelOpt projection read by name and checked: weight, e4m3 block scales, weight_scale_2, input_scale; views.
const std = @import("std");
const ckpt = @import("checkpoint.zig");
const nvfp4 = @import("nvfp4.zig");
const Why = @import("modelopt.zig").Why;

const DType = ckpt.DType;
const Checkpoint = ckpt.Checkpoint;
const Tensor = ckpt.Tensor;

pub const Error = error{UnexpectedTensor};

/// A part's stored form: NVFP4 (u8 codes two to a byte), FP8 (e4m3 under one fp32 scale) or bf16.
pub const Kind = enum { nvfp4, fp8, bf16 };

/// The kind a weight's dtype says (the sniff for checkpoints without recipes), else null.
pub fn kindOf(dtype: DType) ?Kind {
    return switch (dtype) {
        .u8 => .nvfp4,
        .f8_e4m3 => .fp8,
        .bf16 => .bf16,
        else => null,
    };
}

fn dtypeOf(kind: Kind) DType {
    return switch (kind) {
        .nvfp4 => .u8,
        .fp8 => .f8_e4m3,
        .bf16 => .bf16,
    };
}

/// input_scale: left to the family (not read, not marked used), read when stored, or required.
pub const Act = enum { skip, optional, required };

/// A scale's stored shape: one element (shape [], [1] or [1, 1]: Qwen, Nemotron) or rank 0 only (shape []: Dory).
pub const Scalars = enum { one, rank0 };

/// What the family knows: the kind (null: by the weight's dtype), [n, k] if fixed, the input scale, a positive scale.
pub const Want = struct { kind: ?Kind = null, n: ?usize = null, k: ?usize = null, act: Act = .optional, positive: bool = false, scalars: Scalars = .one };

/// A part as stored: the weight, NVFP4's e4m3 blocks, its fp32 scale (weight_scale_2 or FP8's weight_scale), act.
pub const Part = struct { weight: Tensor, blocks: ?Tensor = null, scale: f32 = 1, act: ?f32 = null, kind: Kind, n: usize, k: usize };

/// The [n, k] a weight of `kind` stores (NVFP4: two inputs a byte), checked against `want`; storage-free.
pub fn weightDims(kind: Kind, weight: anytype, want: Want) Error![2]usize {
    if (weight.rank != 2 or weight.dtype != dtypeOf(kind)) return error.UnexpectedTensor;
    const n = weight.shape[0];
    const k = std.math.mul(usize, weight.shape[1], if (kind == .nvfp4) 2 else 1) catch return error.UnexpectedTensor;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    if (want.n) |x| if (x != n) return error.UnexpectedTensor;
    if (want.k) |x| if (x != k) return error.UnexpectedTensor;
    return .{ n, k };
}

/// NVFP4's block scales for an [n, k] weight: e4m3 [n, k / 16] (core.nvfp4.dims' rule); storage-free.
pub fn checkBlocks(blocks: anytype, n: usize, k: usize) Error!void {
    if (blocks.rank != 2) return error.UnexpectedTensor;
    const w: nvfp4.Meta = .{ .dtype = .u8, .shape = &.{ n, k / 2 } };
    const one: nvfp4.Meta = .{ .dtype = .f32, .shape = &.{} };
    const d = nvfp4.dims(w, .{ .dtype = blocks.dtype, .shape = blocks.shape[0..2] }, one, null) catch return error.UnexpectedTensor;
    if (d.n != n or d.k != k) return error.UnexpectedTensor;
}

/// An fp32 tensor of one element (shape [] or [1]).
pub fn scalar(t: Tensor) Error!f32 {
    if (t.dtype != .f32 or t.bytes.len != 4 or t.numel() != 1) return error.UnexpectedTensor;
    return @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little));
}

/// `module`'s part from `ck`, each tensor it reads marked used; a refusal is logged by module and tensor.
pub fn read(ck: *Checkpoint, module: []const u8, want: Want) !Part {
    var why: Why = .{};
    return readWhy(ck, module, want, &why) catch |e| {
        if (why.len > 0) std.log.err("{s}", .{why.text()});
        return e;
    };
}

/// `read`, the refusal kept in `why` instead of the log (tests read it).
pub fn readWhy(ck: *Checkpoint, module: []const u8, want: Want, why: *Why) !Part {
    const w = try fetch(ck, module, "weight", why);
    const kind = want.kind orelse kindOf(w.dtype) orelse return refuse(why, module, "weight", w, want);
    const nk = weightDims(kind, w, want) catch return refuse(why, module, "weight", w, want);
    var p: Part = .{ .weight = w, .kind = kind, .n = nk[0], .k = nk[1] };
    switch (kind) {
        .bf16 => {},
        .fp8 => p.scale = try scalarOf(ck, module, "weight_scale", want, why),
        .nvfp4 => {
            const b = try fetch(ck, module, "weight_scale", why);
            checkBlocks(b, p.n, p.k) catch return refuse(why, module, "weight_scale", b, want);
            p.blocks = b;
            p.scale = try scalarOf(ck, module, "weight_scale_2", want, why);
        },
    }
    if (want.positive and kind != .bf16 and !(p.scale > 0 and std.math.isFinite(p.scale))) {
        why.set("{s}.{s}: {e} is not a positive finite scale", .{ module, if (kind == .fp8) "weight_scale" else "weight_scale_2", p.scale });
        return error.UnexpectedTensor;
    }
    var buf: [320]u8 = undefined;
    const act = try std.fmt.bufPrint(&buf, "{s}.input_scale", .{module});
    if (want.act == .required or (want.act == .optional and ck.has(act))) p.act = try scalarOf(ck, module, "input_scale", want, why);
    return p;
}

/// `{module}.{tail}`, marked used; a missing one refused by name (error.MissingTensor, as Checkpoint.get).
fn fetch(ck: *Checkpoint, module: []const u8, tail: []const u8, why: *Why) !Tensor {
    var buf: [320]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ module, tail });
    if (!ck.has(name)) {
        why.set("{s}: no {s}, a ModelOpt part needs it", .{ module, tail });
        return error.MissingTensor;
    }
    return ck.get(name);
}

/// `{module}.{tail}` as an fp32 scalar of `want.scalars`' shape, refused by name.
fn scalarOf(ck: *Checkpoint, module: []const u8, tail: []const u8, want: Want, why: *Why) !f32 {
    const t = try fetch(ck, module, tail, why);
    const shape = if (want.scalars == .rank0) " of shape []" else "";
    if (want.scalars == .rank0 and t.rank != 0) return badScalar(why, module, tail, t, shape);
    return scalar(t) catch badScalar(why, module, tail, t, shape);
}

fn badScalar(why: *Why, module: []const u8, tail: []const u8, t: Tensor, shape: []const u8) Error {
    why.set("{s}.{s}: {t} {any}, expected an F32 scalar{s}", .{ module, tail, t.dtype, t.shape[0..t.rank], shape });
    return error.UnexpectedTensor;
}

fn refuse(why: *Why, module: []const u8, tail: []const u8, t: Tensor, want: Want) Error {
    const kind = if (want.kind) |k| @tagName(k) else "NVFP4, FP8 or bf16";
    why.set("{s}.{s}: {t} {any}, not a ModelOpt {s} part of [{?d}, {?d}]", .{ module, tail, t.dtype, t.shape[0..t.rank], kind, want.n, want.k });
    return error.UnexpectedTensor;
}

test {
    _ = @import("modelopt_part_test.zig");
}
