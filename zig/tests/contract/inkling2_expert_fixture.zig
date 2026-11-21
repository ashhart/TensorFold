//! Sparse-but-dimensioned checkpoint tensors for the expert probes. All tensors use original names and wire layout.
const std = @import("std");
const tf = @import("tensorfold");
const ck = tf.checkpoint_metal;
const cfg = tf.inkling2_metal.config;
const u = @import("inkling2_util.zig");
const rig_ = @import("rig.zig");
const math = @import("inkling2_expert_math.zig");
pub const Format = math.Format;
pub const experts = 9;
pub const topk = 8;
pub const gate_cap = 4;
pub const up_cap = 25;
pub const den: f32 = 2688;

pub fn configuration(shape: u.c.Shape) cfg.Config {
    var c: cfg.Config = undefined;
    c.text.hidden = @intCast(shape.k);
    c.text.expert_inter = @intCast(shape.n);
    c.text.layers = 1;
    c.text.experts = experts;
    c.text.topk = topk;
    c.text.gate_cap = gate_cap;
    c.text.up_cap = up_cap;
    return c;
}

pub const Index = struct { at: usize, code: u8, scale: u8 };

pub fn support(f: Format, fill: u.g.Fill, gate_up: bool, index: usize, expert: usize, k: usize, out: *[4]Index) []const Index {
    const nv = f == .nvfp4;
    if (fill == .stress) {
        const big: u8 = if (nv) 0x78 else 139;
        const small: u8 = if (nv) 0x28 else 129;
        const pos: u8 = if (nv) 7 else 0x7e;
        const neg: u8 = if (nv) 15 else 0xfe;
        if (gate_up) {
            out[0] = .{ .at = 0, .code = if (nv) 7 else 0x38, .scale = if (nv) 0x78 else 127 };
            return out[0..1];
        }
        const b = f.block();
        out.* = .{
            .{ .at = 0, .code = pos, .scale = big },
            .{ .at = b, .code = pos, .scale = small },
            .{ .at = 2 * b, .code = neg, .scale = big },
            .{ .at = 32 * b, .code = pos, .scale = small },
        };
        return out;
    }
    if (!nv and fill == .nonfinite and expert >= 7) {
        // On expert 7 the tiny input (UE8M0=0) is lifted by a 2^127 weight scale. Expert 8 uses the NaN scale.
        out[0] = .{ .at = 0, .code = 0x38, .scale = if (expert == 8) 255 else if (gate_up) 254 else 127 };
        return out[0..1];
    }
    const j = if (gate_up) index / 2 else index;
    const parity = if (gate_up) index % 2 else 0;
    const at = if (index % 11 == 0) 0 else (j * 37 + expert * 19 + parity * 23) % k;
    const code4 = [_]u8{ 1, 2, 3, 7, 4, 6, 14, 13 };
    const code8 = [_]u8{ 0x38, 0x40, 0xab, 0x3e, 0x54, 0xc3, 0x50, 0xbe };
    const scales4 = [_]u8{ 0x70, 0x78, 0x7c, 0x7e };
    const scales8 = [_]u8{ 125, 126, 127, 128 };
    out[0] = .{
        .at = at,
        .code = (if (nv) code4 else code8)[(j + expert + 3 * parity) % 8],
        .scale = (if (nv) scales4 else scales8)[(j + expert + parity) % 4],
    };
    return out[0..1];
}

pub fn wAmax(e: usize) f32 {
    return 1.0 + @as(f32, @floatFromInt(e)) * 0.5;
}

fn name(cp: *ck.Checkpoint, r: rig_.Ref, dtype: ck.DType, dims: []const usize, comptime part: []const u8, comptime suffix: []const u8) !void {
    var t: ck.Tensor = .{ .buffer = r.buf, .offset = r.off, .dtype = dtype, .rank = dims.len, .bytes = dtype.size() };
    for (dims, 0..) |v, i| {
        t.shape[i] = v;
        t.bytes *= v;
    }
    try cp.set("model.llm.layers.0.mlp.experts." ++ part ++ suffix, t);
}

fn matrix(allocator: std.mem.Allocator, rig: *rig_.Rig, cp: *ck.Checkpoint, f: Format, fill: u.g.Fill, n: usize, k: usize, comptime gate_up: bool) !void {
    const nv = f == .nvfp4;
    const block = f.block();
    const pitch = if (nv) k / 2 else k;
    const scale_pitch = k / block;
    const vals = try allocator.alloc(u8, experts * n * pitch);
    defer allocator.free(vals);
    const scales = try allocator.alloc(u8, experts * n * scale_pitch);
    defer allocator.free(scales);
    @memset(vals, 0);
    @memset(scales, if (nv) 0x38 else 127);
    for (0..experts) |e| for (0..n) |i| {
        const weight = vals[(e * n + i) * pitch ..][0..pitch];
        const scale = scales[(e * n + i) * scale_pitch ..][0..scale_pitch];
        var buf: [4]Index = undefined;
        for (support(f, fill, gate_up, i, e, k, &buf)) |term| {
            if (nv) {
                const sh: u3 = @intCast((term.at % 2) * 4);
                weight[term.at / 2] |= term.code << sh;
            } else weight[term.at] = term.code;
            scale[term.at / block] = term.scale;
        }
    };
    const part = if (gate_up) "w13" else "w2";
    try name(cp, try rig.input(vals), if (nv) .u8 else .f8_e4m3, &.{ experts, n, pitch }, part, "_weight");
    try name(cp, try rig.input(scales), if (nv) .f8_e4m3 else .u8, &.{ experts, n, scale_pitch }, part, "_weight_scale");
    if (nv) {
        var peaks: [experts]f32 = undefined;
        for (&peaks, 0..) |*v, e| v.* = wAmax(e);
        try name(cp, try rig.input(std.mem.asBytes(&peaks)), .f32, &.{experts}, part, "_weight_amax");
        try name(cp, try rig.input(std.mem.asBytes(&den)), .f32, &.{1}, part, "_weight_outer_scale_denominator");
    }
}

pub fn tensors(a: std.mem.Allocator, rig: *rig_.Rig, cp: *ck.Checkpoint, f: Format, fill: u.g.Fill, shape: u.c.Shape) !void {
    try matrix(a, rig, cp, f, fill, shape.n * 2, shape.k, true);
    try matrix(a, rig, cp, f, fill, shape.k, shape.n, false);
}

pub fn rows(f: Format, shape: u.c.Shape, fill: u.g.Fill, seed: u64, x: []u16, ids: []u32, weights: []f32) void {
    var random = std.Random.DefaultPrng.init(seed);
    const rng = random.random();
    // Distinct adjacent codes, mixed signs, a block straddling the 448/480 thresholds, out-of-range route IDs,
    // and a final K block populated by the same test data as the first.
    for (0..shape.rows) |r| {
        const row = x[r * shape.k ..][0..shape.k];
        u.fillDyadic(rng, row, 31, 16);
        row[0] = u.word(3.625);
        row[1] = u.word(-0.875);
        // Both NVFP4 candidates reconstruct this block identically: the calibration chooses six on the tie.
        if (r == 0) {
            @memset(row[1..f.block()], 0);
            row[f.block() - 1] = u.word(-3.625);
        }
        for (0..topk) |s| {
            ids[r * topk + s] = @intCast((r * 3 + s * 5) % experts);
            weights[r * topk + s] = @as(f32, @floatFromInt(1 + (s * 3 + r) % 7)) / 8.0;
        }
        if (r % 2 == 0) ids[r * topk + topk - 1] = std.math.maxInt(u32);
    }
    switch (fill) {
        .stress => {
            @memset(x, 0);
            @memset(ids, std.math.maxInt(u32));
            @memset(weights, 1);
            for (0..shape.rows) |r| {
                x[r * shape.k] = u.word(8);
                ids[r * topk] = 0;
            }
        },
        .nonfinite => {
            x[0] = u.nan_bits;
            x[shape.k] = u.pinf_bits;
            x[2 * shape.k] = u.ninf_bits;
            x[3 * shape.k] = u.ninf_bits;
            x[3 * shape.k + 1] = u.pinf_bits;
            for (ids) |*id| if (id.* >= 7) {
                id.* = std.math.maxInt(u32);
            };
            // One all-inactive token ignores even a non-finite input; one has a NaN routing weight.
            x[4 * shape.k] = u.nan_bits;
            @memset(ids[4 * topk ..][0..topk], std.math.maxInt(u32));
            ids[7 * topk] = 0;
            weights[7 * topk] = std.math.nan(f32);
            if (f == .mxfp8) {
                @memset(x[5 * shape.k ..][0..shape.k], 0);
                x[5 * shape.k] = u.word(0x1p-127);
                @memset(ids[5 * topk ..][0..2 * topk], std.math.maxInt(u32));
                ids[5 * topk] = 7;
                ids[6 * topk] = 8;
            }
        },
        else => {},
    }
}
