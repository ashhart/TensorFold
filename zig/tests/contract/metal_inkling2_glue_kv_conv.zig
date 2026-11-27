//! Paired K/V short convolution helpers for Inkling 2 Mini glue contract probes.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = host.plan;

pub fn isKvConv(kind: p.Kind) bool {
    return kind == .kv_conv_static or kind == .kv_conv_dynamic;
}

fn kvServed(comptime max_n: usize) [64]c.Shape {
    var out: [64]c.Shape = undefined;
    for (0..64) |i| {
        const width: usize = switch (i % 4) {
            0 => 1024,
            1 => 128,
            2 => 512,
            else => max_n,
        };
        out[i] = .{ .rows = i + 1, .n = if (i == 0 or i == 15 or i == 31 or i == 63) 1024 else width, .k = 4 };
    }
    return out;
}

pub const static_served = kvServed(p.hidden);
pub const dynamic_served = kvServed(1024);

pub fn fillFloats(a: std.mem.Allocator, n: usize, rng: std.Random) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*x| x.* = rng.float(f32) * 0.5 - 0.25;
    return out;
}

pub fn uploadBf16(a: std.mem.Allocator, rig: *rig_.Rig, values: []const f32) !rig_.Ref {
    const bits = try a.alloc(u16, values.len);
    for (bits, values) |*b, x| b.* = c.toBf16(x);
    return rig.input(std.mem.sliceAsBytes(bits));
}

fn dynamicRowProj(input_vals: []const f32, proj_vals: []const f32, r: usize) [32]f64 {
    var out: [32]f64 = undefined;
    for (0..32) |t| {
        var dot: f64 = 0;
        for (0..p.hidden) |d| dot += @as(f64, proj_vals[t * p.hidden + d]) * input_vals[r * p.hidden + d];
        out[t] = p.roundBf16(dot);
    }
    return out;
}

fn dynamicAt(taps: []const f32, static_w: []const f32, proj: *const [32]f64, channels: usize, r: usize, col: usize) f64 {
    var sum: f64 = taps[(r + 3) * channels + col];
    for (0..4) |lag| {
        const w = @as(f64, static_w[lag * channels + col]) + proj[4 * (col / 128) + lag];
        sum += w * taps[(r + 3 - lag) * channels + col];
    }
    return sum / 3;
}

/// Populate paired K and V inputs, optional non-finite sentinels, and Metal buffer refs.
pub fn prepare(
    a: std.mem.Allocator,
    rig: *rig_.Rig,
    kind: p.Kind,
    r: usize,
    n: usize,
    nf: bool,
    rng: std.Random,
    f: *[8][]f32,
    refs: *[8]rig_.Ref,
) !void {
    const count: usize = switch (kind) {
        .kv_conv_static => blk: {
            f[0] = try fillFloats(a, (r + 3) * n, rng);
            f[1] = try fillFloats(a, 4 * n, rng);
            f[2] = try fillFloats(a, (r + 3) * n, rng);
            f[3] = try fillFloats(a, 4 * n, rng);
            if (nf) {
                f[0][0] = std.math.nan(f32);
                f[2][0] = std.math.inf(f32);
            }
            break :blk 4;
        },
        .kv_conv_dynamic => blk: {
            f[0] = try fillFloats(a, (r + 3) * n, rng);
            f[1] = try fillFloats(a, r * p.hidden, rng);
            f[2] = try fillFloats(a, 4 * n, rng);
            f[3] = try fillFloats(a, 32 * p.hidden, rng);
            f[4] = try fillFloats(a, (r + 3) * n, rng);
            f[5] = try fillFloats(a, r * p.hidden, rng);
            f[6] = try fillFloats(a, 4 * n, rng);
            f[7] = try fillFloats(a, 32 * p.hidden, rng);
            if (nf) {
                f[1][0] = std.math.nan(f32);
                f[5][0] = std.math.nan(f32);
            }
            break :blk 8;
        },
        else => unreachable,
    };
    for (0..count) |j| refs[j] = try uploadBf16(a, rig, f[j]);
}

/// Run the two un-paired K+V conv dispatches on the exact same inputs and concatenate bytes.
pub fn replaced(
    a: std.mem.Allocator,
    rig: *rig_.Rig,
    pl: *const host.Pipelines,
    kind: p.Kind,
    refs: *const [8]rig_.Ref,
    base_out: *const [2]rig_.Ref,
    out: *std.ArrayList(u8),
    shape: c.Shape,
) ![]const u8 {
    const pass = rig.begin();
    const e = pass.e;
    const r: u32 = @intCast(shape.rows);
    const n: u32 = @intCast(shape.n);
    const half_bytes = shape.rows * shape.n * 2;
    const at = struct {
        fn ref(ref_: rig_.Ref) host.Ref {
            return .{ .buf = ref_.buf, .off = ref_.off };
        }
    }.ref;
    switch (kind) {
        .kv_conv_static => {
            host.convStatic(pl, e, at(refs[0]), at(refs[1]), at(base_out[0]), r, n);
            host.convStatic(pl, e, at(refs[2]), at(refs[3]), at(base_out[1]), r, n);
        },
        .kv_conv_dynamic => {
            host.convDynamic(pl, e, at(refs[0]), at(refs[1]), at(refs[2]), at(refs[3]), at(base_out[0]), r, n);
            host.convDynamic(pl, e, at(refs[4]), at(refs[5]), at(refs[6]), at(refs[7]), at(base_out[1]), r, n);
        },
        else => return error.UnsupportedReplaced,
    }
    try rig.finish(pass);
    out.clearRetainingCapacity();
    try out.appendSlice(a, rig.bytes(base_out[0], half_bytes));
    try out.appendSlice(a, rig.bytes(base_out[1], half_bytes));
    return out.items;
}

pub fn referenceRow(kind: p.Kind, f: *const [8][]f32, r: usize, n: usize, row: usize, want: []f64) void {
    const half = r * n;
    switch (kind) {
        .kv_conv_static => {
            for (0..n) |d| {
                want[row * n + d] = p.staticConv(f[0], f[1], n, row, d);
                want[half + row * n + d] = p.staticConv(f[2], f[3], n, row, d);
            }
        },
        .kv_conv_dynamic => {
            const k_proj = dynamicRowProj(f[1], f[3], row);
            const v_proj = dynamicRowProj(f[5], f[7], row);
            for (0..n) |d| {
                want[row * n + d] = dynamicAt(f[0], f[2], &k_proj, n, row, d);
                want[half + row * n + d] = dynamicAt(f[4], f[6], &v_proj, n, row, d);
            }
        },
        else => unreachable,
    }
}
