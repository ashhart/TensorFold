//! Inkling 2's routed experts on core exl3_experts: core's item plan, grouped kernels and softcapped epilogue.
const std = @import("std");
const cuda = @import("cuda");

const ex = cuda.exl3_experts;
const grouped = cuda.grouped;

/// The two SwiGLU softcaps (config.json mlp_gate_softcap, mlp_up_softcap).
pub const Caps = struct { gate: f32, up: f32 };

/// The least and most half-bit widths (k2) among a layer's expert tensors, as the loader read them.
pub const Bits = struct { lo: u32, hi: u32 };

/// Calls of at least this many rows take 32-member items on core's wide kernel (64 columns a block).
pub const wide_from = 512;

/// Core's experts module and the plan kernels of the shared experts image.
pub const Kernels = struct {
    main: ex.Experts,
    mod: cuda.Module,
    router: grouped.Router,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var main = try ex.Experts.load(d);
        errdefer main.unload();
        var mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer mod.unload();
        return .{ .main = main, .mod = mod, .router = try grouped.Router.resolve(mod) };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
        k.main.unload();
    }
};

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// The plan's buffers past core's scratch: members, items, counts, then the wide plan's rank and hist.
fn planSizes(l: ex.Layer, rows: usize, slots: usize) [5]usize {
    const p = rows * slots;
    const hist = if (p > grouped.small) (p + grouped.small - 1) / grouped.small * (l.experts + 1) else 1;
    return .{ p * 4, grouped.maxItems(p, l.experts + 1, 16) * 12, 8, p * 4, hist * 4 };
}

/// Core's scratch, then the item plan.
pub fn scratchBytes(l: ex.Layer, rows: usize, slots: usize) !usize {
    var n = try ex.scratchBytes(l, rows, slots);
    for (planSizes(l, rows, slots)) |b| n += align256(b);
    return n;
}

/// Where the plan sits in scratch laid out by ex.scratch over scratchBytes().
fn planAt(l: ex.Layer, sc: ex.Scratch) grouped.Plan {
    var at = sc.members + align256(@min(sc.rows * sc.slots, l.experts) * sc.rows * 4);
    var ptr: [5]u64 = undefined;
    for (&ptr, planSizes(l, sc.rows, sc.slots)) |*q, b| {
        q.* = at;
        at += align256(b);
    }
    return .{ .members = ptr[0], .items = ptr[1], .counts = ptr[2], .rank = ptr[3], .hist = ptr[4] };
}

/// exl3_experts.Experts.runWith on core's item plan, with the softcaps as act_mode 3.
pub fn run(e: *const Kernels, s: cuda.Stream, l: ex.Layer, caps: Caps, bits: Bits, x: u64, x_stride: usize, x_dtype: ex.DType, pick: u64, wts: u64, rows: usize, out: u64, sc: ex.Scratch) !void {
    if (rows < 1 or rows > sc.rows or sc.slots < 1 or sc.slots > 32) return error.BadExperts;
    if (l.experts < 1 or l.experts > 512 or l.codebook != .mul1 or l.gu_stride != 0) return error.BadExperts;
    if (bits.lo < 2 or bits.hi > 16 or bits.lo > bits.hi) return error.BadExpertBits; // core decodes 1 to 8 bits
    var c = l;
    c.act_mode = 3;
    c.caps = .{ caps.gate, caps.up };
    const it: ex.Items = .{ .router = &e.router, .plan = planAt(l, sc), .tile = if (rows >= wide_from) 32 else 16, .reject_invalid = true };
    try e.main.runWith(s, c, x, x_stride, x_dtype, pick, wts, rows, out, sc, 0, it);
}

test "experts: Inkling's expert shapes have built tiles (gate/up 4608 -> 2304 whole and a rank's 1152, down back)" {
    try std.testing.expectEqual(@as(u32, 4), try ex.plan(4608, 2304, true));
    try std.testing.expectEqual(@as(u32, 4), try ex.plan(4608, 1152, true));
    _ = try ex.plan(2304, 4608, false);
    _ = try ex.plan(1152, 4608, false);
    // the wide kernel's 64 columns and the rows kernel's 128 divide a rank's 1152 and the 4608 width
    try std.testing.expect(1152 % 128 == 0 and 4608 % 128 == 0);
}

test "experts: the item plan sits past core's scratch and holds a 2,048-row call's items" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const bytes = try scratchBytes(l, 2048, 8);
    const sc = try ex.scratch(0, l, 2048, 8);
    const pl = planAt(l, sc);
    try std.testing.expectEqual(@as(u64, try ex.scratchBytes(l, 2048, 8)), pl.members);
    try std.testing.expect(pl.items - pl.members >= 16384 * 4);
    try std.testing.expect(pl.counts - pl.items >= (16384 / 16 + 385) * 12);
    try std.testing.expect(pl.hist + 16 * 385 * 4 <= bytes);
}

test "experts: run refuses expert widths outside core's 1 to 8 bits on the host" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    for ([_]Bits{ .{ .lo = 1, .hi = 8 }, .{ .lo = 2, .hi = 17 }, .{ .lo = 6, .hi = 4 } }) |b| {
        try std.testing.expectError(error.BadExpertBits, run(undefined, undefined, l, caps, b, 0, 0, .f16, 0, 0, 4, 0, sc));
    }
}

test "experts: run refuses a call or layer its kernels do not take, before any launch" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    const four: Bits = .{ .lo = 8, .hi = 8 };
    try std.testing.expectError(error.BadExperts, run(undefined, undefined, l, caps, four, 0, 0, .f16, 0, 0, 5, 0, sc));
    l.codebook = .mcg;
    try std.testing.expectError(error.BadExperts, run(undefined, undefined, l, caps, four, 0, 0, .f16, 0, 0, 4, 0, sc));
}

test "experts: run refuses expert widths outside the kernels' 1 to 4 bits on the host" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    for ([_]Bits{ .{ .lo = 1, .hi = 8 }, .{ .lo = 2, .hi = 9 }, .{ .lo = 6, .hi = 4 } }) |b| {
        try std.testing.expectError(error.BadExpertBits, run(undefined, undefined, undefined, l, caps, b, 0, 0, .f16, 0, 0, 4, 0, sc));
    }
}

test "experts: run refuses a call or layer its kernels do not take, before any launch" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 384;
    l.dims = 4608;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    const four: Bits = .{ .lo = 8, .hi = 8 };
    try std.testing.expectError(error.BadExperts, run(undefined, undefined, undefined, l, caps, four, 0, 0, .f16, 0, 0, 5, 0, sc));
    l.codebook = .mcg;
    try std.testing.expectError(error.BadExperts, run(undefined, undefined, undefined, l, caps, four, 0, 0, .f16, 0, 0, 4, 0, sc));
}
