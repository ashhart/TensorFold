//! The optional vendor prompt path owns scratch before KV sizing and never receives drafter weights.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");
const Proj = E.Proj;

pub fn owns(e: *const E.Engine, address: u64) bool {
    return address >= e.arena.ptr and address - e.arena.ptr < e.arena.len;
}

fn mode(p: Proj, native: bool) ?cuda.cublaslt_prefill.Mode {
    return switch (p) {
        .a4 => .affine4,
        .ql => |w| switch (w) {
            .nvfp4 => if (native) .nvfp4 else null,
            .fp8 => if (cuda.fp8_prefill.enabled()) .fp8 else null,
            .fp8g => if (cuda.fp8_prefill.enabled()) .fp8_row else null,
            else => null,
        },
        else => null,
    };
}

fn add(e: *const E.Engine, list: *std.ArrayList(Proj), p: Proj, native: bool) !void {
    if (p == .ql and p.ql == .fp8g and !e.g.kinds.ct) return;
    const kind = mode(p, native) orelse return;
    for (list.items) |prior| if (mode(prior, native) == kind and prior.inputs() == p.inputs() and
        prior.outputs() == p.outputs()) return;
    try list.append(e.gpa, p);
}

fn projectionWeight(p: Proj) cuda.qmmf.Weight {
    return switch (p.ql) {
        .nvfp4, .fp8, .fp8g => |w| w,
        else => unreachable,
    };
}

fn projections(e: *const E.Engine, native: bool) !std.ArrayList(Proj) {
    var list: std.ArrayList(Proj) = .empty;
    errdefer list.deinit(e.gpa);
    for (e.layers) |layer| {
        switch (layer.mixer) {
            .delta => |m| for ([_]Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| try add(e, &list, p, native),
            .attention => |m| for ([_]Proj{ m.q, m.k, m.v, m.o }) |p| try add(e, &list, p, native),
        }
        switch (layer.mlp) {
            .dense => |m| for ([_]Proj{ m.gate, m.up, m.down }) |p| try add(e, &list, p, native),
            .moe => {},
        }
    }
    return list;
}

pub fn reserve(k: anytype, e: *const E.Engine) !void {
    if (!try cuda.cublaslt_prefill.requested()) return;
    if (!@import("cuda_geometry.zig").dense27(e.g)) return error.UnsupportedCublasPrefill;
    var list = try projections(e, k.native_requested);
    defer list.deinit(e.gpa);
    if (list.items.len == 0) return error.UnsupportedCublasPrefill;
    var kit = try cuda.cublaslt_prefill.Kit.open(e.d);
    errdefer kit.deinit();
    for (list.items) |p| {
        const npad = if (p == .a4) p.a4.npad else projectionWeight(p).npad;
        try kit.include(mode(p, k.native_requested).?, E.max_rows, p.outputs(), p.inputs(), npad);
    }
    try kit.allocate();
    std.log.info("cuBLAS prompt scratch: {d} bytes", .{kit.bytes()});
    k.vendor = kit;
}

/// Real projection shapes are warmed on zero scratch, before the cache planner reads remaining VRAM.
pub fn warm(e: *E.Engine) !void {
    const k = if (e.mo) |*m| m else return;
    const v = if (k.vendor) |*kit| kit else return;
    var list = try projections(e, k.native_requested);
    defer list.deinit(e.gpa);
    var ran: usize = 0;
    var tuned: usize = 0;
    for (@import("qwen_prefill_config").warm_rows) |rows| for (list.items) |p| {
        const used = if (p == .a4)
            try v.affine4(e.stream, e.act, rows, p.a4, e.up, .bf16)
        else if (p.ql == .nvfp4)
            try v.nvfp4(e.stream, e.act, rows, projectionWeight(p), e.up, .bf16)
        else
            try v.fp8(e.stream, e.act, rows, projectionWeight(p), e.up, .bf16);
        ran += @intFromBool(used);
        if (used and try v.tryTuneLast()) tuned += 1;
    };
    try e.stream.synchronize();
    if (ran == 0) return error.UnsupportedCublasPrefill;
    std.log.info("cuBLAS prompt tuning: {d}/{d} warmed shapes tuned", .{ tuned, ran });
}
