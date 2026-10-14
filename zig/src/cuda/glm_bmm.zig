//! GLM's BF16 matmul (qmm.py's `_bmm`) and router dots (glue.py): the Python engine's K slices and mma order.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// ROUTER_KS: the router's shape-fixed slice count; sliced and single-pass reductions differ in their last bits.
pub const router_ks: u32 = 8;

/// A BF16 matrix [n, k] as the checkpoint stores it (qmm.B16); a call multiplies x [m, k] through it.
pub const B16 = struct {
    w: u64, // bf16 [n, k] row-major, device
    n: u32,
    k: u32,
};

/// qmm.b16_split_k: K slices fixed by the shape alone, in units of 64, so every row's bits hold at any row count.
pub fn b16SplitK(n: usize, k: usize) u32 {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: u32 = 1;
    while (sk < 8 and tiles * sk < 192 and steps % (@as(usize, sk) * 2) == 0 and steps / (@as(usize, sk) * 2) >= 4) {
        sk *= 2;
    }
    return sk;
}

/// qmm.bucket: the row-block size one call runs at; no bucket changes a row's bits.
pub fn bucket(m: usize) u32 {
    for ([_]usize{ 16, 32, 64, 128 }) |b| {
        if (m <= b) return @intCast(b);
    }
    return 128;
}

fn bmIndex(bm: u32) usize {
    return switch (bm) {
        16 => 0,
        32 => 1,
        64 => 2,
        else => 3,
    };
}

const bm_names = [_][]const u8{ "16", "32", "64", "128" };

/// The loaded fatbin: eight matmul instantiations, the router's two passes and the partial reductions.
pub const Bmm = struct {
    mod: Module,
    bmm: [2][4]Function, // [f32 out][bm index]
    routers: [4]Function,
    router_part: [4]Function,
    reduce: Function,
    router_sum: Function,

    pub fn load(d: *const driver.Driver) !Bmm {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var b: Bmm = .{
            .mod = try Module.load(d, kernels.glm_bmm),
            .bmm = undefined,
            .routers = undefined,
            .router_part = undefined,
            .reduce = undefined,
            .router_sum = undefined,
        };
        errdefer b.mod.unload();
        inline for (0..2) |f| {
            inline for (0..4) |i| {
                b.bmm[f][i] = try b.mod.function(std.fmt.comptimePrint("tf_glm_bmm_b16_{s}_{d}", .{ bm_names[i], f }));
            }
        }
        inline for (0..4) |i| {
            b.routers[i] = try b.mod.function(std.fmt.comptimePrint("tf_glm_router_{s}", .{bm_names[i]}));
            b.router_part[i] = try b.mod.function(std.fmt.comptimePrint("tf_glm_router_part_{s}", .{bm_names[i]}));
        }
        b.reduce = try b.mod.function("tf_glm_bmm_reduce");
        b.router_sum = try b.mod.function("tf_glm_router_sum");
        return b;
    }

    pub fn unload(b: *Bmm) void {
        b.mod.unload();
    }

    /// fp32 partial bytes one split call needs (qmm._matmul_b16's `part`), 0 when the shape runs unsplit.
    pub fn partBytes(w: B16, m: usize) usize {
        const sk = b16SplitK(w.n, w.k);
        return if (sk > 1) @as(usize, sk) * m * w.n * 4 else 0;
    }

    /// out [m, n] = x [m, k] through w; `part` (sk*m*n*4 bytes) is required whenever b16SplitK splits K.
    pub fn matmul(b: *const Bmm, s: Stream, w: B16, x: u64, xs: i64, m: usize, out: u64, is_f32: bool, part: u64) !void {
        if (m < 1 or w.k == 0 or w.k % 64 != 0 or w.n == 0) return error.GlmBmmShape;
        const sk = b16SplitK(w.n, w.k);
        if (sk > 1 and part == 0) return error.GlmBmmScratch;
        const bm = bucket(m);
        var r: launch_.Args = .{};
        r.add(x);
        r.add(xs);
        r.add(w.w);
        r.add(out);
        r.add(part);
        r.add(@as(i32, @intCast(m)));
        r.add(@as(i32, @intCast(w.n)));
        r.add(@as(i32, @intCast(w.k)));
        r.add(@as(i32, @intCast(sk)));
        try launch_.launch(b.bmm[@intFromBool(is_f32)][bmIndex(bm)], .{
            .grid = .{ .x = @intCast((m + bm - 1) / bm), .y = @intCast((w.n + 63) / 64), .z = @intCast(sk) },
            .block = .{ .x = @as(u32, if (bm == 128) 256 else 128), .y = 1, .z = 1 },
            .shared = 0,
        }, s, &r);
        if (sk > 1) {
            var rr: launch_.Args = .{};
            rr.add(part);
            rr.add(out);
            rr.add(@as(i32, @intCast(m * w.n)));
            rr.add(@as(i32, @intCast(sk)));
            rr.add(@as(i32, @intFromBool(is_f32)));
            try launch_.launch(b.reduce, .{
                .grid = .{ .x = @intCast(((m * w.n) + 1023) / 1024), .y = 1, .z = 1 },
                .block = .{ .x = 1024, .y = 1, .z = 1 },
                .shared = 0,
            }, s, &rr);
        }
    }

    /// glue.router: fp32 [m, ne] logits = x [m, d] through the gate [ne, d]; the shape's sliced path, else one pass.
    pub fn router(b: *const Bmm, s: Stream, x: u64, xs: i64, w: u64, m: usize, ne: usize, d: usize, out: u64, part: u64) !void {
        if (m < 1 or ne == 0 or d == 0) return error.GlmBmmShape;
        const bm: u32 = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
        const i = bmIndex(bm);
        if (router_ks <= 1 or d % (router_ks * 64) != 0) {
            var r: launch_.Args = .{};
            r.add(x);
            r.add(xs);
            r.add(w);
            r.add(out);
            r.add(@as(u64, 0));
            r.add(@as(i32, @intCast(m)));
            r.add(@as(i32, @intCast(ne)));
            r.add(@as(i32, @intCast(d)));
            r.add(@as(i32, 1));
            try launch_.launch(b.routers[i], .{
                .grid = .{ .x = @intCast((m + bm - 1) / bm), .y = @intCast((ne + 63) / 64), .z = 1 },
                .block = .{ .x = 128, .y = 1, .z = 1 },
                .shared = 0,
            }, s, &r);
            return;
        }
        if (part == 0) return error.GlmBmmScratch;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(xs);
        r.add(w);
        r.add(out);
        r.add(part);
        r.add(@as(i32, @intCast(m)));
        r.add(@as(i32, @intCast(ne)));
        r.add(@as(i32, @intCast(d)));
        r.add(@as(i32, @intCast(router_ks)));
        try launch_.launch(b.router_part[i], .{
            .grid = .{ .x = @intCast((m + bm - 1) / bm), .y = @intCast((ne + 31) / 32), .z = @intCast(router_ks) },
            .block = .{ .x = 128, .y = 1, .z = 1 },
            .shared = 0,
        }, s, &r);
        var rr: launch_.Args = .{};
        rr.add(part);
        rr.add(out);
        rr.add(@as(i32, @intCast(m * ne)));
        rr.add(@as(i32, @intCast(router_ks)));
        try launch_.launch(b.router_sum, .{
            .grid = .{ .x = @intCast(((m * ne) + 1023) / 1024), .y = 1, .z = 1 },
            .block = .{ .x = 1024, .y = 1, .z = 1 },
            .shared = 0,
        }, s, &rr);
    }
};
