//! Idol's DINOv3 kernel tests' shared parts: float64 values at bf16 points, bf16 inputs and the encoder's library.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const id = @import("metal_idol_common.zig");

pub const tf = id.tf;
pub const c = id.c;
pub const g = id.g;
pub const rows = id.rows;
pub const Range = id.Range;
pub const Ends = id.Ends;
pub const Control = id.Control;
pub const kref = id.kref;
pub const noSums = id.noSums;
pub const periodic = id.periodic;
pub const int = id.int;
pub const sign = id.sign;
pub const ks = tf.idol.dino.kernels;
pub const cfg = tf.idol.dino.config;
pub const W = cfg.width;
pub const D = cfg.head_dim;

/// A value rounded once to bf16 (ties to even).
pub const bf = contract.bf16_round.bf16;

/// A range's ends each rounded to bf16.
pub fn bfr(r: Range) Range {
    return .{ .lo = bf(r.lo), .hi = bf(r.hi) };
}

/// fp32 bits rounded to bf16 bits, and back.
pub const bits = contract.bf16_round.bits;
pub const value = contract.bf16_round.value;

/// bf16 values uniform in [lo, hi].
pub fn randomBf(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = bits(lo + rng.float(f32) * (hi - lo));
}

pub const nan: u16 = 0x7fc0;
pub const inf: u16 = 0x7f80;
pub const ninf: u16 = 0xff80;

pub const div_ulps = id.div_ulps;
pub const rsqrt_ulps = id.rsqrt_ulps;
pub const exp_ulps = id.exp_ulps;

/// The encoder's source as the engine compiles it (or a control's copy), its pipelines made as tests ask for them.
pub const Lib = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    ctl: ?Control,
    k: ks.Kernels = undefined,
    built: bool = false,
    made: [ks.names.len]bool = @splat(false),

    fn text(l: *Lib) ![]u8 {
        var out = try tf.frags.source(l.device, l.a, sources.idol_dino);
        const x = l.ctl orelse return out;
        for (x.edits) |e| {
            errdefer l.a.free(out);
            if (std.mem.count(u8, out, e.from) != 1) {
                std.debug.print("control {s}: text not found once: {s}\n", .{ x.what, e.from });
                return error.ControlTextNotFoundOnce;
            }
            const next = try std.mem.replaceOwned(u8, l.a, out, e.from, e.to);
            l.a.free(out);
            out = next;
        }
        if (x.append.len == 0) return out;
        defer l.a.free(out);
        return std.mem.concat(l.a, u8, &.{ out, "\n", x.append });
    }

    /// The pipeline `name` made on this library (compiled on first use).
    pub fn need(l: *Lib, name: []const u8) !void {
        if (!l.built) {
            const src = try l.text();
            defer l.a.free(src);
            l.k.own = try mtl.Library.fromSource(l.device, src, mtl.CompileOptions.mlx());
            l.built = true;
        }
        for (ks.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) {
            if (!l.made[i]) l.k.p[i] = try mtl.Pipeline.init(l.device, l.k.own, n, false);
            l.made[i] = true;
            return;
        };
        return error.NoSuchPipeline;
    }

    pub fn deinit(l: *Lib) void {
        for (l.k.p, l.made) |p, m| if (m) p.deinit();
        if (l.built) l.k.own.deinit();
    }
};

test "bf16 ranges round both ends to nearest even" {
    const r = bfr(.{ .lo = 1 + 0x1p-8, .hi = 1 + 3 * 0x1p-8 });
    try std.testing.expectEqual(@as(f64, 1), r.lo);
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-6), r.hi);
}
