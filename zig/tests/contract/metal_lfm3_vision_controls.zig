//! Planted controls for the image tower: each probe's kernel text with one fault, which one gate must report.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ct = @import("contract");
const vision = @import("metal_lfm3_vision.zig");
const g = ct.gates;

pub const Kind = vision.Kind;

const Edit = struct { from: []const u8, to: []const u8, what: []const u8, gate: g.Gate };

fn edit(kind: Kind) Edit {
    return switch (kind) {
        .dense => .{ .from = "float accum[4][4];", .to = "half accum[4][4];", .what = "sums accumulate in fp16", .gate = .sums },
        .project => .{ .from = "if (BIAS && col < p.n) value += float(bias[col]);", .to = "if (BIAS && col < p.n) value += 0.0f;", .what = "the bias is dropped", .gate = .shapes },
        .rms => .{ .from = "part, sg, lane) / float(L3VW) + 1e-5f);", .to = "part, sg, lane) / float(L3VW / 2) + 1e-5f);", .what = "the mean of squares divides by half the width", .gate = .shapes },
        .layer_norm => .{ .from = "{ v[j] -= mean; sq += v[j] * v[j]; }", .to = "{ sq += v[j] * v[j]; }", .what = "the mean is not subtracted", .gate = .shapes },
        .add => .{ .from = "if (i < count) residual[i]", .to = "if (i + 1 < count) residual[i]", .what = "the last element is not added", .gate = .shapes },
        .conv => .{ .from = "weights[channel * 9 + (dy + 1) * 3 + dx + 1]", .to = "weights[channel * 9 + (dx + 1) * 3 + dy + 1]", .what = "the taps are transposed", .gate = .shapes },
        .qk => .{ .from = "src[lane] = bfloat(na * cs - nb * sn);", .to = "src[lane] = bfloat(na * cs + nb * sn);", .what = "the rotation's first half has the wrong sign", .gate = .shapes },
        .attention => .{ .from = "const uint kh = head / 2;", .to = "const uint kh = head / 4;", .what = "query heads map to the wrong key head", .gate = .shapes },
        .swiglu => .{ .from = "metal::precise::exp(-g)", .to = "metal::precise::exp(g)", .what = "the sigmoid's argument has the wrong sign", .gate = .shapes },
        .gelu => .{ .from = "l3v_erf(value * 0.70710678118654752f)", .to = "l3v_erf(value * 0.5f)", .what = "the erf argument is scaled by one half", .gate = .shapes },
        .image_rows => .{ .from = "if (i < count) dst[i] = src[i];", .to = "if (i < count) dst[i] = (src[i] & 0x7fff) > 0x7f80 ? ushort(0x7fc0) : src[i];", .what = "a NaN is canonicalized", .gate = .nonfinite },
    };
}

pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: metal LFM3 vision " ++ @tagName(k) ++ ", " ++ comptime edit(k).what,
    };
}

/// The kernel's text with its one fault; the edit must change it, or the control proves nothing.
fn source(a: std.mem.Allocator, kind: Kind) ![]u8 {
    const text = if (kind == .image_rows) sources.lfm3_image_rows else sources.lfm3_vision;
    const e = edit(kind);
    if (std.mem.indexOf(u8, text, e.from) == null) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

pub const Probe = struct {
    inner: vision.Probe,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind) !Probe {
        const text = try source(a, kind);
        defer a.free(text);
        return .{ .inner = try vision.Probe.initWith(a, device, queue, kind, text) };
    }
    pub fn deinit(p: *Probe) void {
        p.inner.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        var k = p.inner.kernel();
        k.spec.name = name(p.inner.kind);
        k.control = edit(p.inner.kind).gate;
        return k;
    }
};
