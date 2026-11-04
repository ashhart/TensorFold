//! Flint's Metal kernels under the contract: glue.metal's rows, routing, q/k prep, decode attention, the prompt tiles.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");

const c = contract.spec;
const g = contract.gates;

/// Flint's served widths (RecursiveSI/flint's config.json).
pub const hidden = 2048;
pub const heads = 32;
pub const kv_heads = 4;
pub const head_dim = 128;
pub const experts = 256;
pub const topk = 6;
pub const inter = 6144;
pub const moe_inter = 512;
pub const shared_inter = 1024;
pub const vocab = 150144;
pub const window = 513;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000;
/// state.zig's rows: a stream's decode window, a shared decode forward, a prompt chunk.
pub const window_rows = 16;
pub const batch_rows = 32;
pub const prompt_rows = 2048;
/// A sliding layer's ring: its window and a prompt chunk, to a multiple of 64 (state.zig's slidingRing).
pub const ring = std.mem.alignForward(usize, window + prompt_rows, 64);

/// glue.metal's fl_args (forward.zig's Args).
pub const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, ring: u32 = 0, rope: u32 = 0, eps: f32 = 0, theta: f32 = 0 };

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value rounded to fp32 then to bf16 (to nearest even, twice): an fp32 step on exact inputs, then the bf16 point.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// fp32's unit roundoff, and its least normal (Apple GPUs flush fp32 subnormals).
pub const u32_ulp = 0x1p-24;
pub const tiny = 0x1p-126;

/// A value's spread through fp32 steps and bf16 rounding points; every step is monotonic, so the ends map to ends.
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved by `e` either way.
    pub fn widen(a: Range, e: f64) Range {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    /// One fp32 rounding of each end (the larger end's half ulp either way, and a flushed subnormal).
    pub fn fp32(a: Range) Range {
        return a.widen(u32_ulp * @max(@abs(a.lo), @abs(a.hi)) + tiny);
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    /// Times an exact value of either sign.
    pub fn times(a: Range, v: f64) Range {
        return .{ .lo = @min(a.lo * v, a.hi * v), .hi = @max(a.lo * v, a.hi * v) };
    }
    /// Times another range: the four corners' extremes.
    pub fn mul(a: Range, b: Range) Range {
        const p = [4]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    pub fn plus(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
};

/// The allowance a range gives about `want`: the farther end.
pub fn spread(want: f64, r: Range) f64 {
    return @max(@abs(want - r.lo), @abs(r.hi - want));
}

/// Allowances from a test's `intervals` (low, high): the farther end from float64's value, 0 where it is not finite.
pub fn allowFrom(a: std.mem.Allocator, t: anytype, want: []const f64, allow: []f64) !void {
    const n = want.len;
    const buf = try a.alloc(f64, 2 * n);
    defer a.free(buf);
    try t.intervals(buf[0..n], buf[n..]);
    for (allow, want, buf[0..n], buf[n..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) spread(w, .{ .lo = lo, .hi = hi }) else 0;
}

pub const nan32 = std.math.nan(f32);
pub const inf32 = std.math.inf(f32);
pub const planted = [_]f32{ nan32, inf32, -inf32 };

/// A planted control's fault: in the body of function `kernel`, the first `from` after its name becomes `to`.
pub const Edit = struct { kernel: []const u8, from: []const u8, to: []const u8 };

/// `text` with `e` applied (an owned copy either way); an edit that changes nothing proves nothing.
pub fn edited(a: std.mem.Allocator, text: []const u8, e: ?Edit) ![]u8 {
    const x = e orelse return a.dupe(u8, text);
    const head = try std.fmt.allocPrint(a, " {s}(", .{x.kernel});
    defer a.free(head);
    const k = std.mem.indexOf(u8, text, head) orelse return error.ControlEditMissing;
    const at = k + (std.mem.indexOf(u8, text[k..], x.from) orelse return error.ControlEditMissing);
    return std.mem.concat(a, u8, &.{ text[0..at], x.to, text[at + x.from.len ..] });
}

/// One of glue.metal's kernels, compiled as model.zig compiles it (MLX's language, precise math functions).
pub fn gluePipe(a: std.mem.Allocator, device: mtl.Device, name: []const u8, e: ?Edit) !mtl.Pipeline {
    const text = try edited(a, sources.flint_glue, e);
    defer a.free(text);
    const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

/// The kernel's name, or a control's: "control: " and what its fault drops.
pub fn named(base: []const u8, control: ?[]const u8, buf: []u8) []const u8 {
    const what = control orelse return base;
    return std.fmt.bufPrint(buf, "control: {s}, {s}", .{ base, what }) catch base;
}

pub fn wanted(only: ?[]const u8, name: []const u8) bool {
    const o = only orelse return true;
    return std.mem.indexOf(u8, name, o) != null;
}

/// A launch's two texts: this GPU's and the pre-M5 one (equal unless the source includes nax.h).
pub const Texts = struct { here: []const u8, pre_m5: []const u8 };

const groups = .{ @import("metal_flint_glue.zig"), @import("metal_flint_rows.zig"), @import("metal_flint_route.zig"), @import("metal_flint_prep.zig"), @import("metal_flint_decode.zig"), @import("metal_flint_prompt.zig") };
const fields = .{ "glue", "rows", "route", "prep", "decode", "prompt" };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    glue: groups[0].Suite = undefined,
    rows: groups[1].Suite = undefined,
    route: groups[2].Suite = undefined,
    prep: groups[3].Suite = undefined,
    decode: groups[4].Suite = undefined,
    prompt: groups[5].Suite = undefined,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        inline for (fields) |f| {
            @field(s, f) = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
            try @field(s, f).collect(only);
        }
    }
    pub fn deinit(s: *Suite) void {
        inline for (fields) |f| @field(s, f).deinit();
    }
};

/// Gate 5's texts for a `flint ` or `moe_route ` registration: the group whose launches it names compiles it.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !Texts {
    inline for (groups) |grp| if (try grp.limitsText(a, device, name)) |t| return t;
    return error.UnknownLaunch;
}

/// The tests of one group of kernels, each made once and freed with the group.
pub fn List(comptime T: type) type {
    return struct {
        items: std.ArrayList(*T) = .empty,

        pub fn add(l: *@This(), a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel), x: T) !void {
            const p = try a.create(T);
            p.* = x;
            errdefer {
                p.deinit();
                a.destroy(p);
            }
            try l.items.append(a, p);
            try kernels.append(a, p.kernel());
        }
        pub fn deinit(l: *@This(), a: std.mem.Allocator) void {
            for (l.items.items) |p| {
                p.deinit();
                a.destroy(p);
            }
            l.items.deinit(a);
        }
    };
}

test "a control's edit lands in its own kernel's body, once" {
    const text = "void fl_a(x) { y = bfloat(z); }\nvoid fl_b(x) { y = bfloat(z); }\n";
    const got = try edited(std.testing.allocator, text, .{ .kernel = "fl_b", .from = "bfloat(z)", .to = "(z)" });
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("void fl_a(x) { y = bfloat(z); }\nvoid fl_b(x) { y = (z); }\n", got);
    try std.testing.expectError(error.ControlEditMissing, edited(std.testing.allocator, text, .{ .kernel = "fl_c", .from = "z", .to = "w" }));
}
