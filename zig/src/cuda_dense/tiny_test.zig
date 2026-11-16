//! A tiny dense recipe of the dots shape, with or without a fake image tower, for the engine's host tests.
const std = @import("std");
const cuda = @import("cuda");
const Pass = @import("pass.zig").Pass;
const recipe = @import("recipe.zig");
const Geometry = recipe.Geometry;
const Allocator = std.mem.Allocator;

pub const tiny: Geometry = .{ .hidden = 256, .layers = 2, .heads = 4, .kv_heads = 2, .head_dim = 64, .vocab = 512, .eps = 1e-6, .context = 4096, .acc = 512, .wide = 1024, .mid = 512, .theta = 1e4 };

/// A recipe of the dots shape on tiny sizes: biased q|k|v sums, attention, o, SwiGLU, an untied head; tower `Tw`.
pub fn Recipe(comptime Tw: ?type) type {
    return struct {
        const Self = @This();
        pub const name = "Tiny";
        pub const model_type = "tiny_dense";
        pub const formats: []const []const u8 = &.{"bf16"};
        pub const default_context: i64 = 4096;
        pub const reasoning = false;
        pub const post_embed: ?fn (*const Pass(Self)) anyerror!void = null;
        pub const Tower: ?type = Tw;
        pub const Weights = struct {
            embed: u64 = 0,
            norm: u64 = 0,
            bias: u64 = 0,
            proj: cuda.qmmf.Weight = cuda.modelopt.bf16Weight(0, 512, 256),
            pub fn deinit(_: *Weights) void {}
        };
        pub const Kernels = struct {
            pub fn load(_: *const cuda.Context) !Kernels {
                return .{};
            }
            pub fn deinit(_: *Kernels) void {}
        };
        pub fn geometry(_: Allocator, _: std.Io, _: []const u8) !Geometry {
            return tiny;
        }
        pub fn load(_: Allocator, _: std.Io, _: *const cuda.Driver, _: cuda.Stream, _: []const u8) !Weights {
            return .{};
        }
        pub fn weightBytes(_: Allocator, _: std.Io, _: []const u8) !u64 {
            return 1 << 20;
        }
        pub fn embed(p: *const Pass(Self)) !void {
            try p.k.embed.bf16(p.s, p.m.s.ids, p.w.embed, p.m.s.h, p.rows, p.g.hidden);
        }
        pub fn layer(p: *const Pass(Self), i: usize) !void {
            const x = p.m.s;
            try p.norm(if (i == 0) null else x.r, p.w.norm);
            try p.projectBias(p.w.proj, p.w.bias, x.x, x.wide);
            try p.projectF32(p.w.proj, x.x, x.acc);
            try p.attend(i, x.q, x.kv, x.y);
            try p.project(p.w.proj, x.y, x.r);
            try p.siluMul(x.wide, x.y, p.g.mid);
        }
        pub fn final(p: *const Pass(Self)) !void {
            try p.norm(p.m.s.r, p.w.norm);
        }
        pub fn head(w: *const Weights) cuda.qmmf.Weight {
            return w.proj;
        }
        pub fn explain(err: anyerror) ?[]const u8 {
            return if (err == error.TinyConfig) "this Tiny checkpoint is not the one the recipe serves" else null;
        }
    };
}

/// How a fake tower misses the contract: none, or one decl of another shape (checkTower's red controls).
pub const Flaw = enum { none, limits, width, marks, encode, prepare, explain, sure_encode, fallible_explain };

/// A fake image tower: marks 500 (image) inside 501 and 502, 64 tokens a request, `encode` recorded, no device.
pub fn FakeTower(comptime flaw: Flaw) type {
    return struct {
        const T = @This();
        pub const Limits = if (flaw == .limits) struct { image_tokens: u32 } else struct { image_tokens: u32, max_images: u32 };
        pub const default_image_tokens: u32 = 64;
        pub const template_wraps = true;
        pub const part_type = "image";
        pub const Prepared = if (flaw == .prepare) struct { pixels: []f32, gh: u32, gw: u32, tokens: u32 } else struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
        const Marks = if (flaw == .marks) struct { open: u32, image: u32 } else struct { open: u32, image: u32, close: u32 };
        pub const Call = struct { at: u32, tokens: u32, out: u64 };

        marks: Marks = if (flaw == .marks) .{ .open = 501, .image = 500 } else .{ .open = 501, .image = 500, .close = 502 },
        mark_ids: [3]u32 = .{ 501, 500, 502 },
        limits: Limits = if (flaw == .limits) .{ .image_tokens = 64 } else .{ .image_tokens = 64, .max_images = 4 },
        width: if (flaw == .width) u32 else usize = 256,
        calls: [8]Call = undefined,
        n: usize = 0,
        gpa: ?Allocator = null,

        pub fn open(gpa: Allocator, _: std.Io, _: *const cuda.Context, _: cuda.Stream, _: []const u8, limits: Limits) !*T {
            const t = try gpa.create(T);
            t.* = .{ .limits = limits, .gpa = gpa };
            return t;
        }
        pub fn openBytes(_: Allocator, _: std.Io, _: []const u8) !u64 {
            return 3 << 16;
        }
        pub fn weightBytes(_: Allocator, _: std.Io, _: []const u8) !u64 {
            return 5 << 20;
        }
        /// Four tokens an image: a 4 x 4 grid of patches of one value each.
        pub fn prepare(_: *T, a: Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
            if (bytes.len == 0) return error.FakeImage;
            if (max_tokens < 4) return error.FakeImageTokens;
            const pixels = try a.alloc(f32, 16);
            @memset(pixels, @floatFromInt(bytes[0]));
            if (flaw == .prepare) return .{ .pixels = pixels, .gh = 4, .gw = 4, .tokens = 4 };
            return .{ .pixels = pixels, .gh = 4, .gw = 4, .tokens = 4, .hash = std.hash.Wyhash.hash(0, bytes) };
        }
        pub const encode = switch (flaw) {
            .encode => encodeUntapped,
            .sure_encode => encodeSure,
            else => encodeTapped,
        };
        fn encodeSure(t: *T, s: cuda.Stream, image: *const recipe.Image, out: u64, taps: ?recipe.Tap) void {
            encodeTapped(t, s, image, out, taps) catch unreachable;
        }
        fn encodeTapped(t: *T, _: cuda.Stream, image: *const recipe.Image, out: u64, taps: ?recipe.Tap) !void {
            std.debug.assert(taps == null);
            t.calls[t.n] = .{ .at = image.at, .tokens = image.tokens, .out = out };
            t.n += 1;
        }
        fn encodeUntapped(t: *T, s: cuda.Stream, image: *const recipe.Image, out: u64) !void {
            return encodeTapped(t, s, image, out, null);
        }
        pub const explain = switch (flaw) {
            .explain => explainBytes,
            .fallible_explain => explainFallible,
            else => explainErr,
        };
        fn explainFallible(err: anyerror) !?[]const u8 {
            if (err == error.OutOfMemory) return error.Unworded;
            return explainErr(err);
        }
        fn explainErr(err: anyerror) ?[]const u8 {
            return if (err == error.FakeImage) "the fake tower reads no empty image" else null;
        }
        fn explainBytes(_: []const u8) ?[]const u8 {
            return null;
        }
        pub fn deinit(t: *T) void {
            if (t.gpa) |a| a.destroy(t);
        }
    };
}

pub const Fake = FakeTower(.none);
pub const Tiny = Recipe(null);
pub const TinyTower = Recipe(Fake);
