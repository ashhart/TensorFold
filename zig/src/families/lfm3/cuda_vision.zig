//! The adapter's view of the picture tower: `tower` (the engine runs it) and `Vision` (the server asks).
const std = @import("std");
const cuda = @import("cuda");
const tower_mod = @import("vision.zig");
const image = @import("image.zig");

const Allocator = std.mem.Allocator;

pub const tower = tower_mod;
pub const weights = @import("cuda_vision_weights.zig");
pub const media = @import("cuda_media.zig");

/// What the template writes: LFM3's only marker is the checkpoint's own image token.
pub const image_token_id: u32 = image.image_token_id; // the family\s capability check reads this

pub const Marks = struct {
    image: u32 = image.image_token_id,
    open: ?u32 = null, // the template writes no wrapper
    close: ?u32 = null,
};

/// How much the server may send.
pub const Limits = struct {
    image_tokens: u32 = 1024,
    max_images: u32 = 4,
};

pub const PreparedImage = struct {
    pixels: []const f32, // [gh*gw, 1536], normalized
    gh: u32,
    gw: u32,
    tokens: u32,
    hash: u64,
};

/// The server's entry point for one picture: bytes -> patch tensor + grid + row count.
pub const Vision = struct {
    /// LFM3's chat template writes each image's markers itself, so the server must not wrap the token
    pub const template_wraps = {};
    io: ?std.Io = null, // the shared picture decoder loads its libraries through this
    gpa: Allocator,
    d: *const cuda.Driver,
    tower_v: tower_mod.Vision,
    /// Owns the tower binding: freeing after `Vision.init` leaves the tower on freed memory.
    bound: ?*weights.Bound = null,
    marks: Marks = .{},
    mark_ids: [1]u32 = .{image.image_token_id},
    limits: Limits = .{},

    /// The server's entry point: encoded image bytes in, prepared rows out.
    pub fn prepare(ctx: *Vision, a: Allocator, bytes: []const u8, max_tokens: u32) !PreparedImage {
        const picture = @import("core").picture;
        var rgb = try picture.decode(ctx.io orelse return error.NoIo, a, bytes);
        defer rgb.deinit(a);
        return ctx.prepareRgb(a, rgb.pixels, rgb.width, rgb.height, max_tokens);
    }

    pub fn prepareRgb(ctx: *Vision, a: Allocator, rgb: []const u8, w: usize, h: usize, max_tokens: u32) !PreparedImage {
        // fitGrid, not gridSize: `detail:"low"` shrinks the picture instead of refusing it.
        const grid = image.fitGrid(w, h, max_tokens);
        const px = try image.resample(a, rgb, w, h, grid[0], grid[1]);
        defer a.free(px);
        const p = try image.patchify(a, px, grid[0], grid[1]);
        errdefer a.free(p.rows);
        const tokens = image.tokenCount(p.gh, p.gw);
        if (tokens > max_tokens) return error.ImagePastTokens;
        if (tokens > ctx.limits.image_tokens) return error.ImagePastLimit;
        var hash: u64 = 0xcbf29ce484222325;
        for (p.rows) |v| {
            const bits: u32 = @bitCast(v);
            hash = (hash ^ bits) *% 0x100000001b3;
        }
        return .{ .pixels = p.rows, .gh = @intCast(p.gh), .gw = @intCast(p.gw), .tokens = @intCast(tokens), .hash = hash };
    }

    pub fn deinit(ctx: *Vision) void {
        ctx.tower_v.deinit();
        if (ctx.bound) |b| {
            b.deinit();
            ctx.gpa.destroy(b);
            ctx.bound = null;
        }
    }
    pub fn explain(_: anyerror) ?[]const u8 {
        return null;
    }
};
