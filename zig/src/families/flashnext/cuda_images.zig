//! Flash Next's images through the shared Qwen-VL tower: rank zero holds its weights, a follower receives its rows.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const shared = @import("qwen_vision");
const wt = @import("cuda_prompt_wt.zig");
const media = @import("cuda_media.zig");
pub const Limits = shared.media.Limits;
pub const Vision = struct {
    pub const template_wraps = true;
    a: std.mem.Allocator,
    io: std.Io,
    tower: ?*shared.Tower,
    processor: core.qwen_vl.Config,
    module: ?cuda.Module,
    encoder: shared.media.Encoder,
    limits: Limits,
    marks: struct { open: u32, image: u32, close: u32 },
    mark_ids: [3]u32,
    prep: cuda.Function,
    pool: cuda.Function,
    pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8, limits: Limits, rank: usize, prep: cuda.Function, pool: cuda.Function) !*Vision {
        try shared.media.checkLimits(limits);
        const settings = (try shared.Settings.read(a, io, dir)) orelse return error.NoVisionTower;
        const c = settings.tower;
        if (c.out_hidden != wt.dims) return error.BadVisionWidth;
        // the markers are distinct ids inside the language model's vocabulary
        var marks: [4]u32 = undefined;
        var count: usize = 0;
        for ([_]?u32{ c.image_token, c.video_token, c.vision_start, c.vision_end }) |mark| if (mark) |id| {
            marks[count] = id;
            count += 1;
        };
        for (marks[0..count], 0..) |id, i| {
            if (id >= wt.vocab) return error.UnsupportedQwenVision;
            for (marks[0..i]) |prior| if (prior == id) return error.UnsupportedQwenVision;
        }
        const self = try a.create(Vision);
        errdefer a.destroy(self);
        const t = if (rank == 0) try shared.Tower.load(a, io, d, s, dir, c) else null;
        errdefer if (t) |x| x.deinit();
        var module: ?cuda.Module = if (t == null) try cuda.Module.load(d, cuda.kernels.qwen_vision) else null;
        errdefer if (module) |*m| m.unload();
        const splice = try (if (t) |x| &x.mod else &module.?).function("tf_qv_splice");
        self.* = .{ .a = a, .io = io, .tower = t, .processor = settings.processor, .module = module, .limits = limits, .marks = .{ .open = c.vision_start, .image = c.image_token, .close = c.vision_end }, .mark_ids = .{ c.vision_start, c.image_token, c.vision_end }, .prep = prep, .pool = pool, .encoder = .{ .ctx = self, .config = c, .driver = d, .stream = s, .splice = splice, .encode = encode } };
        return self;
    }
    fn encode(ptr: *anyopaque, pixels: []const f32, grids: []const shared.Grid, out: u64) anyerror!void {
        const self: *Vision = @ptrCast(@alignCast(ptr));
        try (self.tower orelse return error.FollowerCannotEncode).encode(pixels, grids, out);
    }
    pub fn deinit(v: *Vision) void {
        if (v.tower) |t| t.deinit();
        if (v.module) |*m| m.unload();
        v.a.destroy(v);
    }
    pub fn prepare(v: *Vision, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !struct {
        pixels: []f32,
        gh: u32,
        gw: u32,
        tokens: u32,
        hash: u64,
    } {
        if (max_tokens == 0 or max_tokens > v.limits.image_tokens) return error.BadImageLimits;
        const settings: shared.Settings = .{ .tower = v.encoder.config, .processor = v.processor };
        const p = try settings.prepare(v.io, a, bytes, max_tokens);
        return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
    }
    pub fn state(v: *Vision, ids: []const u32, images: []const @import("lanes").stream.Image) !media.State {
        return .{ .image = try shared.media.Media.init(v.a, v.encoder, ids, images, v.limits), .prep = v.prep, .pool = v.pool };
    }
    pub fn receive(v: *Vision, n: usize, runs: []const shared.media.Run) !media.State {
        const p = try shared.media.Plan.init(v.a, n, runs, v.limits);
        return .{ .image = try shared.media.Media.allocate(p, v.encoder), .prep = v.prep, .pool = v.pool };
    }
    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.BadImageLimits, error.BadImageCount => "image count or row budget exceeds the configured limits",
            error.BadImageGrid, error.BadImagePatches, error.BadImagePlaceholders => "image rows do not match the prompt placeholders",
            error.NoVisionTower, error.BadVisionWidth, error.UnsupportedQwenVision => "unsupported Flash Next Qwen-VL tower configuration",
            else => core.qwen_vl.geometry.problem(err),
        };
    }
};
