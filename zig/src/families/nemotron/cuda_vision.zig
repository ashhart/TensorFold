//! Image input for Nemotron Omni on CUDA: the RADIO tower, a prompt pass's image rows, their splice after each embed.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const radio = @import("radio");
pub const video = radio.video;

/// What a request may hold: image tokens in all (the rows buffer), and images.
pub const Limits = struct { image_tokens: u32, max_images: u32, video_tokens: u32 = 0 };

/// Takes a prompt pass's images in the tower's place, so a host test can count them without a GPU.
pub const Encoder = struct {
    ctx: *anyopaque,
    encode: *const fn (ctx: *anyopaque, s: cuda.Stream, images: []const lanes.stream.Image, spans: []const radio.Span, rows: *const radio.Rows) anyerror!void,
};

pub const Vision = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    cfg: radio.Config,
    tower: radio.Tower,
    rows: radio.Rows,
    marks: radio.Marks,
    limits: Limits,
    mark_ids: [3]u32, // the placeholder and its wrapper: text that spells one stays text
    spans: []radio.Span = &.{}, // the prompt pass in progress
    encoder: ?Encoder = null, // null: the tower

    /// The checkpoint's tower and a rows buffer for `limits`; null with `problem` when it cannot take images.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, limits: Limits, problem: *[]const u8) !?*Vision {
        const cfg = (radio.load(gpa, io, dir, problem) catch |e| switch (e) {
            error.NoVisionTower => {
                problem.* = "this checkpoint has no vision tower: serve it without --vision";
                return null;
            },
            else => return e,
        }) orelse return null;
        var tok = try core.tokenizer.loadTokenizer(io, gpa, dir);
        defer tok.deinit();
        const marks = radio.Marks.resolve(&tok, &cfg) orelse {
            problem.* = "the tokenizer lacks <image>, <img> or </img>, or its <image> is not config.json's img_context_token_id";
            return null;
        };
        const capacity = try reservation(limits, cfg.video_temporal == 2);
        var tower = try radio.Tower.init(gpa, io, d, dir, cfg);
        errdefer tower.deinit();
        // a request's images and videos are budgeted apart but planned together: the rows hold both budgets
        var rows = try radio.Rows.init(d, capacity, cfg.text_hidden);
        errdefer rows.deinit();
        const v = try gpa.create(Vision);
        v.* = .{ .gpa = gpa, .io = io, .cfg = cfg, .tower = tower, .rows = rows, .marks = marks, .limits = limits, .mark_ids = marks.markers() };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.end();
        v.rows.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    /// Device bytes the tower and the rows buffer hold.
    pub fn bytes(v: *const Vision) u64 {
        return v.tower.bytes + v.rows.buf.len;
    }

    /// A prompt pass from row `from`: the images whose rows end past it, through the tower into the rows buffer.
    pub fn begin(v: *Vision, s: cuda.Stream, images: []const lanes.stream.Image, from: usize) !void {
        v.end();
        if (images.len == 0) return;
        v.spans = try radio.plan(v.gpa, images, @intCast(from));
        if (radio.rows.total(v.spans) > v.rows.limit) return error.ImageRows;
        if (v.encoder) |x| return x.encode(x.ctx, s, images, v.spans, &v.rows);
        try v.tower.encode(s, images, v.spans, &v.rows);
    }

    /// The pass ended: no rows splice into later passes until another begins.
    pub fn end(v: *Vision) void {
        if (v.spans.len > 0) v.gpa.free(v.spans);
        v.spans = &.{};
    }

    /// Prompt rows [lo, hi) embedded at `emb` take the image rows they cross.
    pub fn splice(v: *const Vision, s: cuda.Stream, lo: usize, hi: usize, emb: u64) !void {
        if (v.spans.len == 0) return;
        try v.rows.splice(s, v.spans, @intCast(lo), @intCast(hi), emb);
    }

    /// The words for a prepare error the server cannot word (a JPEG library this machine lacks).
    pub const explain = radio.decoderProblem;

    /// A decoded video as the tower reads it: its frames sampled, sized and paired, with the words before each pair.
    pub fn prepareVideo(v: *const Vision, a: std.mem.Allocator, src: radio.video.Source, max_tokens: u32, max_frames: u32) !radio.video.Prepared {
        return radio.video.prepare(a, &v.cfg, src, max_tokens, max_frames);
    }

    /// An image's bytes as the tower reads them, at most `max_tokens` tokens (the server's prepare).
    pub fn prepare(v: *const Vision, a: std.mem.Allocator, data: []const u8, max_tokens: u32) !radio.image.Prepared {
        return radio.image.prepare(v.io, a, &v.cfg, data, max_tokens);
    }
};

/// Request rows before any tower or device allocation.
pub fn reservation(limits: Limits, has_video: bool) !u32 {
    return std.math.add(u32, limits.image_tokens, if (has_video) limits.video_tokens else 0) catch error.BadVisionLimits;
}
