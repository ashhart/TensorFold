//! A prompt pass's image rows on every rank: rank 0's tower encodes them, the others receive them, all splice them.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const vision = @import("cuda_vision.zig");
const vcfg = @import("vision_config.zig");
const image_core = @import("image_core.zig");
const image = image_core.For(@import("core").picture);
const video_mod = @import("video_core.zig").For(@import("core").picture);

/// What a request may hold: image tokens in all (the rows buffer), and images.
pub const Limits = struct { image_tokens: u32, max_images: u32, video_tokens: u32 = 0, max_videos: u32 = 0, max_frames: u32 = 0 };

/// Prompt rows [at, at + n) take image rows [row, row + n).
pub const Span = struct { at: u32, n: u32, row: u32 };

const Saved = struct {
    rows: cuda.DeviceBuffer,
    spans: std.ArrayList(Span) = .empty,
    fn deinit(s: *Saved, a: std.mem.Allocator) void {
        s.rows.free();
        s.spans.deinit(a);
    }
};

pub const Images = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    module: cuda.Module,
    run: vision.Run,
    tower: ?vision.Tower = null, // rank 0's; the other ranks receive the rows
    cfg: ?vcfg.Config = null,
    rows: cuda.DeviceBuffer, // bf16 [limits.image_tokens][out_hidden]
    map: cuda.DeviceBuffer, // i32 [max_rows]: a splice's feature row per prompt row
    host_map: []i32,
    limits: Limits,
    spans: std.ArrayList(Span) = .empty,
    saved: std.AutoHashMapUnmanaged(u32, Saved) = .empty,
    slot: u32 = 0, // the sequence the spans belong to: another stream's rows at the same positions take none
    marks: struct { image: u32, open: u32, close: u32 } = .{ .image = 0, .open = 0, .close = 0 },
    mark_ids: [3]u32 = @splat(0), // begin, image, end: text that spells one stays text
    video_ids: ?[3]u32 = null, // begin, video, end: the checkpoint reads videos (each frame pair as an image)

    /// The chat template writes each image's begin and end itself (the server only expands the image token).
    pub const template_wraps = true;
    /// A video's frames to patch pairs (Glm5NextVideoProcessor), over core's decoder: each pair goes in as an image.
    pub const video = video_mod;

    /// Rank 0 (`tower`) or another rank: the rows buffer, and rank 0's tower; null with `problem` when no tower.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, dir: []const u8, limits: Limits, max_rows: usize, tower: bool, ql: cuda.qlinear.Linear, problem: *[]const u8) !?*Images {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cwd = std.Io.Dir.cwd();
        const cfg_text = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
        const proc = cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "processor_config.json" }), a, .limited(1 << 24)) catch null;
        const cfg = (try vcfg.Config.parse(a, cfg_text, proc)) orelse {
            problem.* = "this checkpoint has no vision tower: serve it without --vision";
            return null;
        };
        const v = try gpa.create(Images);
        errdefer gpa.destroy(v);
        var module = try cuda.Module.load(d, cuda.kernels.glm_vision);
        errdefer module.unload();
        v.* = .{ .gpa = gpa, .io = io, .d = d, .module = module, .run = .{ .stream = stream, .f = try vision.Funcs.load(module) }, .cfg = cfg, .rows = undefined, .map = undefined, .host_map = undefined, .limits = limits };
        v.video_ids = cfg.video;
        v.rows = try cuda.DeviceBuffer.alloc(d, @as(usize, try v.rowLimit()) * vcfg.out_hidden * 2);
        errdefer v.rows.free();
        v.map = try cuda.DeviceBuffer.alloc(d, max_rows * 4);
        errdefer v.map.free();
        v.host_map = try gpa.alloc(i32, max_rows);
        errdefer gpa.free(v.host_map);
        if (tower) v.tower = try vision.Tower.load(gpa, io, d, dir, ql);
        v.marks = .{ .image = cfg.image_token, .open = cfg.image_start, .close = cfg.image_end };
        v.mark_ids = .{ cfg.image_start, cfg.image_token, cfg.image_end };
        return v;
    }

    // A mixed request holds both budgets; validate the addition before allocating or receiving rows.
    pub fn rowLimit(v: *const Images) !u32 {
        return std.math.add(u32, v.limits.image_tokens, if (v.video_ids != null) v.limits.video_tokens else 0) catch error.ImageRows;
    }

    pub fn deinit(v: *Images) void {
        @import("cuda_load_owner.zig").drain(v.run.stream, v.d);
        var it = v.saved.valueIterator();
        while (it.next()) |state| state.deinit(v.gpa);
        v.saved.deinit(v.gpa);
        v.spans.deinit(v.gpa);
        if (v.tower) |*t| t.deinit();
        v.gpa.free(v.host_map);
        v.map.free();
        v.rows.free();
        v.module.unload();
        v.gpa.destroy(v);
    }

    /// Device bytes the tower and the buffers hold.
    pub fn bytes(v: *const Images) u64 {
        var total: u64 = (if (v.tower) |t| t.bytes else 0) + v.rows.len + v.map.len;
        var it = v.saved.valueIterator();
        while (it.next()) |owner| total += owner.rows.len;
        return total;
    }

    // The tower and upload map are shared scratch. Feature rows and spans belong to each live sequence.
    pub fn activate(v: *Images, slot: u32) !void {
        if (v.slot == slot) return;
        if (!v.saved.contains(slot)) try v.saved.ensureUnusedCapacity(v.gpa, 1);
        const next = if (v.saved.fetchRemove(slot)) |entry| entry.value else Saved{
            .rows = try cuda.DeviceBuffer.alloc(v.d, @as(usize, try v.rowLimit()) * vcfg.out_hidden * 2),
        };
        v.saved.putAssumeCapacity(v.slot, .{ .rows = v.rows, .spans = v.spans });
        v.rows = next.rows;
        v.spans = next.spans;
        v.slot = slot;
    }

    pub fn release(v: *Images, slot: u32) void {
        @import("cuda_load_owner.zig").drain(v.run.stream, v.d);
        if (slot != v.slot) {
            if (v.saved.fetchRemove(slot)) |entry| {
                var state = entry.value;
                state.deinit(v.gpa);
            }
            return;
        }
        if (slot == 0) {
            v.clear();
            return;
        }
        if (v.saved.fetchRemove(0)) |zero| { // slot 0's rows and spans as its stream left them
            v.rows.free();
            v.spans.deinit(v.gpa);
            v.rows = zero.value.rows;
            v.spans = zero.value.spans;
        } else v.clear(); // slot 0's stream ended while this slot was bound: this buffer is slot 0's now
        v.slot = 0;
    }

    pub fn streamBytes(v: *const Images) usize {
        return @as(usize, v.rowLimit() catch unreachable) * vcfg.out_hidden * 2;
    }

    /// No rows splice into any pass until another begins (each request's prompt pass starts here).
    pub fn clear(v: *Images) void {
        v.spans.clearRetainingCapacity();
    }

    /// Rank 0: a prompt pass on `slot` from row `from`: the images whose rows end past it, through the tower.
    pub fn begin(v: *Images, slot: u32, images: []const lanes.stream.Image, from: usize) !void {
        try v.activate(slot);
        v.clear();
        errdefer v.clear();
        const t = &(v.tower orelse return error.NoVision);
        var row: u32 = 0;
        for (images) |img| {
            if (img.gh * img.gw != img.tokens * vcfg.patches_per_token) return error.ImageOutOfStep;
            if (img.at + img.tokens <= from) continue; // in the kept state already
            if (img.tokens > (try v.rowLimit()) -| row) return error.ImageRows; // the server holds a request to its limits
            try t.encode(&v.run, img.pixels, .{ 1, img.gh, img.gw }, v.rows.ptr + @as(u64, row) * vcfg.out_hidden * 2);
            try v.spans.append(v.gpa, .{ .at = img.at, .n = img.tokens, .row = row });
            row += img.tokens;
        }
        v.slot = slot;
    }

    /// The other ranks: rank 0's spans (as `words`: at, n, row a span) on `slot`; the rows arrive into `rows` next.
    pub fn take(v: *Images, slot: u32, words: []const u32) !void {
        try v.activate(slot);
        v.clear();
        errdefer v.clear();
        if (words.len % 3 != 0) return error.BadStep;
        var i: usize = 0;
        while (i < words.len) : (i += 3) {
            const sp: Span = .{ .at = words[i], .n = words[i + 1], .row = words[i + 2] };
            if (sp.row > try v.rowLimit() or sp.n > (try v.rowLimit()) - sp.row) return error.ImageRows;
            try v.spans.append(v.gpa, sp);
        }
        v.slot = slot;
    }

    /// Image rows the spans hold (what crosses to the other ranks).
    pub fn rowCount(v: *const Images) usize {
        var n: usize = 0;
        for (v.spans.items) |sp| n = @max(n, sp.row + sp.n);
        return n;
    }

    /// Prompt rows [lo, hi) of `slot`, embedded at `emb` in `copies` streams, take the image rows they cross.
    pub fn splice(v: *Images, s: cuda.Stream, slot: u32, lo: usize, hi: usize, emb: u64, copies: usize) !void {
        try v.activate(slot);
        if (v.spans.items.len == 0) return;
        const n = hi - lo;
        if (n > v.host_map.len) return error.BadWindow;
        var any = false;
        for (v.host_map[0..n], lo..) |*m, p| {
            m.* = -1;
            for (v.spans.items) |sp| if (p >= sp.at and p < sp.at + sp.n) {
                m.* = @intCast(sp.row + (p - sp.at));
                any = true;
            };
        }
        if (!any) return;
        try v.map.uploadAsync(0, std.mem.sliceAsBytes(v.host_map[0..n]), s.handle); // ordered after the prior splice
        var a: cuda.Args = .{};
        a.add(v.rows.ptr);
        a.add(emb);
        a.add(v.map.ptr);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(vcfg.out_hidden)));
        a.add(@as(c_int, @intCast(copies)));
        try cuda.launch.launch(v.run.f.inject, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The words for a prepare error the server cannot word (a JPEG library this machine lacks).
    pub const explain = image.decoderProblem;

    /// A video's sampled frames as patch pairs within `max_tokens` and `max_frames` (the server's prepare).
    pub fn prepareVideo(v: *const Images, a: std.mem.Allocator, s: video.Source, max_tokens: u32, max_frames: u32) !video.Prepared {
        return video.prepare(a, s, v.cfg.?, max_tokens, max_frames);
    }

    /// An image's bytes as the tower reads them, at most `max_tokens` tokens (the server's prepare).
    pub fn prepare(v: *const Images, a: std.mem.Allocator, data: []const u8, max_tokens: u32) !image_core.Prepared {
        return image.prepare(v.io, a, data, v.cfg.?, max_tokens);
    }
};

test "a splice's map: rows inside a span take its rows, others keep their embedding" {
    var host: [8]i32 = undefined;
    const spans = [_]Span{ .{ .at = 3, .n = 2, .row = 0 }, .{ .at = 6, .n = 4, .row = 2 } };
    for (&host, 2..) |*m, p| {
        m.* = -1;
        for (spans) |sp| if (p >= sp.at and p < sp.at + sp.n) {
            m.* = @intCast(sp.row + (p - sp.at));
        };
    }
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, -1, 2, 3, 4, 5 }, &host);
}

test "a request with images and a video holds the sum of both row budgets" {
    var v: Images = undefined;
    v.limits = .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 };
    v.video_ids = .{ 1, 2, 3 };
    try std.testing.expectEqual(@as(u32, 6144), try v.rowLimit());
    v.video_ids = null;
    try std.testing.expectEqual(@as(u32, 2048), try v.rowLimit());
}

test "a follower's failed image admission clears its partial spans, and the same admission then succeeds" {
    const a = std.testing.allocator;
    var words: [3 * 128]u32 = undefined;
    for (0..128) |i| words[3 * i ..][0..3].* = .{ @intCast(i), 1, @intCast(i) };
    for (0..8) |offset| {
        var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = offset, .resize_fail_index = 0 });
        var v: Images = undefined;
        v.gpa = fail.allocator();
        v.spans = .empty;
        defer v.spans.deinit(v.gpa);
        v.slot = 9;
        v.limits = .{ .image_tokens = 64, .max_images = 8, .video_tokens = 128 };
        v.video_ids = .{ 1, 2, 3 };
        v.take(9, &words) catch |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(@as(usize, 0), v.spans.items.len);
            try std.testing.expectEqual(@as(u32, 9), v.slot);
        };
        fail.fail_index = std.math.maxInt(usize);
        try v.take(9, &words);
        try std.testing.expectEqual(@as(usize, 128), v.spans.items.len);
        try std.testing.expectEqual(@as(usize, 128), v.rowCount());
    }
}

test "a row budget whose image and video sum overflows is refused" {
    var v: Images = undefined;
    v.limits = .{ .image_tokens = 1, .max_images = 1, .video_tokens = std.math.maxInt(u32) };
    v.video_ids = .{ 1, 2, 3 };
    try std.testing.expectError(error.ImageRows, v.rowLimit());
}
