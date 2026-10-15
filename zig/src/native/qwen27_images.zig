//! Qwen3.8-27B's images (--vision): the shared Qwen-VL tower's rows for a stream's prompt pass, and its M-RoPE layout.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const q = tf.qwen27;
const vl = tf.qwen_vl;
const Tower = tf.qwen_vl_tower.Tower;
const Allocator = std.mem.Allocator;

pub const Vision = struct {
    gpa: Allocator,
    c: vl.Config,
    tower: *Tower,
    rows: mtl.Buffer, // [image tokens, the model's width] bf16: one prompt pass's image rows
    capacity: u32, // image tokens one request holds (--vision-image-tokens)
    max_images: u32, // and images (--vision-max-images)
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text
    spans: std.ArrayList(q.image_rows.Span) = .empty,

    pub fn ids(v: *const Vision) vl.rope.Ids {
        return .{ .start = v.c.vision_start, .image = v.c.image_token, .end = v.c.vision_end };
    }

    /// Buffers to keep resident with the model's.
    pub fn buffers(v: *const Vision) [3]mtl.Buffer {
        const t = v.tower.buffers();
        return .{ t[0], t[1], v.rows };
    }

    pub fn deinit(v: *Vision) void {
        v.spans.deinit(v.gpa);
        v.rows.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }
};

/// The checkpoint's tower for `model`'s width under its rope, for `o.image_tokens` image tokens a request.
pub fn open(gpa: Allocator, io: std.Io, model: *const q.model.Model, dir: []const u8, o: api.VisionOpen, a: Allocator, why: *[]const u8) !*Vision {
    const read = struct {
        fn f(al: Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
            const path = std.fs.path.join(al, &.{ d, name }) catch return null;
            return std.Io.Dir.cwd().readFileAlloc(i, path, al, .limited(16 << 20)) catch null;
        }
    }.f;
    const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
    const c = (vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json")) catch |err| {
        why.* = "--vision: this checkpoint's vision_config or image processor names a tower the native engine does not read";
        return err;
    }) orelse {
        why.* = try std.fmt.allocPrint(a, "--vision: {s} has no image tower (its config.json names no vision_config)", .{dir});
        return error.NoVisionTower;
    };
    const mc = model.config;
    if (c.out_hidden != mc.hidden or !mc.mrope_interleaved or mc.mrope_section[0] + mc.mrope_section[1] + mc.mrope_section[2] != mc.rope_dims / 2) {
        why.* = "--vision: the image tower's output width or the text's M-RoPE sections do not match this model";
        return error.VisionShape;
    }
    const v = try gpa.create(Vision);
    errdefer gpa.destroy(v);
    v.* = .{ .gpa = gpa, .c = c, .tower = undefined, .rows = undefined, .capacity = o.image_tokens, .max_images = o.max_images, .markers = .{ c.vision_start, c.image_token, c.vision_end } };
    // one image a pass: the tower's scratch holds the largest image the processor makes under the request's budget
    const largest: u32 = @intCast(@min(@as(u64, o.image_tokens), c.max_pixels / (@as(u64, c.factor()) * c.factor())));
    v.tower = try Tower.load(gpa, io, model.device, dir, c, largest);
    errdefer v.tower.deinit();
    v.rows = try model.device.buffer(@as(usize, o.image_tokens) * mc.hidden * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    std.log.info("Qwen3.8-27B: image tower loaded ({d:.2} GiB with scratch for {d} image tokens a request)", .{ @as(f64, @floatFromInt(v.tower.bytes + v.rows.length())) / (1 << 30), o.image_tokens });
    return v;
}

/// `s`'s rows' M-RoPE layout: empty for text, its placeholders checked against its images otherwise.
pub fn layout(v: ?*const Vision, s: *const lanes.Stream) !vl.rope.Layout {
    if (s.images.len == 0) return .{};
    const x = v orelse return error.NoVision;
    return vl.rope.Layout.init(s.prompt(), s.images, x.ids(), x.c.merge);
}

/// The tower's rows of `s`'s images past prompt row `from` (an earlier image a kept state already holds runs not).
pub fn encode(v: *Vision, queue: mtl.Queue, s: *const lanes.Stream, slot: u32) !?q.image_rows.Images {
    v.spans.clearRetainingCapacity();
    const from = s.cached;
    const width: usize = v.c.out_hidden;
    var row: u32 = 0;
    for (s.images) |img| {
        if (img.at + img.tokens <= from) continue;
        if (row + img.tokens > v.capacity) return error.ImageTooLarge;
        try v.tower.image(queue, img.pixels, img.gh, img.gw, .{ .buf = v.rows, .off = @as(usize, row) * width * 2 });
        try v.spans.append(v.gpa, .{ .at = img.at, .n = img.tokens, .row = row });
        row += img.tokens;
    }
    if (v.spans.items.len == 0) return null;
    return .{ .slot = slot, .rows = .{ .buffer = v.rows }, .spans = v.spans.items };
}

/// Info.vision's prepare: an image's bytes to the tower's patches (core/qwen_vl geometry), on the HTTP thread.
pub fn prepare(v: *const Vision, io: std.Io, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const p = try vl.geometry.prepare(io, a, bytes, v.c, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
