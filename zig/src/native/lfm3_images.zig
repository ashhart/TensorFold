//! LFM3-VL's tower and image rows share the text model's queue and checkpoint lifetime.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm3;

pub const Vision = struct {
    gpa: std.mem.Allocator,
    model: *q.Model,
    tower: *q.vision.Vision,
    rows: mtl.Buffer,
    copy: mtl.Pipeline,
    capacity: u32,
    max_images: u32,
    spans: std.ArrayList(q.image_rows.Span) = .empty,

    pub fn deinit(v: *Vision) void {
        v.spans.deinit(v.gpa);
        v.copy.deinit();
        v.rows.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    pub fn encode(ptr: *anyopaque, stream: *const tf.lanes.Stream, from: usize) !?q.image_rows.Images {
        const v: *Vision = @ptrCast(@alignCast(ptr));
        try q.image.validate(stream.images, stream.prompt().len, v.capacity, v.max_images);
        v.spans.clearRetainingCapacity();
        var used: u32 = 0;
        for (stream.images) |img| {
            if (@as(usize, img.at) + img.tokens <= from) continue;
            if (img.tokens > v.capacity -| used) return error.ImageTooLarge;
            try v.tower.image(v.model.queue, img.pixels, img.gh, img.gw, .{
                .buf = v.rows,
                .off = @as(usize, used) * q.config.hidden * 2,
            });
            try v.spans.append(v.gpa, .{ .at = img.at, .rows = img.tokens, .source = used });
            used += img.tokens;
        }
        if (v.spans.items.len == 0) return null;
        return .{ .buffer = v.rows, .pipeline = v.copy, .spans = v.spans.items };
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, model: *q.Model, dir: []const u8, requested: api.VisionOpen) !*Vision {
    if (model.config.kind != .vision_language) return error.NoVisionTower;
    const path = try std.fs.path.join(gpa, &.{ dir, "processor_config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(4 << 20));
    defer gpa.free(bytes);
    try q.image.processor(gpa, bytes);
    const capacity = if (requested.image_tokens_set) requested.image_tokens else 256;
    if (capacity < 64 or capacity > 4096 or requested.max_images == 0) return error.BadLfm3ImageBudget;
    const v = try gpa.create(Vision);
    errdefer gpa.destroy(v);
    const tower = try q.vision.Vision.load(gpa, model.device, &model.checkpoint, capacity);
    errdefer tower.deinit();
    const rows = try model.device.buffer(@as(usize, capacity) * q.config.hidden * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer rows.deinit();
    const copy = try q.image_rows.load(model.device);
    v.* = .{ .gpa = gpa, .model = model, .tower = tower, .rows = rows, .copy = copy, .capacity = capacity, .max_images = requested.max_images };
    return v;
}
