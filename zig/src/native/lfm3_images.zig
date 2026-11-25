//! LFM3-VL's tower and a prompt pass's projected image rows share the text model's queue and lifetime.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.lfm3;

pub const Vision = struct {
    gpa: std.mem.Allocator,
    model: *q.Model,
    tower: *q.vision.Vision,
    rows: mtl.Buffer, // the admitted buffer, used by one pass at a time
    busy: bool = false,
    passes: std.AutoHashMapUnmanaged(*const tf.lanes.Stream, Pass) = .empty,
    copy: mtl.Pipeline,
    capacity: u32,
    max_images: u32,

    /// A pass's projected rows and spans: the shared buffer, or one of its own while that is in use.
    const Pass = struct { spans: std.ArrayList(q.image_pass.Span) = .empty, rows: mtl.Buffer, owned: bool };

    pub fn deinit(v: *Vision) void {
        var it = v.passes.valueIterator();
        while (it.next()) |pass| v.closePass(pass);
        v.passes.deinit(v.gpa);
        v.copy.deinit();
        v.rows.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    fn closePass(v: *Vision, pass: *Pass) void {
        pass.spans.deinit(v.gpa);
        if (pass.owned) pass.rows.deinit() else v.busy = false;
    }

    /// One pass's images from `from` on, each distinct one through the tower once before its first chunk.
    pub fn begin(ptr: *anyopaque, stream: *const tf.lanes.Stream, from: usize) !?q.image_rows.Images {
        const v: *Vision = @ptrCast(@alignCast(ptr));
        try q.image.validate(stream.images, stream.prompt().len, v.capacity, v.max_images);
        const total = q.image_pass.rows(stream.images, from);
        if (total == 0) return null;
        if (v.passes.contains(stream)) return error.Lfm3ImagePassOpen;
        var pass: Pass = .{ .rows = v.rows, .owned = false };
        if (v.busy) {
            pass.rows = try v.model.device.buffer(total * q.config.hidden * 2, mtl.ResourceOptions.private | mtl.ResourceOptions.untracked);
            pass.owned = true;
        } else v.busy = true;
        errdefer v.closePass(&pass);
        try q.image_pass.encode(v.gpa, stream.images, from, &pass.spans, project(v, pass.rows));
        try v.passes.put(v.gpa, stream, pass);
        const kept = v.passes.getPtr(stream).?;
        return .{ .buffer = kept.rows, .pipeline = v.copy, .spans = kept.spans.items };
    }

    /// The pass ended (its last chunk, a cancel or a failure): its rows and spans go.
    pub fn end(ptr: *anyopaque, stream: *const tf.lanes.Stream) void {
        const v: *Vision = @ptrCast(@alignCast(ptr));
        if (v.passes.fetchRemove(stream)) |entry| {
            var pass = entry.value;
            v.closePass(&pass);
        }
    }

    fn project(v: *Vision, buf: mtl.Buffer) Project {
        return .{ .v = v, .buf = buf };
    }

    const Project = struct {
        v: *Vision,
        buf: mtl.Buffer,

        pub fn project(p: Project, im: anytype, row: u32) !void {
            try p.v.tower.image(p.v.model.queue, im.pixels, im.gh, im.gw, .{ .buf = p.buf, .off = @as(usize, row) * q.config.hidden * 2 });
        }
    };
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, model: *q.Model, dir: []const u8, requested: api.VisionOpen) !*Vision {
    if (model.config.kind != .vision_language) return error.NoVisionTower;
    try q.image.loadProcessor(io, gpa, dir);
    const capacity = try q.image.admitBudget(requested.image_tokens_set, requested.image_tokens, requested.max_images);
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
