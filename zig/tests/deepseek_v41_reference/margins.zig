//! Prefill, then capture one diagnostic token's per-layer selection scratch for offline margins; no reload.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const index = tf.deepseek_v41.csa2_index;
const gpu_counters = tf.deepseek_v41_profiler;

const copy_source =
    \\kernel void margins_copy(const device uint* x [[buffer(0)]], device uint* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if(i<n) y[i]=x[i]; }
;

const Capture = struct {
    pipeline: mtl.Pipeline,
    buffer: mtl.Buffer,
    per_layer: usize,
    slots: []Slot,
    count: usize = 0,

    const Slot = struct { layer: u32, position: u32, rows: u32, offset: u64 };

    fn init(gpa: std.mem.Allocator, model: *const Model, slots: usize) !Capture {
        const per_layer = index.Scratch.bytes(model.attention_workspace.max_rows);
        const device = model.device;
        const lib = try mtl.Library.fromSource(device, copy_source, .{});
        defer lib.deinit();
        const pipeline = try mtl.Pipeline.init(device, lib, "margins_copy", false);
        errdefer pipeline.deinit();
        const buffer = try device.buffer(per_layer * slots, mtl.ResourceOptions.shared);
        errdefer buffer.deinit();
        return .{ .pipeline = pipeline, .buffer = buffer, .per_layer = per_layer, .slots = try gpa.alloc(Slot, slots) };
    }
    fn deinit(self: *Capture) void {
        self.pipeline.deinit();
        self.buffer.deinit();
    }

    fn hook(ctx: *anyopaque, enc: mtl.ComputeEncoder, layer: u32, stage: gpu_counters.Stage, model: *const Model) void {
        _ = stage;
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (self.count > 0 and self.slots[self.count - 1].layer == layer) return; // the hook fires twice per layer
        const work = &model.attention_workspace;
        if (work.selection_layer == null or work.selection_layer.? != layer) return;
        if (self.count >= self.slots.len) return;
        const scratch_bytes = index.Scratch.bytes(work.selection.rows);
        const slot = &self.slots[self.count];
        slot.* = .{ .layer = layer, .position = work.selection_position, .rows = work.selection_rows, .offset = @as(u64, self.count) * @as(u64, self.per_layer) };
        self.count += 1;
        const uints: u32 = @intCast(scratch_bytes / 4);
        enc.setPipeline(self.pipeline);
        enc.setBuffer(work.selection.buffer, 0, 0);
        enc.setBuffer(self.buffer, @intCast(slot.offset), 1);
        enc.setValue(uints, 2);
        enc.dispatchThreads(.{ .width = uints }, .{ .width = 1024 });
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, model: *Model, ids: []const u32, dir: []const u8, deep: usize) !void {
    if (ids.len <= deep + 1) return error.InvalidMarginsPrompt;
    var indexer_layers: usize = 0;
    for (model.weights.layers) |layer| {
        if (layer.attn.indexer != null) indexer_layers += 1;
    }
    if (indexer_layers == 0) return error.MissingAttentionIndexer;
    var capture = try Capture.init(gpa, model, indexer_layers);
    defer capture.deinit();

    try model.reset();
    var at: usize = 0;
    while (at < deep) {
        const end = @min(at + model.workspace.plan.rows, deep);
        _ = try model.forward(ids[at..end], .last);
        at = end;
    }
    const previous = model.observer;
    model.observer = .{ .ctx = &capture, .encode = Capture.hook };
    defer model.observer = previous;
    _ = try model.forward(&.{ids[deep]}, .last);

    const blob_path = try std.fs.path.join(gpa, &.{ dir, "margins-scratch.bin" });
    defer gpa.free(blob_path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = blob_path, .data = capture.buffer.contents()[0 .. capture.per_layer * capture.count] });
    var meta: std.ArrayList(u8) = .empty;
    defer meta.deinit(gpa);
    for (capture.slots[0..capture.count]) |slot| {
        const line = try std.fmt.allocPrint(gpa, "{{\"layer\":{d},\"position\":{d},\"rows\":{d},\"offset\":{d},\"bytes\":{d}}}\n", .{ slot.layer, slot.position, slot.rows, slot.offset, index.Scratch.bytes(model.attention_workspace.max_rows) });
        defer gpa.free(line);
        try meta.appendSlice(gpa, line);
    }
    const meta_path = try std.fs.path.join(gpa, &.{ dir, "margins-meta.jsonl" });
    defer gpa.free(meta_path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = meta_path, .data = meta.items });
    std.debug.print("margins: {d} layers captured at position {d}\n", .{ capture.count, deep });
}
