//! Run the original 23-layer tower against the independent FP64 tower formula.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const fixture = @import("vision_checkpoint_test.zig");
const oracle = @import("vision_test.zig");
const image = @import("vision_image.zig");
const vision = @import("vision.zig");
const t = std.testing;

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,

    fn init() !Gpu {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        return .{ .device = device, .queue = device.queue() catch return error.SkipZigTest };
    }
    fn deinit(g: Gpu) void {
        g.queue.deinit();
        g.device.deinit();
    }
};

test "original Mini vision projector matches the FP64 oracle on square and padded rectangular images" {
    const dir = try fixture.directory();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try fixture.loadConfig(a, dir);
    try t.expectEqual(@as(u32, 23), cfg.vision.layers);
    const gpu = try Gpu.init();
    defer gpu.deinit();
    const path = try fixture.shard(a, dir, "model.visual.backbone.embeddings.patch_embedding.weight");
    var store = ck.Checkpoint.init(t.allocator);
    defer store.deinit();
    try store.addFileSelected(gpu.device, path, "", "model.visual.");
    try t.expectEqual(@as(usize, 422), store.tensors.count());
    const tower = try vision.Tower.loadFromCheckpoint(t.allocator, gpu.device, gpu.queue, &cfg, .{ .max_tokens = 2 }, &store);
    defer tower.deinit();
    const patch = try store.get("model.visual.backbone.embeddings.patch_embedding.weight");
    try t.expectEqual(patch.buffer.id, tower.patch_w.buf.id);
    try t.expectEqual(patch.offset, tower.patch_w.off);
    try t.expect(tower.bytes() < store.residentBytes());
    for ([_][2]usize{ .{ 40, 40 }, .{ 79, 34 } }) |wh| {
        var rgb = try @import("vision_image_test.zig").rgbImage(a, wh[0], wh[1]);
        defer rgb.deinit(a);
        const p = try image.fromRgb(a, rgb, 1, 2);
        const out = try tower.encodeInput(p.pixels, p.gh, p.gw);
        defer out.deinit();
        const want = try oracle.reference(a, &store, cfg.vision, p.pixels, p.gh, p.gw);
        const got = out.slice(u16, want.len);
        const width = cfg.vision.text_hidden;
        for (0..p.tokens) |row| {
            const err = oracle.compare(got[row * width ..][0..width], want[row * width ..][0..width]);
            if (err.rms >= 0.05) {
                std.debug.print("vision {d}x{d} row {d}: relative RMS error {d}, worst column {d}\n", .{ wh[0], wh[1], row, err.rms, err.max_abs });
                try diagnose(tower, &store, p, got);
                return error.VisionProjectorMismatch;
            }
        }
    }
}

fn diagnose(tower: *vision.Tower, store: *const ck.Checkpoint, p: vision.Prepared, got: []const u16) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = tower.cfg;
    const n = cfg.layers + 1;
    const words = (@as(usize, p.tokens) * cfg.merge * cfg.merge + 1) * cfg.hidden * n;
    const trace = try tower.device.buffer(words * 2, mtl.ResourceOptions.shared);
    defer trace.deinit();
    const out = try tower.traceInput(p.pixels, p.gh, p.gw, trace);
    defer out.deinit();
    try t.expectEqualSlices(u16, got, out.slice(u16, got.len));
    const errs = try a.alloc(oracle.Error, n);
    const expected = try oracle.evaluate(a, store, cfg, p.pixels, p.gh, p.gw, trace.slice(u16, words), errs);
    for (errs, 0..) |err, i| {
        std.debug.print("vision stage {d} with matched BF16 input: rel RMS {d}, max abs {d}\n", .{ i, err.rms, err.max_abs });
    }
    const err = oracle.compare(got, expected);
    std.debug.print("vision projector with matched BF16 input: rel RMS {d}, max abs {d}\n", .{ err.rms, err.max_abs });
}
