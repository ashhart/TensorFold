//! Native teacher-forced LFM3 logits for a public token fixture, without a Python forward in the native process.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm3;

const Dump = struct {
    io: std.Io,
    a: std.mem.Allocator,
    dir: []const u8,
    offset: usize = 0,

    fn take(ptr: *anyopaque, name: []const u8, value: mtl.Buffer, rows: usize, width: usize) !void {
        const self: *Dump = @ptrCast(@alignCast(ptr));
        const path = try std.fmt.allocPrint(self.a, "{s}/{d:0>5}_{s}.bf16", .{ self.dir, self.offset, name });
        defer self.a.free(path);
        try std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = path, .data = value.contents()[0 .. rows * width * 2] });
    }
};

pub fn main(init: std.process.Init) !void {
    const raw_args = try init.minimal.args.toSlice(init.arena.allocator());
    const tiles = raw_args.len > 1 and std.mem.eql(u8, raw_args[raw_args.len - 1], "--tiles");
    const binned = raw_args.len > 1 and std.mem.eql(u8, raw_args[raw_args.len - 1], "--binned");
    const args = raw_args[0 .. raw_args.len - @intFromBool(tiles or binned)];
    if (args.len < 4 or args.len > 8) return error.ExpectedModelTokensOutput;
    if (args.len >= 7 and !std.mem.eql(u8, args[4], "images")) return error.ExpectedImageRowsAndStart;
    if (args.len == 5 and !std.mem.eql(u8, args[4], "embedding") and !std.mem.eql(u8, args[4], "prefix") and !std.mem.eql(u8, args[4], "decode")) return error.ExpectedEmbeddingPrefixOrDecode;
    if (args.len == 6 and !std.mem.eql(u8, args[4], "trace")) return error.ExpectedTraceDirectory;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    const a = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<u4") or a.rank != 1 or a.count() == 0) return error.ExpectedU32TokenVector;
    if (args.len == 5 and std.mem.eql(u8, args[4], "embedding") and a.count() > q.state.batch_rows) return error.EmbeddingRowsExceedScratch;
    const ids = try init.gpa.alloc(u32, a.count());
    defer init.gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), a.data);
    for (ids) |id| if (id >= q.config.vocab) return error.InvalidLfmToken;
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    if (tiles) try m.enableProjectionCandidate();
    if (binned) try m.enableBinnedProjectionCandidate();
    var image_buffer: ?mtl.Buffer = null;
    defer if (image_buffer) |buffer| buffer.deinit();
    var image_pipeline: ?mtl.Pipeline = null;
    defer if (image_pipeline) |pipeline| pipeline.deinit();
    var span: q.image_rows.Span = undefined;
    var images: ?q.image_rows.Images = null;
    if (args.len >= 7) {
        if (m.config.kind != .vision_language) return error.ExpectedVisionLanguageModel;
        const start = try std.fmt.parseInt(u32, args[6], 10);
        const image_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[5], init.gpa, .limited(64 << 20));
        defer init.gpa.free(image_bytes);
        const row_bytes = q.config.hidden * 2;
        if (image_bytes.len == 0 or image_bytes.len % row_bytes != 0) return error.ExpectedBf16ImageRows;
        const rows = image_bytes.len / row_bytes;
        if (start > ids.len or rows > ids.len - start) return error.ImageRowsExceedPrompt;
        for (ids, 0..) |id, at| {
            if ((id == 125030) != (at >= start and at < start + rows)) return error.ImagePlaceholderMismatch;
        }
        var at: usize = 0;
        while (at < image_bytes.len) : (at += 2) {
            const value = std.mem.readInt(u16, image_bytes[at..][0..2], .little);
            if (value & 0x7f80 == 0x7f80) return error.NonfiniteImageRow;
        }
        image_buffer = try m.device.buffer(image_bytes.len, mtl.ResourceOptions.shared);
        @memcpy(image_buffer.?.contents()[0..image_bytes.len], image_bytes);
        image_pipeline = try q.image_rows.load(m.device);
        span = .{ .at = start, .rows = @intCast(rows), .source = 0 };
        images = .{ .buffer = image_buffer.?, .pipeline = image_pipeline.?, .spans = (&span)[0..1] };
    }
    var cache = try q.state.Cache.init(init.gpa, m.device, &m.config.layer_types, ids.len + 64);
    defer cache.deinit();
    var scratch = try q.state.Scratch.init(init.gpa, m.device, &m.config.layer_types, 32);
    defer scratch.deinit();
    if (args.len == 5 and std.mem.eql(u8, args[4], "embedding")) {
        @memcpy(scratch.ids.slice(u32, ids.len), ids);
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        const w = m.weights.embedding;
        e.setPipeline(m.kernels.embed(w.bits));
        e.setBuffer(scratch.ids, 0, 0);
        e.setBuffer(w.w, w.w_off, 1);
        e.setBuffer(w.scales, w.s_off, 2);
        e.setBuffer(w.biases, w.b_off, 3);
        e.setBuffer(scratch.h, 0, 4);
        e.setValue(@as(u32, @intCast(w.group)), 5);
        e.dispatchThreads(mtl.Size.of(q.config.hidden, ids.len, 1), mtl.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = scratch.h.contents()[0 .. ids.len * q.config.hidden * 2] });
        return;
    }
    // `prefix`: each block's history read again in 128-row chunks, the fidelity tool's gate for the tensor path.
    const prefix = args.len > 4 and std.mem.eql(u8, args[4], "prefix");
    var wide = try q.state.Scratch.init(init.gpa, m.device, &m.config.layer_types, if (prefix) 128 else 1);
    defer wide.deinit();
    if (args.len == 5 and std.mem.eql(u8, args[4], "decode")) {
        const history = ids.len - 1;
        var done: usize = 0;
        while (done < history) {
            const rows: usize = @min(history - done, 32);
            try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = rows }}, ids[done..][0..rows], false, .last);
            done += rows;
        }
        try q.forward.run(m, &wide, &.{.{ .cache = &cache, .rows = 1 }}, ids[history..], false, .all);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = wide.logits.contents()[0 .. q.config.vocab * 2] });
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(ids[0..history]), &digest, .{});
        std.debug.print("native LFM3 decode: history {d}, row {d}, token {d}, history_u32le_sha256 {s}\n", .{
            history, history, ids[history], &std.fmt.bytesToHex(digest, .lower),
        });
        return;
    }
    const out = try init.gpa.alloc(u8, ids.len * q.config.vocab * 2);
    defer init.gpa.free(out);
    const tracing = args.len == 6 or args.len == 8;
    var dump: Dump = .{ .io = init.io, .a = init.gpa, .dir = if (args.len == 8) args[7] else if (args.len == 6) args[5] else "" };
    if (tracing) try std.Io.Dir.cwd().createDirPath(init.io, dump.dir);
    var at: usize = 0;
    while (at < ids.len) {
        const n: usize = @min(ids.len - at, 32);
        if (prefix and at > 0) {
            cache.deinit();
            cache = try q.state.Cache.init(init.gpa, m.device, &m.config.layer_types, ids.len + 64);
            var done: usize = 0;
            while (done < at) {
                const rows: usize = @min(at - done, 128);
                try q.forward.run(m, &wide, &.{.{ .cache = &cache, .rows = rows }}, ids[done..][0..rows], false, .last);
                done += rows;
            }
        }
        if (tracing) {
            dump.offset = at;
            try q.forward.runTraceWithImages(m, &scratch, &.{.{ .cache = &cache, .rows = n }}, ids[at..][0..n], .all, images, .{ .ctx = &dump, .take = Dump.take });
        } else try q.forward.runWithImages(m, &scratch, &.{.{ .cache = &cache, .rows = n }}, ids[at..][0..n], false, .all, images);
        @memcpy(out[at * q.config.vocab * 2 ..][0 .. n * q.config.vocab * 2], scratch.logits.contents()[0 .. n * q.config.vocab * 2]);
        at += n;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = out });
    std.debug.print("native LFM3 forward: {d} rows, bf16 logits\n", .{ids.len});
}
