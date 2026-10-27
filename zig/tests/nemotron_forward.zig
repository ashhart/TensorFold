//! Native teacher-forced Nemotron logits for tools/zig/nemotron_fidelity.py: prompt rows fused_rows a forward, then
//! (from --decode-from) decode rows one a forward. With --stages DIR, every stage forward.Point names, of every layer
//! of every forward, is copied out beside the logits (stages.bin, manifest stages.jsonl) for nemotron_stages.py.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const nemotron = tf.nemotron;
const fwd = nemotron.forward;

/// The stages of one forward, copied into one shared buffer as they are written.
const Stages = struct {
    dump: mtl.Buffer,
    cap: usize,
    used: usize = 0,
    layer: usize = 0,
    first: usize = 0, // the forward's first row
    rows: usize = 0,
    list: std.ArrayList(u8) = .empty, // this forward's manifest lines
    gpa: std.mem.Allocator,
    overflow: bool = false,

    fn at(ctx: *anyopaque, f: fwd.Forward, e: *fwd.Enc, p: fwd.Point, src: mtl.Buffer, bytes: usize) void {
        const self: *Stages = @ptrCast(@alignCast(ctx));
        if (p == .h_in) self.layer += 1; // every layer starts with h_in
        if (self.used + bytes > self.cap) {
            self.overflow = true;
            return;
        }
        f.copyIds(e, src, 0, self.dump, self.used, bytes / 4);
        self.list.print(self.gpa, "{{\"at\": {d}, \"rows\": {d}, \"layer\": {d}, \"point\": \"{t}\", \"bytes\": {d}, \"offset\": {d}}}\n", .{ self.first, self.rows, self.layer - 1, p, bytes, self.used }) catch {
            self.overflow = true;
        };
        self.used += bytes;
    }
};

const usage = "usage: tf-nemotron-forward MODEL TOKENS.npy OUT.bin [--decode-from N] [--stages DIR]";

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("{s}\n", .{usage});
        std.process.exit(2);
    }
    var decode_from: ?usize = null;
    var stages_dir: ?[]const u8 = null;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (i + 1 < args.len and std.mem.eql(u8, args[i], "--decode-from")) {
            i += 1;
            decode_from = try std.fmt.parseInt(usize, args[i], 10);
        } else if (i + 1 < args.len and std.mem.eql(u8, args[i], "--stages")) {
            i += 1;
            stages_dir = args[i];
        } else {
            std.debug.print("{s}\n", .{usage});
            std.process.exit(2);
        }
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(4 << 20));
    defer gpa.free(bytes);
    const a = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<u4") or a.rank != 1 or a.count() == 0) return error.ExpectedU32TokenVector;
    const ids = try gpa.alloc(u32, a.count());
    defer gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), a.data);

    const m = try nemotron.Model.load(gpa, init.io, args[1], false);
    defer m.deinit();
    const vocab = m.config.vocab;
    const step = nemotron.backend.fused_rows;
    const b = try nemotron.backend.Metal.init(gpa, m, .{ .capacity = ids.len + 64, .chunk = step, .drafts = false, .streams = 2 });
    defer b.deinit();
    var cache = try nemotron.state.Cache.init(m.device, m.config, ids.len + 64, false);
    defer cache.deinit(&b.pool);
    @memcpy(b.prompt.slice(u32, ids.len), ids);
    const out = try gpa.alloc(u8, ids.len * vocab * 2);
    defer gpa.free(out);

    // --stages: one shared buffer a forward, appended to stages.bin after it; the manifest's offsets are into stages.bin
    var stages: ?Stages = null;
    var stage_bytes: std.ArrayList(u8) = .empty;
    defer stage_bytes.deinit(gpa);
    var stage_lines: std.ArrayList(u8) = .empty;
    defer stage_lines.deinit(gpa);
    if (stages_dir != null) stages = .{ .dump = try m.device.buffer(1 << 30, mtl.ResourceOptions.shared), .cap = 1 << 30, .gpa = gpa };
    defer if (stages) |*s| {
        s.list.deinit(gpa);
        s.dump.deinit();
    };

    var at: usize = 0;
    while (at < ids.len) {
        const width: usize = if (decode_from) |n| (if (at >= n) 1 else @min(step, n - at)) else step;
        const rows = @min(ids.len - at, width);
        const slot = try b.pool.take();
        const inner = mtl.objc.Pool.push();
        defer inner.pop();
        const cb = m.queue.commandBuffer();
        var e = nemotron.forward.Enc{ .e = cb.compute(.concurrent), .concurrent = true };
        const segs = [_]nemotron.forward.Seg{.{ .rows = rows, .cache = &cache, .store = .full, .slot = @intCast(slot) }};
        var f = b.forward();
        if (stages) |*s| {
            s.used = 0;
            s.layer = 0;
            s.first = at;
            s.rows = rows;
            s.list.clearRetainingCapacity();
            f.tap = .{ .ctx = s, .at = Stages.at };
        }
        f.body(&e, &segs, b.prompt, at * 4);
        f.draw(&e, &segs, .all, b.tokens, 0);
        e.e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("GPU failure at row {d}: {s}\n", .{ at, why });
            return error.GpuFailed;
        }
        cache.advance(&b.pool, rows, slot);
        @memcpy(out[at * vocab * 2 ..][0 .. rows * vocab * 2], b.scratch.logits.contents()[0 .. rows * vocab * 2]);
        if (stages) |*s| {
            if (s.overflow) return error.StageDumpOverflow;
            // the forward's offsets are into its own buffer: shift them by what stages.bin already holds
            const base = stage_bytes.items.len;
            var lines = std.mem.splitScalar(u8, s.list.items, '\n');
            while (lines.next()) |line| {
                if (line.len == 0) continue;
                const k = std.mem.indexOf(u8, line, "\"offset\": ").? + 10;
                const off = try std.fmt.parseInt(usize, line[k .. line.len - 1], 10);
                try stage_lines.print(gpa, "{s}{d}}}\n", .{ line[0..k], base + off });
            }
            try stage_bytes.appendSlice(gpa, s.dump.contents()[0..s.used]);
        }
        at += rows;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = out });
    if (stages_dir) |dir| {
        try std.Io.Dir.cwd().createDirPath(init.io, dir);
        const bin = try std.fs.path.join(gpa, &.{ dir, "stages.bin" });
        defer gpa.free(bin);
        const list = try std.fs.path.join(gpa, &.{ dir, "stages.jsonl" });
        defer gpa.free(list);
        // in pieces: one write() of more than 2 GiB fails on macOS
        const file = try std.Io.Dir.cwd().createFile(init.io, bin, .{});
        defer file.close(init.io);
        var off: usize = 0;
        while (off < stage_bytes.items.len) : (off += 256 << 20) {
            const end = @min(off + (256 << 20), stage_bytes.items.len);
            try file.writePositionalAll(init.io, stage_bytes.items[off..end], off);
        }
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = list, .data = stage_lines.items });
    }
    std.debug.print("native Nemotron forward: {d} rows, bf16 logits{s}\n", .{ ids.len, if (stages != null) ", stages dumped" else "" });
}
