//! Native teacher-forced Nemotron logits for tools/zig/nemotron_fidelity.py, fused_rows rows a decode forward.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const nemotron = tf.nemotron;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-nemotron-forward MODEL TOKENS.npy OUT.bin\n", .{});
        std.process.exit(2);
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
    var at: usize = 0;
    while (at < ids.len) {
        const rows = @min(ids.len - at, step);
        const slot = try b.pool.take();
        const inner = mtl.objc.Pool.push();
        defer inner.pop();
        const cb = m.queue.commandBuffer();
        var e = nemotron.forward.Enc{ .e = cb.compute(.concurrent), .concurrent = true };
        const segs = [_]nemotron.forward.Seg{.{ .rows = rows, .cache = &cache, .store = .full, .slot = @intCast(slot) }};
        b.forward().body(&e, &segs, b.prompt, at * 4);
        b.forward().draw(&e, &segs, .all, b.tokens, 0);
        e.e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("GPU failure at row {d}: {s}\n", .{ at, why });
            return error.GpuFailed;
        }
        cache.advance(&b.pool, rows, slot);
        @memcpy(out[at * vocab * 2 ..][0 .. rows * vocab * 2], b.scratch.logits.contents()[0 .. rows * vocab * 2]);
        at += rows;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = out });
    std.debug.print("native Nemotron forward: {d} rows, bf16 logits\n", .{ids.len});
}
