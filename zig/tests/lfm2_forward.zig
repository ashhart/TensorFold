//! Native teacher-forced LFM2 logits for a public token fixture, without a Python forward in the native process.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) return error.ExpectedModelTokensOutput;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    const a = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<u4") or a.rank != 1 or a.count() == 0) return error.ExpectedU32TokenVector;
    const ids = try init.gpa.alloc(u32, a.count());
    defer init.gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), a.data);
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    var cache = try q.state.Cache.init(init.gpa, m.device, &m.config.attention, ids.len + 64);
    defer cache.deinit();
    var scratch = try q.state.Scratch.init(init.gpa, m.device, &m.config.attention, 32);
    defer scratch.deinit();
    // `prefix`: each block's history read again in 2,048-row chunks, the fidelity tool's gate for the tensor path.
    const prefix = args.len > 4 and std.mem.eql(u8, args[4], "prefix");
    var wide = try q.state.Scratch.init(init.gpa, m.device, &m.config.attention, if (prefix) 2048 else 1);
    defer wide.deinit();
    const out = try init.gpa.alloc(u8, ids.len * q.config.vocab * 2);
    defer init.gpa.free(out);
    var at: usize = 0;
    while (at < ids.len) {
        const n: usize = @min(ids.len - at, 32);
        if (prefix and at > 0) {
            cache.deinit();
            cache = try q.state.Cache.init(init.gpa, m.device, &m.config.attention, ids.len + 64);
            var done: usize = 0;
            while (done < at) {
                const rows: usize = @min(at - done, 2048);
                try q.forward.run(m, &wide, &.{.{ .cache = &cache, .rows = rows }}, ids[done..][0..rows], false, .last);
                done += rows;
            }
        }
        try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = n }}, ids[at..][0..n], false, .all);
        @memcpy(out[at * q.config.vocab * 2 ..][0 .. n * q.config.vocab * 2], scratch.logits.contents()[0 .. n * q.config.vocab * 2]);
        at += n;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = out });
    std.debug.print("native LFM2 forward: {d} rows, bf16 logits\n", .{ids.len});
}
