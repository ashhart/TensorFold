//! Teacher-forced Qwen3.5-2B logits on the CPU for a token fixture: raw bf16 rows for tools/zig/cpu_fidelity.py.
const std = @import("std");
const q = @import("qwen35_cpu");
const npy = @import("core").npy;

/// MODEL_DIR tokens.npy out.bf16 [THREADS [ROWS]]: THREADS workers beside the caller, ROWS a pass (1: decode steps).
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4 or args.len > 6) return error.ExpectedModelTokensOutput;
    const threads: ?usize = if (args.len >= 5) try std.fmt.parseInt(usize, args[4], 10) else null;
    const rows: usize = if (args.len == 6) try std.fmt.parseInt(usize, args[5], 10) else q.state.batch_rows;
    if (rows == 0 or rows > q.state.batch_rows) return error.RowsPerPass;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    const a = try npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<u4") or a.rank != 1 or a.count() == 0) return error.ExpectedU32TokenVector;
    const ids = try init.gpa.alloc(u32, a.count());
    defer init.gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), a.data);
    const m = try q.Model.load(init.gpa, init.io, args[1], threads);
    defer m.deinit();
    var cache = try q.state.Cache.init(init.gpa, ids.len + 64);
    defer cache.deinit();
    var scratch = try q.state.Scratch.init(init.gpa, q.state.batch_rows, cache.capacity, m.pool.workers());
    defer scratch.deinit();
    const out = try init.gpa.alloc(u16, ids.len * q.config.vocab);
    defer init.gpa.free(out);
    var at: usize = 0;
    while (at < ids.len) {
        const n: usize = @min(ids.len - at, rows);
        try q.forward.run(m, &scratch, &.{.{ .cache = &cache, .rows = n }}, ids[at..][0..n], false, .all);
        @memcpy(out[at * q.config.vocab ..][0 .. n * q.config.vocab], scratch.logits[0 .. n * q.config.vocab]);
        at += n;
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = std.mem.sliceAsBytes(out) });
    std.debug.print("cpu Qwen forward: {d} rows of bf16 logits on {d} threads, {d} rows a pass\n", .{ ids.len, m.pool.workers(), rows });
}
