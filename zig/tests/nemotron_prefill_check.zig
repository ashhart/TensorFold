//! One prompt chunk on the prefill kernels against tools/zig/prefill_dump.py's rows: embedding, every layer, final norm.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const nemotron = tf.nemotron;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4 and args.len != 5) {
        std.debug.print("usage: tf-nemotron-prefill-check MODEL ID,... DUMP.npy [SCRATCH_DIR]\n", .{});
        std.process.exit(2);
    }
    var list: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, args[2], ',');
    while (it.next()) |t| try list.append(arena, try std.fmt.parseInt(u32, t, 10));
    // TF_PREFILL_PREFIX=N: the first N ids fill the caches as a chunk first; the rest is the checked chunk
    const before: usize = if (std.c.getenv("TF_PREFILL_PREFIX")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 0;
    const all = list.items;
    const ids = all[before..];
    const n = ids.len;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const f = try mtl.MappedFile.open(args[3]);
    defer f.deinit();
    const dump = try tf.npy.parse(f.bytes[0..f.size]);
    const m = try nemotron.Model.load(gpa, init.io, args[1], false);
    defer m.deinit();
    const c = m.config;
    const d = c.hidden;
    const steps = c.layers + 2;
    if (dump.rank != 3 or dump.shape[0] != steps or dump.shape[1] != n or dump.shape[2] != d) return error.BadDump;
    const want = @as([*]const u16, @ptrCast(@alignCast(dump.data.ptr)))[0 .. steps * n * d];

    const b = try nemotron.backend.Metal.init(gpa, m, .{ .capacity = all.len + 64, .chunk = @max(n, before, nemotron.backend.fused_rows + 1), .drafts = false, .streams = 2 });
    defer b.deinit();
    var cache = try nemotron.state.Cache.init(m.device, c, all.len + 64, false);
    defer cache.deinit(&b.pool);
    @memcpy(b.prompt.slice(u32, all.len), all);
    if (before > 0) {
        const first = try b.pool.take();
        const cb = m.queue.commandBuffer();
        var e = nemotron.forward.Enc{ .e = cb.compute(.concurrent), .concurrent = true };
        b.wide.?.chunk(&e, b.forward(), &cache, b.prompt, 0, before, first);
        e.e.end();
        cb.commit();
        cb.wait();
        cache.advance(&b.pool, before, first);
    }
    const fresh = try b.pool.take();
    defer b.pool.give(fresh);
    const p = &b.wide.?;
    var bad: usize = 0;
    for (0..steps) |step| {
        const cb = m.queue.commandBuffer();
        var e = nemotron.forward.Enc{ .e = cb.compute(.concurrent), .concurrent = true };
        const x = p.context(&e, b.forward(), &cache, n, fresh);
        if (step == 0) p.embed(x, b.prompt, before * 4) else if (step <= c.layers) p.layer(x, step - 1) else p.final(x);
        e.e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        const got = (if (step <= c.layers) p.s.h else p.s.x).slice(u16, n * d);
        const ref = want[step * n * d ..][0 .. n * d];
        var differ: usize = 0;
        var first: usize = 0;
        for (got, ref, 0..) |g, r, i| if (g != r) {
            if (differ == 0) first = i;
            differ += 1;
        };
        const name = if (step == 0) "embedding" else if (step <= c.layers) @tagName(c.kinds[step - 1]) else "final norm";
        if (differ == 0) {
            std.debug.print("{d:>2} {s}: equal\n", .{ step, name });
            continue;
        }
        std.debug.print("{d:>2} {s}: {d} of {d} values differ, the first at row {d} column {d}\n", .{ step, name, differ, n * d, first / d, first % d });
        bad += 1;
        if (args.len == 5) return scratch(init.io, p, args[4]);
    }
    if (bad > 0) std.process.exit(1);
}

/// Every scratch buffer after the first differing step, as DIR/<name>.bin (the step's intermediates).
fn scratch(io: std.Io, p: *const nemotron.prefill.Prefill, dir: []const u8) !void {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    inline for (@typeInfo(nemotron.prefill.Scratch).@"struct".field_names) |name| {
        const b = @field(p.s, name);
        try d.writeFile(io, .{ .sub_path = name ++ ".bin", .data = b.contents()[0..b.length()] });
    }
    std.debug.print("scratch written to {s}\n", .{dir});
    std.process.exit(1);
}
