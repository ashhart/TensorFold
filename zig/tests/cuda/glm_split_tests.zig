//! GLM-5.3-Flash's prompt chunks with their indexer rows split across the ranks (TF_GLM_INDEX_SPLIT) against twins.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// `glm-split-check`: prompts a chunk at a time, split and not; first token, next window and every state byte equal.
pub fn splitCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192 }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    try check.expect(world > 1, "glm-split-check needs two ranks or more", .{});
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const C = glm.engine.prompt_rows;
    const capacity = @max(dc.capacity, 4 * C);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's split and whole prompt chunks to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    const from = C; // every chunk from the second on: each holds rows past the dense limit
    // three chunks and a tail of 300, a tail of 2 (one rank without a row), a tail of 5, two chunks
    const lens = [_]usize{ 3 * C + 300, 2 * C + 2, 2 * C + 5, 2 * C };
    var failed = false;
    for (lens, 0..) |len, i| {
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, j| u.* = @intCast(100 + (j * 7919 + i * 31) % 5000);
        var firsts: [2]u32 = undefined;
        var outs: [2][4]u32 = undefined;
        var logits: [2][]u16 = undefined;
        var splits: [2]u64 = undefined;
        for (0..2) |twin| { // twin 0 whole, twin 1 split from `from`
            e.settings.index_split = if (twin == 1) from else 0;
            try x.bind(@intCast(2 * i + twin));
            try x.begin(null);
            const before = e.index_splits;
            var at: usize = 0;
            const t = while (true) : (at += C) {
                const r = x.chunk(ids, at, null, null, null) catch |err| {
                    std.debug.print("FAIL split: prompt of {d}, twin {d}: the chunk at {d}: {s}\n", .{ len, twin, at, @errorName(err) });
                    return err;
                };
                if (r) |f| break f;
            };
            splits[twin] = e.index_splits - before;
            firsts[twin] = t;
            const win = [_]u32{ t, 101, 202, 303 };
            try x.verify(&win, &outs[twin], 0);
            logits[twin] = try a.alloc(u16, win.len * V);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(logits[twin].ptr, e.bd.logits, win.len * V * 2), "cuMemcpyDtoH");
        }
        e.settings.index_split = 0;
        var want: u64 = 0; // the chunks glm.sparse.splitAt splits over this prompt
        var start: usize = 0;
        while (start < len) : (start += C) want += @intFromBool(glm.sparse.splitAt(start, from, e.share.world));
        if (splits[0] != 0 or splits[1] != want) {
            std.debug.print("FAIL split: prompt of {d}: {d} split chunks whole, {d} split (want {d})\n", .{ len, splits[0], splits[1], want });
            failed = true;
        }
        if (firsts[0] != firsts[1] or !std.mem.eql(u32, &outs[0], &outs[1]) or !std.mem.eql(u16, logits[0], logits[1])) {
            std.debug.print("FAIL split: prompt of {d}: first {d}/{d}, the next window's tokens or logits differ\n", .{ len, firsts[0], firsts[1] });
            failed = true;
        }
        try e.stream.synchronize();
        const st = &e.seqs.items[2 * i].?.st;
        const tw = &e.seqs.items[2 * i + 1].?.st;
        const n = st.arena.buf.len;
        const h0 = try a.alloc(u8, n);
        const h1 = try a.alloc(u8, n);
        try st.arena.buf.download(0, h0);
        try tw.arena.buf.download(0, h1);
        const diff = std.mem.indexOfDiff(u8, h0, h1);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or !std.mem.eql(u1, st.cur, tw.cur) or diff != null) {
            std.debug.print("FAIL split: prompt of {d}: the split state differs (pos {d}/{d}, first byte {?d} of {d})\n", .{ len, st.pos, tw.pos, diff, n });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("split indexer rows at TP={d}: prompts of {any} rows equal their whole twins in the first token, the next window's logits and tokens, and every state byte", .{ world, lens });
}
