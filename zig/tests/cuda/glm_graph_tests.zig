//! TF_GLM_GRAPHS on CUDA: graphed streams' windows against their eager twins', byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lanes_tests = @import("glm_lanes_tests.zig");
const Gpu = check.Gpu;

/// Prompts under seams: lone windows cross 512 and 4,096 keys, then shared ones a score grid's.
const lengths = [_]usize{ 500, 2330, 3990 };
/// A window of 8,192 and a verify window, as cuda_native.capacity makes it.
const capacity = 8256;
const S = lengths.len; // graphed slots 0 to S - 1; their eager twins S to 2S - 1
const steady = 48; // rounds a stream takes lone windows of 4 rows (3 kept), then all streams' shared windows likewise
const mixed = 90; // then rounds of 1 to 4 rows a stream, lone and 2 or 3 streams shared
const rounds = steady * S + steady + mixed;

/// A round's windows: `n` streams from `first` on (one: a lone window), `rows` and `keep` each.
const Round = struct { n: usize, first: usize, rows: [S]usize = @splat(4), keep: [S]usize = @splat(3) };

/// The steady rounds repeat one key a slot, so a held graph replays and moves in place at a seam; then mixed ones.
fn plan(round: usize) Round {
    if (round < steady * S) return .{ .n = 1, .first = round / steady };
    if (round < steady * S + steady) return .{ .n = S, .first = 0 };
    const m = round - steady * S - steady;
    var r: Round = .{ .n = if (m % 3 == 0) 1 else 2 + m % 2, .first = (m / 3 + m) % S };
    for (0..S) |k| {
        r.rows[k] = 1 + (m + k) % 4;
        r.keep[k] = 1 + (m + k * 3) % r.rows[k];
    }
    return r;
}

/// `glm-graph-check`: lone and shared windows through the graph cache against eager twins; then the twins' states.
pub fn graphCheck(gpu: Gpu, checkpoint: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try lanes_tests.openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    set.graphs = true; // every rank's engine holds the caches; rank 0 turns them off for the twins' windows
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's graphed and eager windows to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const V = e.share.vocab.len();
    const mine = try a.alloc(u16, 4 * S * V);
    const twin = try a.alloc(u16, 4 * S * V);
    var failed = false;
    for (lengths, 0..) |len, i| {
        const ids = try a.alloc(u32, len);
        for (ids, 0..) |*u, n| u.* = @intCast(100 + (n * 7919 + i * 31) % 5000);
        var first: [2]u32 = undefined;
        var drafts: [2][glm.engine.max_drafts]u32 = undefined;
        for ([_]u32{ @intCast(i), @intCast(S + i) }, 0..) |slot, j| {
            try x.bind(slot);
            try x.begin(null);
            first[j] = try x.prefill(ids, null, null, null);
            try x.draft(true, &.{first[j]}, 2, &drafts[j], null);
        }
        if (first[0] != first[1] or !std.mem.eql(u32, drafts[0][0..2], drafts[1][0..2])) {
            std.debug.print("FAIL graphs: a {d}-token prompt's first token or drafts differ from its twin's\n", .{len});
            failed = true;
        }
    }
    var compared: usize = 0;
    for (0..rounds) |round| {
        const p = plan(round);
        const n = p.n;
        const rows = p.rows;
        var ids: [S][4]u32 = undefined;
        var outs: [2][S][4]u32 = undefined;
        var total: usize = 0;
        for (0..n) |k| {
            const s = (p.first + k) % S;
            for (ids[k][0..rows[k]], 0..) |*u, r| u.* = @intCast(100 + (round * 31 + s * 7 + r * 13) % 5000);
            total += rows[k];
        }
        for (0..2) |j| { // j 0: the graphed slots, j 1: their twins with the caches off
            const off = j == 1;
            const held = if (off) e.graphs else null;
            if (off) e.graphs = null;
            defer if (off) {
                e.graphs = held;
            };
            const base: u32 = if (j == 0) 0 else S;
            if (n == 1) {
                try x.bind(base + @as(u32, @intCast(p.first)));
                try x.verify(ids[0][0..rows[0]], outs[j][0][0..rows[0]], 0);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bd.logits, rows[0] * V * 2), "cuMemcpyDtoH");
            } else {
                var parts: [S]glm.engine.Part = undefined;
                for (0..n) |k| parts[k] = .{ .slot = base + @as(u32, @intCast((p.first + k) % S)), .ids = ids[k][0..rows[k]], .out = outs[j][k][0..rows[k]] };
                try x.verifyShared(parts[0..n]);
                try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(if (j == 0) mine.ptr else twin.ptr, e.bs.?.logits, total * V * 2), "cuMemcpyDtoH");
            }
        }
        var same = std.mem.eql(u16, mine[0 .. total * V], twin[0 .. total * V]);
        for (0..n) |k| same = same and std.mem.eql(u32, outs[0][k][0..rows[k]], outs[1][k][0..rows[k]]);
        if (!same) {
            std.debug.print("FAIL graphs: round {d} ({d} streams, {d} rows): logits or tokens differ from the eager twins'\n", .{ round, n, total });
            failed = true;
        }
        compared += total;
        for (0..n) |k| { // the same rows kept and drafted from on both, the twin first
            const s: u32 = @intCast((p.first + k) % S);
            const keep = p.keep[k];
            var d: [2][glm.engine.max_drafts]u32 = undefined;
            for ([_]u32{ @as(u32, S) + s, s }, [_]usize{ 1, 0 }) |slot, j| {
                try x.bind(slot);
                try x.keep(keep);
                try x.draft(false, outs[j][k][0..keep], 2, &d[j], null);
            }
            if (!std.mem.eql(u32, d[0][0..2], d[1][0..2])) {
                std.debug.print("FAIL graphs: round {d}, stream {d}: drafts {any} against its twin's {any}\n", .{ round, s, d[0][0..2], d[1][0..2] });
                failed = true;
            }
        }
    }
    try e.stream.synchronize();
    for (0..S) |i| { // the whole states after every round's commit
        const st = &e.seqs.items[i].?.st;
        const tw = &e.seqs.items[S + i].?.st;
        const h0 = try a.alloc(u8, st.arena.buf.len);
        const h1 = try a.alloc(u8, tw.arena.buf.len);
        try st.arena.buf.download(0, h0);
        try tw.arena.buf.download(0, h1);
        const at = std.mem.indexOfDiff(u8, h0, h1);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or !std.mem.eql(u1, st.cur, tw.cur) or at != null) {
            std.debug.print("FAIL graphs: stream {d}'s state differs from its twin's (pos {d}/{d}, first byte {?d})\n", .{ i, st.pos, tw.pos, at });
            failed = true;
        }
    }
    const g = e.graphs.?;
    const counts = [_]u64{ g.one.captures, g.one.updates, g.one.replays, g.shared.captures, g.shared.updates, g.shared.replays };
    for (counts) |c| if (c == 0) {
        std.debug.print("FAIL graphs: a cache never captured, updated or replayed (lone {d}/{d}/{d}, shared {d}/{d}/{d})\n", .{ counts[0], counts[1], counts[2], counts[3], counts[4], counts[5] });
        failed = true;
        break;
    };
    if (failed) return error.TestFailed;
    check.pass("graphs at TP={d}: {d} window rows over {d} rounds (lone and 2 or 3 streams shared, 1 to 4 rows each, past 512- and 4,096-key seams and a score grid's) equal their eager twins' logits, tokens, drafts and states byte for byte; lone {d} captured, {d} updated, {d} replayed; shared {d}, {d}, {d}", .{ world, compared, rounds, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5] });
}
