//! GLM-5.3-Flash's lane backend under upstream's round loop, no server: rank 0 drives lanes.Engine.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// A communicator of `world` ranks: rank 0 writes its unique id to `id_path` as hex, the others read it there.
pub fn openComm(gpu: Gpu, lib: *const cuda.nccl.Library, rank: c_int, world: c_int, id_path: []const u8) !cuda.nccl.Comm {
    var id: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        const hex = std.fmt.bytesToHex(id.internal, .lower);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = id_path, .data = &hex });
    } else {
        var hex: [256]u8 = undefined;
        const got = try std.Io.Dir.cwd().readFile(gpu.io, id_path, &hex);
        try check.expect(got.len == 256, "unique id file {s}: {d} hex digits", .{ id_path, got.len });
        _ = try std.fmt.hexToBytes(&id.internal, &hex);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    return comm;
}

const DcKind = struct { kind: []const u8, prompt: []const i32, want: []const i32 };

/// `glm-lanes-cover`: CONTRIBUTING's single-stream contracts on one Engine: drafts, cancel, discard, eos, prompt cache.
pub fn coverCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192, kinds: []const DcKind }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = dc.capacity, .mtp = true, .settings = set });
    defer e.deinit();
    try e.prepareSnapshots(2 * e.stateBytes(4096));
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, dc.capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's steps to the end, through every cancel and discard", .{rank});
        return;
    }
    var gl: glm.lanes.Lanes = .{ .x = .{ .e = e, .link = &link } };
    defer gl.deinit();
    var cfg = try lanes.Config.init(gpu.gpa, gl.facts(), glm.lanes.max_rows, glm.lanes.max_rows - 1);
    defer cfg.deinit(gpu.gpa);
    var wall: lanes.backend.WallClock = .{ .io = gpu.io };
    var le = lanes.Engine.init(gpu.gpa, &cfg, gl.backend(), wall.clock());
    defer le.deinit();
    defer gl.x.end() catch {};
    var c: Cover = .{ .gpa = gpu.gpa, .a = a, .le = &le };

    const k = dc.kinds[0];
    const p = try a.alloc(u32, k.prompt.len);
    for (p, k.prompt) |*u, i| u.* = @intCast(i);
    // W, what every contract below holds to: the model's own plain greedy pass ("draft": false)
    const w: []const u32 = (try c.run(.{ .prompt = p, .max_new = @intCast(k.want.len), .drafts = false })).ids;
    const copy_prompt = try std.mem.concat(a, u32, &.{ p, w[0..256], p });

    // copy drafts, greedy and sampled
    for ([_]bool{ false, true }) |keyed| {
        const sampling: ?lanes.Sampling = if (keyed) .{ .seed = lanes.sampling.seedFor(copy_prompt, 0) } else null;
        const plain = try c.run(.{ .prompt = copy_prompt, .max_new = 256, .drafts = false, .sampling = sampling });
        const drafted = try c.run(.{ .prompt = copy_prompt, .max_new = 256, .sampling = sampling, .copies = true });
        try c.same(if (keyed) "copy drafts, sampled" else "copy drafts, greedy", drafted, plain.ids, .length);
        // a thin model's sampled pass does not repeat W, so nothing is there to copy: its count is only reported
        if (keyed and set.layers > 0) check.pass("copy drafts, sampled, thin model: the proposer judged {d} copied rows ({d} kept)", .{ drafted.judged, drafted.copied }) else try c.expect(drafted.judged > 0, "copy drafts{s}: the proposer judged {d} copied rows ({d} kept)", .{ if (keyed) ", sampled" else "", drafted.judged, drafted.copied });
    }
    // cancels in the prompt pass (P + W[0..256] + P is three 2,048-row chunks: asks 0, 1, 2 before them, 3 after)
    for ([_]u32{ 0, 1, 3 }) |at| {
        var cut: Cut = .{ .at = at };
        const r = c.run(.{ .prompt = copy_prompt, .max_new = 256, .cancel = &cut });
        try c.expect(if (r) |_| false else |err| err == error.Cancelled, "a cancel at ask {d} of the prompt pass ended it with error.Cancelled", .{at});
        try c.expect(cut.asked == at + 1, "the cancel was heard at ask {d} ({d} asks)", .{ at, cut.asked });
        try c.same(try std.fmt.allocPrint(a, "the request after a cancel at prompt ask {d}", .{at}), try c.run(.{ .prompt = p, .max_new = 64 }), w[0..64], .length);
    }
    // discards between rounds: after 1, after 8, and right after a round that rejected a draft (its rows uncommitted)
    for ([_]?u32{ 1, 8, null }) |rounds| {
        const left = try c.discardAfter(p, rounds);
        try c.expect(left, "a stream was discarded mid-decode ({s})", .{if (rounds) |n| try std.fmt.allocPrint(a, "after {d} rounds", .{n}) else "after a rejected draft"});
        try c.same(try std.fmt.allocPrint(a, "the request after a discard {s}", .{if (rounds) |n| try std.fmt.allocPrint(a, "after {d} rounds", .{n}) else "after a rejected draft"}), try c.run(.{ .prompt = p, .max_new = 64 }), w[0..64], .length);
    }
    // eos at W[j]'s first occurrence (the first at or past each of 0, 9, 33, 100; a thin model's W repeats more)
    var last_j: ?usize = null;
    for ([_]usize{ 0, 9, 33, 100 }) |from| {
        const j = for (from..w.len) |i| {
            if (std.mem.indexOfScalar(u32, w[0..i], w[i]) == null) break i;
        } else continue;
        if (last_j == j) continue;
        last_j = j;
        for ([_]bool{ true, false }) |drafts| {
            const r = try c.run(.{ .prompt = p, .max_new = 512, .eos = &.{w[j]}, .drafts = drafts });
            try c.same(try std.fmt.allocPrint(a, "eos at token {d}{s}", .{ j, if (drafts) " with drafts" else "" }), r, w[0 .. j + 1], .stop);
        }
    }
    // max_new at the window's edges
    for (1..6) |n| try c.same(try std.fmt.allocPrint(a, "max_new {d} with drafts", .{n}), try c.run(.{ .prompt = p, .max_new = @intCast(n) }), w[0..n], .length);
    // resumed = fresh: states kept at marks, as the lane host keeps them, then passes resumed from them
    {
        var keeper: Keeper = .{ .e = e };
        const marks = [_]u32{ 2048, 4096 };
        const fresh = try c.run(.{ .prompt = copy_prompt, .max_new = 64, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &keeper, .at = Keeper.at } } });
        try c.expect(keeper.n == marks.len, "the pass kept {d} states at its marks ({d} asked)", .{ keeper.n, marks.len });
        for (keeper.kept[0..keeper.n], marks[0..keeper.n]) |kept, at| {
            const got = try c.run(.{ .prompt = copy_prompt, .max_new = 64, .reuse = .{ .saved = kept, .at = at } });
            try c.expect(got.cached == at, "the pass resumed at {d} ({d} restored)", .{ at, got.cached });
            try c.same(try std.fmt.allocPrint(a, "resumed at {d}, against the fresh pass", .{at}), got, fresh.ids, .length);
            const plain = try c.run(.{ .prompt = copy_prompt, .max_new = 64, .drafts = false, .reuse = .{ .saved = kept, .at = at } });
            try c.same("cached drafted against cached plain", got, plain.ids, .length);
            const cached = try c.together(&.{ .{ .p = 0, .reuse = .{ .saved = kept, .at = at } }, .{ .p = 0, .reuse = .{ .saved = kept, .at = at } } }, &.{copy_prompt}, &.{.{ .seed = 1234 }}, 64);
            for (cached.ids) |ids| try c.expect(std.mem.eql(u32, ids, fresh.ids), "cached concurrent equals fresh solo at {d}", .{at});
        }
        const other = try std.mem.concat(a, u32, &.{ copy_prompt[0..4096], p[0..300] });
        const other_fresh = try c.run(.{ .prompt = other, .max_new = 64 });
        if (keeper.n == marks.len) {
            const got = try c.run(.{ .prompt = other, .max_new = 64, .reuse = .{ .saved = keeper.kept[1], .at = 4096 } });
            try c.expect(got.cached == 4096, "another tail resumed at 4096 ({d} restored)", .{got.cached});
            try c.same("another tail after the kept 4096, against its fresh pass", got, other_fresh.ids, .length);
        }
        for (keeper.kept[0..keeper.n]) |kept| glm.native.snap_drop(e, kept);
    }
    try c.same("the last request, all of W with drafts", try c.run(.{ .prompt = p, .max_new = @intCast(w.len) }), w, .length);
    if (c.failed) return error.TestFailed;
}

/// `glm-lanes-concurrent`: each concurrent stream on one lanes.Engine equals its solo run (2+ streams a round).
pub fn concurrentCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192, kinds: []const DcKind }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = dc.capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, dc.capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's steps to the end, every stream's sequence on its slot", .{rank});
        return;
    }
    var gl: glm.lanes.Lanes = .{ .x = .{ .e = e, .link = &link } };
    defer gl.deinit();
    var cfg = try lanes.Config.init(gpu.gpa, gl.facts(), glm.lanes.max_rows, glm.lanes.max_rows - 1);
    defer cfg.deinit(gpu.gpa);
    var wall: lanes.backend.WallClock = .{ .io = gpu.io };
    var le = lanes.Engine.init(gpu.gpa, &cfg, gl.backend(), wall.clock());
    defer le.deinit();
    defer gl.x.end() catch {};
    var c: Cover = .{ .gpa = gpu.gpa, .a = a, .le = &le };

    const n = 128;
    const ps = try a.alloc([]const u32, dc.kinds.len);
    for (ps, dc.kinds) |*p, k| {
        const ids = try a.alloc(u32, k.prompt.len);
        for (ids, k.prompt) |*u, i| u.* = @intCast(i);
        p.* = ids;
    }
    const seeds = [_]lanes.Sampling{ .{ .seed = 11 }, .{ .seed = 22 }, .{ .seed = 33 } };
    const solo = try a.alloc([]const u32, ps.len);
    const solo_s = try a.alloc([]const u32, ps.len);
    for (ps, solo, solo_s, 0..) |p, *g, *gs, i| {
        g.* = (try c.run(.{ .prompt = p, .max_new = n })).ids;
        gs.* = (try c.run(.{ .prompt = p, .max_new = n, .sampling = seeds[i % seeds.len] })).ids;
    }
    const P = ps.len;
    var most: usize = 0;
    const cases = [_]struct { name: []const u8, joins: []const Join }{
        .{ .name = "all at once", .joins = &.{ .{ .p = 0 }, .{ .p = 1 }, .{ .p = 2 } } },
        .{ .name = "staggered by 5 rounds", .joins = &.{ .{ .p = 0 }, .{ .p = 1, .at = 5 }, .{ .p = 2, .at = 10 } } },
        .{ .name = "in reverse", .joins = &.{ .{ .p = 2 }, .{ .p = 1 }, .{ .p = 0 } } },
        .{ .name = "the same prompt twice", .joins = &.{ .{ .p = 0 }, .{ .p = 0 } } },
        .{ .name = "sampled", .joins = &.{ .{ .p = 0, .sampled = true }, .{ .p = 1, .sampled = true }, .{ .p = 2, .sampled = true } } },
        .{ .name = "the second discarded after 6 rounds", .joins = &.{ .{ .p = 0 }, .{ .p = 1, .discard_at = 6 }, .{ .p = 2 } } },
        .{ .name = "prompts a chunk at a time while the others decode", .joins = &.{ .{ .p = 0 }, .{ .p = 1, .at = 2, .filled = true }, .{ .p = 2, .at = 4, .filled = true } } },
        .{ .name = "every prompt a chunk at a time", .joins = &.{ .{ .p = 0, .filled = true }, .{ .p = 1, .filled = true }, .{ .p = 2, .filled = true } } },
        .{ .name = "sampled, prompts a chunk at a time", .joins = &.{ .{ .p = 0, .sampled = true }, .{ .p = 1, .at = 2, .sampled = true, .filled = true }, .{ .p = 2, .at = 4, .sampled = true, .filled = true } } },
    };
    for (cases) |cs| {
        const got = try c.together(cs.joins, ps, seeds[0..], n);
        most = @max(most, got.most);
        for (cs.joins, got.ids, 0..) |j, ids, i| {
            if (j.discard_at != null or j.p >= P) continue;
            try c.same(try std.fmt.allocPrint(a, "{s}: stream {d} (prompt {d}{s}) against its solo run", .{ cs.name, i, j.p, if (j.sampled) ", sampled" else "" }), .{ .ids = ids, .reason = .length, .rounds = got.rounds[i], .judged = 0, .copied = 0, .cached = 0 }, if (j.sampled) solo_s[j.p] else solo[j.p], .length);
        }
        try c.expect(got.most >= 2, "{s}: {d} streams at most in one round, {d} rounds with two or more", .{ cs.name, got.most, got.shared });
        for (cs.joins) |j| if (j.filled) {
            try c.expect(got.between > 0, "{s}: {d} rounds of the others between a prompt's chunks", .{ cs.name, got.between });
            break;
        };
    }
    try c.expect(gl.shared > 0, "{d} forwards held several streams' windows (Engine.verifyShared)", .{gl.shared});
    if (c.failed) return error.TestFailed;
}

/// `glm-shared-check`: streams verified in one shared forward equal lone twins byte for byte.
pub fn sharedCheck(gpu: Gpu, checkpoint: []const u8, dc_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dc = try std.json.parseFromSliceLeaky(struct { capacity: usize = 8192, kinds: []const DcKind }, a, try std.Io.Dir.cwd().readFileAlloc(gpu.io, dc_path, a, .limited(1 << 26)), .{ .ignore_unknown_fields = true });
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    const set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = dc.capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, dc.capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's shared and lone windows to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const S = 5; // streams: every prompt, the first twice, the last sampled
    const V = e.share.vocab.len();
    var failed = false;
    var first: [S]u32 = undefined;
    for (0..2 * S) |slot| {
        const i = slot % S;
        const k = dc.kinds[i % dc.kinds.len];
        const ids = try a.alloc(u32, k.prompt.len);
        for (ids, k.prompt) |*u, v| u.* = @intCast(v);
        try x.bind(@intCast(slot));
        try x.begin(if (i == S - 1) .{ .seed = 77, .temperature = 0.7, .top_k = 20, .top_p = 0.95 } else null);
        const t = try x.prefill(ids, null, null, null);
        if (slot >= S and t != first[i]) {
            std.debug.print("FAIL shared: stream {d}'s twin drew {d} from the prompt, the stream {d}\n", .{ i, t, first[i] });
            failed = true;
        }
        first[i] = t;
        var held: [glm.engine.max_drafts]u32 = undefined;
        try x.draft(true, &.{t}, 1, &held);
    }
    const lone = try a.alloc(u16, 4 * V);
    const mine = try a.alloc(u16, 4 * V);
    var compared: usize = 0;
    var round: usize = 0;
    while (round < 12) : (round += 1) {
        const n = 2 + round % (S - 1); // streams this round: 2 to S
        var parts: [S]glm.engine.Part = undefined;
        var ids: [S][4]u32 = undefined;
        var outs: [S][4]u32 = undefined;
        var twin: [S][4]u32 = undefined;
        for (0..n) |i| {
            const rows = 1 + (round + i) % 4;
            for (0..rows) |r| ids[i][r] = @intCast(100 + (round * 31 + i * 7 + r * 13) % 5000);
            parts[i] = .{ .slot = @intCast(i), .ids = ids[i][0..rows], .out = outs[i][0..rows] };
        }
        try x.verifyShared(parts[0..n]);
        var row0: usize = 0;
        for (0..n) |i| { // each stream's rows of the shared forward against its twin's lone round (verify, keep, draft)
            const rows = parts[i].ids.len;
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(mine.ptr, e.bs.?.logits + row0 * V * 2, rows * V * 2), "cuMemcpyDtoH");
            try x.bind(@intCast(S + i));
            try x.verify(parts[i].ids, twin[i][0..rows], 0);
            try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(lone.ptr, e.bd.logits, rows * V * 2), "cuMemcpyDtoH");
            if (!std.mem.eql(u16, mine[0 .. rows * V], lone[0 .. rows * V]) or !std.mem.eql(u32, outs[i][0..rows], twin[i][0..rows])) {
                std.debug.print("FAIL shared: round {d}, stream {d} ({d} rows from row {d} of the shared window): logits or tokens differ from its twin's\n", .{ round, i, rows, row0 });
                failed = true;
            }
            compared += rows;
            row0 += rows;
            const keep = 1 + (round * 7 + i) % rows; // the same rows kept and drafted from on both
            var d0: [glm.engine.max_drafts]u32 = undefined;
            var d1: [glm.engine.max_drafts]u32 = undefined;
            for ([_]u32{ @intCast(S + i), @intCast(i) }, [_]*[glm.engine.max_drafts]u32{ &d1, &d0 }) |slot, d| {
                try x.bind(slot);
                try x.keep(keep);
                try x.draft(false, outs[i][0..keep], 2, d);
            }
            if (!std.mem.eql(u32, d0[0..2], d1[0..2])) {
                std.debug.print("FAIL shared: round {d}, stream {d}: drafts {any} against its twin's {any}\n", .{ round, i, d0[0..2], d1[0..2] });
                failed = true;
            }
        }
    }
    try e.stream.synchronize();
    for (0..S) |i| { // the whole states after every round's commit
        const st = &e.seqs.items[i].?.st;
        const tw = &e.seqs.items[S + i].?.st;
        const len = st.arena.buf.len;
        const h0 = try a.alloc(u8, len);
        const h1 = try a.alloc(u8, len);
        try st.arena.buf.download(0, h0);
        try tw.arena.buf.download(0, h1);
        const at = std.mem.indexOfDiff(u8, h0, h1);
        if (st.pos != tw.pos or st.mtp_len != tw.mtp_len or !std.mem.eql(u1, st.cur, tw.cur) or at != null) {
            std.debug.print("FAIL shared: stream {d}'s state differs from its twin's (pos {d}/{d}, first byte {?d} of {d})\n", .{ i, st.pos, tw.pos, at, len });
            failed = true;
        }
    }
    if (failed) return error.TestFailed;
    check.pass("shared windows at TP={d}: {d} rows of {d} streams over 12 rounds (2 to {d} streams a forward, 1 to 4 rows each, greedy and sampled) equal their lone twins' logits, tokens and drafts byte for byte, and every state its twin's", .{ world, compared, S, S });
}

/// A concurrent case's stream: prompt `p` joining at round `at`; `filled`: its prompt a chunk at a time between rounds.
const Join = struct { p: usize, at: u32 = 0, sampled: bool = false, discard_at: ?u32 = null, filled: bool = false, reuse: lanes.stream.Reuse = .{} };

/// A mark hook that keeps the state as the lane host's prompt cache does (cuda_native's snap_save).
const Keeper = struct {
    e: *@import("glm_cuda").engine.Engine,
    kept: [4]*anyopaque = undefined,
    n: usize = 0,

    fn at(ptr: *anyopaque, s: *lanes.Stream, pos: u32) void {
        _ = s;
        const k: *Keeper = @ptrCast(@alignCast(ptr));
        k.kept[k.n] = @import("glm_cuda").native.snap_save(k.e, null, pos) catch |err| {
            std.debug.print("FAIL keeping the state at {d}: {s}\n", .{ pos, @errorName(err) });
            return;
        };
        k.n += 1;
    }
};

/// A cancel hook that answers yes from its `at`-th ask on.
const Cut = struct {
    at: u32,
    asked: u32 = 0,

    fn check(ptr: *anyopaque) bool {
        const cut: *Cut = @ptrCast(@alignCast(ptr));
        cut.asked += 1;
        return cut.asked > cut.at;
    }
};

const Cover = struct {
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    le: *lanes.Engine,
    failed: bool = false,

    const Ask = struct {
        prompt: []const u32,
        max_new: u32,
        eos: []const u32 = &.{},
        drafts: bool = true,
        sampling: ?lanes.Sampling = null,
        copies: bool = false, // the server's proposer
        cancel: ?*Cut = null,
        reuse: lanes.stream.Reuse = .{},
    };
    const Got = struct { ids: []const u32, reason: lanes.stream.Reason, rounds: u64, judged: i64, copied: i64, cached: u32 };

    /// One request through the shared lanes.Engine, as the server's lane host runs it.
    fn run(c: *Cover, ask: Ask) !Got {
        var lookup = try lanes.SuffixLookup.init(c.gpa, .{ .min_match = 4 });
        defer lookup.deinit();
        var s = try lanes.Stream.init(c.gpa, .{ .id = "cover", .prompt = ask.prompt, .max_new = ask.max_new, .eos = ask.eos, .drafts = ask.drafts, .sampling = ask.sampling, .proposer = if (ask.copies) lookup.proposer() else null, .cancel_check = if (ask.cancel) |cut| .{ .ptr = cut, .check = Cut.check } else null, .reuse = ask.reuse });
        defer s.deinit(c.gpa);
        try c.le.addStream(&s);
        while (c.le.activeCount() > 0) try c.le.step();
        return .{ .ids = try c.a.dupe(u32, s.emitted()), .reason = s.reason, .rounds = s.rounds, .judged = lookup.judged_tokens, .copied = lookup.accepted_tokens, .cached = s.cached };
    }

    const Together = struct { ids: []const []const u32, rounds: []const u64, most: usize, shared: u32, between: u32 };

    /// The streams of `joins` through the rounds together, each joining at its round; how many were live at once.
    fn together(c: *Cover, joins: []const Join, ps: []const []const u32, seeds: []const lanes.Sampling, max_new: u32) !Together {
        const ss = try c.a.alloc(lanes.Stream, joins.len);
        const made = try c.a.alloc(bool, joins.len);
        @memset(made, false);
        defer for (ss, made) |*s, m| if (m) s.deinit(c.gpa);
        var most: usize = 0;
        var shared: u32 = 0;
        var between: u32 = 0;
        var round: u32 = 0;
        var filling: std.ArrayList(*lanes.Stream) = .empty; // the lane host's fillOne: the oldest's next chunk a round
        defer filling.deinit(c.gpa);
        var began = false;
        while (true) : (round += 1) {
            for (joins, ss, made) |j, *s, *m| if (!m.* and j.at <= round) {
                s.* = try lanes.Stream.init(c.gpa, .{ .id = "joined", .prompt = ps[j.p], .max_new = max_new, .reuse = j.reuse, .sampling = if (j.sampled) seeds[j.p % seeds.len] else null });
                m.* = true;
                if (j.filled) try filling.append(c.gpa, s) else try c.le.addStream(s);
            };
            if (filling.items.len > 0) {
                if (try c.le.fillStream(filling.items[0], !began)) {
                    _ = filling.orderedRemove(0);
                    began = false;
                } else began = true;
            }
            for (joins, ss, made) |j, *s, m| if (m and j.discard_at == round and !s.finished) c.le.discard(s);
            const live = c.le.activeCount();
            most = @max(most, live);
            shared += @intFromBool(live >= 2);
            if (live == 0) {
                if (std.mem.indexOfScalar(bool, made, false) == null and filling.items.len == 0) break;
                continue;
            }
            between += @intFromBool(began);
            try c.le.step();
        }
        const ids = try c.a.alloc([]const u32, joins.len);
        const rounds = try c.a.alloc(u64, joins.len);
        for (ss, ids, rounds) |*s, *i, *r| {
            i.* = try c.a.dupe(u32, s.emitted());
            r.* = s.rounds;
        }
        return .{ .ids = ids, .rounds = rounds, .most = most, .shared = shared, .between = between };
    }

    /// A drafted stream over `prompt`, discarded after `rounds` (null: first rejecting round); false if it ended first.
    fn discardAfter(c: *Cover, prompt: []const u32, rounds: ?u32) !bool {
        var s = try lanes.Stream.init(c.gpa, .{ .id = "discarded", .prompt = prompt, .max_new = 512 });
        defer s.deinit(c.gpa);
        try c.le.addStream(&s);
        var n: u32 = 0;
        while (c.le.activeCount() > 0) : (n += 1) {
            if (rounds) |r| if (n == r) break;
            const rejected = s.drafted - s.accepted;
            try c.le.step();
            if (rounds == null and s.drafted - s.accepted > rejected) break;
        }
        if (c.le.activeCount() == 0) return false;
        c.le.discard(&s);
        return true;
    }

    fn same(c: *Cover, what: []const u8, got: Got, want: []const u32, reason: lanes.stream.Reason) !void {
        const n = @min(got.ids.len, want.len);
        const first = for (got.ids[0..n], want[0..n], 0..) |x, y, i| {
            if (x != y) break i;
        } else if (got.ids.len != want.len) n else null;
        if (first == null and got.reason == reason) {
            check.pass("{s}: {d} ids equal, finished by {s} ({d} rounds)", .{ what, want.len, reason.name(), got.rounds });
            return;
        }
        c.failed = true;
        if (first) |i| {
            std.debug.print("FAIL {s}: token {d} of {d} is {d}, want {d} (emitted {d}, {s})\n", .{ what, i, want.len, if (i < got.ids.len) got.ids[i] else 0, if (i < want.len) want[i] else 0, got.ids.len, got.reason.name() });
        } else std.debug.print("FAIL {s}: finished by {s}, want {s}\n", .{ what, got.reason.name(), reason.name() });
    }

    fn expect(c: *Cover, ok: bool, comptime fmt: []const u8, args: anytype) !void {
        if (ok) return check.pass(fmt, args);
        c.failed = true;
        std.debug.print("FAIL " ++ fmt ++ "\n", args);
    }
};

fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    const now = std.Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(now.toNanoseconds() - since.toNanoseconds())) / 1e9;
}
