//! tf-inkling2-engine: the serving engine's checks on one or two ranks (both run the same loop, no plan words).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const check = ink.engine_check;

const usage = "usage: tf-inkling2-engine graphs|greedy <model dir> <ids.json> <out dir> [context] [new] [options]\n" ++
    "  graphs: decode rows eager in some slots and graphed in others on the same rows; logits must match bit for bit\n" ++
    "  greedy: each text's prompt pass then greedy decode rows, eager; writes greedy.json (name -> tokens),\n" ++
    "          logits.r<rank>.json (name -> sha256 of the rank's logits of each drawn row), run.json and, from rank 0,\n" ++
    "          logits.<name>.f32 (each drawn row over the whole vocabulary) with logits.json (each row's sha256)\n" ++
    "  --streams K: greedy decodes K texts together, each on its own slot (K slots reserved)\n" ++
    "  --image FILE: the picture for every image placeholder run, widened to its visual tokens\n" ++
    "  --virtual: greedy at two ranks on one GPU, each with its own weights, plans and partial sums\n" ++
    "  --serial: opt-in serial reference on one GPU: keeps both ranks' partials, combines rank 0's then rank 1's\n" ++
    "          at the same rounding points as two ranks, and rebuilds each row from both padded vocabulary spans;\n" ++
    "          ordinary one-rank routing and the two-rank path are unchanged\n" ++
    "  --graphed: greedy's decode rows replayed from CUDA graphs\n" ++
    "  two ranks: TF_INKLING2_RANK, TF_INKLING2_WORLD and NCCL_COMM_ID on both\n";

const Text = struct { name: []const u8, ids: []u32 };

/// What every rank runs: the texts, the run's shape and an optional picture (prepared on every rank for its size).
const Run = struct {
    o: check.Options,
    texts: []const Text,
    picture: ?ink.image.Prepared = null,
    image_sha: [32]u8 = @splat(0),
    ids_sha: [32]u8,
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const o = check.parse(args[1..]) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const text = try std.Io.Dir.cwd().readFileAlloc(io, o.ids, a, .limited(1 << 26));
    var run: Run = .{ .o = o, .texts = &.{}, .ids_sha = sha(text) };
    if (o.image) |p| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(1 << 28));
        run.picture = try ink.image.prepare(io, a, bytes, ink.pictures.picture_tokens);
        run.image_sha = sha(bytes);
    }
    var doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    var list: std.ArrayList(Text) = .empty;
    var it = doc.value.object.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.array.items;
        var ids = try a.alloc(u32, v.len);
        for (v, ids) |x, *id| id.* = @intCast(x.integer);
        if (run.picture) |p| ids = try check.widen(a, ids, p.tokens());
        try list.append(a, .{ .name = e.key_ptr.*, .ids = ids });
    }
    run.texts = list.items;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.Inkling2RequiresSm121;
    const place = try ink.admit.place();
    const lim: ?ink.pictures.Limits = if (run.picture != null) .{ .image_tokens = ink.pictures.picture_tokens, .max_images = 8 } else null;
    const opts: ink.engine.Options = .{ .capacity = ink.native_exl3.capacity(o.context), .vision = lim };
    try std.Io.Dir.cwd().createDirPath(io, o.out);
    var out = try std.Io.Dir.cwd().openDir(io, o.out, .{});
    defer out.close(io);
    if (o.serial) {
        if (place.world != 1) return error.SerialRanksRunInOneProcess; // unset TF_INKLING2_RANK and TF_INKLING2_WORLD
        const e = try ink.engine.Engine.init(gpa, io, &driver, o.dir, 0, 2, opts);
        defer e.deinit();
        try ink.serial.Serial.attach(e, o.dir);
        try greedyRank(e, 0, &run, out, io, gpa);
        return 0;
    }
    if (o.virtual) {
        if (place.world != 1) return error.VirtualRanksRunInOneProcess; // unset TF_INKLING2_RANK and TF_INKLING2_WORLD
        var pair = try ink.pair.Pair.init(&driver);
        defer pair.deinit();
        var es: [2]*ink.engine.Engine = undefined;
        es[0] = try ink.engine.Engine.init(gpa, io, &driver, o.dir, 0, 2, opts);
        defer es[0].deinit();
        es[1] = try ink.engine.Engine.init(gpa, io, &driver, o.dir, 1, 2, opts);
        defer es[1].deinit();
        var ranks: [2]Rank = undefined;
        for (&ranks, es, 0..) |*r, e, q| {
            e.exchange = .{ .pair = .{ .p = &pair, .rank = q } };
            r.* = .{ .e = e, .rank = q, .run = &run, .out = out, .io = io, .gpa = gpa, .ctx = &ctx, .meet = &pair.meet };
        }
        const t = try std.Thread.spawn(.{}, Rank.go, .{&ranks[1]});
        ranks[0].go();
        t.join();
        for (ranks) |r| if (r.err) |err| return err;
        return 0;
    }
    var group: ?cuda.tp_group.Group = null;
    defer if (group) |*g| g.close(); // after the engine (its graphs hold the comm's kernels), as the server closes
    const e = try ink.engine.Engine.init(gpa, io, &driver, o.dir, place.rank, place.world, opts);
    defer e.deinit();
    if (place.world > 1) {
        group = try cuda.tp_group.Group.open(&driver, place.rank, place.world);
        try group.?.openRoce(gpa, io, &driver, e.stream, e.roceBytes());
        e.exchange = .{ .nccl = .{ .lib = &group.?.lib, .comm = group.?.comm, .group = &group.? } };
    }
    if (!o.graphs) {
        try greedyRank(e, place.rank, &run, out, io, gpa);
        return 0;
    }
    try e.reserve(16);
    return graphRuns(e, a, place.rank == 0, run.texts);
}

fn sha(bytes: []const u8) [32]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return h;
}

fn hex(h: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(h, .lower);
}

/// One virtual rank's thread: the context made current, then the greedy run a real rank makes.
const Rank = struct {
    e: *ink.engine.Engine,
    rank: usize,
    run: *const Run,
    out: std.Io.Dir,
    io: std.Io,
    gpa: std.mem.Allocator,
    ctx: *const cuda.Context,
    meet: *ink.pair.Meet,
    err: ?anyerror = null,

    fn go(r: *Rank) void {
        r.ctx.makeCurrent() catch |err| return r.failed(err);
        greedyRank(r.e, r.rank, r.run, r.out, r.io, r.gpa) catch |err| r.failed(err);
    }

    fn failed(r: *Rank, err: anyerror) void {
        std.log.err("virtual rank {d}: {s}", .{ r.rank, @errorName(err) });
        r.err = err;
        r.meet.fail();
    }
};

/// This rank's greedy run, `streams` texts at a time: tokens, whole rows and run.json on rank 0, hashes on every rank.
fn greedyRank(e: *ink.engine.Engine, rank: usize, run: *const Run, out: std.Io.Dir, io: std.Io, gpa: std.mem.Allocator) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const k = run.o.streams;
    e.graphs.settings.on = run.o.graphed;
    try e.reserve(k);
    const w = ink.serial.widest(e.spans);
    var rec: Rec = .{ .scratch = try cuda.DeviceBuffer.alloc(e.d, e.w.share.world * w * 4), .host = try a.alloc(u8, e.w.share.world * w * 4), .out = try a.alloc(u8, e.w.padded_vocab * 4) };
    defer rec.scratch.free();
    var toks: std.ArrayList(u8) = .empty;
    var lists: [3]std.ArrayList(u8) = .{ .empty, .empty, .empty }; // this rank's span, rank 1's (serial), whole rows
    try toks.appendSlice(a, "{");
    for (&lists) |*l| try l.appendSlice(a, "{");
    var at: usize = 0;
    while (at < run.texts.len) : (at += k) {
        const group = run.texts[at..@min(run.texts.len, at + k)];
        const got = try greedy(e, a, rank, run, group, &rec);
        for (group, got, 0..) |t, g, j| {
            const sep = if (at + j > 0) ", " else "";
            try toks.print(a, "{s}\"{s}\": [", .{ sep, t.name });
            for (g.toks, 0..) |x, n| try toks.print(a, "{s}{d}", .{ if (n > 0) ", " else "", x });
            try toks.appendSlice(a, "]");
            for (&lists, [_][]const [32]u8{ g.rows, g.peer, g.whole }) |*l, rows| try hashes(l, a, sep, t.name, rows);
            if (rank == 0) try out.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "logits.{s}.f32", .{t.name}), .data = g.full });
            if (rank == 0) std.debug.print("{s}: {d} prompt tokens, {d} greedy tokens\n", .{ t.name, t.ids.len, g.toks.len });
        }
    }
    try toks.appendSlice(a, "}\n");
    for (&lists) |*l| try l.appendSlice(a, "}\n");
    try out.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "logits.r{d}.json", .{rank}), .data = lists[0].items });
    if (rank != 0) return;
    if (e.serial != null) try out.writeFile(io, .{ .sub_path = "logits.r1.json", .data = lists[1].items });
    try out.writeFile(io, .{ .sub_path = "logits.json", .data = lists[2].items });
    try out.writeFile(io, .{ .sub_path = "greedy.json", .data = toks.items });
    // the digest the server's ranks compare at ready: shard bytes, config, stream shape and graph settings
    const shards = try (try ink.native_exl3.startIdentity(gpa, io, run.o.dir)).finish();
    const digest = ink.native_exl3.readyDigest(shards, &e.c, e.capacity, e.vision, (cuda.graph_cache.Settings{}).fromEnv().digest());
    const yes = [2][]const u8{ "false", "true" };
    const info = try std.fmt.allocPrint(a, "{{\"ranks\": {d}, \"virtual\": {s}, \"serial\": {s}, \"graphed\": {s}, \"context\": {d}, \"new\": {d}, " ++
        "\"streams\": {d}, \"padded_vocab\": {d}, \"ready_digest\": \"{s}\", \"ids_sha256\": \"{s}\", \"image_sha256\": \"{s}\"}}\n", .{ e.w.share.world, yes[@intFromBool(run.o.virtual)], yes[@intFromBool(run.o.serial)], yes[@intFromBool(run.o.graphed)], run.o.context, run.o.new, k, e.w.padded_vocab, &hex(digest), &hex(run.ids_sha), &hex(run.image_sha) });
    try out.writeFile(io, .{ .sub_path = "run.json", .data = info });
}

/// One text's row hashes as a JSON entry: `"name": ["<sha256>", ...]`.
fn hashes(l: *std.ArrayList(u8), a: std.mem.Allocator, sep: []const u8, name: []const u8, rows: []const [32]u8) !void {
    try l.print(a, "{s}\"{s}\": [", .{ sep, name });
    for (rows, 0..) |x, n| try l.print(a, "{s}\"{s}\"", .{ if (n > 0) ", " else "", &hex(x) });
    try l.appendSlice(a, "]");
}

/// Whole rows: the gather's device scratch and host copy ([world][widest span] fp32) and one row of the vocabulary.
const Rec = struct { scratch: cuda.DeviceBuffer, host: []u8, out: []u8 };

/// A text's tokens; per drawn row its rank's span hash, rank 1's (serial), the whole row and its hash (rank 0).
const Got = struct { toks: []u32, rows: [][32]u8, peer: [][32]u8, whole: [][32]u8, full: []u8 };

/// Up to `streams` texts: each one's prompt pass on its own slot, then `new` shared greedy decode rows of all of them.
fn greedy(e: *ink.engine.Engine, a: std.mem.Allocator, rank: usize, run: *const Run, texts: []const Text, rec: *const Rec) ![]Got {
    const got = try a.alloc(Got, texts.len);
    var pend: [ink.engine.max_shared]u32 = undefined;
    const rows = run.o.new + 1;
    for (texts, got, 0..) |t, *g, i| {
        const r1: usize = if (e.serial != null) rows else 0;
        const whole: usize = if (rank == 0) rows else 0;
        g.* = .{ .toks = try a.alloc(u32, run.o.new), .rows = try a.alloc([32]u8, rows), .peer = try a.alloc([32]u8, r1), .whole = try a.alloc([32]u8, whole), .full = try a.alloc(u8, whole * rec.out.len) };
        try e.bind(@intCast(i));
        e.begin(null);
        if (run.picture) |p| {
            const sp = try ink.vision.spans(a, t.ids);
            const pics = try a.alloc(ink.image.Prepared, sp.len);
            @memset(pics, p);
            if (sp.len > 0) try e.images(if (rank == 0) pics else null, sp);
        }
        var at: usize = 0;
        pend[i] = while (true) : (at += ink.engine.prompt_rows) {
            if (try e.prefillChunk(t.ids, at)) |x| break x;
        };
        const most = @max(1, e.bp.parts.slots / (e.shape_p.heads * ink.forward.stride(&e.c, t.ids.len - 1)));
        try record(e, rec, g, 0, true, check.lastRow(t.ids.len, most), rank);
    }
    for (0..run.o.new) |step| {
        var parts: [ink.engine.max_shared]ink.engine.Part = undefined;
        var outs: [ink.engine.max_shared][1]u32 = undefined;
        for (texts, 0..) |_, i| {
            got[i].toks[step] = pend[i];
            parts[i] = .{ .slot = @intCast(i), .ids = pend[i..][0..1], .out = &outs[i] };
        }
        try e.verifyShared(parts[0..texts.len]);
        for (texts, 0..) |_, i| {
            try record(e, rec, &got[i], step + 1, false, i, rank);
            pend[i] = outs[i][0];
        }
    }
    return got;
}

/// Row `row` of the last call as drawn row `k`: this rank's span hash, then (rank 0) the whole row and its hashes.
fn record(e: *ink.engine.Engine, rec: *const Rec, g: *Got, k: usize, prompt: bool, row: usize, rank: usize) !void {
    g.rows[k] = try rowHash(e, if (prompt) &e.bp else &e.bd, row);
    try ink.serial.rebuild(e, prompt, row, rec.scratch, rec.host, rec.out);
    if (rank != 0) return;
    @memcpy(g.full[k * rec.out.len ..][0..rec.out.len], rec.out);
    g.whole[k] = sha(rec.out);
    if (e.serial != null) g.peer[k] = sha(rec.out[e.spans[1].lo * 4 ..][0 .. e.spans[1].len() * 4]);
}

/// sha256 of this rank's fp32 logits of row `row` (its padded vocabulary span).
fn rowHash(e: *ink.engine.Engine, b: *ink.forward.Buffers, row: usize) ![32]u8 {
    const nv = e.w.share.vocab.len();
    const bytes = try e.gpa.alloc(u8, nv * 4);
    defer e.gpa.free(bytes);
    try e.stream.synchronize();
    try b.logits.download(row * nv * 4, bytes);
    return sha(bytes);
}

fn graphRuns(e: *ink.engine.Engine, a: std.mem.Allocator, r0: bool, texts: []const Text) !u8 {
    // cases: k streams, `steps` decode rows each; a text of 16,380+ tokens also crosses a global chunk
    const Case = struct { k: usize, steps: usize, at: usize };
    var cases: std.ArrayList(Case) = .empty;
    for ([_]usize{ 1, 2, 3, 4, 8 }) |k| try cases.append(a, .{ .k = k, .steps = 6, .at = 0 });
    for (texts) |t| if (t.ids.len >= 16390) {
        try cases.append(a, .{ .k = 1, .steps = 8, .at = 16380 });
        try cases.append(a, .{ .k = 2, .steps = 8, .at = 16380 });
        break;
    };
    var pass: usize = 0;
    var total: usize = 0;
    for (cases.items) |cs| {
        const diff = try graphCase(e, a, texts, cs.k, cs.steps, cs.at);
        total += cs.steps;
        pass += cs.steps - diff;
        if (r0) std.debug.print("case {d} streams from {s}: {d}/{d} steps equal\n", .{ cs.k, if (cs.at > 0) "16380" else "prompt end", cs.steps - diff, cs.steps });
    }
    const st = e.graphs.stats;
    if (r0) std.debug.print("graphs: {d}/{d} steps bit-identical; captured {d}, replayed {d}, eager {d}\n", .{ pass, total, st.captured, st.replayed, st.eager });
    return if (pass == total) 0 else 1;
}

/// One case: the same streams in slots 0..k (eager) and k..2k (graphed); returns the steps whose logits differ.
fn graphCase(e: *ink.engine.Engine, a: std.mem.Allocator, texts: []const Text, k: usize, steps: usize, at: usize) !usize {
    var pend: [8]u32 = undefined;
    var src: [8][]const u32 = undefined;
    var long: ?[]const u32 = null;
    for (texts) |t| if (t.ids.len >= 16390) {
        long = t.ids;
    };
    for (0..k) |i| {
        src[i] = if (at > 0) long.?[0..at] else texts[i % texts.len].ids;
        for ([_]usize{ i, k + i }) |slot| {
            try e.bind(@intCast(slot));
            e.begin(null);
            var p: usize = 0;
            pend[i] = while (true) : (p += ink.engine.prompt_rows) {
                if (try e.prefillChunk(src[i], p)) |t| break t;
            };
        }
    }
    const nv = e.w.share.vocab.len();
    var diff: usize = 0;
    for (0..steps) |_| {
        var outs: [2][8][1]u32 = undefined;
        var logits: [2][]u8 = undefined;
        for (0..2) |arm| {
            var parts: [8]ink.engine.Part = undefined;
            for (0..k) |i| parts[i] = .{ .slot = @intCast(arm * k + i), .ids = pend[i..][0..1], .out = &outs[arm][i] };
            e.graphs.settings.on = arm == 1;
            try e.verifyShared(parts[0..k]);
            try e.stream.synchronize();
            logits[arm] = try a.alloc(u8, k * nv * 4);
            try e.bd.logits.download(0, logits[arm]);
        }
        if (!std.mem.eql(u8, logits[0], logits[1])) diff += 1;
        for (0..k) |i| pend[i] = outs[0][i][0];
    }
    return diff;
}
