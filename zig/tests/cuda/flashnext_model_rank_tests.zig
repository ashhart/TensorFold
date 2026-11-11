//! Full-checkpoint experiment: two-rank prompt, snapshot replay, concurrent decode, drafted windows beside the rounds.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const ranks = flash.rank_engine;

/// `mode` "leave": rank 0 decodes until its peer is stopped from outside and must see an error; `master` hosts rank 0.
pub fn run(gpu: check.Gpu, dir: []const u8, rank: usize, port: u16, ids_path: []const u8, mode: []const u8, master: []const u8) !void {
    const leave = std.mem.eql(u8, mode, "leave");
    if (mode.len > 0 and !leave and !std.mem.eql(u8, mode, "rounds")) return error.UnknownMode;
    const context = 1024;
    // TF_FLASHNEXT_CHUNK's rows a pass: past 64 a pass takes the expert exchange, at most 64 the all-reduce
    const chunk = try prompt.chunkRows();
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(source);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpu.gpa);
    var pieces = std.mem.tokenizeAny(u8, source, " \r\n\t");
    while (pieces.next()) |word| try ids.append(gpu.gpa, try std.fmt.parseInt(u32, word, 10));
    if (ids.items.len == 0 or ids.items.len > context / 2) return error.BadPrompt;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    var identity = try ranks.identity(gpu.gpa, gpu.io, dir, context, chunk);
    identity.mtp = 1; // both ranks take verify and keep frames
    const session = try ranks.Session.init(gpu.gpa, gpu.io, master, port, rank, identity, ranks.linkWait());
    const engine = prompt.Engine.initRank(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir, 2, rank) catch |err| {
        session.poison();
        session.deinit();
        return err;
    };
    session.engine = engine;
    defer {
        session.deinit();
        engine.deinit();
    }
    engine.collective = &session.collective;
    if (engine.chunk != chunk) return error.RankIdentity;
    const room = try session.room((try gpu.ctx.memInfo()).free);
    try check.expect(room > prompt.seqBytes(context, engine.heads) * 6, "both ranks have room for experimental caches", .{});
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, 2 * prompt.vocabulary * 2);
    defer logits.free();
    const graph = identity.decode_graph == 1;
    if (rank == 1) {
        try session.follow(flash.Tri, tri, logits.ptr);
        if (!leave) try expectCounts(engine, graph);
        check.pass("rank 1 replay completed (dense split {d})", .{identity.dense_split});
        return;
    }
    var first = try session.create(context);
    defer session.drop(&first);
    if (leave) return untilPeerLeaves(session, tri, &first, ids.items, logits.ptr, identity.decode_graph == 1);
    try session.forward(flash.Tri, tri, &first, ids.items, .{});
    check.pass("two-rank prompt completed ({d} tokens)", .{ids.items.len});
    std.debug.print("LOGITS {x:0>16}\n", .{std.hash.Wyhash.hash(0, first.logits)});
    for (0..prompt.vocabulary) |i| {
        const value: f32 = @bitCast(@as(u32, std.mem.readInt(u16, first.logits[i * 2 ..][0..2], .little)) << 16);
        try check.expect(std.math.isFinite(value), "finite initial logits at {d}", .{i});
    }
    const kept = try session.save(&first, null);
    defer session.dropSaved(kept);
    const token = prompt.argmax(first.logits);
    try session.forward(flash.Tri, tri, &first, &.{token}, .{});
    const expected = try gpu.gpa.dupe(u8, first.logits);
    defer gpu.gpa.free(expected);
    try session.restore(&first, null, kept);
    try session.forward(flash.Tri, tri, &first, &.{token}, .{});
    try check.sameBytes("two-rank snapshot restore and replay", first.logits, expected);
    try session.restore(&first, null, kept);
    var second = try session.create(context);
    defer session.drop(&second);
    try session.forward(flash.Tri, tri, &second, ids.items, .{});
    try session.round(flash.Tri, tri, &.{ &first, &second }, &.{ token, token }, logits.ptr);
    try check.sameBytes("two-rank concurrent first stream", first.logits, expected);
    try check.sameBytes("two-rank concurrent second stream", second.logits, expected);
    check.pass("two-rank full-model prompt, snapshot replay and concurrent byte equality", .{});
    // eager twins: a one-row forward never takes the round graph, so each round's rows must equal theirs
    var twin = try session.create(context);
    defer session.drop(&twin);
    var twin2 = try session.create(context);
    defer session.drop(&twin2);
    for ([_]*prompt.Seq{ &twin, &twin2 }) |q| try session.forward(flash.Tri, tri, q, ids.items, .{});
    try session.forward(flash.Tri, tri, &twin, &.{token}, .{});
    try session.forward(flash.Tri, tri, &twin2, &.{token}, .{});
    var held: ?cuda.abi.GraphExec = null;
    for (0..24) |i| {
        try roundPair(session, tri, &.{&first}, &.{&twin}, logits.ptr, graph);
        // the first round captures; the rest replay it with their own ids and positions staged
        const exec = engine.round_graph.exec orelse continue;
        if (held == null) held = exec.handle;
        try check.expect(exec.handle == held.?, "round {d} replayed the first round's graph", .{i});
    }
    // two lanes at different positions, then the other owner order: another key, captured anew
    for (0..8) |_| try roundPair(session, tri, &.{ &first, &second }, &.{ &twin, &twin2 }, logits.ptr, graph);
    for (0..8) |_| try roundPair(session, tri, &.{ &second, &first }, &.{ &twin2, &twin }, logits.ptr, graph);
    // the 512-key attention seam: the next chunk count is another key
    var filler: [32]u32 = undefined;
    for (&filler, 0..) |*f, i| f.* = ids.items[i % ids.items.len];
    while (first.fed < 506) {
        const n = @min(filler.len, 506 - first.fed);
        try session.forward(flash.Tri, tri, &first, filler[0..n], .{});
        try session.forward(flash.Tri, tri, &twin, filler[0..n], .{});
    }
    try check.sameBytes("a filled prompt's rows", first.logits, twin.logits);
    for (0..12) |_| try roundPair(session, tri, &.{&first}, &.{&twin}, logits.ptr, graph);
    // a drafted stream beside: its windows verified and kept between the plain rounds leave their graph held
    var third = try session.create(context);
    defer session.drop(&third);
    var twin3 = try session.create(context);
    defer session.drop(&twin3);
    for ([_]*prompt.Seq{ &third, &twin3 }) |q| try session.forward(flash.Tri, tri, q, ids.items, .{});
    var draft = try flash.mtp.Draft.init(gpu.gpa, engine.driver, stream, context, 1, chunk, engine.heads);
    defer draft.deinit();
    try roundPair(session, tri, &.{&first}, &.{&twin}, logits.ptr, graph);
    const held_captures = engine.round_graph.captures;
    for (0..4) |i| {
        try window(session, tri, &third, &twin3, &draft, ids.items, if (i % 2 == 0) 4 else 2);
        try roundPair(session, tri, &.{&first}, &.{&twin}, logits.ptr, graph);
    }
    try check.expect(engine.round_graph.captures == held_captures, "plain rounds between verifies captured {d} more", .{engine.round_graph.captures - held_captures});
    // the drafted lane's own rounds: a whole keep swaps its states, so the round after it is recorded again
    for (0..3) |_| {
        for (0..2) |_| try roundPair(session, tri, &.{&third}, &.{&twin3}, logits.ptr, graph);
        try window(session, tri, &third, &twin3, &draft, ids.items, 4);
    }
    try expectCounts(engine, graph);
    check.pass("63 rounds equal eager rows byte for byte (round graph {}, dense split {d}): replays, lanes, owner order, 512 seam, verifies", .{ graph, identity.dense_split });
}

/// A 4-row drafted window on `q` verified and its first `keep` rows kept; its eager twin fed the same rows.
fn window(session: *ranks.Session, tri: flash.Tri, q: *prompt.Seq, twin: *prompt.Seq, d: *flash.mtp.Draft, ids: []const u32, keep: usize) !void {
    const rows = [4]u32{ prompt.argmax(twin.logits), ids[0], ids[1 % ids.len], ids[2 % ids.len] };
    try session.verify(flash.Tri, tri, q, d, &rows);
    try session.keep(flash.Tri, tri, q, d, keep);
    try session.forward(flash.Tri, tri, twin, rows[0..keep], .{});
    try check.expect(q.fed == twin.fed, "a kept window's position is its twin's", .{});
}

/// Both ranks capture the same keys: concurrent, one lane, two lanes, their swap, the seam, then the drafted lane's.
fn expectCounts(engine: *prompt.Engine, graph: bool) !void {
    const want: [2]u64 = if (graph) .{ 10, 54 } else .{ 0, 0 };
    try check.expect(std.meta.eql(want, [2]u64{ engine.round_graph.captures, engine.round_graph.replays }), "round graphs captured and replayed {any}, want {any}", .{ [2]u64{ engine.round_graph.captures, engine.round_graph.replays }, want });
}

/// A round of `seqs` on their twins' argmax tokens (graphed when on), then each twin's eager row on it: equal bytes.
fn roundPair(session: *ranks.Session, tri: flash.Tri, seqs: []const *prompt.Seq, twins: []const *prompt.Seq, out: u64, graph: bool) !void {
    var next: [2]u32 = undefined;
    for (twins, 0..) |q, i| next[i] = prompt.argmax(q.logits);
    std.debug.print("TOKEN {d}\n", .{next[0]});
    try session.round(flash.Tri, tri, seqs, next[0..seqs.len], out);
    try check.expect(graph == (session.engine.round_graph.exec != null), "the round took the graph exactly when it is on", .{});
    for (seqs, twins, 0..) |q, twin, i| {
        try session.forward(flash.Tri, tri, twin, next[i..][0..1], .{});
        try check.sameBytes("a two-rank round's row equals the eager one-row pass", q.logits, twin.logits);
    }
}

/// Rounds until the peer is stopped: the leader must surface an error, not wait on its graph or communicator.
fn untilPeerLeaves(session: *ranks.Session, tri: flash.Tri, q: *prompt.Seq, ids: []const u32, out: u64, graph: bool) !void {
    try session.forward(flash.Tri, tri, q, ids, .{});
    var rounds: usize = 0;
    while (q.fed < q.cap) : (rounds += 1) {
        if (rounds == 4) {
            try check.expect(graph == (session.engine.round_graph.exec != null), "the round took the graph exactly when it is on", .{});
            std.debug.print("LEAVE-READY\n", .{});
        }
        session.round(flash.Tri, tri, &.{q}, &.{prompt.argmax(q.logits)}, out) catch |err| {
            const lost = err == error.RankDisconnected or err == error.RankTimeout;
            try check.expect(rounds >= 4 and lost, "the leader stopped on a lost peer after LEAVE-READY: {t} after {d} rounds", .{ err, rounds });
            check.pass("the leader stopped after {d} rounds when its peer left: {t}", .{ rounds, err });
            return;
        };
    }
    return check.expect(false, "the peer did not leave within {d} rounds", .{rounds});
}
