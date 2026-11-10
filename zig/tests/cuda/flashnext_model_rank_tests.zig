//! Full-checkpoint experiment: two-rank prompt, snapshot replay, concurrent decode and every round equal eager rows.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const ranks = flash.rank_engine;

/// `mode` "leave": rank 0 decodes until its peer is stopped from outside, and must see it as an error.
pub fn run(gpu: check.Gpu, dir: []const u8, rank: usize, port: u16, ids_path: []const u8, mode: []const u8) !void {
    const leave = std.mem.eql(u8, mode, "leave");
    if (mode.len > 0 and !leave) return error.UnknownMode;
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
    const identity = try ranks.identity(gpu.gpa, gpu.io, dir, context, chunk);
    const session = try ranks.Session.init(gpu.gpa, gpu.io, "127.0.0.1", port, rank, identity, ranks.linkWait());
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
    if (rank == 1) {
        try session.follow(flash.Tri, tri, logits.ptr);
        check.pass("rank 1 replay completed", .{});
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
    const graph = identity.decode_graph == 1;
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
    check.pass("52 rounds equal eager rows byte for byte (round graph {}): replays, two lanes, owner order, 512 seam", .{graph});
}

/// A round of `seqs` on their argmax tokens (graphed when on), then each twin's eager row on that token: equal bytes.
fn roundPair(session: *ranks.Session, tri: flash.Tri, seqs: []const *prompt.Seq, twins: []const *prompt.Seq, out: u64, graph: bool) !void {
    var next: [2]u32 = undefined;
    for (seqs, 0..) |q, i| next[i] = prompt.argmax(q.logits);
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
