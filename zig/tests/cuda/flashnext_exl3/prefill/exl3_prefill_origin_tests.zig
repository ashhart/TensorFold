const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const graph = flash.exl3.graph;
const paging = flash.exl3.kvpage;
const prefill = flash.exl3.prefill;
const step_rows = 256;
const steps = 24;

fn require(name: [:0]const u8, expected: []const u8) !void {
    const setting = std.c.getenv(name);
    try check.expect(setting != null and std.mem.eql(u8, std.mem.span(setting.?), expected), "prefill-origin fixture requires {s}={s}", .{ name, expected });
}

fn stateBytes(sequence: *const flash.prompt.Seq) usize {
    var bytes = sequence.logits.len + sequence.tail.len;
    for ([_][]cuda.DeviceBuffer{ sequence.state, sequence.conv, sequence.ikc, sequence.pooled }) |buffers| for (buffers) |buffer| {
        bytes += buffer.len;
    };
    const pager = sequence.paged.?;
    const row_bytes = pager.hostMut(0, false).len / sequence.cap;
    return bytes + 2 * sequence.ikc.len * sequence.fed * row_bytes;
}

fn readState(sequence: *flash.prompt.Seq, stream: cuda.Stream, target: []u8) !void {
    const pager = sequence.paged.?;
    try pager.flush(stream, sequence.fed);
    try stream.synchronize();
    var offset: usize = 0;
    @memcpy(target[0..sequence.logits.len], sequence.logits);
    offset += sequence.logits.len;
    for ([_][]cuda.DeviceBuffer{ sequence.state, sequence.conv, sequence.ikc, sequence.pooled }) |buffers| for (buffers) |buffer| {
        try buffer.download(0, target[offset..][0..buffer.len]);
        offset += buffer.len;
    };
    try sequence.tail.download(0, target[offset..][0..sequence.tail.len]);
    offset += sequence.tail.len;
    for (0..sequence.ikc.len) |layer| for ([_]bool{ false, true }) |values| {
        const canonical = pager.hostMut(layer, values);
        const valid_bytes = sequence.fed * (canonical.len / sequence.cap);
        @memcpy(target[offset..][0..valid_bytes], canonical[0..valid_bytes]);
        offset += valid_bytes;
    };
    try check.expect(offset == target.len, "prefill-origin state read covers every allocated comparison byte", .{});
}

fn eagerForward(engine: anytype, sequence: *flash.prompt.Seq, ids: []const u32, comptime operations: anytype) !void {
    std.mem.swap(flash.prompt.Seq, &engine.seq, sequence);
    defer std.mem.swap(flash.prompt.Seq, &engine.seq, sequence);
    graph.force = .off;
    try poisonScratch(engine);
    try operations.forward(engine, ids, .{ .logits = .last });
}

fn poisonScratch(engine: anytype) !void {
    try engine.e.scores.?.fill32(0x7fc00000, engine.stream.handle);
    const scratch = engine.e.batch_s.?;
    for (scratch.bufs.items) |buffer| {
        if (buffer.ptr == scratch.po or buffer.ptr == scratch.pm or buffer.ptr == scratch.pl) try buffer.fill32(0x7fc00000, engine.stream.handle);
    }
}

fn finiteLogits(bytes: []const u8) bool {
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += 2) {
        if (std.mem.readInt(u16, bytes[offset..][0..2], .little) & 0x7f80 == 0x7f80) return false;
    }
    return true;
}

pub fn run(gpu: check.Gpu, rank: usize, world: usize, port: u16, directory: []const u8, storage: []const u8, comptime operations: anytype) !void {
    try require("TF_FLASHNEXT_EXL3_GRAPH_PREFILL", "1");
    try require("TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER", "1");
    try require("TF_FLASHNEXT_CHUNK", "256");
    try check.expect(world == 2 and !paging.force_off, "prefill-origin fixture requires TP2 and BF16 RAM paging", .{});
    const shared = std.mem.eql(u8, storage, "shared");
    try check.expect(shared or std.mem.eql(u8, storage, "private"), "prefill-origin storage must be private or shared", .{});
    const config: paging.Config = .{ .window = 2052, .rows = step_rows, .slots = 512, .shared_lanes = if (shared) 2 else 0 };
    const previous_paging = paging.force;
    paging.force = config;
    defer paging.force = previous_paging;
    const previous_options = prefill.state.options;
    prefill.state.options = .{ .rows = step_rows, .serve_rows = step_rows, .min_rows = step_rows, .ring = false, .prefetch = false, .tiles = false };
    defer prefill.state.options = previous_options;
    const previous_mode = graph.force;
    defer graph.force = previous_mode;
    const engine = try operations.init(gpu, rank, world, port, directory, config);
    defer operations.deinit(engine);
    var eager = try engine.e.newSeq(engine.seq.cap);
    defer eager.deinit(gpu.gpa);
    try operations.reset(engine);
    std.mem.swap(flash.prompt.Seq, &engine.seq, &eager);
    operations.reset(engine) catch |failure| {
        std.mem.swap(flash.prompt.Seq, &engine.seq, &eager);
        return failure;
    };
    std.mem.swap(flash.prompt.Seq, &engine.seq, &eager);
    const pager = engine.seq.paged orelse return error.PrefillOriginPagingRequired;
    try check.expect(engine.seq.kv8 == null and engine.e.chunk == step_rows and pager.cfg.rows == step_rows and pager.cfg.shared_lanes == config.shared_lanes, "prefill-origin fixture uses the exact native BF16 private/shared geometry", .{});
    const generation: ?paging.shared_contract.Sequence = if (shared) pager.shared_sequence else null;
    graph.clearPrefill();
    var tokens: [steps * step_rows]u32 = undefined;
    var generator = std.Random.DefaultPrng.init(0x70726566696c6c);
    for (&tokens) |*token| token.* = 200 + generator.random().uintLessThan(u32, 60000);
    var observed_chunk_grid = false;
    var observed_score_grid = false;
    var observed_origin = false;
    var observed_sparse_decline = false;
    var observed_split_decline = false;
    const initial = graph.prefill_stats;
    for (0..steps) |step| {
        const first = step * step_rows;
        const ids = tokens[first..][0..step_rows];
        try eagerForward(engine, &eager, ids, operations);
        const reference = try gpu.gpa.alloc(u8, stateBytes(&eager));
        defer gpu.gpa.free(reference);
        try readState(&eager, engine.stream, reference);
        const before = graph.prefill_stats;
        const before_deep = prefill.state.stats;
        const old_origin = pager.win0[0];
        const old_page_rows = try paging.pagingRows(pager.cfg, old_origin, step_rows);
        graph.force = .graph;
        try poisonScratch(engine);
        try operations.forward(engine, ids, .{ .logits = .last });
        const after = graph.prefill_stats;
        const rebound = after.rebinds - before.rebinds;
        const new_page_rows = try paging.pagingRows(pager.cfg, pager.win0[0], step_rows);
        const actual = try gpu.gpa.alloc(u8, stateBytes(&engine.seq));
        defer gpu.gpa.free(actual);
        try readState(&engine.seq, engine.stream, actual);
        const exact = finiteLogits(engine.seq.logits) and std.mem.eql(u8, reference, actual) and eager.fed == engine.seq.fed and
            std.meta.eql(eager.history, engine.seq.history) and std.meta.eql(eager.paged.?.win0, pager.win0);
        const peer_exact = try operations.swap(engine, @intFromBool(exact));
        if (!exact) try check.sameBytes("prefill-origin logits, recurrent/conv/PLE, indexer and canonical BF16 K/V", actual, reference);
        try check.expect(exact and peer_exact == 1, "rank {d}: {s} step {d} exact eager/rebound logits and state", .{ rank, storage, step });
        try check.expect(prefill.state.stats.chunks == before_deep.chunks + 1 and prefill.state.stats.rows == before_deep.rows + step_rows, "rank {d}: step {d} uses one actual deep-prefill chunk", .{ rank, step });
        const end = first + step_rows;
        const sparse_transition = first / 4 <= 512 and end / 4 > 512;
        const split_transition = old_page_rows != new_page_rows;
        if (sparse_transition or split_transition) {
            const declined = rebound == 0 and after.captures > before.captures;
            const peer_declined = try operations.swap(engine, @intFromBool(declined));
            try check.expect(declined and peer_declined == 1, "rank {d}: step {d} topology transition recaptures with zero origin rebinds", .{ rank, step });
            observed_sparse_decline = observed_sparse_decline or sparse_transition;
            observed_split_decline = observed_split_decline or split_transition;
        } else if (step > 0 and rebound > 0) {
            const old_chunks = (@min(first, 2051) + 511) / 512;
            const new_chunks = (@min(end, 2051) + 511) / 512;
            observed_chunk_grid = observed_chunk_grid or old_chunks != new_chunks;
            observed_score_grid = observed_score_grid or (first > 2048 and (first + 255) / 256 != (end + 255) / 256);
            observed_origin = observed_origin or old_origin != pager.win0[0];
        }
        if (generation) |owner| try check.expect(std.meta.eql(owner, pager.shared_sequence), "rank {d}: origin replay preserves the exact paging generation/incarnation", .{rank});
        const status = try pager.total();
        try check.expect(!status.overflow, "rank {d}: step {d} paging stays within its slot capacity", .{ rank, step });
        std.debug.print("PREFILL-ORIGIN rank={d}/{d} store={s} step={d} end={d} origin={d} page_rows={d} rebinds={d} captures={d} exact=true\n", .{ rank, world, storage, step, end, pager.win0[0], new_page_rows, rebound, after.captures - before.captures });
    }
    const covered = observed_chunk_grid and observed_score_grid and observed_origin and observed_sparse_decline and observed_split_decline;
    const peer_covered = try operations.swap(engine, @intFromBool(covered));
    try check.expect(covered and peer_covered == 1, "rank {d}: real prefill covers chunk/score grids, sliding origins and both topology declines", .{rank});
    check.pass("rank {d}/{d}: native BF16 {s} prefill exact eager/rebound logits and state, {d} actual origin rebinds, both topology declines", .{ rank, world, storage, graph.prefill_stats.rebinds - initial.rebinds });
}
