//! flashnext-exl3-tpe: the routing trace over a prompt schedule.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");

const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const xpass = X.pass;
const tp = X.tp;
const fpool = flash.exl3_pool;
const wire = flash.rank_wire;
const Collective = flash.collective.Collective;
const vocab = prompt.vocabulary;

const engine = @import("flashnext_exl3_tpe_engine.zig");
const cap_default = engine.cap_default;
const out_rows = engine.out_rows;
const promote = engine.promote;
const ms = engine.ms;
const gib = engine.gib;
const Ranks = engine.Ranks;
const Eng = engine.Eng;
const printPool = engine.printPool;
const Golden = engine.Golden;
const ints = engine.ints;
const num = engine.num;
const top1 = engine.top1;
const margin = engine.margin;

fn traceSchedule(allocator: std.mem.Allocator, text: []const u8, prompts: usize) ![]usize {
    var result: std.ArrayList(usize) = .empty;
    errdefer result.deinit(allocator);
    var fields = std.mem.splitScalar(u8, text, ',');
    while (fields.next()) |field| {
        const index = std.fmt.parseInt(usize, field, 10) catch return error.InvalidTraceSchedule;
        if (index >= prompts) return error.InvalidTraceSchedule;
        try result.append(allocator, index);
    }
    return result.toOwnedSlice(allocator);
}

test "trace schedule preserves prompt order and repeated selections" {
    const schedule = try traceSchedule(std.testing.allocator, "2,0,2,1", 3);
    defer std.testing.allocator.free(schedule);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0, 2, 1 }, schedule);
    for ([_][]const u8{ "", "0,", "3", "-1" }) |invalid|
        try std.testing.expectError(error.InvalidTraceSchedule, traceSchedule(std.testing.allocator, invalid, 3));
}

pub fn traceMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, golden_path: []const u8, prefix: []const u8, rounds: usize, schedule_text: []const u8, repeats: usize) !void {
    if (rounds == 0) return error.InvalidTraceRounds;
    if (repeats == 0) return error.InvalidTraceSchedule;
    if (world != 2) return error.TraceRequiresTp2;
    if (!X.graph.enabled()) return error.TraceRequiresGraphs;
    var golden = try Golden.open(gpu, golden_path);
    defer golden.parsed.deinit();
    const a = gpu.gpa;
    const schedule = try traceSchedule(a, schedule_text, golden.prompts().len);
    defer a.free(schedule);
    const segments = try std.math.mul(usize, schedule.len, repeats);
    const total_rounds = try std.math.mul(usize, segments, rounds);
    var record_bound: usize = 0;
    for (schedule) |index| {
        const count = golden.prompts()[index].object.get("ids").?.array.items.len;
        if (count == 0 or count >= cap_default or rounds >= cap_default - count) return error.InvalidTraceRounds;
        record_bound = try std.math.add(usize, record_bound, try std.math.add(usize, count, rounds));
    }
    record_bound = try std.math.mul(usize, record_bound, repeats);
    X.dplan.demand_trace.requested = true; // the planner keeps its demand trace
    defer X.dplan.demand_trace.requested = false;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    const initial_planner = (try X.dplan.forLayers(a, gpu.d, &g.e.exl3_dplan, g.e.layers[0..g.e.n_layers])) orelse return error.TracePlannerMissing;
    if (initial_planner.trace_mem == null) return error.TraceDisabled;
    for (initial_planner.pools, initial_planner.pool_states, 0..) |pool, entry, index| {
        const header = try X.dplan.demand_trace.header(index, pool.opt.layer, entry.lines, initial_planner.max_rows);
        if (record_bound > header[8]) return error.TraceScheduleExceedsCapacity;
    }
    std.debug.print("TRACE sequential shared-pool workload (not concurrent): schedule={s}, repeats={d}, segments={d}, decode rounds={d}, conservative record bound={d}\n", .{ schedule_text, repeats, segments, total_rounds, record_bound });
    const generated = try a.alloc(u32, try std.math.add(usize, rounds, 1));
    defer a.free(generated);
    var output_hash: u64 = 0;
    for (0..segments) |segment| {
        const selection = schedule[segment % schedule.len];
        if (segment != 0) {
            try gpu.d.synchronize();
            g.seq.deinit(a);
            g.seq_live = false;
            g.seq = try g.e.newSeq(cap_default);
            g.seq_live = true;
        }
        const ids = try ints(a, golden.prompts()[selection].object.get("ids").?);
        defer a.free(ids);
        try g.reset();
        try g.forward(ids, .{ .logits = .last });
        var token = prompt.argmax(g.seq.logits);
        const before = X.graph.coverage[0][1][@backingInt(X.graph.Outcome.replay)];
        generated[0] = token;
        for (1..generated.len) |index| {
            token = try g.round(token);
            generated[index] = token;
        }
        if (X.graph.coverage[0][1][@backingInt(X.graph.Outcome.replay)] == before) return error.TraceRequiresGraphReplay;
        const segment_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(generated));
        try check.expect(try g.swap(segment_hash) == segment_hash, "trace segment {d}: tp=2 output hashes differ", .{segment});
        output_hash = std.hash.Wyhash.hash(output_hash, std.mem.sliceAsBytes(generated));
        std.debug.print("TRACE segment={d} prompt={d} rounds={d} output={x}\n", .{ segment, selection, rounds, segment_hash });
    }
    try check.expect(try g.swap(output_hash) == output_hash, "trace run rank {d}: tp=2 output hashes differ", .{rank});
    if (X.dplan.current(&g.e.exl3_dplan)) |planner| {
        if (planner != initial_planner) return error.TracePlannerChanged;
        try X.dplan.toHost(&g.e.exl3_dplan);
        var decoder_pools: usize = 0;
        for (planner.pools, 0..) |pool, index| {
            var snapshot = try planner.downloadTraceAfterConsumers(a, index);
            defer snapshot.deinit();
            const view = try snapshot.view();
            if (view.dropped() != 0) return error.TraceDroppedRecords;
            if (pool.opt.layer < 48) {
                decoder_pools += 1;
                if (view.recorded() < total_rounds) return error.TraceMissingDecoderPool;
            } else if (view.recorded() == 0) continue;
            const entry = planner.pool_states[index];
            const terminal_words = try a.alloc(u32, entry.words);
            defer a.free(terminal_words);
            try planner.state.download(entry.byteOffset(), std.mem.sliceAsBytes(terminal_words));
            const replay = try X.dplan.replay.verifyTraceWithFinal(a, view, terminal_words);
            const bytes = try view.encode(a);
            defer a.free(bytes);
            const path = try std.fmt.allocPrintSentinel(a, "{s}.rank-{d}.pool-{d}.exdpt", .{ prefix, rank, index }, 0);
            defer a.free(path);
            try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = path, .data = bytes });
            std.debug.print("TRACE v{d} rank {d} pool {d} layer {d}: rounds={d}/{d}, checkpoints={d}, verified transitions={d}, picks hit/miss={d}/{d}, copies={d}, evictions={d}, in-place experts/picks={d}/{d}, bytes={d}, output={x}\n", .{ X.dplan.demand_trace.version, rank, index, pool.opt.layer, view.recorded(), view.words[8], view.checkpoints(), replay.matched_transitions, replay.delta.hit_picks, replay.delta.miss_picks, replay.delta.copies, replay.delta.evictions, replay.delta.inplace, replay.delta.mapped_picks, bytes.len, output_hash });
        }
        if (decoder_pools != 48) return error.TraceDecoderPoolCount;
    } else return error.TracePlannerMissing;
}
