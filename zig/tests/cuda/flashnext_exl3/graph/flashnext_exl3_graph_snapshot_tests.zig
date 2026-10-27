//! flashnext-exl3-graph snapshot: paged lanes saved, restored and continued under eager and graph steps.

const std = @import("std");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const Gpu = check.Gpu;
const prompt = flash.prompt;
const X = flash.exl3;
const dplan = X.dplan;
const graph = X.graph;
const vocab = prompt.vocabulary;
const harness = @import("flashnext_exl3_graph_engine.zig");
const Eng = harness.Eng;
const cap_default = harness.cap_default;

fn expectSnapshotState(rank: usize, previous: *const flash.snapshot.Snap, saved: *const flash.snapshot.Snap) !void {
    const canonical = saved.paged.?.layout.canonical;
    const state_offset = saved.paged.?.layout.total;
    try check.expect(previous.at == saved.at and std.meta.eql(previous.history, saved.history), "rank {d}: exact snapshot position and history", .{rank});
    try check.expect(std.mem.eql(u8, previous.host.?.bytes[0..canonical], saved.host.?.bytes[0..canonical]), "rank {d}: exact canonical BF16 history", .{rank});
    try check.expect(std.mem.eql(u8, previous.host.?.bytes[state_offset..previous.buf.len], saved.host.?.bytes[state_offset..saved.buf.len]), "rank {d}: exact recurrent/index/PLE state and logits", .{rank});
}

fn expectOwnedRestore(engine: *Eng, sequence: *prompt.Seq, wanted: *const flash.snapshot.Snap) !void {
    const saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, sequence, null);
    defer flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
    try expectSnapshotState(engine.rank, wanted, saved);
    try check.expect(std.meta.eql(wanted.paged, saved.paged) and std.mem.eql(u8, wanted.host.?.bytes[0..wanted.buf.len], saved.host.?.bytes[0..saved.buf.len]), "rank {d}: exact destination windows, marks, controls and snapshot metadata", .{engine.rank});
    const previous = wanted.owned_blocks orelse return error.SharedKvSnapshotRequired;
    const current = saved.owned_blocks orelse return error.SharedKvSnapshotRequired;
    try check.expect(previous.entries.len > 0 and previous.entries.len == current.entries.len, "rank {d}: restore retains every owned resident entry, not a cold cache", .{engine.rank});
    try check.expect(std.meta.eql(current.source, sequence.paged.?.shared_sequence), "rank {d}: restored slots belong to destination generation", .{engine.rank});
    const block_bytes = try previous.geometry.blockBytes();
    for (previous.entries, 0..) |entry, old_index| {
        const new_index = for (current.entries, 0..) |candidate, index| {
            if (std.meta.eql(entry.key, candidate.key)) break index;
        } else return error.SharedKvSnapshotMissingEntry;
        try check.expect(previous.clock - entry.stamp == current.clock - current.entries[new_index].stamp, "rank {d}: owned entry retains relative LRU age", .{engine.rank});
        try check.expect(std.mem.eql(u8, previous.payload[old_index * block_bytes ..][0..block_bytes], current.payload[new_index * block_bytes ..][0..block_bytes]), "rank {d}: owned entry remaps exact K/V payload", .{engine.rank});
    }
}

const SharedSnapshotProbe = struct {
    sequence: prompt.Seq,
    saved: *flash.snapshot.Snap,
    continued: ?*flash.snapshot.Snap = null,
    key: [6]u64,
    buffers: [3]u64,

    fn init(engine: *Eng, ids: []const u32) !*SharedSnapshotProbe {
        const probe = try engine.gpu.gpa.create(SharedSnapshotProbe);
        errdefer engine.gpu.gpa.destroy(probe);
        probe.* = .{ .sequence = try engine.e.newSeq(cap_default), .saved = undefined, .key = undefined, .buffers = undefined };
        errdefer probe.sequence.deinit(engine.gpu.gpa);
        try engine.e.forward(flash.Tri, engine.tri(), &probe.sequence, ids, .{ .logits = .last });
        probe.saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, &probe.sequence, null);
        errdefer flash.snapshot.drop(engine.gpu.gpa, engine.stream, probe.saved);
        const arena = engine.e.shared_arena orelse return error.SharedKvArenaRequired;
        try check.expect(probe.saved.version == 2 and probe.saved.owned_blocks.?.entries.len > 0, "rank {d}: foreign probe holds shared resident blocks", .{engine.rank});
        probe.key = probe.sequence.paged.?.graphKey();
        probe.buffers = .{ arena.device.ptr, arena.tables.ptr, arena.work.ptr };
        return probe;
    }

    fn deinit(probe: *SharedSnapshotProbe, engine: *Eng) void {
        if (probe.continued) |saved| flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
        flash.snapshot.drop(engine.gpu.gpa, engine.stream, probe.saved);
        probe.sequence.deinit(engine.gpu.gpa);
        engine.gpu.gpa.destroy(probe);
    }

    fn unchanged(probe: *SharedSnapshotProbe, engine: *Eng) !void {
        const arena = engine.e.shared_arena.?;
        try check.expect(std.meta.eql(probe.buffers, [3]u64{ arena.device.ptr, arena.tables.ptr, arena.work.ptr }), "rank {d}: arena addresses survive restore and sequence churn", .{engine.rank});
        try check.expect(std.meta.eql(probe.key, probe.sequence.paged.?.graphKey()), "rank {d}: foreign graph bindings remain legal after restore", .{engine.rank});
        const saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, &probe.sequence, null);
        defer flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
        try check.expect(std.meta.eql(probe.saved.paged, saved.paged), "rank {d}: foreign window origins and counters unchanged", .{engine.rank});
        try check.expect(probe.saved.at == saved.at and std.meta.eql(probe.saved.history, saved.history), "rank {d}: foreign logical position unchanged", .{engine.rank});
        try check.expect(std.mem.eql(u8, probe.saved.host.?.bytes[0..probe.saved.buf.len], saved.host.?.bytes[0..saved.buf.len]), "rank {d}: restore preserves foreign canonical history, full windows, marks, controls, ordinary state and logits", .{engine.rank});
    }

    fn continueAndRestore(probe: *SharedSnapshotProbe, engine: *Eng) !void {
        try engine.e.forward(flash.Tri, engine.tri(), &probe.sequence, &.{ 417, 819, 1237 }, .{ .logits = .last });
        const saved = try flash.snapshot.save(engine.gpu.gpa, engine.gpu.d, engine.stream, &probe.sequence, null);
        if (probe.continued) |previous| {
            defer flash.snapshot.drop(engine.gpu.gpa, engine.stream, saved);
            try expectSnapshotState(engine.rank, previous, saved);
        } else probe.continued = saved;
        try flash.snapshot.restore(engine.stream, &probe.sequence, null, probe.saved);
        try expectOwnedRestore(engine, &probe.sequence, probe.saved);
        try probe.unchanged(engine);
    }
};

const SnapshotStorage = enum { legacy, private, shared };

pub fn snapshotStorageMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize, storage: SnapshotStorage) !void {
    if (world != 2) return error.SharedSnapshotRequiresTp2;
    if (steps < 8 or steps > 64) return error.SharedSnapshotDecodeSteps;
    if (X.kvpage.force_off) return error.SharedKvPagingDisabled;
    var paging = X.kvpage.configFromEnv() orelse X.kvpage.Config{ .shared_lanes = 8 };
    if (storage == .shared) {
        if (paging.shared_lanes < 3) return error.SharedSnapshotNeedsThreeLanes;
    } else paging.shared_lanes = 0;
    paging.slots = 1024;
    paging.window = 4096;
    paging.rows = 4096;
    if ((X.prefill.state.options orelse X.prefill.state.optionsFromEnv()).serve_rows > paging.rows) return error.SharedSnapshotPrefillRows;
    try paging.validate();
    const previous = X.kvpage.force;
    X.kvpage.force = paging;
    defer X.kvpage.force = previous;
    return snapshotMode(gpu, rank, world, port, dir, steps, storage);
}

pub fn snapshotMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize, storage: SnapshotStorage) !void {
    const snaps = flash.snapshot;
    const allocator = gpu.gpa;
    const shared = storage == .shared;
    const paired = storage != .legacy;
    const g = if (paired) try Eng.initPaged(gpu, rank, world, port, dir, X.kvpage.force.?) else try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    defer graph.force = null;
    try check.expect(g.seq.paged != null, "snapshot test requires BF16 paging", .{});
    if (paired) try check.expect(g.seq.kv8 == null and g.seq.paged.?.cfg.slots == 1024 and g.seq.paged.?.cfg.rows == 4096, "rank {d}: paired fixture keeps fixed BF16 paging geometry", .{rank});
    if (shared) try check.expect(g.seq.paged.?.shared_arena == g.e.shared_arena and g.seq.paged.?.cfg.shared_lanes == X.kvpage.force.?.shared_lanes and g.seq.paged.?.cfg.slots == 1024 and g.seq.paged.?.cfg.rows == 4096, "rank {d}: shared snapshot fixture cannot silently use private or different paging geometry", .{rank});
    if (shared) {
        const owner = g.seq.paged.?.shared_sequence;
        const peer_generation = try g.swap(owner.generation);
        try check.expect(owner.generation != peer_generation, "rank {d}: shared TP2 fixture requires legal cross-rank generation asymmetry", .{rank});
        const logical = g.seq.paged.?.logicalGraphKey();
        const logical_hash = std.hash.Wyhash.hash(0, std.mem.asBytes(&logical));
        try check.expect(try g.swap(logical_hash) == logical_hash, "rank {d}: asymmetric physical generations retain identical logical TP geometry", .{rank});
        std.debug.print("SHARED rank {d}/{d}: primary lane {d} generation {d}, peer generation {d}; logical TP key {x}\n", .{ rank, world, owner.lane, owner.generation, peer_generation, logical_hash });
    }
    if (storage == .private) try check.expect(g.seq.paged.?.cfg.shared_lanes == 0 and g.seq.paged.?.shared_arena == null, "rank {d}: private oracle uses independent per-sequence slots", .{rank});
    var second = try g.e.newSeq(cap_default);
    var second_live = true;
    defer if (second_live) second.deinit(allocator);
    const ids = try allocator.alloc(u32, if (paired) 24576 else 16384);
    defer allocator.free(ids);
    var random = std.Random.DefaultPrng.init(0x736e6170);
    for (ids) |*token| token.* = 200 + random.random().uintLessThan(u32, 60000);
    if (paired) std.debug.print("SNAPSHOT-ORACLE store={s} rank={d}/{d} seed=736e6170 ids={x:0>16} first=8707 second=4096+8709 continuation=8192 decode={d}; compare matching rank/run/lane hashes between snapshot-private and snapshot-shared, not a golden\n", .{ @tagName(storage), rank, world, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ids)), steps });
    graph.force = .graph;
    const first_end: usize = if (paired) 8707 else 3075;
    const second_end: usize = if (paired) 12805 else 6917;
    try g.forward(ids[0..first_end], .{ .logits = .last });
    var first_saved: *snaps.Snap = undefined;
    if (paired) first_saved = try snaps.save(allocator, gpu.d, g.stream, &g.seq, null);
    defer if (paired) snaps.drop(allocator, g.stream, first_saved);
    try g.e.forward(flash.Tri, g.tri(), &second, ids[4096..second_end], .{ .logits = .last });
    if (!paired) first_saved = try snaps.save(allocator, gpu.d, g.stream, &g.seq, null);
    defer if (!paired) snaps.drop(allocator, g.stream, first_saved);
    const second_saved = try snaps.save(allocator, gpu.d, g.stream, &second, null);
    defer snaps.drop(allocator, g.stream, second_saved);
    const source_key = second.paged.?.graphKey();
    const probe: ?*SharedSnapshotProbe = if (shared) try SharedSnapshotProbe.init(g, ids[14336..23045]) else null;
    defer if (probe) |foreign| foreign.deinit(g);
    const expected = try allocator.alloc(u8, (steps + 1) * 2 * vocab * 2);
    defer allocator.free(expected);
    var reference: [2]?*snaps.Snap = .{ null, null };
    defer for (reference) |item| if (item) |saved| snaps.drop(allocator, g.stream, saved);
    for ([_]graph.Mode{ .off, .graph, .graph, .graph }, 0..) |mode, run_index| {
        graph.force = mode;
        try dplan.toHost();
        g.e.round_graph.clear();
        if (shared) {
            const previous = second.paged.?.shared_sequence;
            const window = second.paged.?.devRows(0, false);
            const reuse_before = X.recycle.stats.seq_reused;
            graph.force = .graph;
            second_live = false;
            second.deinit(allocator);
            second = try g.e.newSeq(cap_default);
            second_live = true;
            graph.force = mode;
            try check.expect(X.recycle.stats.seq_reused == reuse_before + 1, "rank {d}: request recycle reused the retained sequence allocations", .{rank});
            const current = second.paged.?.shared_sequence;
            try check.expect(previous.arena == current.arena and previous.lane == current.lane and current.generation > previous.generation and current.generation > second_saved.owned_blocks.?.source.generation, "rank {d}: saved source restores into recycled lane with a new generation", .{rank});
            try check.expect(window == second.paged.?.devRows(0, false), "rank {d}: recycled lane retains its fixed window address", .{rank});
            try probe.?.unchanged(g);
            std.debug.print("SHARED rank {d}/{d}: recycled lane {d}, sequence reuse count +1, source generation {d}, destination generation {d}, stable window {x}\n", .{ rank, world, current.lane, second_saved.owned_blocks.?.source.generation, current.generation, window });
        }
        try snaps.restore(g.stream, &g.seq, null, first_saved);
        if (shared) {
            try expectOwnedRestore(g, &g.seq, first_saved);
            try probe.?.unchanged(g);
        }
        try snaps.restore(g.stream, &second, null, second_saved);
        if (shared) {
            try expectOwnedRestore(g, &second, second_saved);
            try check.expect(source_key[4] != second.paged.?.graphKey()[4], "rank {d}: identical restored geometry at same window address has a different strict graph generation key", .{rank});
            try probe.?.unchanged(g);
        }
        const before = graph.stats;
        const prefill_before = graph.prefill_stats;
        const deep_before = X.prefill.state.stats;
        try g.forward(ids[first_end..][0..8192], .{ .logits = .last });
        try g.e.forward(flash.Tri, g.tri(), &second, ids[second_end..][0..8192], .{ .logits = .last });
        if (paired) try check.expect(X.prefill.state.stats.chunks > deep_before.chunks and X.prefill.state.stats.rows >= deep_before.rows + 16384, "rank {d}: both paired lanes actually execute deep prefill", .{rank});
        var tokens = [_]u32{ prompt.argmax(g.seq.logits), prompt.argmax(second.logits) };
        const sequences = [_]*prompt.Seq{ &g.seq, &second };
        for (0..steps + 1) |step| {
            if (step > 0) {
                try g.e.round(flash.Tri, g.tri(), &sequences, &tokens, g.out.ptr);
                tokens = .{ prompt.argmax(g.seq.logits), prompt.argmax(second.logits) };
            }
            for (sequences, 0..) |sequence, lane| {
                const wanted = expected[(step * 2 + lane) * vocab * 2 ..][0 .. vocab * 2];
                if (run_index == 0) @memcpy(wanted, sequence.logits) else try check.expect(std.mem.eql(u8, wanted, sequence.logits), "rank {d} snapshot {s} lane {d} step {d} BF16 logits", .{ rank, @tagName(mode), lane, step });
            }
            if (paired) {
                for (sequences, 0..) |sequence, lane| std.debug.print("SNAPSHOT-ORACLE store={s} rank={d}/{d} mode={s} run={d} lane={d} step={d} logits={x:0>16}\n", .{ @tagName(storage), rank, world, @tagName(mode), run_index, lane, step, std.hash.Wyhash.hash(0, sequence.logits) });
                const digest = std.hash.Wyhash.hash(0, g.seq.logits) ^ std.hash.Wyhash.hash(1, second.logits);
                try check.expect(try g.swap(digest) == digest, "rank {d}: paired TP2 logits agree at step {d}", .{ rank, step });
            }
        }
        for (sequences, 0..) |sequence, lane| {
            const saved = try snaps.save(allocator, gpu.d, g.stream, sequence, null);
            if (paired) {
                const canonical_hash = std.hash.Wyhash.hash(0, saved.host.?.bytes[0..saved.paged.?.layout.canonical]);
                const state_hash = std.hash.Wyhash.hash(1, saved.host.?.bytes[saved.paged.?.layout.total..saved.buf.len]);
                const history_hash = std.hash.Wyhash.hash(2, std.mem.asBytes(&saved.history)) ^ std.hash.Wyhash.hash(3, std.mem.asBytes(&saved.at));
                std.debug.print("SNAPSHOT-ORACLE store={s} rank={d}/{d} mode={s} run={d} lane={d} at={d} canonical={x:0>16} state-logits={x:0>16} history={x:0>16} full={x:0>16}\n", .{ @tagName(storage), rank, world, @tagName(mode), run_index, lane, saved.at, canonical_hash, state_hash, history_hash, canonical_hash ^ state_hash ^ history_hash });
            }
            if (run_index == 0) {
                reference[lane] = saved;
            } else {
                defer snaps.drop(allocator, g.stream, saved);
                const previous = reference[lane].?;
                const canonical = saved.paged.?.layout.canonical;
                const state_offset = saved.paged.?.layout.total;
                try check.expect(std.mem.eql(u8, previous.host.?.bytes[0..canonical], saved.host.?.bytes[0..canonical]), "rank {d} snapshot lane {d} exact canonical BF16 history", .{ rank, lane });
                try check.expect(std.mem.eql(u8, previous.host.?.bytes[state_offset..previous.buf.len], saved.host.?.bytes[state_offset..saved.buf.len]), "rank {d} snapshot lane {d} exact recurrent/index/PLE state", .{ rank, lane });
            }
        }
        if (mode == .graph) {
            std.debug.print("SNAPSHOT rank {d}/{d} graph cache run={d}: prefill captures +{d}, updates +{d}, replays +{d}, paged replays +{d}, rebinds +{d}, sighting hits +{d}, misses +{d}, eager +{d}; decode replays +{d}\n", .{ rank, world, run_index, graph.prefill_stats.captures - prefill_before.captures, graph.prefill_stats.updates - prefill_before.updates, graph.prefill_stats.replays - prefill_before.replays, graph.prefill_stats.paged_replays - prefill_before.paged_replays, graph.prefill_stats.rebinds - prefill_before.rebinds, graph.prefill_stats.sighting_hits - prefill_before.sighting_hits, graph.prefill_stats.sighting_misses - prefill_before.sighting_misses, graph.prefill_stats.eager - prefill_before.eager, graph.stats.replays - before.replays });
            if (run_index == 1) {
                try check.expect(graph.prefill_stats.captures + graph.prefill_stats.updates + graph.prefill_stats.replays + graph.prefill_stats.eager > prefill_before.captures + prefill_before.updates + prefill_before.replays + prefill_before.eager, "rank {d} restored prefill graph warmup runs", .{rank});
            } else if (run_index == 2) {
                try check.expect(graph.prefill_stats.captures + graph.prefill_stats.updates + graph.prefill_stats.replays > prefill_before.captures + prefill_before.updates + prefill_before.replays, "rank {d} restored prefill graph captures or rebinds after warmup", .{rank});
            } else {
                try check.expect(graph.prefill_stats.paged_replays > prefill_before.paged_replays, "rank {d} paged-attention graph replay after snapshot restore", .{rank});
            }
            try check.expect(graph.stats.replays > before.replays, "rank {d} concurrent decode graph replay after restore", .{rank});
        }
        if (probe) |foreign| {
            for (sequences) |sequence| {
                const paged = sequence.paged.?;
                const status = try paged.total();
                try check.expect(!status.overflow and status.misses > 0 and status.rebases > 0, "rank {d}: shared paging actually fetches blocks and rebases without overflow", .{rank});
                for (paged.win0) |origin| try check.expect(origin > 0 and try X.kvpage.pagingRows(paged.cfg, origin, 256) < 256, "rank {d}: slots1024 forces paging subdivision", .{rank});
            }
            try foreign.unchanged(g);
            try foreign.continueAndRestore(g);
        }
        std.debug.print("SNAPSHOT rank {d}/{d} {s}: two paged lanes, {d} decode steps, exact state and logits; prefill replays {d}, decode replays {d}\n", .{ rank, world, @tagName(mode), steps, graph.prefill_stats.replays - prefill_before.replays, graph.stats.replays - before.replays });
    }
    if (shared) check.pass("rank {d}/{d}: shared slots1024 TP2 snapshot remapping, recycled generation, foreign windows/history/continuation, exact eager/graph state/logits and actual prefill/decode replays", .{ rank, world }) else check.pass("rank {d}/{d}: concurrent paged snapshots preserve BF16 continuation and graph prefill/decode", .{ rank, world });
}
