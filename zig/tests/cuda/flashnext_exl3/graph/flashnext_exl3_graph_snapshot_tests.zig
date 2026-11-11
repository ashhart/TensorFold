//! flashnext-exl3-graph snapshot: two lanes saved, restored and continued under eager and graph steps.

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

/// A snapshot's bytes on the host: its pinned buffer, or a copy of its device buffer.
fn bytesOf(gpu: Gpu, saved: *const flash.snapshot.Snap) ![]u8 {
    const out = try gpu.gpa.alloc(u8, saved.buf.len);
    errdefer gpu.gpa.free(out);
    if (saved.host) |host| @memcpy(out, host.bytes[0..out.len]) else try gpu.d.check(gpu.d.api.cuMemcpyDtoH_v2(out.ptr, saved.buf.ptr, out.len), "cuMemcpyDtoH");
    return out;
}

pub fn snapshotMode(gpu: Gpu, rank: usize, world: usize, port: u16, dir: []const u8, steps: usize) !void {
    const snaps = flash.snapshot;
    const allocator = gpu.gpa;
    const g = try Eng.init(gpu, rank, world, port, dir);
    defer g.deinit();
    defer graph.force = null;
    var second = try g.e.newSeq(cap_default);
    defer second.deinit(allocator);
    const ids = try allocator.alloc(u32, 16384);
    defer allocator.free(ids);
    var random = std.Random.DefaultPrng.init(0x736e6170);
    for (ids) |*token| token.* = 200 + random.random().uintLessThan(u32, 60000);
    graph.force = .graph;
    const first_end: usize = 3075;
    const second_end: usize = 6917;
    try g.forward(ids[0..first_end], .{ .logits = .last });
    try g.e.forward(flash.Tri, g.tri(), &second, ids[4096..second_end], .{ .logits = .last });
    const first_saved = try snaps.save(allocator, gpu.d, g.stream, &g.seq, null);
    defer snaps.drop(allocator, g.stream, first_saved);
    const second_saved = try snaps.save(allocator, gpu.d, g.stream, &second, null);
    defer snaps.drop(allocator, g.stream, second_saved);
    const expected = try allocator.alloc(u8, (steps + 1) * 2 * vocab * 2);
    defer allocator.free(expected);
    var reference: [2]?*snaps.Snap = .{ null, null };
    defer for (reference) |item| if (item) |saved| snaps.drop(allocator, g.stream, saved);
    for ([_]graph.Mode{ .off, .graph, .graph, .graph }, 0..) |mode, run_index| {
        graph.force = mode;
        try dplan.toHost(&g.e.exl3_dplan);
        g.e.round_graph.clear();
        try snaps.restore(g.stream, &g.seq, null, first_saved);
        try snaps.restore(g.stream, &second, null, second_saved);
        const before = graph.stats;
        const prefill_before = graph.prefill_stats;
        try g.forward(ids[first_end..][0..8192], .{ .logits = .last });
        try g.e.forward(flash.Tri, g.tri(), &second, ids[second_end..][0..8192], .{ .logits = .last });
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
        }
        for (sequences, 0..) |sequence, lane| {
            const saved = try snaps.save(allocator, gpu.d, g.stream, sequence, null);
            if (run_index == 0) {
                reference[lane] = saved;
            } else {
                defer snaps.drop(allocator, g.stream, saved);
                const previous = reference[lane].?;
                try check.expect(previous.at == saved.at and std.meta.eql(previous.history, saved.history), "rank {d} snapshot lane {d} exact position and history", .{ rank, lane });
                const old_bytes = try bytesOf(gpu, previous);
                defer allocator.free(old_bytes);
                const new_bytes = try bytesOf(gpu, saved);
                defer allocator.free(new_bytes);
                try check.expect(std.mem.eql(u8, old_bytes, new_bytes), "rank {d} snapshot lane {d} exact cache and recurrent state", .{ rank, lane });
            }
        }
        if (mode == .graph) {
            std.debug.print("SNAPSHOT rank {d}/{d} graph cache run={d}: prefill captures +{d}, updates +{d}, replays +{d}, eager +{d}; decode replays +{d}\n", .{ rank, world, run_index, graph.prefill_stats.captures - prefill_before.captures, graph.prefill_stats.updates - prefill_before.updates, graph.prefill_stats.replays - prefill_before.replays, graph.prefill_stats.eager - prefill_before.eager, graph.stats.replays - before.replays });
            try check.expect(graph.stats.replays > before.replays, "rank {d} concurrent decode graph replay after restore", .{rank});
        }
        std.debug.print("SNAPSHOT rank {d}/{d} {s}: two lanes, {d} decode steps, exact state and logits\n", .{ rank, world, @tagName(mode), steps });
    }
    check.pass("rank {d}/{d}: concurrent snapshots preserve the BF16 continuation under eager and graph steps", .{ rank, world });
}
