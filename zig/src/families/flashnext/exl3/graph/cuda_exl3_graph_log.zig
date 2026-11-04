//! Coverage counters of the EXL3 decode graphs, logged every `log_every` steps.

const std = @import("std");
const hub = @import("cuda_exl3_graph.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const Key = gkey.Key;
const Outcome = hub.Outcome;

pub var steps_seen: u64 = 0;
const log_every: u64 = 256;

pub fn note(key: Key, o: Outcome) void {
    hub.coverage[key.kind - 1][@min(key.rows, 16)][@backingInt(o)] += 1;
    steps_seen += 1;
    if (steps_seen % log_every != 0) return;
    std.log.info("EXL3 state rebind attempts/succeeded/failed/declined {d}/{d}/{d}/{d}", .{ hub.stats.rebind_attempts, hub.stats.rebinds, hub.stats.rebind_failures, hub.stats.rebind_declined });
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (0..2) |k| for (1..17) |r| {
        const c = hub.coverage[k][r];
        if (c[0] + c[1] + c[2] + c[3] == 0) continue;
        w.print(" {s}{d}:{d}/{d}/{d}/{d}", .{ if (k == 0) "round" else "win", r, c[0], c[1], c[2], c[3] }) catch break;
    };
    const rs = @import("../offload/cuda_exl3_recycle.zig").stats;
    const cs = @import("../model/cuda_exl3_chain.zig").stats;
    std.log.info("flashnext exl3 graph coverage after {d} steps (replay/eager/instantiated/updated):{s}; recycled seqs {d}/{d} drafts {d}/{d} (reused/kept); draft chains {d}: {d} replays, {d} captures ({d} updates), {d} eager; decode prepare {d} ms/{d}, capture {d} ms; prefill graphs {d}/{d}/{d}/{d}, capture {d} ms, sightings {d}/{d}, admission misses {d}", .{ steps_seen, w.buffered(), rs.seq_reused, rs.seq_kept, rs.draft_reused, rs.draft_kept, cs.chains, cs.replays, cs.captures, cs.updates, cs.eager, hub.stats.prepare_ns / std.time.ns_per_ms, hub.stats.prepares, hub.stats.capture_ns / std.time.ns_per_ms, hub.prefill_stats.replays, hub.prefill_stats.eager, hub.prefill_stats.captures, hub.prefill_stats.updates, hub.prefill_stats.capture_ns / std.time.ns_per_ms, hub.prefill_stats.sighting_hits, hub.prefill_stats.sighting_misses, hub.prefill_stats.admission_misses });
    std.log.info("EXL3 sequence retained-buffer evictions {d}", .{rs.seq_evicted});
}
