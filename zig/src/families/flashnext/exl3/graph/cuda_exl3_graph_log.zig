//! Coverage counters, the profile and the key-change log of the EXL3 decode graphs.

const std = @import("std");
const hub = @import("cuda_exl3_graph.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const Key = gkey.Key;
const Outcome = hub.Outcome;

pub var steps_seen: u64 = 0;

pub fn note(key: Key, o: Outcome) void {
    hub.coverage[key.kind - 1][@min(key.rows, 16)][@backingInt(o)] += 1;
    steps_seen += 1;
    const every: u64 = if (std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_LOG")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 256 else 256;
    if (every == 0 or steps_seen % every != 0) return;
    if (hub.profileEnabled()) logProfile();
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
    std.log.info("flashnext exl3 graph coverage after {d} steps (replay/eager/instantiated/updated):{s}; recycled seqs {d}/{d} drafts {d}/{d} (reused/kept), shared lanes reassigned {d}; draft chains {d}: {d} replays, {d} captures ({d} updates), {d} eager; decode prepare {d} ms/{d}, capture {d} ms; prefill graphs {d}/{d}/{d}/{d}, capture {d} ms, paged replays {d}, origin rebinds {d}, sightings {d}/{d}, admission misses {d}", .{ steps_seen, w.buffered(), rs.seq_reused, rs.seq_kept, rs.draft_reused, rs.draft_kept, rs.shared_lane_reassigned, cs.chains, cs.replays, cs.captures, cs.updates, cs.eager, hub.stats.prepare_ns / std.time.ns_per_ms, hub.stats.prepares, hub.stats.capture_ns / std.time.ns_per_ms, hub.prefill_stats.replays, hub.prefill_stats.eager, hub.prefill_stats.captures, hub.prefill_stats.updates, hub.prefill_stats.capture_ns / std.time.ns_per_ms, hub.prefill_stats.paged_replays, hub.prefill_stats.rebinds, hub.prefill_stats.sighting_hits, hub.prefill_stats.sighting_misses, hub.prefill_stats.admission_misses });
    std.log.info("EXL3 sequence retained-buffer evictions {d}", .{rs.seq_evicted});
}

pub fn logProfile() void {
    std.log.info("EXL3 graph profile: misses mean absent keys, not missed replays; present-unready means pending or no exec at entry, before settlement; readiness is not replay proof; use runtime replay counters; fingerprint estimates, not allocation sizes; lane/key overflow {d}/{d}; changes overlap and compare within width except width transitions", .{ hub.profile.lane_overflow, hub.profile.key_overflow });
    for (hub.profile.counters, 0..) |kind, kind_index| for (kind, 0..) |counts, width| {
        if (counts.accesses == 0) continue;
        std.log.info("EXL3 graph profile {s}{d}: accesses/key-presence-misses/present-unready/new/returning-key-misses/evicted/post-eviction-key-misses {d}/{d}/{d}/{d}/{d}/{d}/{d}; width/state/keep/seam/paging/allocation changes {d}/{d}/{d}/{d}/{d}/{d}; orientation overflow {d}", .{ if (kind_index == 0) "round" else "win", width, counts.accesses, counts.misses, counts.present_unready, counts.first_keys, counts.returning_misses, counts.pressure_evictions, counts.post_eviction_misses, counts.width_changes, counts.state_changes, counts.keep_changes, counts.seam_changes, counts.paging_changes, counts.allocation_changes, counts.orientation_overflow });
    };
    for (hub.profile.lanes, 0..) |maybe, index| if (maybe) |lane| {
        std.log.info("EXL3 graph profile lane {d}: width mask {x}; accesses/key-presence-misses/present-unready/new/returning-key-misses/evicted {d}/{d}/{d}/{d}/{d}/{d}; orientation A/B/other accesses {d}/{d}/{d}", .{ index, lane.width_mask, lane.counters.accesses, lane.counters.misses, lane.counters.present_unready, lane.counters.first_keys, lane.counters.returning_misses, lane.counters.pressure_evictions, lane.orientation_accesses[0], lane.orientation_accesses[1], lane.orientation_accesses[2] });
        std.log.info("EXL3 graph profile lane {d}: new fingerprints by width 0..16 {any}", .{ index, lane.new_keys_by_width });
    };
}

/// TF_FLASHNEXT_EXL3_GRAPH_DEBUG=1: which key fields changed since the last step of the same kind and rows.
pub var last_key: [2][17]?Key = @splat(@splat(null));
pub fn debugKey(key: Key) void {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_DEBUG") orelse return;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return;
    const lk = &last_key[key.kind - 1][@min(key.rows, 16)];
    defer lk.* = key;
    const prev = lk.* orelse return;
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    inline for (@typeInfo(Key).@"struct".field_names) |f| {
        if (!std.meta.eql(@field(prev, f), @field(key, f))) w.print(" {s}", .{f}) catch {};
    }
    std.log.info("flashnext exl3 graph key ({s} of {d} rows) changed:{s} keep {x} seq {x} {x} {x} {d} {x} out {x} streams {x} scratch {x} scores {x}", .{ if (key.kind == 1) "round" else "window", key.rows, w.buffered(), key.keep, key.seq[0][0], key.seq[0][1], key.seq[0][2], key.seq[0][3], key.seq[0][4], key.output, key.streams, key.scratch, key.scores });
}
