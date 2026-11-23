//! TF_DSV41_PROMPT_PROF: where a prompt's time went, its Engram rows by origin, and the lines it logs.
const std = @import("std");
const host = @import("dsv41");
const forward = @import("forward.zig");

const Config = host.config.Config;

/// Where a prompt's time went: host fills, host encoding, host waits, GPU time, the Engram rows (TF_DSV41_PROMPT_PROF).
pub const Prof = struct {
    prep_ns: u64 = 0,
    enc_ns: u64 = 0,
    stall_ns: u64 = 0,
    gpu_s: f64 = 0,
    chunks: u32 = 0,
    /// the prompt's start, its first commit after it, and the first chunk's Engram reads queued (hashes included)
    t0: u64 = 0,
    first_ns: u64 = 0,
    issue_ns: u64 = 0,
    eng: [host.engram_host.max_layers]Eng = @splat(.{}),
    /// the decoder's cone: wall and GPU time
    dec_ns: u64 = 0,
    dec_gpu_s: f64 = 0,
    /// the decoder's commits to their GPU starts; the encoder's buffers' waits past their commits and predecessors
    dec_late_s: f64 = 0,
    enc_late_s: f64 = 0,

    pub fn report(f: *const Prof, c: *const Config, rows: usize, total_ns: u64) void {
        logLines(f, c, rows, total_ns);
    }
};

const Stats = host.engram_rows.Stats;

/// An Engram layer's rows over a prompt: host time, the reader's wait, lookups, and their unique rows by origin.
pub const Eng = struct {
    ns: u64 = 0,
    wait_ns: u64 = 0,
    lookups: u64 = 0,
    hits: u64 = 0,
    waited: u64 = 0,
    misses: u64 = 0,
    decode_ns: u64 = 0,
    settle_ns: u64 = 0,

    pub fn add(e: *Eng, ns: u64, s0: Stats, s1: Stats, lookups: u64) void {
        e.ns += ns;
        e.lookups += lookups;
        e.hits += s1.hits - s0.hits;
        e.misses += s1.misses - s0.misses;
        if (@hasField(Stats, "waited")) {
            e.waited += s1.waited - s0.waited;
            e.wait_ns += s1.wait_ns - s0.wait_ns;
        }
        if (@hasField(Stats, "decode_ns")) {
            e.decode_ns += s1.decode_ns - s0.decode_ns;
            e.settle_ns += s1.settle_ns - s0.settle_ns;
        }
    }
};

pub fn statsOf(lookup: ?forward.Lookup) Stats {
    const l = lookup orelse return .{};
    return if (l.stats) |s| s.* else .{};
}

pub fn now() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn logLines(f: *const Prof, c: *const Config, rows: usize, total_ns: u64) void {
    const log = std.log.scoped(.dsv41);
    log.info("prompt {d} rows: {d:.0} ms, encoder {d} chunks: gpu {d:.0} ms, host fill {d:.0} ms, host encode {d:.0} ms, host wait {d:.0} ms, buffers started {d:.1} ms late", .{ rows, ms(total_ns), f.chunks, f.gpu_s * 1e3, ms(f.prep_ns), ms(f.enc_ns), ms(f.stall_ns), f.enc_late_s * 1e3 });
    log.info("prompt {d} rows: first commit at {d:.1} ms, Engram reads queued in {d:.1} ms; decoder {d:.0} ms, gpu {d:.0} ms, started {d:.1} ms after its commit", .{ rows, ms(f.first_ns), ms(f.issue_ns), ms(f.dec_ns), f.dec_gpu_s * 1e3, f.dec_late_s * 1e3 });
    for (c.engram_layers.items(), 0..) |layer, li| {
        const e = f.eng[li];
        log.info("prompt Engram layer {d}: {d} lookups, unique rows {d} cached, {d} read ahead, {d} read late; host {d:.1} ms: waited {d:.1}, decoded {d:.1}, settled {d:.1}", .{ layer, e.lookups, e.hits, e.waited, e.misses, ms(e.ns), ms(e.wait_ns), ms(e.decode_ns), ms(e.settle_ns) });
    }
}
