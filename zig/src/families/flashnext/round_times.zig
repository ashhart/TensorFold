//! FZ_ROUND_TIMES=1: a shared round's host, GPU and idle milliseconds, averaged and logged every 128 rounds.
const std = @import("std");
const mtl = @import("metal");

const EVERY = 128;
const names = [_][]const u8{ "settle", "verify encode", "verify wait", "verify gpu", "host to draft", "draft encode", "draft wait", "draft gpu", "host to verify", "gpu idle verify-draft", "gpu idle draft-verify" };

pub const RoundTimes = struct {
    on: bool = false,
    n: u64 = 0,
    rows: u64 = 0,
    streams: u64 = 0,
    t0: f64 = 0, // the phase's start (host clock)
    ts: f64 = 0, // the verify's settle done
    tc: f64 = 0, // the phase's commit
    verify_end: f64 = 0,
    draft_end: f64 = 0,
    gpu_verify_end: f64 = 0,
    gpu_draft_end: f64 = 0,
    sum: [names.len]f64 = @splat(0),

    pub fn init() RoundTimes {
        return .{ .on = if (std.c.getenv("FZ_ROUND_TIMES")) |v| v[0] == '1' else false };
    }

    pub fn begin(r: *RoundTimes, drafting: bool) void {
        if (!r.on) return;
        r.t0 = mtl.clock.seconds();
        r.ts = r.t0;
        if (drafting) {
            if (r.verify_end > 0) r.sum[4] += r.t0 - r.verify_end;
        } else if (r.draft_end > 0) r.sum[8] += r.t0 - r.draft_end;
    }

    pub fn settled(r: *RoundTimes) void {
        if (r.on) r.ts = mtl.clock.seconds();
    }

    pub fn commit(r: *RoundTimes) void {
        if (r.on) r.tc = mtl.clock.seconds();
    }

    pub fn end(r: *RoundTimes, cb: mtl.CommandBuffer, drafting: bool, rows: usize, streams: usize) void {
        if (!r.on) return;
        const t = mtl.clock.seconds();
        const g0 = cb.gpuStart();
        const g1 = cb.gpuEnd();
        if (drafting) {
            r.sum[5] += r.tc - r.t0;
            r.sum[6] += t - r.tc;
            r.sum[7] += g1 - g0;
            if (r.gpu_verify_end > 0) r.sum[9] += g0 - r.gpu_verify_end;
            r.draft_end = t;
            r.gpu_draft_end = g1;
            return;
        }
        r.sum[0] += r.ts - r.t0;
        r.sum[1] += r.tc - r.ts;
        r.sum[2] += t - r.tc;
        r.sum[3] += g1 - g0;
        if (r.gpu_draft_end > 0) r.sum[10] += g0 - r.gpu_draft_end;
        r.verify_end = t;
        r.gpu_verify_end = g1;
        r.n += 1;
        r.rows += rows;
        r.streams += streams;
        if (r.n % EVERY == 0) r.log();
    }

    fn log(r: *RoundTimes) void {
        var buf: [512]u8 = undefined;
        var len: usize = 0;
        for (names, r.sum) |name, s| {
            const w = std.fmt.bufPrint(buf[len..], " {s} {d:.2}", .{ name, s * 1e3 / EVERY }) catch break;
            len += w.len;
        }
        std.log.info("flashnext rounds: streams {d:.1} rows {d:.1} ms:{s}", .{ @as(f64, @floatFromInt(r.streams)) / EVERY, @as(f64, @floatFromInt(r.rows)) / EVERY, buf[0..len] });
        r.sum = @splat(0);
        r.rows = 0;
        r.streams = 0;
    }
};
