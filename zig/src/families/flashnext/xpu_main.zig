//! Standalone CLI runner and verifier for Qwen3.8 Flash Next (qwen4_exp) on Intel Xe2 GPUs (TP={1,2,4} + PLE + MTP).
const std = @import("std");
const eng = @import("xpu_engine.zig");
const Engine = eng.Engine;

const gpa = std.heap.c_allocator;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-flashnext MODEL_DIR (--bench=N | IDS_FILE) [STEPS]\n", .{});
        return;
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    const env = @import("sys").environment;
    try env.checkScope(.xpu, env.Process{});
    const tp: usize = try env.read(u32, "XPU_TP", 4);
    const ctx: usize = try env.read(usize, "XPU_CTX", 16384);
    const want_mtp: bool = try env.read(bool, "XPU_MTP", true);
    const verify_test: bool = try env.read(bool, "XPU_VERIFY_TEST", false);

    var bench_lens: std.ArrayList(usize) = .empty;
    defer bench_lens.deinit(gpa);
    var file_ids: std.ArrayList(u32) = .empty;
    defer file_ids.deinit(gpa);
    if (std.mem.startsWith(u8, argv[2], "--bench=")) {
        var bit = std.mem.tokenizeScalar(u8, argv[2][8..], ',');
        while (bit.next()) |part| try bench_lens.append(gpa, try std.fmt.parseInt(usize, part, 10));
    } else {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, argv[2], gpa, .limited(16 << 20));
        defer gpa.free(text);
        var it = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
        while (it.next()) |w| try file_ids.append(gpa, try std.fmt.parseInt(u32, w, 10));
    }

    const e = try Engine.init(gpa, io, argv[1], .{ .context = ctx, .tp = tp, .mtp = want_mtp });
    defer e.deinit();
    e.r.ngram.waitWarm();

    std.debug.print(
        "loaded Qwen3.8-Flash-Next (tp={d}, mtp={}, vram/rank={d:.2} GiB, ple_ram={d:.2} GiB) in {d:.1} s\n",
        .{ tp, e.head != null, @as(f64, @floatFromInt(e.allocatedBytesPerRank())) / (1 << 30), @as(f64, @floatFromInt(e.r.ngram.total_bytes)) / (1 << 30), e.load_seconds },
    );

    if (bench_lens.items.len > 0) {
        var can_sweep = bench_lens.items.len > 1 and !verify_test;
        for (bench_lens.items, 0..) |n, idx| {
            if (n < 64 or n % 64 != 0 or (idx > 0 and n <= bench_lens.items[idx - 1])) can_sweep = false;
        }
        if (can_sweep) {
            try runIncrementalSweep(e, bench_lens.items, steps);
        } else {
            for (bench_lens.items) |n| {
                var ids: std.ArrayList(u32) = .empty;
                defer ids.deinit(gpa);
                for (0..n) |i| try ids.append(gpa, @intCast((i * 7919 + 13) % 200000));
                std.debug.print("\n=== Bench prompt_len={d} ===\n", .{n});
                try runOnePrompt(e, ids.items, steps, verify_test);
            }
        }
    } else {
        try runOnePrompt(e, file_ids.items, steps, verify_test);
    }
}

fn runIncrementalSweep(e: *Engine, lens: []const usize, steps: usize) !void {
    const max_n = lens[lens.len - 1];
    const ids = try gpa.alloc(u32, max_n);
    defer gpa.free(ids);
    for (0..max_n) |i| ids[i] = @intCast((i * 7919 + 13) % 200000);

    try e.reset();
    var save_dsts: [eng.ld.max_tp]*anyopaque = undefined;
    for (0..e.r.tp) |rk| save_dsts[rk] = e.r.acts[rk].prompt_streams;

    var prev_n: usize = 0;
    var cum_plain_sec: f64 = 0;
    var cum_mtp_sec: f64 = 0;

    for (lens, 0..) |target_n, idx| {
        std.debug.print("\n=== Bench prompt_len={d} ===\n", .{target_n});
        var t = prev_n;
        while (t + 64 < target_n) : (t += 64) {
            const tp0 = eng.nowSec();
            try e.r.forwardPass(ids[t .. t + 64], true, .none, if (e.r.has_mtp) save_dsts else null);
            e.m.pos = e.r.cur.pos;
            cum_plain_sec += eng.nowSec() - tp0;
            if (e.r.has_mtp) {
                const tm0 = eng.nowSec();
                _ = try e.r.forwardMtp(ids[t + 1 .. t + 65], save_dsts, t, false, false);
                cum_mtp_sec += eng.nowSec() - tm0;
            }
        }
        // Final 64-token chunk `[target_n - 64 .. target_n]`
        const tp0 = eng.nowSec();
        try e.r.forwardPass(ids[t..target_n], true, .last, if (e.r.has_mtp) save_dsts else null);
        e.m.pos = e.r.cur.pos;
        const dt_last_plain = eng.nowSec() - tp0;

        var dt_last_mtp: f64 = 0;
        if (e.r.has_mtp) {
            const tm0 = eng.nowSec();
            _ = try e.r.forwardMtp(ids[t + 1 .. target_n], save_dsts, t, false, false);
            dt_last_mtp = eng.nowSec() - tm0;
        }

        const plain_sec = cum_plain_sec + dt_last_plain;
        const full_sec = plain_sec + cum_mtp_sec + dt_last_mtp;
        std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.1} tok/s", .{ target_n, plain_sec, @as(f64, @floatFromInt(target_n)) / plain_sec });
        if (e.r.has_mtp) {
            std.debug.print("; w/ MTP absorb: {d:.3} s, {d:.1} tok/s", .{ full_sec, @as(f64, @floatFromInt(target_n)) / full_sec });
        }
        std.debug.print(")\n", .{});

        const first_tok = try e.argmax();
        const sn = try e.snapSave(@intCast(target_n));

        var serial: std.ArrayList(u32) = .empty;
        defer serial.deinit(gpa);
        try serial.append(gpa, first_tok);
        const td0 = eng.nowSec();
        for (1..steps) |_| {
            try e.stepRow(serial.items[serial.items.len - 1]);
            try serial.append(gpa, try e.argmax());
        }
        const td1 = eng.nowSec();
        std.debug.print("decode (plain): {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, td1 - td0, @as(f64, @floatFromInt(steps - 1)) / (td1 - td0) });
        std.debug.print("tokens:", .{});
        for (serial.items) |tok| std.debug.print(" {d}", .{tok});
        std.debug.print("\n", .{});

        if (e.head != null) {
            try e.snapRestore(sn);
            try runMtpFromReady(e, first_tok, serial.items);
        }

        if (idx + 1 < lens.len) {
            try e.snapRestore(sn);
            e.snapDrop(sn);
            if (e.r.has_mtp) {
                var st: [eng.ld.max_tp]*anyopaque = undefined;
                for (0..e.r.tp) |rk| st[rk] = e.r.cur.ranks[rk].last_streams;
                const tm0 = eng.nowSec();
                _ = try e.r.forwardMtp(ids[target_n .. target_n + 1], st, target_n - 1, false, false);
                cum_mtp_sec += eng.nowSec() - tm0;
            }
            cum_plain_sec = plain_sec;
            cum_mtp_sec += dt_last_mtp;
            prev_n = target_n;
        } else {
            e.snapDrop(sn);
        }
    }
}

fn prefillPlain(e: *Engine, prompt_ids: []const u32) !void {
    var t: usize = 0;
    while (t < prompt_ids.len) {
        const n = @min(@as(usize, 64), prompt_ids.len - t);
        const last = (t + n == prompt_ids.len);
        try e.r.forwardPass(prompt_ids[t .. t + n], true, if (last) .last else .none, null);
        e.m.pos = e.r.cur.pos;
        t += n;
    }
}

fn runOnePrompt(e: *Engine, prompt_ids: []const u32, steps: usize, verify_test: bool) !void {
    // 1. Plain prefill + serial greedy decode
    try e.reset();
    const t0 = eng.nowSec();
    try prefillPlain(e, prompt_ids);
    const t1 = eng.nowSec();

    var serial: std.ArrayList(u32) = .empty;
    defer serial.deinit(gpa);
    try serial.append(gpa, try e.argmax());
    for (1..steps) |_| {
        try e.stepRow(serial.items[serial.items.len - 1]);
        try serial.append(gpa, try e.argmax());
    }
    const t2 = eng.nowSec();

    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ prompt_ids.len, t1 - t0, @as(f64, @floatFromInt(prompt_ids.len)) / (t1 - t0) });
    std.debug.print("decode (plain): {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (serial.items) |tok| std.debug.print(" {d}", .{tok});
    std.debug.print("\n", .{});

    // 2. Optional stand-in verify rollback exactness check (perfect & all-wrong 8-row windows)
    if (verify_test) {
        try runVerifyTest(e, prompt_ids, serial.items);
    }

    // 3. Real MTP speculative decode benchmark & bit-exactness check vs serial
    if (e.head != null) {
        try runMtpBench(e, prompt_ids, serial.items);
    }
}

fn runVerifyTest(e: *Engine, prompt: []const u32, serial: []const u32) !void {
    for ([_]bool{ true, false }) |perfect| {
        try e.reset();
        try prefillPlain(e, prompt);
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(gpa);
        try got.append(gpa, try e.argmax());
        var rounds: usize = 0;
        while (got.items.len < serial.len) {
            var block: [8]u32 = undefined;
            block[0] = got.items[got.items.len - 1];
            for (1..8) |j| {
                const at = got.items.len - 1 + j;
                block[j] = if (perfect and at < serial.len) serial[at] else 0;
            }
            try e.verifyRows(&block);
            var g: [8]i32 = undefined;
            try e.rowTokens(&g);
            var acc: u32 = 0;
            while (acc + 1 < 8 and @as(u32, @intCast(g[acc])) == block[acc + 1]) acc += 1;
            try e.keepRows(acc + 1);
            for (g[0 .. acc + 1]) |t| try got.append(gpa, @intCast(t));
            rounds += 1;
        }
        const same = std.mem.eql(u32, got.items[0..serial.len], serial);
        std.debug.print("verify test ({s} drafts): {d} rounds, equal to serial: {}\n", .{ if (perfect) "perfect" else "wrong", rounds, same });
        if (!same) return error.VerifyMismatch;
    }
}

fn runMtpBench(e: *Engine, prompt: []const u32, serial: []const u32) !void {
    try e.reset();
    try e.prefillSpan(prompt, 0, prompt.len, 64);
    var g0: [1]i32 = undefined;
    try e.rowTokens(&g0);
    try runMtpFromReady(e, @intCast(g0[0]), serial);
}

fn runMtpFromReady(e: *Engine, first_tok: u32, serial: []const u32) !void {
    const env = @import("sys").environment;
    const depth: u32 = try env.read(u32, "XPU_MTP_DEPTH", 6);
    const conf: f64 = try env.read(f64, "XPU_MTP_CONF", -0.4);

    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(gpa);
    try got.append(gpa, first_tok);

    var drafts: [16]u32 = undefined;
    var rounds: usize = 0;
    var total_drafted: usize = 0;
    var total_accepted: usize = 0;
    var t_draft: f64 = 0;
    var t_verify: f64 = 0;

    // First round drafts from prompt's final row + first token
    var first_round = true;
    var prev_start: u32 = 0;
    var prev_kept: [16]u32 = undefined;
    var prev_kept_n: usize = 0;

    const t0 = eng.nowSec();
    while (got.items.len < serial.len) {
        const d0 = eng.nowSec();
        const nd = if (first_round) blk: {
            first_round = false;
            break :blk try draftWithConf(e, true, got.items[0], &.{}, 0, depth, conf, &drafts);
        } else try draftWithConf(e, false, 0, prev_kept[0..prev_kept_n], prev_start, depth, conf, &drafts);
        const d1 = eng.nowSec();
        t_draft += d1 - d0;

        const win_len = 1 + nd;
        var win: [16]u32 = undefined;
        win[0] = got.items[got.items.len - 1];
        @memcpy(win[1..win_len], drafts[0..nd]);

        prev_start = e.m.pos;
        const v0 = eng.nowSec();
        if (win_len == 1) {
            try e.stepRow(win[0]);
            const next_tok = try e.argmax();
            prev_kept[0] = next_tok;
            prev_kept_n = 1;
            try got.append(gpa, next_tok);
        } else {
            try e.verifyRows(win[0..win_len]);
            var g: [16]i32 = undefined;
            try e.rowTokens(g[0..win_len]);
            var acc: usize = 0;
            while (acc < nd and @as(u32, @intCast(g[acc])) == drafts[acc]) acc += 1;
            try e.keepRows(@intCast(acc + 1));
            for (0..acc + 1) |k| {
                const tok: u32 = @intCast(g[k]);
                prev_kept[k] = tok;
                try got.append(gpa, tok);
            }
            prev_kept_n = acc + 1;
            total_drafted += nd;
            total_accepted += acc;
        }
        t_verify += eng.nowSec() - v0;
        rounds += 1;
    }
    const dt = eng.nowSec() - t0;
    const gen_n = got.items.len - 1;
    const same = std.mem.eql(u32, got.items[0..serial.len], serial);
    std.debug.print(
        "decode (MTP depth={d}, conf={d:.2}): {d} tokens in {d} rounds ({d:.2} tok/round, accepted {d}/{d}), {d:.1} tok/s (draft {d:.2} ms + verify {d:.2} ms/round), equal to serial: {}\n",
        .{
            depth,
            conf,
            gen_n,
            rounds,
            @as(f64, @floatFromInt(gen_n)) / @as(f64, @floatFromInt(rounds)),
            total_accepted,
            total_drafted,
            @as(f64, @floatFromInt(gen_n)) / dt,
            t_draft * 1e3 / @as(f64, @floatFromInt(rounds)),
            t_verify * 1e3 / @as(f64, @floatFromInt(rounds)),
            same,
        },
    );
    if (!same) return error.MtpSerialMismatch;
}

fn draftWithConf(e: *Engine, is_first: bool, first_tok: u32, follow: []const u32, start: u32, max_d: u32, conf: f64, out: []u32) !usize {
    if (max_d == 0) return 0;
    var st: [eng.ld.max_tp]*anyopaque = undefined;
    const pk0 = if (is_first) blk: {
        for (0..e.r.tp) |rk| st[rk] = e.r.cur.ranks[rk].last_streams;
        break :blk (try e.r.forwardMtp(&.{first_tok}, st, e.m.pos - 1, true, true)).?;
    } else blk: {
        for (0..e.r.tp) |rk| st[rk] = e.r.cur.ranks[rk].verify_streams;
        break :blk (try e.r.forwardMtp(follow, st, start, true, true)).?;
    };

    var prod: f64 = 1.0;
    var n: usize = 0;
    var cur = pk0;
    var mtp_st: [eng.ld.max_tp]*anyopaque = undefined;
    for (0..e.r.tp) |rk| mtp_st[rk] = e.r.cur.ranks[rk].mtp_out;

    var j: usize = 0;
    while (j < @min(@as(usize, max_d), 15)) : (j += 1) {
        var propose = true;
        var more = true;
        if (conf > 0) {
            const low = cur.prob < conf;
            if (low and j > 0) {
                propose = false;
                more = false;
            } else {
                more = !low;
            }
        } else if (conf < 0) {
            prod *= cur.prob;
            if (prod < -conf) {
                propose = (j == 0);
                more = false;
            }
        }
        if (!propose) break;
        out[n] = cur.token;
        n += 1;
        if (!more or j + 1 >= @min(@as(usize, max_d), 15)) break;
        cur = (try e.r.forwardMtp(out[n - 1 .. n], mtp_st, e.m.pos + (n - 1), true, true)).?;
    }
    return n;
}
