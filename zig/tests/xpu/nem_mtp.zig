//! Nemotron-H MTP drafter against plain greedy decoding (outputs must be identical), acceptance and tokens/s.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;
const nm = @import("nemotron_xpu").mtp;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn parseIds(s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, std.mem.trim(u8, t, " "), 10));
    return out.toOwnedSlice(gpa);
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) return error.Usage;
    const dir = args[1];
    const prompt = try parseIds(args[2]);
    const n_new = try std.fmt.parseInt(usize, args[3], 10);
    const ks = try parseIds(args[4]);
    if (prompt.len == 0 or n_new < 2) return error.BadArgument; // before the device is opened
    var ctx_len: u32 = 4096;
    var plain_run = true;
    var i: usize = 5;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--ctx")) {
            i += 1;
            ctx_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-plain")) plain_run = false;
    }
    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ctx_len);
    m.bf16_logits = true;
    var w = try nw.Win.init(gpa, &m);
    var mt = try nm.Mtp.load(gpa, &m, &l, dir);
    const eos = cfg.value.eos_token_id;

    var plain: std.ArrayList(u32) = .empty;
    if (plain_run) {
        try w.reset(&m);
        var t: usize = 0;
        var next: u32 = 0;
        while (t < prompt.len) {
            const rows = @min(16, prompt.len - t);
            const last = t + rows == prompt.len;
            try w.forward(&m, prompt[t .. t + rows], true, if (last) 1 else 0);
            if (last) {
                var g: [1]i32 = undefined;
                try w.rowArgmax(&m, &g);
                next = @intCast(g[0]);
            } else try r.sync();
            t += rows;
        }
        try plain.append(gpa, next);
        const first = next;
        // identity baseline: one-row windows (the numerics of the window kernels)
        const t0 = nowNs();
        while (plain.items.len < n_new) {
            try nw.stopCheck(&r);
            try w.forward(&m, &[_]u32{next}, true, 1);
            var one: [1]i32 = undefined;
            try w.rowArgmax(&m, &one);
            next = @intCast(one[0]);
            try plain.append(gpa, next);
        }
        const ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        std.debug.print("plain greedy on one-row windows: {d} tokens in {d:.1} ms = {d:.2} tokens/s\n", .{ n_new - 1, ms, @as(f64, @floatFromInt(n_new - 1)) / ms * 1e3 });
        // speed baseline: model.zig's single-row decode (own kernels; tokens may differ from the window path at ties)
        try w.reset(&m);
        {
            var t2: usize = 0;
            while (t2 < prompt.len) {
                const rows = @min(16, prompt.len - t2);
                try w.forward(&m, prompt[t2 .. t2 + rows], true, 0);
                t2 += rows;
            }
            try r.sync();
        }
        next = first;
        const t1 = nowNs();
        for (1..n_new) |_| {
            try nw.stopCheck(&r);
            try m.forward(next, true);
            next = try m.argmax();
        }
        const ms1 = @as(f64, @floatFromInt(nowNs() - t1)) / 1e6;
        std.debug.print("plain greedy with model.zig decode: {d} tokens in {d:.1} ms = {d:.2} tokens/s\n", .{ n_new - 1, ms1, @as(f64, @floatFromInt(n_new - 1)) / ms1 * 1e3 });
    }
    for (ks) |k| {
        try nw.stopCheck(&r);
        try w.reset(&m);
        var st: nm.Stats = .{};
        var drafts: [16]u32 = undefined;
        const pf0 = nowNs();
        const pf = try nm.prefill(&w, &mt, &m, prompt, k, &drafts, &st);
        const pf_ms = @as(f64, @floatFromInt(nowNs() - pf0)) / 1e6;
        var ctx: std.ArrayList(u32) = .empty;
        try ctx.append(gpa, pf.first);
        const t0 = nowNs();
        try nm.runSpec(&w, &mt, &m, gpa, &ctx, drafts[0..pf.nd], n_new - 1, k, eos, true, &st);
        const ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        var same = true;
        var first_bad: usize = 0;
        if (plain_run) {
            for (0..@min(ctx.items.len, plain.items.len)) |j| if (ctx.items[j] != plain.items[j]) {
                same = false;
                first_bad = j;
                break;
            };
            if (ctx.items.len != plain.items.len) same = false;
        }
        const nwn = @as(f64, @floatFromInt(@max(st.windows, 1)));
        std.debug.print("mtp k={d}: {d} tokens in {d:.1} ms = {d:.2} tokens/s; identical to plain: {s}; prefill {d:.1} ms\n", .{ k, ctx.items.len - 1, ms, @as(f64, @floatFromInt(ctx.items.len - 1)) / ms * 1e3, if (!plain_run) "n/a" else if (same) "yes" else "NO", pf_ms });
        if (!same and plain_run) std.debug.print("  first difference at generated index {d}\n", .{first_bad});
        std.debug.print("  windows {d}, plain steps {d}, drafted {d}, accepted {d} ({d:.2} per window, {d:.1}% of drafts); tokens per window {d:.2}\n", .{ st.windows, st.plain, st.drafted, st.accepted, @as(f64, @floatFromInt(st.accepted)) / nwn, @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(@max(st.drafted, 1))) * 100.0, @as(f64, @floatFromInt(st.tokens)) / nwn });
        std.debug.print("  cost: verify windows {d:.2} ms each; MTP {d:.1} ms total ({d} head steps, {d} absorb-only); per token {d:.2} ms\n", .{ @as(f64, @floatFromInt(st.win_ns)) / 1e6 / nwn, @as(f64, @floatFromInt(st.mtp_ns)) / 1e6, st.steps, st.absorbs, ms / @as(f64, @floatFromInt(ctx.items.len - 1)) });
    }
}
