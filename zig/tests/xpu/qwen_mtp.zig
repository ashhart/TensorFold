//! MTP speculative decoding must match plain greedy. usage: xpu-qwen_mtp-test CKPT IDS N_NEW K [--prefill ROWS]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const mtp = @import("qwen_xpu").mtp;
const qw = @import("qwen_xpu").win;
const qg = @import("qwen_xpu").gguf;

const gpa = std.heap.page_allocator;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

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
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) return error.Usage;
    const dir = args[1];
    const prompt = try parseIds(args[2]);
    const n_new = try std.fmt.parseInt(usize, args[3], 10);
    if (n_new < 2) return error.BadArgument; // before the device is opened
    const ks = try parseIds(args[4]);
    var ctx_len: u32 = 4096;
    var plain_run = true;
    var win_rows: u32 = 16; // --prefill N: prompt windows of N rows for the plain run and the MTP prefill alike
    var mtp_dir: ?[]const u8 = null;
    var i: usize = 5;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--ctx")) {
            i += 1;
            ctx_len = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--prefill")) {
            i += 1;
            win_rows = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--no-plain")) plain_run = false else if (std.mem.eql(u8, args[i], "--mtp-from")) {
            i += 1;
            mtp_dir = args[i];
        }
    }
    var r = try rt.open();
    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    if (win_rows > 16) qw.default_rows = win_rows;
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ctx_len);
    m.bf16_logits = true;
    var ml = if (mtp_dir) |d| try ld.Loader.init(gpa, &r, d) else l;
    var mt = try mtp.Mtp.load(gpa, &m, &ml, ctx_len);
    std.debug.print("target {s}; MTP head from {s}; device memory {d:.2} GB\n", .{ dir, mtp_dir orelse dir, @as(f64, @floatFromInt(l.total + if (mtp_dir != null) ml.total else 0)) / 1e9 });
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    const eos = cfg.value.eos_token_id;

    // plain greedy reference (prefill in windows of win_rows, then one token at a time)
    var plain: std.ArrayList(u32) = .empty;
    var plain_ms: f64 = 0;
    if (plain_run) {
        try m.reset(zeros);
        var t: usize = 0;
        var next: u32 = 0;
        while (t < prompt.len) {
            const rows = @min(@max(16, @min(win_rows, m.win.rcap)), prompt.len - t);
            const last = t + rows == prompt.len;
            try m.forwardRows(prompt[t .. t + rows], if (last) 1 else 0, true);
            try r.sync(); // windows read host-written rope/token uploads: no queueing ahead across chunks
            if (last) {
                var g: [1]i32 = undefined;
                try m.rowArgmax(&g);
                next = @intCast(g[0]);
            }
            t += rows;
        }
        try plain.append(gpa, next);
        const t0 = nowNs();
        while (plain.items.len < n_new) {
            try m.forward(next, true);
            next = try m.argmax();
            try plain.append(gpa, next);
        }
        plain_ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        std.debug.print("plain greedy: {d} tokens in {d:.1} ms = {d:.2} tokens/s ({d:.2} ms/token)\n", .{ n_new - 1, plain_ms, @as(f64, @floatFromInt(n_new - 1)) / plain_ms * 1e3, plain_ms / @as(f64, @floatFromInt(n_new - 1)) });
    }
    for (ks) |k| {
        try m.reset(zeros);
        var st: mtp.Stats = .{};
        var drafts: [16]u32 = undefined;
        const pf0 = nowNs();
        const pf = try mtp.prefill(&m, &mt, prompt, k, &drafts, &st, win_rows);
        const pf_ms = @as(f64, @floatFromInt(nowNs() - pf0)) / 1e6;
        var ctx: std.ArrayList(u32) = .empty;
        try ctx.append(gpa, pf.first);
        const t0 = nowNs();
        try mtp.run(&m, &mt, gpa, &ctx, drafts[0..pf.nd], n_new - 1, k, eos, true, &st);
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
        const nw = @as(f64, @floatFromInt(@max(st.windows, 1)));
        std.debug.print("mtp k={d}: {d} tokens in {d:.1} ms = {d:.2} tokens/s; identical to plain: {s}{s}; prefill {d:.1} ms\n", .{ k, ctx.items.len - 1, ms, @as(f64, @floatFromInt(ctx.items.len - 1)) / ms * 1e3, if (!plain_run) "n/a" else if (same) "yes" else "NO (first difference at ", if (!plain_run or same) "" else "token)", pf_ms });
        if (!same and plain_run) {
            std.debug.print("  first difference at generated index {d}; plain", .{first_bad});
            for (plain.items[0..@min(8, plain.items.len)]) |x| std.debug.print(" {d}", .{x});
            std.debug.print("; mtp", .{});
            for (ctx.items[0..@min(8, ctx.items.len)]) |x| std.debug.print(" {d}", .{x});
            std.debug.print("\n", .{});
        }
        std.debug.print("  windows {d}, plain steps {d}, drafted {d}, accepted {d} ({d:.2} per window, {d:.1}% of drafts); tokens per window {d:.2}\n", .{ st.windows, st.plain, st.drafted, st.accepted, @as(f64, @floatFromInt(st.accepted)) / nw, @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(@max(st.drafted, 1))) * 100.0, @as(f64, @floatFromInt(st.tokens)) / nw });
        if (mt.timing) std.debug.print("  MTP step cost (synced): head step {d:.3} ms ({d} steps), absorb step {d:.3} ms ({d} steps)\n", .{ @as(f64, @floatFromInt(mt.t_head_ns)) / 1e6 / @as(f64, @floatFromInt(@max(mt.n_head, 1))), mt.n_head, @as(f64, @floatFromInt(mt.t_abs_ns)) / 1e6 / @as(f64, @floatFromInt(@max(mt.n_abs, 1))), mt.n_abs });
        if (mt.top2_on) {
            std.debug.print("  top-2 simulation per level (reach / first choice right / second choice would be right):", .{});
            for (0..k) |j| std.debug.print(" L{d}: {d} / {d} / {d};", .{ j, st.reach[j], st.hit1[j], st.hit2[j] });
            std.debug.print("\n", .{});
        }
        mt.t_head_ns = 0;
        mt.n_head = 0;
        mt.t_abs_ns = 0;
        mt.n_abs = 0;
        std.debug.print("  cost: verify windows {d:.2} ms each ({d:.1} ms total); MTP {d:.1} ms total ({d} head steps, {d} absorb-only steps); per accepted token {d:.2} ms\n", .{ @as(f64, @floatFromInt(st.win_ns)) / 1e6 / nw, @as(f64, @floatFromInt(st.win_ns)) / 1e6, @as(f64, @floatFromInt(st.mtp_ns)) / 1e6, st.steps, st.absorbs, ms / @as(f64, @floatFromInt(ctx.items.len - 1)) });
    }
}
