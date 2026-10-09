//! Greedy generation for prompt files, one model load. usage: xpu-qwen_genbatch-test CKPT PROMPTS.ids OUT N_NEW [opts]

const std = @import("std");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;
const qw = @import("qwen_xpu").win;
const core = @import("kl_core.zig");
const stop = @import("xpu").stop;

fn parseLine(gpa: std.mem.Allocator, s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, std.mem.trim(u8, t, " \r"), 10));
    return out.toOwnedSlice(gpa);
}

fn top2(logits: []const f32) struct { id: u32, gap: f32 } {
    var b1: f32 = -std.math.inf(f32);
    var b2: f32 = -std.math.inf(f32);
    var id: u32 = 0;
    for (logits, 0..) |v, i| {
        if (v > b1) {
            b2 = b1;
            b1 = v;
            id = @intCast(i);
        } else if (v > b2) b2 = v;
    }
    return .{ .id = id, .gap = b1 - b2 };
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !u8 {
    stop.install();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print("usage: xpu-qwen_genbatch-test CHECKPOINT PROMPTS.ids OUT N_NEW [--force REF.ids] [--prefill ROWS] [--ctx N]\n", .{});
        return 2;
    }
    const dir = args[1];
    const n_new = try std.fmt.parseInt(u32, args[4], 10);
    var force_path: ?[]const u8 = null;
    var prefill: u32 = 512;
    var ctx: u32 = 32768 + 512;
    var i: usize = 5;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--force")) {
            i += 1;
            force_path = args[i];
        } else if (std.mem.eql(u8, args[i], "--prefill")) {
            i += 1;
            prefill = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, args[i], "--ctx")) {
            i += 1;
            ctx = try std.fmt.parseInt(u32, args[i], 10);
        } else return error.BadArgument;
    }
    const out_path = try std.fmt.allocPrintSentinel(gpa, "{s}", .{args[3]}, 0);
    const prompts_txt = try core.slurp(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}", .{args[2]}, 0));
    const ref_txt: []u8 = if (force_path) |p| try core.slurp(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}", .{p}, 0)) else &.{};
    const done_txt = try core.slurp(gpa, out_path);
    var done: usize = 0;
    for (done_txt) |ch| done += @intFromBool(ch == '\n');

    var r = try rt.open();
    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    const vocab: usize = cfg.value.text_config.vocab_size;
    if (prefill > 16) qw.default_rows = prefill;
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ctx);
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    const logits = try gpa.alloc(f32, vocab * 16);

    var plines = std.mem.splitScalar(u8, prompts_txt, '\n');
    var rlines = std.mem.splitScalar(u8, ref_txt, '\n');
    var idx: usize = 0;
    var line: std.ArrayList(u8) = .empty;
    while (plines.next()) |pl| {
        const rl = rlines.next() orelse "";
        if (pl.len == 0) continue;
        defer idx += 1;
        if (idx < done) continue;
        if (stop.requested()) return error.Interrupted;
        const prompt = try parseLine(gpa, pl);
        const forced: ?[]u32 = if (force_path != null) try parseLine(gpa, rl) else null;
        if (prompt.len + n_new > ctx) return error.ContextTooSmall;
        try m.reset(zeros);
        const t0 = core.nowNs();
        var t: usize = 0;
        while (t < prompt.len) { // prompt windows; the last window's last row carries the logits
            const rows = @min(prefill, prompt.len - t);
            const last = t + rows == prompt.len;
            try m.forwardRows(prompt[t .. t + rows], if (last) 1 else 0, true);
            t += rows;
        }
        line.clearRetainingCapacity();
        var step: u32 = 0;
        var prev: u32 = 0;
        const steps: u32 = if (forced) |f| @min(n_new, @as(u32, @intCast(f.len))) else n_new; // a reference that stopped at EOS is shorter
        while (step < steps) : (step += 1) {
            var tok: u32 = undefined;
            var gap: f32 = 0;
            if (step == 0) {
                try m.fetchRowLogits(logits);
                const tt = top2(logits[0..vocab]);
                tok = tt.id;
                gap = tt.gap;
            } else {
                const fed: u32 = if (forced) |f| f[step - 1] else prev;
                try m.forward(fed, true);
                try m.fetchLogits(logits[0..vocab]);
                const tt = top2(logits[0..vocab]);
                tok = tt.id;
                gap = tt.gap;
            }
            if (step > 0) try line.append(gpa, ',');
            var tmp: [48]u8 = undefined;
            const s = if (forced != null) try std.fmt.bufPrint(&tmp, "{d}/{d:.4}", .{ tok, gap }) else try std.fmt.bufPrint(&tmp, "{d}", .{tok});
            try line.appendSlice(gpa, s);
            prev = tok;
        }
        try line.append(gpa, '\n');
        try core.appendFile(out_path, line.items);
        std.debug.print("prompt {d}: {d} prompt tokens, {d} new in {d:.1} s\n", .{ idx, prompt.len, n_new, @as(f64, @floatFromInt(core.nowNs() - t0)) / 1e9 });
    }
    return 0;
}

