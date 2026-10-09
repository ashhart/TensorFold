//! Prefill-vs-decode logit drift on a real sequence. usage: xpu-qwen_drift-test CHECKPOINT IDS_FILE S CHUNK [CHUNK ...]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qw = @import("qwen_xpu").win;
const qg = @import("qwen_xpu").gguf;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn lse(v: []const f32) f64 {
    var mx = v[0];
    for (v) |x| mx = @max(mx, x);
    var s: f64 = 0;
    for (v) |x| s += @exp(@as(f64, x - mx));
    return @as(f64, mx) + @log(s);
}

fn top1(v: []const f32) usize {
    var b: usize = 0;
    for (v, 0..) |x, i| if (x > v[b]) {
        b = i;
    };
    return b;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) return error.Usage;
    const dir = args[1];
    const ids_text = try ld.readFile(gpa, "/", args[2][1..]);
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, ids_text, ", \n");
    while (it.next()) |t| try ids.append(gpa, try std.fmt.parseInt(u32, t, 10));
    const stride = try std.fmt.parseInt(usize, args[3], 10);
    var chunks: std.ArrayList(u32) = .empty;
    var big_rows: u32 = 16;
    for (args[4..]) |a| {
        const c = try std.fmt.parseInt(u32, a, 10);
        try chunks.append(gpa, c);
        big_rows = @max(big_rows, c);
    }
    qw.default_rows = big_rows;

    var r = try rt.open();

    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    const n = ids.items.len;
    var m = try model.Model.load(gpa, &r, &l, cfg.value, @intCast(n + 64));
    const vocab = cfg.value.text_config.vocab_size;
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    const big_ok = m.ops.ex != null or m.ops.gg != null;

    var pos: std.ArrayList(usize) = .empty;
    var p: usize = stride - 1;
    while (p < n) : (p += stride) try pos.append(gpa, p);
    const np = pos.items.len;
    const dec = try gpa.alloc(f32, np * vocab);
    const pre = try gpa.alloc(f32, vocab);

    var t0 = nowNs();
    var j: usize = 0;
    for (ids.items, 0..) |tok, t| {
        const need = j < np and pos.items[j] == t;
        try m.forward(tok, need);
        if (need) {
            try m.fetchLogits(dec[j * vocab ..][0..vocab]);
            j += 1;
        } else try r.sync();
    }
    std.debug.print("decode: {d} tokens in {d:.0} s, {d} scored positions\n", .{ n, @as(f64, @floatFromInt(nowNs() - t0)) / 1e9, np });

    for (chunks.items) |c| {
        try m.reset(zeros);
        t0 = nowNs();
        var maxabs = try gpa.alloc(f32, np);
        var klv = try gpa.alloc(f64, np);
        var same = try gpa.alloc(u8, np);
        var start: usize = 0; // first token not yet fed
        j = 0;
        while (j < np) {
            // feed tokens up to and including position pos[j] (segments end at scored positions)
            const end = pos.items[j] + 1;
            if (big_ok and c > 16) {
                // windows of c rows; the one containing pos[j] keeps its hidden rows
                while (start < end) {
                    const rows = @min(c, n - start);
                    const win_end = start + rows;
                    if (pos.items[j] < win_end) {
                        try m.forwardRows(ids.items[start..win_end], qw.keep_hidden, true);
                        while (j < np and pos.items[j] < win_end) : (j += 1) {
                            try m.headBatch(@intCast(pos.items[j] - start), 1);
                            try m.fetchRowLogits(pre);
                            const d = dec[j * vocab ..][0..vocab];
                            var mx: f32 = 0;
                            for (d, pre) |a, b| mx = @max(mx, @abs(a - b));
                            maxabs[j] = mx;
                            const ld_ = lse(d);
                            const lp = lse(pre);
                            var kl: f64 = 0;
                            for (d, pre) |a, b| {
                                const pa = @as(f64, a) - ld_;
                                kl += @exp(pa) * (pa - (@as(f64, b) - lp));
                            }
                            klv[j] = kl;
                            same[j] = @intFromBool(top1(d) == top1(pre));
                        }
                    } else try m.forwardRows(ids.items[start..win_end], 0, true);
                    start = win_end;
                }
            } else {
                while (start < end) {
                    const rows = @min(@min(c, 16), end - start);
                    const last = start + rows == end;
                    try m.forwardRows(ids.items[start .. start + rows], if (last) 1 else 0, true);
                    start += rows;
                }
                try m.fetchRowLogits(pre);
                const d = dec[j * vocab ..][0..vocab];
                var mx: f32 = 0;
                for (d, pre) |a, b| mx = @max(mx, @abs(a - b));
                maxabs[j] = mx;
                const ld_ = lse(d);
                const lp = lse(pre);
                var kl: f64 = 0;
                for (d, pre) |a, b| {
                    const pa = @as(f64, a) - ld_;
                    kl += @exp(pa) * (pa - (@as(f64, b) - lp));
                }
                klv[j] = kl;
                same[j] = @intFromBool(top1(d) == top1(pre));
                j += 1;
            }
        }
        const secs = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
        std.debug.print("chunk {d}: prefill of {d} tokens + {d} score batches in {d:.1} s\n", .{ c, start, np, secs });
        var b0: usize = 0;
        while (b0 < n) : (b0 += 1024) {
            var cnt: usize = 0;
            var kl_sum: f64 = 0;
            var mxa: f32 = 0;
            var same_n: usize = 0;
            for (pos.items, 0..) |pp, k| if (pp >= b0 and pp < b0 + 1024) {
                cnt += 1;
                kl_sum += klv[k];
                mxa = @max(mxa, maxabs[k]);
                same_n += same[k];
            };
            if (cnt > 0) std.debug.print("  positions {d:>5}-{d:>5}: {d} points, mean KL(decode||prefill) {e:.2}, max |logit diff| {d:.3}, top-1 agree {d}/{d}\n", .{ b0, b0 + 1023, cnt, kl_sum / @as(f64, @floatFromInt(cnt)), mxa, same_n, cnt });
        }
    }
}
