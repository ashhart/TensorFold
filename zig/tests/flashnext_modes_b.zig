const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const tf = @import("tensorfold");
const fz = tf.flashnext_replay;

const opts = fz.opts;
const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const CAP = fz.CAP;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const Buf = fz.Buf;
const Entry = fz.Entry;
const Variant = fz.Variant;
const Site = fz.Site;
const readAll = fz.readAll;
const Run = fz.Run;
const TOP = fz.TOP;
const KW = fz.KW;
const Select = fz.Select;
const GSelect = fz.GSelect;
const Hc = fz.Hc;
const Layer = fz.Layer;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const Ple = fz.Ple;
const Tmp = fz.Tmp;
const f32Buf = fz.f32Buf;
const i32Buf = fz.i32Buf;
const Mtp = fz.Mtp;
const Model = fz.Model;
const DepthRule = fz.DepthRule;
const copyDrafts = fz.copyDrafts;
const PMAX = fz.PMAX;
const Prompt = fz.Prompt;
const jsonInt = fz.jsonInt;

/// FZ_PROMPT_N: only the prompt's first n tokens (all when unset).
fn promptCut() !usize {
    const v = platform.c.getenv("FZ_PROMPT_N") orelse return std.math.maxInt(usize);
    return std.fmt.parseInt(usize, std.mem.span(v), 10);
}

pub fn profile(r: *Run, m: *Model, ref: std.json.Value, io: anytype, arena: std.mem.Allocator) !void {
    if (platform.c.getenv("TF_FLASHNEXT_TP")) |path| r.tp = try fz.Tp2.init(arena, io, r.device, std.mem.span(path));
    var hcskips: std.ArrayList(u32) = .empty; // FZ_HCSKIP=1,2,4: the profile again under each hc knock-out
    if (platform.c.getenv("FZ_HCSKIP")) |v| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |x| try hcskips.append(arena, try std.fmt.parseInt(u32, x, 10));
    } else try hcskips.append(arena, 0);
    const names = [_][]const u8{ "none", "hc", "dense", "experts", "router", "gdn", "attn", "ple", "head", "tp" };
    var toks: [MAXR]u32 = undefined;
    for (0..MAXR) |i| toks[i] = @intCast(ref.object.get("prompt").?.array.items[i].integer);
    var pk: [MAXR]u32 = undefined;
    m.reset();
    for (0..3) |_| try m.window(toks[0..1], &pk);
    if (platform.c.getenv("FZ_PROFILE_AB") != null) { // in-model A/B, interleaved: recorded, core lane, fz_gdn, both
        const gpipe = try fz.gdn_step.compile(r, false);
        const arms = [_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true }, .{ true, true } };
        for ([_]usize{ 1, 2, 4, 6, 8, 16 }) |rows| {
            var ms: [4]f64 = @splat(0);
            for (0..5) |_| for (arms, 0..) |arm, a| {
                r.lane_new = arm[0];
                r.gdn_pipe = if (arm[1]) gpipe else null;
                m.gpu_seconds = 0;
                for (0..20) |_| try m.window(toks[0..rows], &pk);
                ms[a] += m.gpu_seconds * 1e3 / 100;
            };
            std.debug.print("rows {d:2}: window ms recorded {d:6.3}, core lane {d:6.3}, fz_gdn {d:6.3}, both {d:6.3}\n", .{ rows, ms[0], ms[1], ms[2], ms[3] });
        }
        return;
    }
    var ups: std.ArrayList(bool) = .empty; // FZ_HC_UP=0,1: the profile with the recorded up projection, then fz_hc_up
    if (platform.c.getenv("FZ_HC_UP")) |v| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |x| try ups.append(arena, !std.mem.eql(u8, x, "0"));
    } else try ups.append(arena, false);
    var prow: std.ArrayList(usize) = .empty; // FZ_PROFILE_ROWS=2,3: the windows' widths (default 1, 4, 8)
    if (platform.c.getenv("FZ_PROFILE_ROWS")) |v| {
        var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
        while (it.next()) |x| try prow.append(arena, try std.fmt.parseInt(usize, x, 10));
    } else try prow.appendSlice(arena, &.{ 1, 4, 8 });
    for (ups.items) |up| for (hcskips.items) |hs| for (prow.items) |rows| {
        r.hc_up = up;
        r.hcskip = hs;
        var base: f64 = 0;
        for (names, 0..) |name, c| {
            const only = platform.c.getenv("FZ_PROFILE_ONLY") != null; // every class but this one knocked out
            r.skip = if (c == 0) (if (only) 0x1ff else 0) else if (only) 0x1ff & ~(@as(u32, 1) << @intCast(c - 1)) else @as(u32, 1) << @intCast(c - 1);
            m.gpu_seconds = 0;
            for (0..20) |_| try m.window(toks[0..rows], &pk);
            const ms = m.gpu_seconds * 1e3 / 20;
            if (c == 0) base = ms;
            std.debug.print("hc_up {d} hcskip {d} rows {d}: without {s:8} {d:6.2} ms GPU  ({d:5.2} ms)\n", .{ @intFromBool(up), hs, rows, name, ms, base - ms });
        }
    };
    r.hcskip = 0;
    r.skip = 0;
    for ([_]usize{ 1, 4, 8 }) |rows| { // the MTP head: a chain step (one row) and a window's catch-up
        m.gpu_seconds = 0;
        for (0..20) |_| {
            m.mtp.drafted = 0;
            _ = try m.mtpRun(toks[0..rows], m.mtp.last);
            m.mtp.pos -= rows;
        }
        std.debug.print("mtp rows {d}: {d:6.2} ms GPU\n", .{ rows, m.gpu_seconds * 1e3 / 20 });
    }
    return;
}

// two lane groups on two queues: one window, two in a row, two at once
pub fn dual(r: *Run, m: *Model, ref: std.json.Value, device: mtl.Device) !void {
    const want0 = ref.object.get("tokens").?.array.items;
    const pr = ref.object.get("prompt").?.array.items;
    var pk: [MAXR]u32 = undefined;
    m.reset();
    for (pr) |x| {
        try m.window(&.{@intCast(x.integer)}, &pk);
        m.keepRows(&.{@intCast(x.integer)}, 1);
    }
    const q2 = try device.queue();
    const ta = m.t;
    var tb = m.t;
    const info = @typeInfo(Tmp).@"struct";
    inline for (info.field_names, info.field_types) |fname, ftype| {
        const f = .{ .name = fname, .type = ftype };
        if (f.type == Buf) {
            const old = @field(ta, f.name);
            const n = old.b.length() - old.off;
            const nb = try r.buffer(n);
            @memcpy(nb.contents()[0..n], old.b.contents()[old.off .. old.off + n]);
            @field(tb, f.name) = .{ .b = nb };
        } else if (f.type == [2]Buf) {
            for (0..2) |k| {
                const old = @field(ta, f.name)[k];
                const n = old.b.length() - old.off;
                @field(tb, f.name)[k] = .{ .b = try r.buffer(n) };
            }
        }
    }
    for ([_]usize{ 1, 4, 8 }) |rows| {
        var toks: [MAXR]u32 = undefined;
        for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
        m.windowMeta(rows);
        const ids = ta.ids8.b.slice(u32, MAXR);
        for (0..MAXR) |i| ids[i] = if (i < rows) toks[i] else 0;
        m.pleIds(toks[0..rows]);
        var toks_b: [MAXR]u32 = undefined; // the second group: other tokens at the same positions
        for (0..rows) |i| toks_b[i] = @intCast(want0[16 + i].integer);
        const ids_b = tb.ids8.b.slice(u32, MAXR);
        for (0..MAXR) |i| ids_b[i] = if (i < rows) toks_b[i] else 0;
        m.t = tb;
        m.pleIds(toks_b[0..rows]);
        m.t = ta;
        var wall: [3]f64 = undefined;
        for (0..3) |mode| {
            const n_it: usize = 20;
            const a = mtl.clock.seconds();
            for (0..n_it) |_| {
                const cb1 = r.queue.commandBuffer();
                r.enc = cb1.compute(.serial);
                m.t = ta;
                try m.windowEncode(rows, ta.ids8);
                r.enc.end();
                if (mode == 0) {
                    cb1.commit();
                    cb1.wait();
                    continue;
                }
                const cb2 = (if (mode == 1) r.queue else q2).commandBuffer();
                r.enc = cb2.compute(.serial);
                m.t = tb;
                try m.windowEncode(rows, tb.ids8);
                r.enc.end();
                cb1.commit();
                cb2.commit();
                cb1.wait();
                cb2.wait();
            }
            wall[mode] = (mtl.clock.seconds() - a) * 1e3 / @as(f64, @floatFromInt(n_it));
        }
        m.t = ta;
        std.debug.print("{d} lanes: one group {d:.2} ms, two groups in a row {d:.2} ms ({d:.2}x), two groups at once {d:.2} ms ({d:.2}x)\n", .{ rows, wall[0], wall[1], wall[1] / wall[0], wall[2], wall[2] / wall[0] });
    }
    return;
}

// fz_hc_up against the recorded up projection: every row's logits and streams, bit for bit
pub fn hcCheck(r: *Run, m: *Model, ref: std.json.Value, gpa: std.mem.Allocator) !void {
    const want0 = ref.object.get("tokens").?.array.items;
    const pr = ref.object.get("prompt").?.array.items;
    var pk: [MAXR]u32 = undefined;
    m.reset();
    for (pr) |x| {
        try m.window(&.{@intCast(x.integer)}, &pk);
        m.keepRows(&.{@intCast(x.integer)}, 1);
    }
    const keep = try gpa.alloc(u8, MAXR * VOCAB * 2);
    defer gpa.free(keep);
    const keep_h = try gpa.alloc(u8, MAXR * WIDE * 2);
    defer gpa.free(keep_h);
    const Diff = struct {
        fn of(a: []const u8, b: Buf, n: usize) usize {
            var d: usize = 0;
            const c = b.b.contents()[b.off .. b.off + 2 * n];
            for (0..n) |i| d += @intFromBool(a[2 * i] != c[2 * i] or a[2 * i + 1] != c[2 * i + 1]);
            return d;
        }
    };
    var toks: [MAXR]u32 = undefined;
    for (0..MAXR) |i| toks[i] = @intCast(want0[i].integer);
    var bad: usize = 0;
    for (1..MAXR + 1) |rows| {
        r.hc_up = false;
        try m.window(toks[0..rows], &pk);
        @memcpy(keep[0 .. rows * VOCAB * 2], m.t.logits.b.contents()[0 .. rows * VOCAB * 2]);
        @memcpy(keep_h[0 .. rows * WIDE * 2], m.last.b.contents()[m.last.off .. m.last.off + rows * WIDE * 2]);
        r.hc_up = true;
        try m.window(toks[0..rows], &pk);
        const dl = Diff.of(keep, m.t.logits, rows * VOCAB);
        const dh = Diff.of(keep_h, m.last, rows * WIDE);
        std.debug.print("rows {d}: {d} of {d} logits and {d} of {d} stream values differ\n", .{ rows, dl, rows * VOCAB, dh, rows * WIDE });
        bad += dl + dh;
    }
    std.debug.print("hc check: {s}\n", .{if (bad == 0) "bit-identical" else "DIFFERENT"});
    return;
}

// grouped or fused experts against fz_xgu/fz_xdown: every row's logits, bit for bit
pub fn gCheck(r: *Run, m: *Model, ref: std.json.Value, gpa: std.mem.Allocator) !void {
    const want0 = ref.object.get("tokens").?.array.items;
    const pr = ref.object.get("prompt").?.array.items;
    var pk: [MAXR]u32 = undefined;
    m.reset();
    for (pr) |x| {
        try m.window(&.{@intCast(x.integer)}, &pk);
        m.keepRows(&.{@intCast(x.integer)}, 1);
    }
    const keep = try gpa.alloc(u16, MAXR * VOCAB);
    defer gpa.free(keep);
    for (1..MAXR + 1) |rows| {
        var toks: [MAXR]u32 = undefined;
        for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
        const g0, const f0, const x0 = .{ r.grouped, r.xfused, r.xsx };
        r.grouped, r.xfused, r.xsx = .{ false, false, false };
        try m.window(toks[0..rows], &pk);
        @memcpy(keep[0 .. rows * VOCAB], m.t.logits.b.slice(u16, rows * VOCAB));
        r.grouped, r.xfused, r.xsx = .{ g0, f0, x0 };
        try m.window(toks[0..rows], &pk);
        const now = m.t.logits.b.slice(u16, rows * VOCAB);
        var diff: usize = 0;
        for (keep[0 .. rows * VOCAB], now) |a, b| diff += @intFromBool(a != b);
        std.debug.print("rows {d}: {d} of {d} logits differ\n", .{ rows, diff, rows * VOCAB });
    }
    return;
}

// the experts' share of a window: chain rows against one token repeated
pub fn xtime(r: *Run, m: *Model, ref: std.json.Value) !void {
    const want0 = ref.object.get("tokens").?.array.items;
    var pk: [MAXR]u32 = undefined;
    m.reset();
    for (0..3) |_| try m.window(&.{@intCast(want0[0].integer)}, &pk);
    if (platform.c.getenv("FZ_GPARTS") != null) for ([_]usize{ 2, 8 }) |rows| {
        var toks: [MAXR]u32 = undefined;
        for (0..rows) |i| toks[i] = @intCast(want0[i].integer);
        for ([_]u32{ 0, 1, 4, 6 }) |gs| {
            r.gskip = gs;
            m.gpu_seconds = 0;
            for (0..30) |_| try m.window(toks[0..rows], &pk);
            std.debug.print("rows {d} gskip {d}: window {d:6.2} ms\n", .{ rows, gs, m.gpu_seconds * 1e3 / 30 });
        }
        r.gskip = 0;
    };
    for ([_]usize{ 1, 2, 4, 8 }) |rows| {
        for ([_]bool{ false, true }) |same| {
            var toks: [MAXR]u32 = undefined;
            for (0..rows) |i| toks[i] = @intCast(want0[if (same) 0 else i].integer);
            var ms: [2]f64 = undefined;
            for ([_]u32{ 0, 1 << 2 }, 0..) |skip, j| {
                r.skip = skip;
                m.gpu_seconds = 0;
                for (0..30) |_| try m.window(toks[0..rows], &pk);
                ms[j] = m.gpu_seconds * 1e3 / 30;
            }
            r.skip = 0;
            std.debug.print("rows {d} {s}: window {d:6.2} ms, experts {d:5.2} ms\n", .{ rows, if (same) "one token" else "chain    ", ms[0], ms[0] - ms[1] });
        }
    }
    return;
}

// the prompt path against the exact 8-row windows, then one-row decode
pub fn prefill(r: *Run, m: *Model, ref: std.json.Value, gpa: std.mem.Allocator, args: anytype) !void {
    const all_items = ref.object.get("prompt").?.array.items;
    const pr_items = all_items[0..@min(all_items.len, try promptCut())]; // FZ_PROMPT_N: the prompt's first n tokens
    const toks = try gpa.alloc(u32, pr_items.len);
    defer gpa.free(toks);
    for (pr_items, 0..) |x, i| toks[i] = @intCast(x.integer);
    if (platform.c.getenv("FZ_SATTN_CHECK") != null) { // prompt attention: the decode's kernels vs the tensor units
        var pr = try Prompt.init(r, args[2], r.xnew_header);
        if (platform.c.getenv("FZ_STEP")) |v| pr.step = try std.fmt.parseInt(usize, std.mem.span(v), 10);
        const ab0: []const u8 = if (platform.c.getenv("FZ_AB")) |v| std.mem.span(v) else "attn";
        const seg2 = std.mem.eql(u8, ab0, "seg2");
        // seg2: chunks vs staggered segments (core/segments.zig); FZ_SEGS=2 engine rule, 1 one, 3-4 that many
        const nseg: usize = if (platform.c.getenv("FZ_SEGS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2;
        const seg_min: usize = if (platform.c.getenv("FZ_SEG_MIN")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 1;
        if (seg2 and (nseg == 0 or nseg > tf.segments.MAX)) return error.Segments;
        var extra: [3]Prompt = undefined;
        var ps: [4]*Prompt = undefined;
        ps[0] = &pr;
        if (seg2) for (1..nseg) |k| {
            extra[k - 1] = try pr.sibling();
            ps[k] = &extra[k - 1];
        };
        var outs: [2][48]u32 = undefined;
        const logits: [2][]u16 = .{ try gpa.alloc(u16, VOCAB), try gpa.alloc(u16, VOCAB) };
        var best: [2]f64 = .{ 1e9, 1e9 };
        const ab: []const u8 = if (platform.c.getenv("FZ_AB")) |v| std.mem.span(v) else "attn";
        for (0..2) |arm| { // arm 0 the old kernels, arm 1 the new: FZ_AB attn (default), scan or tiles
            if (std.mem.eql(u8, ab, "scan")) pr.scan4 = arm == 1 else if (std.mem.eql(u8, ab, "tiles")) pr.tall_tiles = arm == 1 else if (std.mem.eql(u8, ab, "gu")) pr.fused_gu = arm == 1 else if (std.mem.eql(u8, ab, "bm32")) pr.expert_bm = if (arm == 1) 32 else 0 else if (std.mem.eql(u8, ab, "scan8")) pr.scan8 = arm == 1 else if (std.mem.eql(u8, ab, "seg2")) {} else pr.fast_attn = arm == 1;
            for (0..3) |run| {
                m.reset();
                const c0 = mtl.clock.seconds();
                var at: usize = 0;
                var first: u32 = 0;
                while (at < toks.len) {
                    const left = toks.len - at;
                    if (seg2 and arm == 1) {
                        const c: tf.segments.Call = switch (nseg) {
                            1 => .{ .rows = @min(pr.step, left), .parts = 1 },
                            2 => tf.segments.next(left, pr.step, seg_min),
                            else => .{ .rows = @min(nseg * pr.step, left), .parts = (@min(nseg * pr.step, left) + pr.step - 1) / pr.step },
                        };
                        first = try Prompt.chunkN(ps[0..c.parts], m, gpa, toks[at .. at + c.rows]);
                        at += c.rows;
                    } else {
                        const n = @min(pr.step, left);
                        first = try pr.chunk(m, gpa, toks[at .. at + n]);
                        at += n;
                    }
                }
                best[arm] = @min(best[arm], mtl.clock.seconds() - c0);
                if (run > 0) continue;
                @memcpy(logits[arm], m.t.logits.b.slice(u16, VOCAB));
                var pk: [MAXR]u32 = undefined;
                outs[arm][0] = first;
                for (1..48) |k| {
                    try m.window(&.{outs[arm][k - 1]}, &pk);
                    m.keepRows(&.{outs[arm][k - 1]}, 1);
                    outs[arm][k] = pk[0];
                }
            }
        }
        var worst: f64 = 0;
        for (logits[0], logits[1]) |x, y| {
            const fx: f64 = @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)));
            const fy: f64 = @floatCast(@as(f32, @bitCast(@as(u32, y) << 16)));
            worst = @max(worst, @abs(fx - fy));
        }
        var same: usize = 0;
        while (same < 48 and outs[0][same] == outs[1][same]) same += 1;
        std.debug.print("prompt {d} tokens, chunks of {d}: old {d:.3} s ({d:.0} tok/s), new {d:.3} s ({d:.0} tok/s)\n", .{ toks.len, pr.step, best[0], @as(f64, @floatFromInt(toks.len)) / best[0], best[1], @as(f64, @floatFromInt(toks.len)) / best[1] });
        std.debug.print("first token {d} vs {d}; last-row logits max diff {d:.4}; 48 decoded after each: {d} equal\n", .{ outs[0][0], outs[1][0], worst, same });
        std.debug.print("  old {any}\n  new {any}\n", .{ outs[0][0..16], outs[1][0..16] });
        return;
    }
    var pk: [MAXR]u32 = undefined;
    const n_dec: usize = 48;
    // reference: windows up to 8 rows (each row equals a one-row step); FZ_PF_REF=n stops after n tokens (0: none)
    m.reset();
    const a0 = mtl.clock.seconds();
    var at: usize = 0;
    var last_n: usize = 1;
    const ref_upto: usize = if (platform.c.getenv("FZ_PF_REF")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else toks.len;
    while (at < @min(ref_upto, toks.len)) {
        const n: usize = @min(MAXR, toks.len - at);
        try m.window(toks[at .. at + n], &pk);
        m.keepRows(toks[at .. at + n], n);
        at += n;
        last_n = n;
    }
    const a1 = mtl.clock.seconds();
    const ref_logits = try gpa.alloc(u16, VOCAB);
    defer gpa.free(ref_logits);
    @memcpy(ref_logits, m.t.logits.b.slice(u16, MAXR * VOCAB)[(last_n - 1) * VOCAB .. last_n * VOCAB]);
    var want_t: std.ArrayList(u32) = .empty;
    var margins: [64]f32 = @splat(0);
    try want_t.append(gpa, pk[last_n - 1]);
    while (want_t.items.len < n_dec) {
        const tk = want_t.items[want_t.items.len - 1];
        try m.window(&.{tk}, &pk);
        m.keepRows(&.{tk}, 1);
        { // the top-two logit margin of this step's pick
            var top1: f32 = -std.math.inf(f32);
            var top2: f32 = -std.math.inf(f32);
            for (m.t.logits.b.slice(u16, VOCAB)) |raw| {
                const v: f32 = @bitCast(@as(u32, raw) << 16);
                if (v > top1) {
                    top2 = top1;
                    top1 = v;
                } else if (v > top2) top2 = v;
            }
            margins[want_t.items.len] = top1 - top2;
        }
        try want_t.append(gpa, pk[0]);
    }
    // the prompt path
    var pr = try Prompt.init(r, args[2], r.xnew_header);
    if (platform.c.getenv("FZ_STEP")) |v| pr.step = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    m.reset();
    const b0 = mtl.clock.seconds();
    var first: u32 = 0;
    at = 0;
    while (at < toks.len) {
        const n: usize = @min(pr.step, toks.len - at);
        first = try pr.chunk(m, gpa, toks[at .. at + n]);
        at += n;
    }
    const b1 = mtl.clock.seconds();
    const lg = m.t.logits.b.slice(u16, VOCAB);
    var worst: f64 = 0;
    var scale: f64 = 0;
    for (lg, ref_logits) |x, y| {
        const fx: f64 = @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)));
        const fy: f64 = @floatCast(@as(f32, @bitCast(@as(u32, y) << 16)));
        worst = @max(worst, @abs(fx - fy));
        scale = @max(scale, @abs(fy));
    }
    var got_t: std.ArrayList(u32) = .empty;
    try got_t.append(gpa, first);
    while (got_t.items.len < n_dec) {
        const tk = got_t.items[got_t.items.len - 1];
        try m.window(&.{tk}, &pk);
        m.keepRows(&.{tk}, 1);
        try got_t.append(gpa, pk[0]);
    }
    var same: usize = 0;
    while (same < n_dec and got_t.items[same] == want_t.items[same]) same += 1;
    std.debug.print("prompt {d} tokens: 8-row windows {d:.3} s ({d:.0} tok/s); prompt path {d:.3} s ({d:.0} tok/s)\n", .{ toks.len, a1 - a0, @as(f64, @floatFromInt(toks.len)) / (a1 - a0), b1 - b0, @as(f64, @floatFromInt(toks.len)) / (b1 - b0) });
    std.debug.print("first token {d} vs {d}; last-row logits max diff {d:.4} (max |logit| {d:.2}); decode after it: {d}/{d} tokens equal\n", .{ first, want_t.items[0], worst, scale, same, n_dec });
    if (same < n_dec) std.debug.print("  the exact path's top-two margin where they part (token {d}): {d:.4}; median margin {d:.3}\n", .{ same, margins[same], blk: {
        var sorted = margins;
        std.mem.sort(f32, sorted[1..n_dec], {}, std.sort.asc(f32));
        break :blk sorted[n_dec / 2];
    } });
    try pr.classes(m, @min(toks.len, pr.step));
    const skips = [_]u32{ 0, 4 | 1024, 1024, 2, 16, 1 };
    const what = [_][]const u8{ "none", "attention", "sparse attention", "DeltaNet", "hyper-connections", "experts" };
    var base_s: f64 = 0;
    for (skips, what) |sk, w| {
        pr.skip = sk;
        var best: f64 = 1e9;
        for (0..3) |_| {
            m.reset();
            m.gpu_seconds = 0;
            const c0 = mtl.clock.seconds();
            at = 0;
            while (at < toks.len) {
                const n: usize = @min(pr.step, toks.len - at);
                _ = try pr.chunk(m, gpa, toks[at .. at + n]);
                at += n;
            }
            best = @min(best, mtl.clock.seconds() - c0);
        }
        if (sk == 0) base_s = best;
        std.debug.print("warm, without {s:18}: {d:.3} s ({d:.0} tok/s), saves {d:.3} s\n", .{ w, best, @as(f64, @floatFromInt(toks.len)) / best, base_s - best });
    }
    pr.skip = 0;
    return;
}
