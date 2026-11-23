//! Synthetic layers of the release's shapes timed per row: dispatch classes at 1-16 rows and 4 slots x 1 row.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const timing = @import("timing_real.zig");
const timing_fp8 = @import("timing_fp8.zig");

const Ids = host.config.Ids;
const limit = 1280;
const p0 = 1101;

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

/// One layer of each kind the release serves, at its widths: window, full r2, reuse r2, full r1, reindex, reuse r1.
pub fn config(experts: u32) host.config.Config {
    var c = fixture.config();
    const r: host.config.Config = .{};
    inline for (.{ "hidden", "heads", "q_lora", "o_lora", "o_groups", "expert_width", "index_heads", "index_dim", "index_topk", "candidate_blocks" }) |f| @field(c, f) = @field(r, f);
    c.layers = 6;
    c.vocab = 1024;
    c.experts = experts;
    c.compress_ratios = Ids.of(&.{ 0, 2, 2, 1, 1, 1 });
    c.kv_sources = Ids.of(&.{ 1, 3 });
    c.index_sources = Ids.of(&.{ 1, 3, 4 });
    c.candidate_source = 3;
    return c;
}

/// How many release layers each synthetic layer stands for (sums to 40).
const weight = [6]f64{ 2, 3, 15, 1, 4, 15 };

/// Kernel classes a per-row cost is split into, by the pipeline's field path.
const Class = enum { fp8, experts, attention, glue };
const class_names = [_][]const u8{ "FP8 rows", "experts", "attention", "glue" };

fn classOf(name: []const u8) Class {
    const has = struct {
        fn f(n: []const u8, t: []const u8) bool {
            return std.mem.indexOf(u8, n, t) != null;
        }
    }.f;
    if (has(name, "p.fp8") or has(name, "p.tile") or has(name, "p.mma")) return .fp8;
    if (has(name, "p.experts") or has(name, "p.route") or has(name, "p.xmma")) return .experts;
    if (has(name, "p.attn") or has(name, "p.index") or has(name, "p.cand") or has(name, "p.topk512") or has(name, "p.compress")) return .attention;
    return .glue;
}

const Split = struct { gpu: f64, sum: f64, by: [4]f64, n: usize };

fn split(t: timing.Timing) Split {
    var s: Split = .{ .gpu = t.gpu, .sum = t.sum, .by = @splat(0), .n = t.each.len };
    for (t.each, t.each_names) |v, nm| s.by[@backingInt(classOf(nm))] += v;
    return s;
}

/// Every row's input is row 0's, so the rows share their experts (a greedy draft block, or T0 streams).
fn sameRows(r: *fixture.Rig, rows: u32) void {
    const x = r.io[0].slice(u16, rows * 4 * r.c.hidden);
    for (1..rows) |i| @memcpy(x[i * 4 * r.c.hidden ..][0 .. 4 * r.c.hidden], x[0 .. 4 * r.c.hidden]);
    const pre = r.io[2].slice(f32, rows * 4);
    for (1..rows) |i| @memcpy(pre[i * 4 ..][0..4], pre[0..4]);
}

const Shape = struct { slots: u32, rows: u32 };

fn launchOf(r: *fixture.Rig, l: u32, sh: Shape, slots: []pkg.kv.Slot, spans: []pkg.scratch.Span, segs: []pkg.ctx.Seg) timing.Launch {
    const total = sh.slots * sh.rows;
    if (sh.slots == 1) {
        spans[0] = .{ .p0 = p0, .n = sh.rows };
        return .{ .w = &r.layers[l], .ctx = r.ctx(&slots[0], p0, sh.rows), .spans = spans[0..1] };
    }
    for (0..sh.slots) |i| {
        const at: u32 = p0 + 3 * @as(u32, @intCast(i));
        spans[i] = .{ .p0 = at, .n = sh.rows };
        segs[i] = .{ .slot = &slots[i], .p0 = at, .r0 = @intCast(i * sh.rows), .n = sh.rows };
    }
    var x = r.ctx(&slots[0], p0, total);
    x.segs = segs[0..sh.slots];
    return .{ .w = &r.layers[l], .ctx = x, .spans = spans[0..sh.slots] };
}

fn printRow(label: []const u8, sh: Shape, s: Split, base: ?Split) void {
    const rows = sh.slots * sh.rows;
    std.debug.print("  {s:<10} {d:>2}x{d:<2} concurrent {d:>8.1} us, alone {d:>8.1} us:", .{ label, sh.slots, sh.rows, s.gpu * 1e6, s.sum * 1e6 });
    for (class_names, s.by) |nm, v| std.debug.print(" {s} {d:.1}", .{ nm, v * 1e6 });
    if (base) |b| if (rows > 1) {
        const extra: f64 = @floatFromInt(rows - 1);
        std.debug.print(" | a row: {d:.1}", .{(s.gpu - b.gpu) / extra * 1e6});
        for (class_names, s.by, b.by) |nm, v, w| std.debug.print(" {s} {d:.1}", .{ nm, (v - w) / extra * 1e6 });
    };
    std.debug.print("\n", .{});
}

test "dsv41 metal synthetic release-shape layers timed by dispatch class at 1-16 rows and 4 slots x 1 row" {
    const n = if (env("TF_DSV41_TIMING_N")) |v| try std.fmt.parseInt(usize, v, 10) else 7;
    const experts = if (env("TF_DSV41_SYNTH_EXPERTS")) |v| try std.fmt.parseInt(u32, v, 10) else 16;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try fixture.Rig.initMode(std.heap.page_allocator, std.testing.io, config(experts), limit, 5, .mapped);
    defer r.deinit();
    const junk = try r.device.buffer(timing.flush_bytes, mtl.ResourceOptions.private);
    defer junk.deinit();
    // TF_DSV41_SYNTH_FP8: each FP8 projection alone, the row kernel against the tile at every simdgroup count
    if (env("TF_DSV41_SYNTH_FP8") != null) return timing_fp8.run(r, junk, n);
    var names: timing.Names = .{};
    try names.walk(a, pkg.pipelines.Pipelines, &r.p, "p");
    var slots: [4]pkg.kv.Slot = undefined;
    slots[0] = r.slot;
    for (slots[1..]) |*s| s.* = try pkg.kv.Slot.init(std.heap.page_allocator, r.device, &r.c, limit);
    defer for (slots[1..]) |*s| s.deinit();
    for (&slots, 0..) |*s, i| fixture.fillPast(&r.c, s, p0 + 3 * @as(u32, @intCast(i)), 21 + i);
    const same = env("TF_DSV41_TIMING_DIFFERENT") == null;
    // TF_DSV41_SYNTH_ROWS: a comma list of one slot's row counts (default 1,2,4,8,16), then 4 slots x 1 row
    var list: [17]Shape = undefined;
    var len: usize = 0;
    var it = std.mem.tokenizeScalar(u8, env("TF_DSV41_SYNTH_ROWS") orelse "1,2,4,8,16", ',');
    while (it.next()) |tok| : (len += 1) list[len] = .{ .slots = 1, .rows = try std.fmt.parseInt(u32, tok, 10) };
    list[len] = .{ .slots = 4, .rows = 1 };
    const shapes = list[0 .. len + 1];
    var round: [17]Split = @splat(.{ .gpu = 0, .sum = 0, .by = @splat(0), .n = 0 });
    std.debug.print("synthetic release-shape layers, {d} experts, {s} rows, fastest of {d} (us):\n", .{ experts, if (same) "identical" else "different", n });
    for (0..r.c.layers) |li| {
        const l: u32 = @intCast(li);
        std.debug.print("layer {d} ({s}, ratio {d}):\n", .{ l, @tagName(r.c.mode(l)), r.c.compressRatio(l) });
        var base: ?Split = null;
        for (shapes, 0..) |sh, i| {
            fixture.fillInput(r, sh.slots * sh.rows, 31);
            if (same) sameRows(r, sh.slots * sh.rows);
            var spans: [16]pkg.scratch.Span = undefined;
            var segs: [16]pkg.ctx.Seg = undefined;
            const t = try timing.time(a, r, junk, &names, launchOf(r, l, sh, &slots, &spans, &segs), n);
            const s = split(t);
            if (i == 0) base = s;
            if (env("TF_DSV41_TIMING_EACH") != null) t.print(l, sh.slots * sh.rows, n, true);
            printRow("", sh, s, base);
            round[i].gpu += weight[li] * s.gpu;
            round[i].sum += weight[li] * s.sum;
            for (&round[i].by, s.by) |*x, v| x.* += weight[li] * v;
        }
    }
    std.debug.print("40-layer round (layers weighted by kind; no head, Engram or host):\n", .{});
    for (shapes, round[0..shapes.len]) |sh, s| printRow("round", sh, s, round[0]);
}
