//! tf-cuda-test inkling2-mini-xcheck DIR N: XCHECK json lines (top 2 a step, prompt top 5) for #4337's Metal check.
const std = @import("std");
const core = @import("core");
const ink = @import("inkling2");
const check = @import("check.zig");
const tests = @import("inkling2_mini_tests.zig");
const Gpu = check.Gpu;

const mm = ink.mini_model;
const Pick = ink.mini_ref.Pick;

const Top = struct { id: u32, v: f32 };

/// The k best of `x`, best first (ties: the lower id).
fn top(x: []const f32, comptime k: usize) [k]Top {
    var out: [k]Top = @splat(.{ .id = 0, .v = -std.math.inf(f32) });
    for (x, 0..) |v, i| {
        if (!(v > out[k - 1].v)) continue;
        var at: usize = k - 1;
        while (at > 0 and v > out[at - 1].v) : (at -= 1) out[at] = out[at - 1];
        out[at] = .{ .id = @intCast(i), .v = v };
    }
    return out;
}

fn ids(w: anytype, xs: []const u32) !void {
    try w.writeAll("[");
    for (xs, 0..) |x, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", x });
    try w.writeAll("]");
}

fn pairsGpu(w: anytype, row: []const f32, comptime k: usize) !void {
    const t = top(row, k);
    try w.writeAll("[");
    for (t, 0..) |p, i| try w.print("{s}[{d},{d:.6}]", .{ if (i == 0) "" else ",", p.id, p.v });
    try w.writeAll("]");
}

fn pairsRef(w: anytype, p: Pick, k: usize) !void {
    try w.writeAll("[");
    for (0..k) |i| try w.print("{s}[{d},{d:.6}]", .{ if (i == 0) "" else ",", p.ids[i], p.logits[i] });
    try w.writeAll("]");
}

fn forcedIds(gpa: std.mem.Allocator, name: [:0]const u8) !?[]u32 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, v, ", ");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return try out.toOwnedSlice(gpa);
}

pub fn run(gpu: Gpu, model: []const u8, count_text: []const u8) !void {
    const gpa = gpu.gpa;
    const count = try std.fmt.parseInt(usize, count_text, 10);
    var tok = try core.tokenizer.loadTokenizer(gpu.io, gpa, model);
    defer tok.deinit();
    const b = try tok.encode(gpa, "The capital of France is");
    defer gpa.free(b);
    const prompts = [_]struct { name: []const u8, ids: []const u32 }{
        .{ .name = "A", .ids = &.{ 1, 100, 200, 300, 400, 500, 600, 700 } },
        .{ .name = "B", .ids = b },
    };
    var rig: tests.Rig = undefined;
    try rig.init(gpu, model, 4096, 1);
    defer rig.deinit();
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, gpu.io, model);
    defer ck.close();
    var r: ink.mini_ref.Ref = .{ .gpa = gpa, .c = rig.w.config, .p = rig.w.plan, .ck = &ck, .round_caps = tests.roundCaps() };
    const v = rig.w.config.text.unpadded_vocab;
    var disagree: usize = 0;
    for (prompts) |pr| {
        const n0 = pr.ids.len;
        const all = try gpa.alloc(u32, n0 + count);
        defer gpa.free(all);
        @memcpy(all[0..n0], pr.ids);
        const rows = try gpa.alloc(f32, (count + 1) * v); // the last prompt row's logits, then each step's
        defer gpa.free(rows);
        const first = try rig.prompt(0, pr.ids, 0, mm.prompt_rows);
        @memcpy(rows[0..v], first);
        gpa.free(first);
        // INKLING2_FORCED_A / _B: comma-separated ids fed in place of the argmax (teacher-forced on another engine's)
        const forced = try forcedIds(gpa, if (pr.name[0] == 'A') "INKLING2_FORCED_A" else "INKLING2_FORCED_B");
        defer if (forced) |f| gpa.free(f);
        if (forced) |f| if (f.len < count) return error.ForcedTooShort;
        const next = struct {
            fn at(f: ?[]const u32, row: []const f32, i: usize) u32 {
                return if (f) |ff| ff[i] else tests.argmax(row);
            }
        }.at;
        all[n0] = next(forced, rows[0..v], 0);
        for (1..count) |i| {
            const row = try rig.step(&.{.{ .slot = 0, .p0 = n0 + i - 1, .tokens = all[n0 + i - 1 .. n0 + i] }});
            defer gpa.free(row);
            @memcpy(rows[i * v ..][0..v], row);
            all[n0 + i] = next(forced, row, i);
        }
        const picks = try gpa.alloc(Pick, n0 + count - 1);
        defer gpa.free(picks);
        try r.run(all[0 .. n0 + count - 1], picks);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        var aw: std.Io.Writer.Allocating = .fromArrayList(gpa, &buf);
        const w = &aw.writer;
        try w.print("{{\"prompt\":\"{s}\",\"forced\":{},\"round_caps\":{},\"ids\":", .{ pr.name, forced != null, tests.roundCaps() });
        try ids(w, pr.ids);
        try w.writeAll(",\"tokens\":");
        try ids(w, all[n0..]);
        try w.writeAll(",\"cuda\":{\"last_prompt_top5\":");
        try pairsGpu(w, rows[0..v], 5);
        try w.writeAll(",\"top2\":[");
        for (0..count) |i| {
            if (i > 0) try w.writeAll(",");
            try pairsGpu(w, rows[i * v ..][0..v], 2);
        }
        try w.writeAll("]},\"ref_f64\":{\"last_prompt_top5\":");
        try pairsRef(w, picks[n0 - 1], 5);
        try w.writeAll(",\"top2\":[");
        for (0..count) |i| {
            if (i > 0) try w.writeAll(",");
            try pairsRef(w, picks[n0 - 1 + i], 2);
            if (picks[n0 - 1 + i].id != tests.argmax(rows[i * v ..][0..v])) disagree += 1;
        }
        try w.writeAll("]}}");
        buf = aw.toArrayList();
        std.debug.print("XCHECK {s}\n", .{buf.items});
    }
    std.debug.print("RESULT inkling2 mini xcheck: {d} greedy tokens of 2 x {d} where the float64 reference's top 1 and the engine's differ\n", .{ disagree, count });
}
