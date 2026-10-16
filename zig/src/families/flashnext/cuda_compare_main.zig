//! Our own glue against a captured set on one loaded engine: every prompt row's scores, then each path's greedy reply.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const prompt = flash.prompt;
const vocab = prompt.vocabulary;

/// Prompt rows a teacher-forced pass takes at once (every row's scores come back).
const rows_at_once = 256;

fn score(raw: u16) f64 {
    return @as(f32, @bitCast(@as(u32, raw) << 16));
}

/// What two paths' scores over the same rows say: top-1 agreement, the largest score gap, KL and each path's NLL.
const Tally = struct {
    rows: usize = 0,
    agree: usize = 0,
    max_abs: f64 = 0,
    kl: f64 = 0,
    kl_max: f64 = 0,
    nll_a: f64 = 0,
    nll_b: f64 = 0,
    scored: usize = 0,
    /// Among rows whose top-1 differs, the largest gap between the captured path's best two scores.
    tie_gap: f64 = 0,
    /// Each row's KL, for its median and the share of rows past 0.1, and each scored row's NLL own minus captured.
    kls: std.ArrayList(f64) = .empty,
    dnll: std.ArrayList(f64) = .empty,

    fn row(t: *Tally, a: []const u16, b: []const u16, next: ?u32) void {
        var best_a: usize = 0;
        var best_b: usize = 0;
        var max_a = -std.math.inf(f64);
        var max_b = -std.math.inf(f64);
        for (a, b, 0..) |ra, rb, i| {
            const va = score(ra);
            const vb = score(rb);
            t.max_abs = @max(t.max_abs, @abs(va - vb));
            if (va > max_a) {
                max_a = va;
                best_a = i;
            }
            if (vb > max_b) {
                max_b = vb;
                best_b = i;
            }
        }
        var sum_a: f64 = 0;
        var sum_b: f64 = 0;
        for (a, b) |ra, rb| {
            sum_a += @exp(score(ra) - max_a);
            sum_b += @exp(score(rb) - max_b);
        }
        const lse_a = max_a + @log(sum_a);
        const lse_b = max_b + @log(sum_b);
        var kl: f64 = 0;
        for (a, b) |ra, rb| {
            const la = score(ra) - lse_a;
            kl += @exp(la) * (la - (score(rb) - lse_b));
        }
        t.rows += 1;
        t.kl += kl;
        t.kls.append(std.heap.page_allocator, kl) catch {};
        t.kl_max = @max(t.kl_max, kl);
        if (best_a == best_b) {
            t.agree += 1;
        } else {
            var second = -std.math.inf(f64);
            for (a, 0..) |ra, i| if (i != best_a) {
                second = @max(second, score(ra));
            };
            t.tie_gap = @max(t.tie_gap, max_a - second);
        }
        if (next) |n| {
            t.nll_a += lse_a - score(a[n]);
            t.nll_b += lse_b - score(b[n]);
            t.dnll.append(std.heap.page_allocator, (lse_b - score(b[n])) - (lse_a - score(a[n]))) catch {};
            t.scored += 1;
        }
    }
};

/// One line of a tally: agreement, score gaps, KL (mean, median, max, rows past 0.1) and each path's perplexity.
fn report(what: []const u8, t: *Tally) void {
    if (t.rows == 0) return;
    const n: f64 = @floatFromInt(t.rows);
    const s: f64 = @floatFromInt(@max(t.scored, 1));
    std.mem.sort(f64, t.kls.items, {}, std.sort.asc(f64));
    std.mem.sort(f64, t.dnll.items, {}, std.sort.asc(f64));
    var past: usize = 0;
    for (t.kls.items) |v| past += @intFromBool(v > 0.1);
    const mid = if (t.dnll.items.len > 0) t.dnll.items[t.dnll.items.len / 2] else 0;
    std.debug.print("{s}: {d} rows, top-1 agree {d}/{d}, max |score diff| {d:.4}, KL(captured||own) mean {e:.3} median {e:.3} max {e:.3}, {d} rows past 0.1, perplexity captured {d:.4} own {d:.4} over {d} next tokens (median NLL own - captured {e:.3}), largest top-2 gap where top-1 differs {d:.4}\n", .{ what, t.rows, t.agree, t.rows, t.max_abs, t.kl / n, t.kls.items[t.kls.items.len / 2], t.kl_max, past, @exp(t.nll_a / s), @exp(t.nll_b / s), t.scored, mid, t.tie_gap });
}

/// Rows from this position on attend through the indexer's picks (512 blocks of 4 keys).
const sparse_from = 2048;

fn scores(raw: []const u8) []const u16 {
    return @alignCast(std.mem.bytesAsSlice(u16, raw));
}

fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \t\r\n");
    while (it.next()) |part| try ids.append(gpa, try std.fmt.parseInt(u32, part, 10));
    return ids.toOwnedSlice(gpa);
}

fn envCount(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: flashnext-compare <model-dir> <kernel-dir> <ids-file>... (COUNT greedy tokens, default 32)\n", .{});
        return 2;
    }
    const count = envCount("COUNT", 32);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer set.deinit();
    var glue = try flash.Glue.load(&driver);
    defer glue.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const captured: flash.Tri = .{ .set = &set, .s = stream };
    // TF_FLASHNEXT_OWNED: only the entries it names run ours on this side, the rest the captured set's
    const own: flash.Tri = if (glue.mixed != 0) .{ .set = &set, .own = &glue, .s = stream } else .{ .own = &glue, .s = stream };
    var e = try prompt.Engine.init(gpa, io, &driver, &stream, args[1]);
    defer e.deinit();
    var out_a = try cuda.DeviceBuffer.alloc(&driver, rows_at_once * vocab * 2);
    defer out_a.free();
    var out_b = try cuda.DeviceBuffer.alloc(&driver, rows_at_once * vocab * 2);
    defer out_b.free();
    const host_a = try gpa.alloc(u16, rows_at_once * vocab);
    defer gpa.free(host_a);
    const host_b = try gpa.alloc(u16, rows_at_once * vocab);
    defer gpa.free(host_b);
    for (args[3..]) |path| {
        const ids = try readIds(gpa, io, path);
        defer gpa.free(ids);
        var seq_a = try e.newSeq(ids.len + count);
        defer seq_a.deinit(gpa);
        var seq_b = try e.newSeq(ids.len + count);
        defer seq_b.deinit(gpa);
        // every prompt row's scores on both paths, the next prompt token's NLL under each; dense rows and sparse ones
        var t: Tally = .{};
        var dense: Tally = .{};
        var sparse: Tally = .{};
        var at: usize = 0;
        while (at < ids.len) {
            const m = @min(rows_at_once, ids.len - at);
            try e.forward(flash.Tri, captured, &seq_a, ids[at..][0..m], .{ .logits = .all, .logits_out = out_a.ptr });
            try e.forward(flash.Tri, own, &seq_b, ids[at..][0..m], .{ .logits = .all, .logits_out = out_b.ptr });
            try stream.synchronize();
            try out_a.download(0, std.mem.sliceAsBytes(host_a[0 .. m * vocab]));
            try out_b.download(0, std.mem.sliceAsBytes(host_b[0 .. m * vocab]));
            for (0..m) |r| {
                const next: ?u32 = if (at + r + 1 < ids.len) ids[at + r + 1] else null;
                t.row(host_a[r * vocab ..][0..vocab], host_b[r * vocab ..][0..vocab], next);
                if (ids.len > sparse_from) (if (at + r < sparse_from) &dense else &sparse).row(host_a[r * vocab ..][0..vocab], host_b[r * vocab ..][0..vocab], next);
            }
            at += m;
        }
        var label: [96]u8 = undefined;
        report(try std.fmt.bufPrint(&label, "rows {s}", .{std.fs.path.basename(path)}), &t);
        report(try std.fmt.bufPrint(&label, "rows {s} before {d}", .{ std.fs.path.basename(path), sparse_from }), &dense);
        report(try std.fmt.bufPrint(&label, "rows {s} from {d}", .{ std.fs.path.basename(path), sparse_from }), &sparse);
        // each path's own greedy reply from the prompt, and where they part
        const reply_a = try gpa.alloc(u32, count);
        defer gpa.free(reply_a);
        const reply_b = try gpa.alloc(u32, count);
        defer gpa.free(reply_b);
        var g: Tally = .{};
        var parted: ?usize = null;
        for (0..count) |i| {
            reply_a[i] = prompt.argmax(seq_a.logits);
            reply_b[i] = prompt.argmax(seq_b.logits);
            // the same tokens so far: the same context on both paths, until (and at) the token where they part
            if (parted == null) g.row(scores(seq_a.logits), scores(seq_b.logits), null);
            if (parted == null and reply_a[i] != reply_b[i]) parted = i;
            if (i + 1 == count) break;
            try e.predict(flash.Tri, captured, &seq_a, reply_a[i..][0..1]);
            try e.predict(flash.Tri, own, &seq_b, reply_b[i..][0..1]);
        }
        std.debug.print("reply {s}: {d} greedy tokens, ", .{ std.fs.path.basename(path), count });
        if (parted) |p| std.debug.print("part at token {d} (top-2 gap on the captured path there {d:.4})", .{ p, g.tie_gap }) else std.debug.print("all equal", .{});
        std.debug.print(", KL up to there mean {e:.3}\n", .{g.kl / @as(f64, @floatFromInt(@max(g.rows, 1)))});
        std.debug.print("tokens captured", .{});
        for (reply_a) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\ntokens own", .{});
        for (reply_b) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\n", .{});
    }
    return 0;
}
