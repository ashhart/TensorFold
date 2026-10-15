//! `tf-qwen35-cuda <model> <step|window|tree|prompt|mrope> <ids> <out>`, or `vision <fixtures> <out> [min mean] [min row]`:
//! rows to compare; vision exits 1 when the tower's rows fall under its bounds.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const is_vision = args.len >= 3 and std.mem.eql(u8, args[2], "vision");
    if (args.len != 5 and !(is_vision and args.len <= 7)) {
        std.debug.print("usage: tf-qwen35-cuda <model dir> <step|window|tree|prompt|mrope> <ids> <out>\n" ++
            "       tf-qwen35-cuda <model dir> vision <fixtures> <out> [min mean cosine, 0.999] [min row cosine, 0.94]\n", .{});
        return 2;
    }
    if (is_vision) {
        const bounds: Bounds = .{
            .mean = if (args.len > 5) try std.fmt.parseFloat(f64, args[5]) else 0.999,
            .row = if (args.len > 6) try std.fmt.parseFloat(f64, args[6]) else 0.94,
        };
        return vision(init, a, args[1], args[3], args[4], bounds);
    }
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, args[3], ", ");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1]);
    defer e.deinit();
    // TF_Q35_PREFILL=N starts the stream after N cycled ids; TF_Q35_BESIDE=N steps an N-token stream beside it
    const prefill = try envCount("TF_Q35_PREFILL");
    const beside = try envCount("TF_Q35_BESIDE");
    var cache = try E.Cache.init(init.gpa, e, 4096 + prefill);
    defer cache.deinit(init.gpa);
    try fill(e, &cache, ids.items, prefill);
    var other = try E.Cache.init(init.gpa, e, 4096 + beside);
    defer other.deinit(init.gpa);
    try fill(e, &other, ids.items, beside);
    var f: Fwd = .{ .e = e, .other = if (beside > 0) &other else null, .ids = ids.items };
    var out: std.ArrayList(u8) = .empty;
    if (std.mem.eql(u8, args[2], "step")) {
        var first: i64 = 0;
        var rest: i64 = 0;
        for (ids.items, 0..) |id, i| {
            const t0 = std.Io.Clock.awake.now(init.io).toNanoseconds();
            try f.run(.{ .cache = &cache, .rows = 1 }, &.{id});
            const dt: i64 = @intCast(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0);
            if (i == 0) first = dt else rest += dt;
            @memcpy(cache.logits, e.rowLogits(0));
            try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
        }
        const n: f64 = @floatFromInt(ids.items.len - 1);
        std.debug.print("{d} steps: the first {d:.1} ms, then {d:.1} ms a step ({d:.1} tok/s)\n", .{ ids.items.len, @as(f64, @floatFromInt(first)) / 1e6, @as(f64, @floatFromInt(rest)) / 1e6 / n, n * 1e9 / @as(f64, @floatFromInt(rest)) });
    } else if (std.mem.eql(u8, args[2], "window")) {
        // windows of 1, 2, 3, ... rows: every row's logits, which must equal step's bit for bit
        var at: usize = 0;
        var w: usize = 1;
        while (at < ids.items.len) : (w = w % e.window_rows + 1) {
            const n = @min(w, ids.items.len - at);
            try f.run(.{ .cache = &cache, .rows = n }, ids.items[at..][0..n]);
            for (0..n) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
            at += n;
        }
    } else if (std.mem.eql(u8, args[2], "tree")) {
        // tree windows of growing widths, junk branches beside the kept true one, whose logits must equal step's bits
        var at: usize = 0;
        var w: usize = 2;
        while (at < ids.items.len) : (w = w % e.window_rows + 1) {
            var toks: [E.most_window]u32 = undefined;
            var parents: [E.most_window]i32 = undefined;
            var path: [E.most_window]u32 = undefined;
            toks[0] = ids.items[at];
            parents[0] = -1;
            path[0] = 0;
            var n: usize = 1;
            var kept: usize = 1;
            var last: i32 = 0;
            var depth: usize = 1;
            while (n + 2 <= w and at + depth < ids.items.len) : (depth += 1) {
                toks[n] = (ids.items[at + depth] + 7) % @as(u32, @intCast(e.g.vocab));
                parents[n] = last;
                toks[n + 1] = ids.items[at + depth];
                parents[n + 1] = last;
                path[kept] = @intCast(n + 1);
                last = @intCast(n + 1);
                n += 2;
                kept += 1;
            }
            if (n < w and n >= 3) {
                toks[n] = 1;
                parents[n] = 1;
                n += 1;
            }
            try f.run(.{ .cache = &cache, .rows = n, .parents = parents[0..n] }, toks[0..n]);
            for (path[0..kept]) |r| try out.appendSlice(a, std.mem.sliceAsBytes(e.rowLogits(r)));
            if (n > 1) try cache.keep(e, path[0..kept]);
            at += kept;
        }
    } else {
        // mrope: chunks through an image prompt's table (text positions, no image rows): prompt's bits
        const table = std.mem.eql(u8, args[2], "mrope");
        const n_ids = ids.items.len;
        const src = try a.alloc(i32, n_ids);
        @memset(src, -1);
        const pos = try a.alloc(i32, 3 * n_ids);
        // TF_Q35_SHIFT=k moves the h axis by k: a control whose logits must differ, as the table is read
        const shift = try envCount("TF_Q35_SHIFT");
        for (0..3) |ax| for (0..n_ids) |i| {
            pos[ax * n_ids + i] = @intCast(cache.len + i + if (ax == 1) shift else 0);
        };
        var none = try cuda.DeviceBuffer.alloc(ctx.d, 256);
        defer none.free();
        const im: E.Images = .{ .features = none.ptr, .src = src, .pos = pos };
        const p0 = cache.len;
        var at: usize = 0;
        while (at < n_ids) {
            const n = @min(E.max_rows, n_ids - at);
            if (table) {
                // the table's rows are the prompt's from position p0: shift it so row `at` is the cache's next
                const part: E.Images = .{ .features = im.features, .src = try padded(a, src, p0), .pos = try paddedPos(a, pos, p0, n_ids) };
                try e.forward(&.{.{ .cache = &cache, .rows = n, .images = &part }}, ids.items[at..][0..n], .last);
            } else try e.forward(&.{.{ .cache = &cache, .rows = n }}, ids.items[at..][0..n], .last);
            at += n;
        }
        try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = out.items });
    return 0;
}

/// One forward keeping logits: the stream's part first (its rows' logits come first), then the second stream's step.
const Fwd = struct {
    e: *E.Engine,
    other: ?*E.Cache,
    ids: []const u32,
    n: usize = 0,

    fn run(f: *Fwd, part: E.Part, toks: []const u32) !void {
        const o = f.other orelse return f.e.forward(&.{part}, toks, .logits);
        var all: [E.most_window + 1]u32 = undefined;
        @memcpy(all[0..toks.len], toks);
        all[toks.len] = f.ids[f.n % f.ids.len];
        f.n += 1;
        try f.e.forward(&.{ part, .{ .cache = o, .rows = 1 } }, all[0 .. toks.len + 1], .logits);
    }
};

fn envCount(name: [:0]const u8) !usize {
    const v = std.c.getenv(name) orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10);
}

/// `n` positions of the ids cycled, through prompt forwards.
fn fill(e: *E.Engine, cache: *E.Cache, ids: []const u32, n: usize) !void {
    var buf: [E.max_rows]u32 = undefined;
    var at: usize = 0;
    while (at < n) {
        const k = @min(E.max_rows, n - at);
        for (buf[0..k], at..) |*t, i| t.* = ids[i % ids.len];
        try e.forward(&.{.{ .cache = cache, .rows = k }}, buf[0..k], .last);
        at += k;
    }
}

/// What the tower's rows must reach against 0.6.6's bf16 rows, each image: their mean cosine and their worst row's.
/// With a fixture's fp32 reference rows (qwen_vision_fixtures.py --fp32), the tower must also be as close to them as
/// 0.6.6's bf16 tower is, within `slack` on the mean, so a fragile tower (the 2B's) is judged against the reference.
const Bounds = struct { mean: f64, row: f64, slack: f64 = 0.001 };

/// Rows' cosine, mean and worst, of bf16 `got` against `want` (bf16, or fp32 when `want32`), and the identical rows.
const Cosines = struct { mean: f64, least: f64, same: usize };

fn cosines(got: []align(1) const u16, want: []const u8, want32: bool, rows: usize, width: usize) Cosines {
    var sum: f64 = 0;
    var least: f64 = 1;
    var same: usize = 0;
    for (0..rows) |r| {
        var dot: f64 = 0;
        var ng: f64 = 0;
        var nw: f64 = 0;
        var equal = !want32;
        for (0..width) |i| {
            const x = bf(got[r * width + i]);
            const y: f64 = if (want32) @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, want[(r * width + i) * 4 ..][0..4], .little)))) else bf(std.mem.readInt(u16, want[(r * width + i) * 2 ..][0..2], .little));
            if (!want32 and got[r * width + i] != std.mem.readInt(u16, want[(r * width + i) * 2 ..][0..2], .little)) equal = false;
            dot += x * y;
            ng += x * x;
            nw += y * y;
        }
        const cos = dot / @sqrt(ng * nw);
        sum += cos;
        least = @min(least, cos);
        same += @intFromBool(equal);
    }
    return .{ .mean = sum / @as(f64, @floatFromInt(rows)), .least = least, .same = same };
}

/// Each fixture's patches (tools/zig/qwen_vision_fixtures.py) through the tower: rows' cosine to 0.6.6's, bounded.
fn vision(init: std.process.Init, a: std.mem.Allocator, model: []const u8, dir: []const u8, out_path: []const u8, bounds: Bounds) !u8 {
    const V = q.vision;
    const io = init.io;
    const cfg = try V.Config.read(a, io, model) orelse return error.QwenCheckpointHasNoVision;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const tower = try V.Tower.load(init.gpa, io, ctx.d, stream, model, cfg);
    defer tower.deinit();
    const cwd = std.Io.Dir.cwd();
    const manifest = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(1 << 20));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, manifest, .{});
    var out: std.ArrayList(u8) = .empty;
    var worst: f64 = 1;
    var failed: usize = 0;
    for (parsed.object.get("images").?.array.items) |im| {
        const o = im.object;
        const grid = o.get("grid").?.array.items;
        const g: V.Grid = .{ .h = @intCast(grid[0].integer), .w = @intCast(grid[1].integer) };
        const tokens: usize = @intCast(o.get("tokens").?.integer);
        const px_bytes = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, o.get("pixels").?.string }), a, .limited(1 << 30));
        const want = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, o.get("features").?.string }), a, .limited(1 << 30));
        const px = try a.alloc(f32, px_bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(px), px_bytes);
        var features = try cuda.DeviceBuffer.alloc(ctx.d, tokens * cfg.out_hidden * 2);
        defer features.free();
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        try tower.encode(px, &.{g}, features.ptr);
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
        const got = try a.alloc(u8, tokens * cfg.out_hidden * 2);
        try features.download(0, got);
        try out.appendSlice(a, got);
        if (want.len != got.len) {
            std.debug.print("{s}: 0.6.6 has {d} bytes of rows, the tower {d}\n", .{ o.get("name").?.string, want.len, got.len });
            return 1;
        }
        const gw: []align(1) const u16 = std.mem.bytesAsSlice(u16, got);
        const name = o.get("name").?.string;
        const c = cosines(gw, want, false, tokens, cfg.out_hidden);
        worst = @min(worst, c.least);
        std.debug.print("{s}: grid {d}x{d}, {d} rows, row cosine mean {d:.5} min {d:.5}, {d} rows identical, {d:.1} ms", .{ name, g.h, g.w, tokens, c.mean, c.least, c.same, ms });
        var ok = c.mean >= bounds.mean and c.least >= bounds.row;
        if (o.get("reference")) |ref| {
            // both bf16 towers against 0.6.6's fp32 rows: this one may not be the further, beyond the slack
            const r32 = try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, ref.string }), a, .limited(1 << 31));
            if (r32.len != tokens * cfg.out_hidden * 4) {
                std.debug.print("\n{s}: the fp32 reference has {d} bytes, {d} rows need {d}\n", .{ name, r32.len, tokens, tokens * cfg.out_hidden * 4 });
                return 1;
            }
            const ours = cosines(gw, r32, true, tokens, cfg.out_hidden);
            const theirs = cosines(std.mem.bytesAsSlice(u16, want), r32, true, tokens, cfg.out_hidden);
            std.debug.print("; against fp32: mean {d:.5} (0.6.6 bf16 {d:.5})", .{ ours.mean, theirs.mean });
            // with a reference, the absolute bounds give way to it: 0.6.6's own bf16 rows can sit under them
            ok = ours.mean + bounds.slack >= theirs.mean;
        }
        std.debug.print("{s}\n", .{if (ok) "" else "  FAIL"});
        failed += @intFromBool(!ok);
    }
    try cwd.writeFile(io, .{ .sub_path = out_path, .data = out.items });
    std.debug.print("worst row cosine {d:.5}; {d} image{s} outside the bounds (mean {d}, row {d}, fp32 slack {d})\n", .{ worst, failed, if (failed == 1) "" else "s", bounds.mean, bounds.row, bounds.slack });
    return if (failed == 0) 0 else 1;
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// `src` behind `p0` rows of text (-1): the table indexed by the cache's position, as a prompt from position 0 is.
fn padded(a: std.mem.Allocator, src: []const i32, p0: usize) ![]i32 {
    const out = try a.alloc(i32, p0 + src.len);
    @memset(out[0..p0], -1);
    @memcpy(out[p0..], src);
    return out;
}

/// `pos` ([3][n]) behind `p0` rows, each axis: the positions indexed by the cache's position.
fn paddedPos(a: std.mem.Allocator, pos: []const i32, p0: usize, n: usize) ![]i32 {
    const m = p0 + n;
    const out = try a.alloc(i32, 3 * m);
    for (0..3) |ax| {
        for (0..p0) |i| out[ax * m + i] = @intCast(i);
        @memcpy(out[ax * m + p0 ..][0..n], pos[ax * n ..][0..n]);
    }
    return out;
}
