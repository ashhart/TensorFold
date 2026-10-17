//! `tf-qwen35-cuda <model> <step|window|tree|prompt|mrope|probe|mtp|vision> ...`: rows to compare; usage below.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const q = @import("qwen35");
const E = q.engine;
const vl = core.qwen_vl;

const usage =
    \\usage: tf-qwen35-cuda <model dir> <step|window|tree|prompt|mrope> <ids> <out> [fp8]
    \\       tf-qwen35-cuda <model dir> probe <ids> <layer>: a routed model's routing and rows at that layer
    \\       tf-qwen35-cuda <model dir> mtp <ids> <out>: the MTP drafter's tree regressions after a prompt
    \\       tf-qwen35-cuda <model dir> vision <out dir> <tokens> [--bound MEAN WORST] <picture or NAME.f32@HxW>...
    \\vision prepares each picture with core/qwen_vl under a <tokens> budget and runs the CUDA tower twice: <out dir>
    \\gets NAME.pixels.f32, NAME.rows.f32 (bf16 widened) and meta.json; NAME.ref.f32 there (float32 reference rows) is
    \\compared row by row (relative L2, worst and mean); --bound needs every reference and exits 1 past either bound.
    \\fp8: the attention caches as e4m3 rows (--kv-cache fp8); TF_Q35_PREFILL, _BESIDE, _DELTA, _RESUME, _SHIFT: below.
    \\       tf-qwen35-cuda <model dir> kl <ids file> <out.json> <skip> <rows> [control]
    \\kl: bf16 caches against fp8 ones, teacher-forced: the first <skip> ids as prompt chunks, then <rows> ids in
    \\windows, each row's KL(bf16 || fp8) of the next-token distributions, top-1 agreement and both losses on the next
    \\id; control: bf16 against bf16 whose every other window rotates three positions on (a uniform shift cancels).
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len >= 3 and std.mem.eql(u8, args[2], "vision")) {
        const bounded = args.len > 5 and std.mem.eql(u8, args[5], "--bound");
        const first: usize = if (bounded) 8 else 5;
        if (args.len <= first) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const bound: ?[2]f64 = if (bounded) .{ try std.fmt.parseFloat(f64, args[6]), try std.fmt.parseFloat(f64, args[7]) } else null;
        return vision(init, a, args[1], args[3], try std.fmt.parseInt(u32, args[4], 10), bound, args[first..]);
    }
    if (args.len >= 3 and std.mem.eql(u8, args[2], "kl")) {
        if (args.len != 7 and args.len != 8) {
            std.debug.print("{s}", .{usage});
            return 2;
        }
        const skip = try std.fmt.parseInt(usize, args[5], 10);
        const control = args.len == 8 and std.mem.eql(u8, args[7], "control");
        return kl(init, a, args[1], args[3], args[4], skip, try std.fmt.parseInt(usize, args[6], 10), control);
    }
    if (args.len != 5 and !(args.len == 6 and std.mem.eql(u8, args[5], "fp8"))) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, args[3], ", ");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1], args.len == 6);
    defer e.deinit();
    if (std.mem.eql(u8, args[2], "mtp")) return mtpCheck(init.gpa, init.io, e, args[1], ids.items);
    // TF_Q35_PREFILL=N starts the stream after N cycled ids; TF_Q35_BESIDE=N steps an N-token stream beside it
    const prefill = try envCount("TF_Q35_PREFILL");
    const beside = try envCount("TF_Q35_BESIDE");
    var cache = try E.Cache.init(init.gpa, e, 4096 + prefill);
    defer cache.deinit(init.gpa);
    try fill(e, &cache, ids.items, prefill);
    // TF_Q35_DELTA=d: the rows after the prefill rotate at position + d, as an image prompt's reply does
    if (std.c.getenv("TF_Q35_DELTA")) |v| cache.delta = try std.fmt.parseInt(i32, std.mem.span(v), 10);
    // TF_Q35_RESUME=1: the stream goes on from a kept copy of its state, restored into another (poisoned) cache
    const resume_at = try envCount("TF_Q35_RESUME");
    if (resume_at > 0 and !std.mem.eql(u8, args[2], "prompt") and !std.mem.eql(u8, args[2], "mrope")) try resumeFrom(init.gpa, e, &cache);
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
    } else if (std.mem.eql(u8, args[2], "probe")) {
        // the first id through layers 0..N (N = the out argument): that layer's routing and rows, for a reference
        const layer = try std.fmt.parseInt(usize, args[4], 10);
        const all = e.layers;
        e.layers = all[0 .. layer + 1];
        e.graphs.clearRetainingCapacity(); // the warm-up captured every layer; this forward captures the cut
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, ids.items[0..1], .logits);
        e.layers = all;
        const g = e.g;
        if (!g.moe()) return error.NotRouted;
        const count = g.experts + 1;
        const logits_f = try a.alloc(f32, count);
        const picks = try a.alloc(i32, g.slots());
        const wts = try a.alloc(f32, g.slots());
        var xrow: [8]u16 = undefined;
        var rrow: [8]u16 = undefined;
        try e.arena.download(e.routed.logits - e.arena.ptr, std.mem.sliceAsBytes(logits_f));
        try e.arena.download(e.routed.picks - e.arena.ptr, std.mem.sliceAsBytes(picks));
        try e.arena.download(e.routed.wts - e.arena.ptr, std.mem.sliceAsBytes(wts));
        try e.arena.download(e.x - e.arena.ptr, std.mem.sliceAsBytes(&xrow));
        try e.arena.download(e.r - e.arena.ptr, std.mem.sliceAsBytes(&rrow));
        std.debug.print("probe layer {d} x[:8]", .{layer});
        for (xrow) |h| std.debug.print(" {d:.5}", .{@as(f32, @bitCast(@as(u32, h) << 16))});
        std.debug.print("\nprobe top logits", .{});
        for (0..10) |_| {
            var best: usize = 0;
            for (logits_f[0 .. count - 1], 0..) |v, i| if (v > logits_f[best]) {
                best = i;
            };
            std.debug.print(" ({d}, {d:.5})", .{ best, logits_f[best] });
            logits_f[best] = -std.math.inf(f32);
        }
        std.debug.print(" shared gate {d:.5}\nprobe picks", .{logits_f[count - 1]});
        for (picks) |pk| std.debug.print(" {d}", .{pk});
        std.debug.print(" weights", .{});
        for (wts) |w| std.debug.print(" {d:.5}", .{w});
        std.debug.print("\nprobe r[:8]", .{});
        for (rrow) |h| std.debug.print(" {d:.5}", .{@as(f32, @bitCast(@as(u32, h) << 16))});
        std.debug.print("\n", .{});
        return 0;
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
            // TF_Q35_RESUME=K in prompt and mrope: the pass goes on from a kept copy at position K
            if (resume_at > 0 and cache.len == resume_at) try resumeFrom(init.gpa, e, &cache);
        }
        try out.appendSlice(a, std.mem.sliceAsBytes(cache.logits));
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = out.items });
    return 0;
}

/// `cache` replaced by a fresh one, poisoned, that takes a kept copy of its state (cuda_reuse.zig's save and restore).
fn resumeFrom(gpa: std.mem.Allocator, e: *E.Engine, cache: *E.Cache) !void {
    const copy = try q.reuse.saveCache(e, cache, cache.len, &.{});
    defer q.reuse.dropCopy(e, copy);
    var fresh = try E.Cache.init(gpa, e, cache.cap);
    errdefer fresh.deinit(gpa);
    try fresh.state.fill8(0x5a, e.stream.handle);
    try fresh.kv.fill8(0x5a, e.stream.handle);
    try q.reuse.restoreCache(e, &fresh, copy, &.{});
    if (fresh.len != cache.len or fresh.delta != cache.delta) return error.ResumeMismatch;
    cache.deinit(gpa);
    cache.* = fresh;
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

/// The MTP drafter's tree regressions (cuda_mtp_check.zig) after a prompt over the ids fills the target's tap ring.
fn mtpCheck(gpa: std.mem.Allocator, io: std.Io, e: *E.Engine, dir: []const u8, ids: []const u32) !u8 {
    const d = try q.mtp.Mtp.init(gpa, io, e, dir, 4096);
    defer d.deinit();
    try e.setTaps(d.taps());
    var cache = try E.Cache.init(gpa, e, 4096);
    defer cache.deinit(gpa);
    try fill(e, &cache, ids, ids.len);
    return if (try q.mtp_check.run(gpa, d, cache.tap.?.ptr, ids)) 0 else 1;
}

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

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ?[]const u8 {
    const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch null;
}

/// Each picture prepared as the server prepares it, through the tower twice (bit-equal), against its reference.
fn vision(init: std.process.Init, a: std.mem.Allocator, model: []const u8, out: []const u8, tokens: u32, bound: ?[2]f64, pictures: []const [:0]const u8) !u8 {
    const V = q.vision;
    const io = init.io;
    const config = read(a, io, model, "config.json") orelse return error.NoConfig;
    const pre = read(a, io, model, "preprocessor_config.json");
    const proc = read(a, io, model, "processor_config.json");
    const vc = (try vl.Config.parse(a, config, pre, proc)) orelse return error.NoVisionTower;
    const cfg = try V.Config.of(a, vc, config);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const tower = try V.Tower.load(init.gpa, io, ctx.d, stream, model, cfg);
    defer tower.deinit();
    const width = cfg.out_hidden;
    var meta: std.ArrayList(u8) = .empty;
    try meta.appendSlice(a, "{");
    var failed = false;
    for (pictures, 0..) |pic, n| {
        const p = prepared(a, io, pic, vc, @min(tokens, V.max_patches / (V.merge * V.merge))) catch |err| {
            std.debug.print("{s}: {t}\n", .{ pic, err });
            return 1;
        };
        const count = @as(usize, p.tokens) * width;
        var features = try cuda.DeviceBuffer.alloc(ctx.d, count * 2);
        defer features.free();
        const first = try a.alloc(u16, count);
        const again = try a.alloc(u16, count);
        var ms: f64 = 0;
        for ([_][]u16{ first, again }) |dst| {
            const t0 = std.Io.Clock.awake.now(io);
            try tower.encode(p.pixels, &.{.{ .h = p.gh, .w = p.gw }}, features.ptr);
            ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
            try features.download(0, std.mem.sliceAsBytes(dst));
        }
        const same = std.mem.eql(u16, first, again);
        const wide = try a.alloc(f32, count);
        for (wide, first) |*w, b| w.* = @bitCast(@as(u32, b) << 16);
        const base = std.fs.path.basename(pic);
        const name = base[0 .. std.mem.indexOfScalar(u8, base, '.') orelse base.len];
        const fid = compare(a, io, out, name, wide, width);
        const ref: [2]f64 = if (fid) |f| .{ f.worst, f.mean } else .{ -1, -1 };
        if (bound) |b| {
            const ok = if (fid) |f| f.mean <= b[0] and f.worst <= b[1] else false;
            const why = if (fid == null) "no reference of this shape" else "past it";
            if (!ok) std.debug.print("{s}: FAIL the bound (mean {d}, worst {d}): {s}\n", .{ name, b[0], b[1], why });
            failed = failed or !ok;
        }
        const cwd = std.Io.Dir.cwd();
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.pixels.f32", .{ out, name }), .data = std.mem.sliceAsBytes(p.pixels) });
        try cwd.writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.rows.f32", .{ out, name }), .data = std.mem.sliceAsBytes(wide) });
        const sha: [64]u8 = if (fid) |f| std.fmt.bytesToHex(f.sha, .lower) else @splat('-');
        try meta.print(a, "{s}\"{s}\": {{\"gh\": {d}, \"gw\": {d}, \"tokens\": {d}, \"hash\": \"{x}\", \"ms\": {d:.1}, \"repeat_equal\": {}, \"ref_worst\": {d:.6}, \"ref_mean\": {d:.6}, \"ref_sha256\": \"{s}\"}}", .{ if (n > 0) ", " else "", name, p.gh, p.gw, p.tokens, p.hash, ms, same, ref[0], ref[1], sha });
        std.debug.print("{s}: grid {d} x {d}, {d} tokens, tower {d:.1} ms, second pass bit-equal {}, against the reference worst {d:.6} mean {d:.6}\n", .{ name, p.gh, p.gw, p.tokens, ms, same, ref[0], ref[1] });
        if (!same) return 1;
    }
    try meta.appendSlice(a, "}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/meta.json", .{out}), .data = meta.items });
    return if (failed) 1 else 0;
}

/// bf16 caches against fp8 ones (or a control) on the same ids: each window row's KL and top-1, summed up into `out`.
fn kl(init: std.process.Init, a: std.mem.Allocator, model: []const u8, ids_path: []const u8, out: []const u8, skip: usize, rows: usize, control: bool) !u8 {
    const io = init.io;
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, try std.Io.Dir.cwd().readFileAlloc(io, ids_path, a, .limited(64 << 20)), ", \n");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    if (skip + rows >= ids.items.len) return error.FixtureTooShort;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const eb = try E.Engine.init(init.gpa, io, &ctx, model, false);
    defer eb.deinit();
    const e8 = try E.Engine.init(init.gpa, io, &ctx, model, !control);
    defer e8.deinit();
    var cb = try E.Cache.init(init.gpa, eb, skip + rows + 64);
    defer cb.deinit(init.gpa);
    var c8 = try E.Cache.init(init.gpa, e8, skip + rows + 64);
    defer c8.deinit(init.gpa);
    try fill(eb, &cb, ids.items[0..skip], skip);
    try fill(e8, &c8, ids.items[0..skip], skip);
    const kls = try a.alloc(f64, rows);
    var top1: usize = 0;
    var loss: [2]f64 = .{ 0, 0 };
    var at: usize = 0;
    while (at < rows) {
        const n = @min(eb.window_rows, rows - at);
        const w = ids.items[skip + at ..][0..n];
        try eb.forward(&.{.{ .cache = &cb, .rows = n }}, w, .logits);
        try e8.forward(&.{.{ .cache = &c8, .rows = n }}, w, .logits);
        for (0..n) |r| {
            const d = divergence(eb.rowLogits(r), e8.rowLogits(r), ids.items[skip + at + r + 1]);
            kls[at + r] = d.kl;
            top1 += @intFromBool(d.same_top);
            for (&loss, d.loss) |*l, x| l.* += x;
        }
        at += n;
        if (control) c8.delta = 3 - c8.delta;
    }
    var sum: f64 = 0;
    for (kls) |x| sum += x;
    std.mem.sort(f64, kls, {}, std.sort.asc(f64));
    const nr: f64 = @floatFromInt(rows);
    const report = try std.fmt.allocPrint(a, "{{\"rows\": {d}, \"skip\": {d}, \"control\": {}, \"kl_mean\": {d:.6}, \"kl_p99\": {d:.6}, \"kl_max\": {d:.6}, \"top1\": {d:.5}, \"loss_bf16\": {d:.5}, \"loss_other\": {d:.5}}}\n", .{ rows, skip, control, sum / nr, kls[(rows * 99) / 100], kls[rows - 1], @as(f64, @floatFromInt(top1)) / nr, loss[0] / nr, loss[1] / nr });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = report });
    std.debug.print("{s}", .{report});
    return 0;
}

/// KL(ref || got) in float64 of two bf16 logit rows, whether their top tokens agree, and each one's loss on `next`.
fn divergence(ref: []const u16, got: []const u16, next: u32) struct { kl: f64, same_top: bool, loss: [2]f64 } {
    var mp: f64 = -std.math.inf(f64);
    var mq: f64 = -std.math.inf(f64);
    var ap: usize = 0;
    var aq: usize = 0;
    for (ref, got, 0..) |x, y, i| {
        if (bf(x) > mp) {
            mp = bf(x);
            ap = i;
        }
        if (bf(y) > mq) {
            mq = bf(y);
            aq = i;
        }
    }
    var zp: f64 = 0;
    var zq: f64 = 0;
    for (ref, got) |x, y| {
        zp += @exp(bf(x) - mp);
        zq += @exp(bf(y) - mq);
    }
    const lp = mp + @log(zp);
    const lq = mq + @log(zq);
    var d: f64 = 0;
    for (ref, got) |x, y| {
        const a = bf(x) - lp;
        d += @exp(a) * (a - (bf(y) - lq));
    }
    return .{ .kl = @max(d, 0), .same_top = ap == aq, .loss = .{ lp - bf(ref[next]), lq - bf(got[next]) } };
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// A picture file prepared as the server prepares it, or NAME.f32@HxW: patches another machine prepared, on that grid.
fn prepared(a: std.mem.Allocator, io: std.Io, pic: []const u8, vc: vl.Config, tokens: u32) !vl.geometry.Prepared {
    const cwd = std.Io.Dir.cwd();
    const at = std.mem.indexOfScalar(u8, pic, '@') orelse
        return vl.geometry.prepare(io, a, try cwd.readFileAlloc(io, pic, a, .limited(64 << 20)), vc, tokens);
    const x = std.mem.indexOfScalar(u8, pic[at..], 'x') orelse return error.BadGrid;
    const gh = try std.fmt.parseInt(u32, pic[at + 1 .. at + x], 10);
    const gw = try std.fmt.parseInt(u32, pic[at + x + 1 ..], 10);
    const bytes = try cwd.readFileAlloc(io, pic[0..at], a, .limited(1 << 30));
    if (bytes.len != @as(usize, gh) * gw * vc.patchWidth() * 4) return error.BadPatches;
    const px = try a.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(px), bytes);
    return .{ .pixels = px, .gh = gh, .gw = gw, .tokens = gh * gw / (vc.merge * vc.merge), .hash = 0 };
}

/// A reference's rows against ours: each row's relative L2 error, worst and mean, and the reference file's SHA-256.
const Fidelity = struct { worst: f64, mean: f64, sha: [32]u8 };

/// OUT/NAME.ref.f32 against `rows`; null without a reference of this shape.
fn compare(a: std.mem.Allocator, io: std.Io, out: []const u8, name: []const u8, rows: []const f32, width: usize) ?Fidelity {
    const path = std.fmt.allocPrint(a, "{s}/{s}.ref.f32", .{ out, name }) catch return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30)) catch return null;
    if (bytes.len != rows.len * 4) return null;
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &sha, .{});
    const ref: []align(1) const f32 = std.mem.bytesAsSlice(f32, bytes);
    var worst: f64 = 0;
    var sum: f64 = 0;
    for (0..rows.len / width) |r| {
        var d: f64 = 0;
        var e: f64 = 0;
        for (rows[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
            d += (@as(f64, x) - y) * (@as(f64, x) - y);
            e += @as(f64, y) * y;
        }
        const rel = @sqrt(d) / @max(@sqrt(e), 1e-30);
        worst = @max(worst, rel);
        sum += rel;
    }
    return .{ .worst = worst, .mean = sum / @as(f64, @floatFromInt(rows.len / width)), .sha = sha };
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
