//! A dumped pocket lesson's read rows fitted as the learner fits them, timed; each fit's numbers and weights written.
const std = @import("std");
const platform = @import("sys");
const tf = @import("tensorfold");

const pocket_fit = tf.nemotron.pocket_fit;
const Kind = pocket_fit.Kind;
const Fit = pocket_fit.Fit;

/// What a read row is to the neurons keyed `key`, as pocket.zig's kindFor reads its label.
fn kindFor(kind: Kind, key: i32, k: usize) Kind {
    if (key == @as(i32, @intCast(k))) return kind;
    return if (kind == .steady) .steady else .other;
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("usage: tf-pocket-fit-check DUMP_DIR FACT LAYERS OUT_DIR\n", .{});
        std.process.exit(2);
    }
    const dir = std.Io.Dir.cwd();
    var name: [1024]u8 = undefined;
    const labels = try dir.readFileAlloc(io, try std.fmt.bufPrint(&name, "{s}/fact-{s}-labels.txt", .{ args[1], args[2] }), gpa, .limited(1 << 30));
    var lines = std.mem.tokenizeScalar(u8, labels, '\n');
    var head = std.mem.tokenizeScalar(u8, lines.next() orelse return error.EmptyDump, ' ');
    const slots = try std.fmt.parseInt(usize, head.next().?, 10);
    const d = try std.fmt.parseInt(usize, head.next().?, 10);
    var layers: std.ArrayList(usize) = .empty;
    while (head.next()) |t| try layers.append(gpa, try std.fmt.parseInt(usize, t, 10));
    var kinds: std.ArrayList(Kind) = .empty;
    var keys: std.ArrayList(i32) = .empty;
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        try kinds.append(gpa, @fromBackingInt(@intCast(try std.fmt.parseInt(u8, f.next().?, 10))));
        try keys.append(gpa, try std.fmt.parseInt(i32, f.next().?, 10));
    }
    const m = kinds.items.len;
    const bytes = try dir.readFileAlloc(io, try std.fmt.bufPrint(&name, "{s}/fact-{s}-rows.bf16", .{ args[1], args[2] }), gpa, .limited(1 << 34));
    const rows = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(bytes)));
    if (rows.len != m * slots * d) return error.DumpSizes;
    // the keys a fact's rows name, each fitted at the slots pocket.zig's `candidate` admits
    const total_layers = try std.fmt.parseInt(usize, args[3], 10);
    var jobs: std.ArrayList([2]usize) = .empty;
    for (0..3) |k| {
        var any = false;
        for (keys.items, kinds.items) |key, kind| any = any or (key == @as(i32, @intCast(k)) and kind == .fact);
        if (!any) continue;
        const last = if (k == 1) total_layers * 5 / 8 else total_layers * 7 / 8;
        for (layers.items, 0..) |l, j| if (l >= total_layers / 5 and l <= last) try jobs.append(gpa, .{ k, j });
    }
    const all_kinds = try gpa.alloc(Kind, 3 * m);
    for (0..3) |k| for (0..m) |i| {
        all_kinds[k * m + i] = kindFor(kinds.items[i], keys.items[i], k);
    };
    const us = try gpa.alloc(f32, jobs.items.len * d);
    const fits = try gpa.alloc(Fit, jobs.items.len);
    // TF_FIT_THREADS: that many threads take the fits in turn (default: a thread a fit, as the learner runs them)
    const want = if (platform.c.getenv("TF_FIT_THREADS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else jobs.items.len;
    const threads = try gpa.alloc(std.Thread, @min(@max(want, 1), jobs.items.len));
    var next = std.atomic.Value(usize).init(0);
    const work: Work = .{ .gpa = gpa, .rows = rows, .kinds = all_kinds, .m = m, .slots = slots, .d = d, .jobs = jobs.items, .us = us, .fits = fits, .next = &next };
    const t0 = std.Io.Clock.awake.now(io);
    for (threads) |*th| th.* = try std.Thread.spawn(.{}, Work.run, .{work});
    for (threads) |th| th.join();
    const secs = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0.toNanoseconds())) / 1e9;
    std.debug.print("RESULT {d} fits on {d} read rows of {d} slots in {d:.2} s ({d} threads)\n", .{ jobs.items.len, m, slots, secs, threads.len });
    for (jobs.items, fits) |jb, f| std.debug.print("FIT key {d} layer {d}: fact least {d:.4} on {d:.4}, held mean {d:.4} on {d:.4}, steady most {d:.4}, other most {d:.4}, fires {d}\n", .{ jb[0], layers.items[jb[1]], f.fact_least, f.fact_on, f.held_mean, f.held_on, f.steady_most, f.other_most, f.fires });
    const out = try std.fmt.bufPrint(&name, "{s}/u.f32", .{args[4]});
    try dir.writeFile(io, .{ .sub_path = out, .data = std.mem.sliceAsBytes(us) });
}

/// The fits a thread takes in turn until none is left.
const Work = struct {
    gpa: std.mem.Allocator,
    rows: []const u16,
    kinds: []const Kind,
    m: usize,
    slots: usize,
    d: usize,
    jobs: []const [2]usize,
    us: []f32,
    fits: []Fit,
    next: *std.atomic.Value(usize),

    fn run(w: Work) void {
        while (true) {
            const i = w.next.fetchAdd(1, .monotonic);
            if (i >= w.jobs.len) return;
            const jb = w.jobs[i];
            pocket_fit.fitLayer(w.gpa, w.rows, w.kinds[jb[0] * w.m ..][0..w.m], w.slots, jb[1], w.d, w.us[i * w.d ..][0..w.d], &w.fits[i]);
        }
    }
};
