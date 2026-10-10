//! Output projections lessons changed on CUDA, folded into bf16 and written into the model's shards (learned.zig on Metal).
const std = @import("std");
const core = @import("core");
const train = @import("cuda_train.zig");
const sites_mod = @import("cuda_sites.zig");
const shape = @import("learn_shape.zig");
const subspace = @import("subspace.zig");

const host4 = core.affine4_host;

/// One layer's output projection as the trainer loaded it (bf16, or 4-bit words, scales and biases), and its fold.
const Fold = struct {
    site: *const sites_mod.Site,
    ranks: usize,
    plain: []u16 = &.{},
    words: []u32 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    out: []u16,

    fn run(f: Fold) !void {
        const d = f.site.out;
        const k = f.site.in;
        const values = try std.heap.page_allocator.alloc(f32, d * k);
        defer std.heap.page_allocator.free(values);
        if (f.plain.len > 0) {
            for (values, f.plain) |*v, x| v.* = host4.f32of(x);
        } else host4.dequantize(f.words, f.scales, f.biases, values);
        const a = f.site.a.slice(f32, shape.max_rank * k);
        const b = f.site.b.slice(f32, shape.max_rank * d);
        for (0..d) |j| {
            const row = values[j * k ..][0..k];
            for (0..f.ranks) |q| {
                const c = shape.scale * b[q * d + j];
                if (c != 0) subspace.axpy(row, c, a[q * k ..][0..k]);
            }
        }
        for (f.out, values) |*o, v| o.* = host4.bf16of(v);
    }
};

/// Every output projection the first `ranks` of the change touch, folded into it as loaded and written into the shards.
pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, t: *const train.Trainer, ranks: usize) !usize {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const o = t.e.ops();
    try t.e.stream.synchronize();
    var folds: std.ArrayList(Fold) = .empty;
    var edits: std.ArrayList(core.shard_edit.Replacement) = .empty;
    var bytes: usize = 0;
    for (t.sites.list) |*site| {
        const b = site.b.slice(f32, shape.max_rank * site.out)[0 .. ranks * site.out];
        if (std.mem.allEqual(f32, b, 0)) continue;
        const m = t.outProj(site.layer);
        if (m.n != site.out or m.k != site.in) return error.UnexpectedTensor;
        const n = m.n * m.k;
        var f: Fold = .{ .site = site, .ranks = ranks, .out = try arena.alloc(u16, n) };
        if (m.bf16) {
            f.plain = try arena.alloc(u16, n);
            try o.download(std.mem.sliceAsBytes(f.plain), m.w);
        } else {
            f.words = try arena.alloc(u32, n / 8);
            f.scales = try arena.alloc(u16, n / host4.group);
            f.biases = try arena.alloc(u16, n / host4.group);
            try o.download(std.mem.sliceAsBytes(f.words), m.w);
            try o.download(std.mem.sliceAsBytes(f.scales), m.s);
            try o.download(std.mem.sliceAsBytes(f.biases), m.b);
        }
        try folds.append(arena, f);
        var name: [160]u8 = undefined;
        const base = try arena.dupe(u8, try shape.moduleName(&name, t.c, site.layer));
        // a 4-bit projection leaves its scales and biases behind; a bf16 one (an earlier save) is written over
        const drop: []const []const u8 = if (m.bf16) &.{} else try arena.dupe([]const u8, &.{ try std.fmt.allocPrint(arena, "{s}.scales", .{base}), try std.fmt.allocPrint(arena, "{s}.biases", .{base}) });
        try edits.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}.weight", .{base}), .dtype = .bf16, .shape = try arena.dupe(usize, &.{ m.n, m.k }), .bytes = std.mem.sliceAsBytes(f.out), .drop = drop });
        bytes += n * 2;
    }
    if (folds.items.len == 0) return 0;
    try t.e.stream.synchronize();
    var failed = std.atomic.Value(bool).init(false);
    var next = std.atomic.Value(usize).init(0);
    const Worker = struct {
        fn go(all: []const Fold, counter: *std.atomic.Value(usize), bad: *std.atomic.Value(bool)) void {
            while (true) {
                const at = counter.fetchAdd(1, .monotonic);
                if (at >= all.len) return;
                all[at].run() catch bad.store(true, .monotonic);
            }
        }
    };
    var threads: [12]?std.Thread = @splat(null);
    for (&threads) |*th| th.* = std.Thread.spawn(.{}, Worker.go, .{ folds.items, &next, &failed }) catch null;
    Worker.go(folds.items, &next, &failed);
    for (threads) |th| if (th) |x| x.join();
    if (failed.load(.monotonic)) return error.OutOfMemory;
    try core.shard_edit.bake(gpa, io, dir, edits.items);
    return bytes;
}
