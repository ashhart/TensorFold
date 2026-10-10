//! Output projections a CUDA learner's lessons changed, folded in and written into the model's shards as bf16.
const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const dims = @import("slide_dims.zig");
const sites = @import("cuda_sites.zig");
const subspace = @import("subspace.zig");
const host4 = core.affine4_host;
const Tensor = core.checkpoint.Tensor;

/// Layer i's output projection as the checkpoint names its module: the one a lesson's change sits after.
pub fn moduleName(buf: []u8, c: cfg.Config, i: usize) ![]const u8 {
    const leaf = switch (c.kinds[i]) {
        .moe => "shared_experts.down_proj",
        .mamba => "out_proj",
        .attention => "o_proj",
    };
    return std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.{s}", .{ i, leaf });
}

/// One layer's output projection as the shards hold it (bf16, or 4-bit with its scales and biases), the change added.
const Fold = struct {
    d: usize,
    k: usize,
    weight: Tensor,
    scales: ?Tensor,
    biases: ?Tensor,
    site: *const sites.Site,
    ranks: usize,
    out: []u16,

    fn run(f: Fold) !void {
        const a = std.heap.page_allocator;
        const n = f.d * f.k;
        const values = try a.alloc(f32, n);
        defer a.free(values);
        try base(f, values);
        const av = f.site.a.slice(f32, dims.max_rank * f.k);
        const bv = f.site.b.slice(f32, dims.max_rank * f.d);
        for (0..f.d) |j| {
            const row = values[j * f.k ..][0..f.k];
            for (0..f.ranks) |q| {
                const c = dims.scale * bv[q * f.d + j];
                if (c != 0) subspace.axpy(row, c, av[q * f.k ..][0..f.k]);
            }
        }
        for (f.out, values) |*o, v| o.* = host4.bf16of(v);
    }

    /// The weight as stored: bf16 exactly, else 4-bit codes dequantized (read unaligned from the mapped shard).
    fn base(f: Fold, values: []f32) !void {
        const w = f.weight.bytes;
        if (f.weight.dtype == .bf16) {
            for (values, 0..) |*v, i| v.* = host4.f32of(std.mem.readInt(u16, w[2 * i ..][0..2], .little));
            return;
        }
        const a = std.heap.page_allocator;
        const n = values.len;
        const words = try a.alloc(u32, n / 8);
        defer a.free(words);
        const sb = try a.alloc(u16, 2 * (n / host4.group));
        defer a.free(sb);
        for (words, 0..) |*x, i| x.* = std.mem.readInt(u32, w[4 * i ..][0..4], .little);
        for (sb[0 .. n / host4.group], 0..) |*x, i| x.* = std.mem.readInt(u16, f.scales.?.bytes[2 * i ..][0..2], .little);
        for (sb[n / host4.group ..], 0..) |*x, i| x.* = std.mem.readInt(u16, f.biases.?.bytes[2 * i ..][0..2], .little);
        host4.dequantize(words, sb[0 .. n / host4.group], sb[n / host4.group ..], values);
    }
};

/// Every output projection the first `ranks` of the change touch, folded in and written into the model's shards.
pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config, s: *const sites.Sites, ranks: usize) !usize {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    var open = true;
    defer if (open) ck.close();
    var folds: std.ArrayList(Fold) = .empty;
    var edits: std.ArrayList(core.shard_edit.Replacement) = .empty;
    var total: usize = 0;
    for (s.list) |*site| {
        const bv = site.b.slice(f32, dims.max_rank * site.out)[0 .. ranks * site.out];
        if (std.mem.allEqual(f32, bv, 0)) continue;
        var name: [160]u8 = undefined;
        const mod = try arena.dupe(u8, try moduleName(&name, c, site.layer));
        const weight = try ck.get(try std.fmt.allocPrint(arena, "{s}.weight", .{mod}));
        const quantized = weight.dtype != .bf16;
        if (quantized and weight.dtype != .u32) return error.UnsupportedQuantization;
        const d = c.hidden;
        const k = site.in;
        if (weight.dim(0) != d or weight.dim(1) * @as(usize, if (quantized) 8 else 1) != k) return error.UnexpectedTensor;
        const out = try arena.alloc(u16, d * k);
        try folds.append(arena, .{
            .d = d,
            .k = k,
            .weight = weight,
            .scales = if (quantized) try ck.get(try std.fmt.allocPrint(arena, "{s}.scales", .{mod})) else null,
            .biases = if (quantized) try ck.get(try std.fmt.allocPrint(arena, "{s}.biases", .{mod})) else null,
            .site = site,
            .ranks = ranks,
            .out = out,
        });
        const drop = try arena.dupe([]const u8, &.{ try std.fmt.allocPrint(arena, "{s}.scales", .{mod}), try std.fmt.allocPrint(arena, "{s}.biases", .{mod}) });
        try edits.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}.weight", .{mod}), .dtype = .bf16, .shape = try arena.dupe(usize, &.{ d, k }), .bytes = std.mem.sliceAsBytes(out), .drop = drop });
        total += out.len * 2;
    }
    if (folds.items.len == 0) return 0;
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
    for (&threads) |*t| t.* = std.Thread.spawn(.{}, Worker.go, .{ folds.items, &next, &failed }) catch null;
    Worker.go(folds.items, &next, &failed);
    for (threads) |t| if (t) |th| th.join();
    if (failed.load(.monotonic)) return error.OutOfMemory;
    ck.close(); // every fold has read its shard: the edits may now rewrite the files it mapped
    open = false;
    try core.shard_edit.bake(gpa, io, dir, edits.items);
    return total;
}

test "a layer's output projection module name" {
    var c: cfg.Config = undefined;
    c.kinds[0] = .mamba;
    c.kinds[1] = .moe;
    c.kinds[2] = .attention;
    var buf: [160]u8 = undefined;
    try std.testing.expectEqualStrings("backbone.layers.0.mixer.out_proj", try moduleName(&buf, c, 0));
    try std.testing.expectEqualStrings("backbone.layers.1.mixer.shared_experts.down_proj", try moduleName(&buf, c, 1));
    try std.testing.expectEqualStrings("backbone.layers.2.mixer.o_proj", try moduleName(&buf, c, 2));
}
