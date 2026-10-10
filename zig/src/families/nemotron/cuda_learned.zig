//! Output projections a CUDA learner's lessons changed, folded in and written into the model's shards as bf16.
const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const dims = @import("slide_dims.zig");
const sites = @import("cuda_sites.zig");
const subspace = @import("subspace.zig");
const host4 = core.affine4_host;

/// Layer i's output projection as the checkpoint names its module: the one a lesson's change sits after.
pub fn moduleName(buf: []u8, c: cfg.Config, i: usize) ![]const u8 {
    const leaf = switch (c.kinds[i]) {
        .moe => "shared_experts.down_proj",
        .mamba => "out_proj",
        .attention => "o_proj",
    };
    return std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.{s}", .{ i, leaf });
}

/// A layer's output projection as the shards held it before this session first saved it: bf16, or 4-bit codes.
const Base = struct {
    quantized: bool,
    weight: []const u8,
    scales: []const u8 = &.{},
    biases: []const u8 = &.{},
};

/// Each saved layer's projection as loaded, owned: a later save folds onto it, never onto what an earlier one wrote.
pub const Bases = struct {
    list: [cfg.max_layers]?Base = @splat(null),

    pub fn deinit(b: *Bases, gpa: std.mem.Allocator) void {
        for (b.list) |x| if (x) |v| for ([_][]const u8{ v.weight, v.scales, v.biases }) |bytes| gpa.free(bytes);
        b.* = .{};
    }
};

/// One layer's change as a save reads it: directions a [ranks, in] and outputs b [ranks, out] (out is the hidden size).
pub const Change = struct { layer: usize, in: usize, a: []const f32, b: []const f32 };

/// One layer's output projection as loaded, the change added.
const Fold = struct {
    d: usize,
    k: usize,
    loaded: Base,
    change: Change,
    ranks: usize,
    out: []u16,

    fn run(f: Fold) !void {
        const a = std.heap.page_allocator;
        const n = f.d * f.k;
        const values = try a.alloc(f32, n);
        defer a.free(values);
        try base(f, values);
        for (0..f.d) |j| {
            const row = values[j * f.k ..][0..f.k];
            for (0..f.ranks) |q| {
                const c = dims.scale * f.change.b[q * f.d + j];
                if (c != 0) subspace.axpy(row, c, f.change.a[q * f.k ..][0..f.k]);
            }
        }
        for (f.out, values) |*o, v| o.* = host4.bf16of(v);
    }

    /// The weight as loaded: bf16 exactly, else 4-bit codes dequantized (read unaligned).
    fn base(f: Fold, values: []f32) !void {
        const w = f.loaded.weight;
        if (!f.loaded.quantized) {
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
        for (sb[0 .. n / host4.group], 0..) |*x, i| x.* = std.mem.readInt(u16, f.loaded.scales[2 * i ..][0..2], .little);
        for (sb[n / host4.group ..], 0..) |*x, i| x.* = std.mem.readInt(u16, f.loaded.biases[2 * i ..][0..2], .little);
        host4.dequantize(words, sb[0 .. n / host4.group], sb[n / host4.group ..], values);
    }
};

/// Every output projection the first `ranks` of the change touch, folded in and written into the model's shards.
pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config, s: *const sites.Sites, ranks: usize, bases: *Bases) !usize {
    var changes: [cfg.max_layers]Change = undefined;
    for (s.list, changes[0..s.list.len]) |*site, *ch| ch.* = .{ .layer = site.layer, .in = site.in, .a = site.a.slice(f32, dims.max_rank * site.in), .b = site.b.slice(f32, dims.max_rank * site.out) };
    return fold(gpa, io, dir, c, changes[0..s.list.len], ranks, bases);
}

/// `write` over the changes as slices; each touched layer's projection is kept in `bases` from its first save on.
fn fold(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config, changes: []const Change, ranks: usize, bases: *Bases) !usize {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    var open = true;
    defer if (open) ck.close();
    var folds: std.ArrayList(Fold) = .empty;
    var edits: std.ArrayList(core.shard_edit.Replacement) = .empty;
    var total: usize = 0;
    for (changes) |ch| {
        const d = c.hidden;
        if (std.mem.allEqual(f32, ch.b[0 .. ranks * d], 0)) continue;
        var name: [160]u8 = undefined;
        const mod = try arena.dupe(u8, try moduleName(&name, c, ch.layer));
        if (bases.list[ch.layer] == null) bases.list[ch.layer] = try keep(gpa, arena, &ck, mod, d, ch.in);
        const out = try arena.alloc(u16, d * ch.in);
        try folds.append(arena, .{ .d = d, .k = ch.in, .loaded = bases.list[ch.layer].?, .change = ch, .ranks = ranks, .out = out });
        const drop = try arena.dupe([]const u8, &.{ try std.fmt.allocPrint(arena, "{s}.scales", .{mod}), try std.fmt.allocPrint(arena, "{s}.biases", .{mod}) });
        try edits.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}.weight", .{mod}), .dtype = .bf16, .shape = try arena.dupe(usize, &.{ d, ch.in }), .bytes = std.mem.sliceAsBytes(out), .drop = drop });
        total += out.len * 2;
    }
    ck.close(); // every base is copied out: the edits may now rewrite the files it mapped
    open = false;
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
    try core.shard_edit.bake(gpa, io, dir, edits.items);
    return total;
}

/// The module's projection (with its scales and biases when 4-bit) copied out of the shards as they are now.
fn keep(gpa: std.mem.Allocator, arena: std.mem.Allocator, ck: *core.Checkpoint, mod: []const u8, d: usize, k: usize) !Base {
    const weight = try ck.get(try std.fmt.allocPrint(arena, "{s}.weight", .{mod}));
    const quantized = weight.dtype != .bf16;
    if (quantized and weight.dtype != .u32) return error.UnsupportedQuantization;
    if (weight.dim(0) != d or weight.dim(1) * @as(usize, if (quantized) 8 else 1) != k) return error.UnexpectedTensor;
    const w = try gpa.dupe(u8, weight.bytes);
    errdefer gpa.free(w);
    if (!quantized) return .{ .quantized = false, .weight = w };
    const sc = try gpa.dupe(u8, (try ck.get(try std.fmt.allocPrint(arena, "{s}.scales", .{mod}))).bytes);
    errdefer gpa.free(sc);
    const bi = try gpa.dupe(u8, (try ck.get(try std.fmt.allocPrint(arena, "{s}.biases", .{mod}))).bytes);
    return .{ .quantized = true, .weight = w, .scales = sc, .biases = bi };
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

test "a second save folds the change onto the projection as loaded, not onto what the first save wrote" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const head =
        \\{"backbone.layers.0.mixer.out_proj.weight":{"dtype":"U32","shape":[2,8],"data_offsets":[0,64]},
        \\"backbone.layers.0.mixer.out_proj.scales":{"dtype":"BF16","shape":[2,1],"data_offsets":[64,68]},
        \\"backbone.layers.0.mixer.out_proj.biases":{"dtype":"BF16","shape":[2,1],"data_offsets":[68,72]}}
    ;
    var shard: [8 + head.len + 72]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], head.len, .little);
    @memcpy(shard[8..][0..head.len], head);
    const data = shard[8 + head.len ..];
    for (0..16) |i| {
        var word: u32 = 0;
        for (0..8) |j| word |= @as(u32, @intCast((i * 8 + j) % 16)) << @intCast(4 * j);
        std.mem.writeInt(u32, data[4 * i ..][0..4], word, .little);
    }
    for (0..2) |r| {
        std.mem.writeInt(u16, data[64 + 2 * r ..][0..2], host4.bf16of(1.0 / 64.0), .little);
        std.mem.writeInt(u16, data[68 + 2 * r ..][0..2], host4.bf16of(-0.125), .little);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &shard });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"quantization\":{\"group_size\":64,\"bits\":4}}" });
    const dir = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(dir);
    var c: cfg.Config = undefined;
    c.kinds[0] = .mamba;
    c.hidden = 2;
    var a: [64]f32 = undefined;
    for (&a, 0..) |*v, i| v.* = 0.25 + 0.01 * @as(f32, @floatFromInt(i % 7));
    const b = [2]f32{ 0.002, -0.003 };
    const changes = [_]Change{.{ .layer = 0, .in = 64, .a = &a, .b = &b }};
    var first: [128]u16 = undefined;
    for (0..2) |session| {
        var bases: Bases = .{};
        defer bases.deinit(gpa);
        var loaded: [128]f32 = undefined; // the 4-bit values, then the bf16 the first session saved
        for (&loaded, 0..) |*v, i| v.* = if (session == 0) @as(f32, @floatFromInt(i % 16)) / 64.0 - 0.125 else host4.f32of(first[i]);
        var kept: ?[128]u16 = null;
        for (0..2) |_| {
            try std.testing.expectEqual(@as(usize, 256), try fold(gpa, io, dir, c, &changes, 1, &bases));
            var ck = try core.Checkpoint.openModel(gpa, io, dir);
            defer ck.close();
            const w = try ck.get("backbone.layers.0.mixer.out_proj.weight");
            try std.testing.expect(w.dtype == .bf16 and w.dim(0) == 2 and w.dim(1) == 64);
            var now: [128]u16 = undefined;
            for (&now, loaded, 0..) |*x, base, i| {
                x.* = std.mem.readInt(u16, w.bytes[2 * i ..][0..2], .little);
                const want = base + dims.scale * b[i / 64] * a[i % 64];
                try std.testing.expect(@abs(host4.f32of(x.*) - want) <= @abs(want) / 256);
            }
            if (kept) |k| try std.testing.expectEqualSlices(u16, &k, &now) else kept = now;
        }
        first = kept.?;
    }
}
