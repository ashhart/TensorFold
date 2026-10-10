//! Output projections lessons changed, in the shards as bf16, loaded as the kernels' 4-bit codes plus the rest.
const std = @import("std");
const mtl = @import("metal");
const ckpt = @import("../../core/checkpoint_metal.zig");
const host4 = @import("../../core/affine4_host.zig");
const shard_edit = @import("../../core/shard_edit.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const adapters = @import("adapters.zig");
const subspace = @import("subspace.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Each layer's bf16 [out, in] residual, its weight minus its codes' values: the forward adds it after the projection.
pub const Residuals = [cfg.max_layers]?mtl.Buffer;

/// Layer i's output projection as the checkpoint names its module: the one a lesson's change sits after.
pub fn moduleName(buf: []u8, c: cfg.Config, i: usize) ![]const u8 {
    const leaf = switch (c.kinds[i]) {
        .moe => "shared_experts.down_proj",
        .mamba => "out_proj",
        .attention => "o_proj",
    };
    return std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.{s}", .{ i, leaf });
}

/// Layer i's output projection's input width.
pub fn inWidth(c: cfg.Config, i: usize) usize {
    return switch (c.kinds[i]) {
        .moe => c.shared_width,
        .mamba => c.inner(),
        .attention => c.heads * c.head_dim,
    };
}

/// Every layer whose output projection is bf16 gets codes, scales and biases in the checkpoint, and its residual.
pub fn split(gpa: std.mem.Allocator, device: mtl.Device, ck: *ckpt.Checkpoint, c: cfg.Config) !Residuals {
    var out: Residuals = @splat(null);
    errdefer free(out);
    for (0..c.layers) |i| {
        var name: [160]u8 = undefined;
        const base = try moduleName(&name, c, i);
        var full: [192]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&full, "{s}.weight", .{base}));
        if (w.dtype != .bf16) continue;
        if (w.rank != 2 or w.shape[0] != c.hidden or w.shape[1] != inWidth(c, i)) return error.UnexpectedTensor;
        out[i] = try splitOne(gpa, device, ck, base, w);
    }
    return out;
}

fn splitOne(gpa: std.mem.Allocator, device: mtl.Device, ck: *ckpt.Checkpoint, base: []const u8, w: ckpt.Tensor) !mtl.Buffer {
    const n = w.count();
    const values = try gpa.alloc(f32, n);
    defer gpa.free(values);
    for (values, w.host(u16)) |*v, b| v.* = host4.f32of(b);
    const groups = n / host4.group;
    const codes = try device.buffer(n / 2 + 4 * groups, opts);
    ck.own(codes, n / 2 + 4 * groups) catch |e| {
        codes.deinit();
        return e;
    };
    const words = codes.slice(u32, n / 8);
    const sb = @as([*]u16, @ptrCast(@alignCast(codes.contents() + n / 2)))[0 .. 2 * groups];
    host4.quantize(values, words, sb[0..groups], sb[groups..]);
    const rest = try device.buffer(n * 2, opts);
    errdefer rest.deinit();
    const back = try gpa.alloc(f32, n);
    defer gpa.free(back);
    host4.dequantize(words, sb[0..groups], sb[groups..], back);
    for (rest.slice(u16, n), values, back) |*r, v, q| r.* = host4.bf16of(v - q);
    const d = w.shape[0];
    const k = w.shape[1];
    var full: [192]u8 = undefined;
    try ck.set(try std.fmt.bufPrint(&full, "{s}.weight", .{base}), .{ .buffer = codes, .offset = 0, .bytes = n / 2, .dtype = .u32, .shape = .{ d, k / 8, 1, 1, 1 }, .rank = 2 });
    try ck.set(try std.fmt.bufPrint(&full, "{s}.scales", .{base}), .{ .buffer = codes, .offset = n / 2, .bytes = 2 * groups, .dtype = .bf16, .shape = .{ d, k / 64, 1, 1, 1 }, .rank = 2 });
    try ck.set(try std.fmt.bufPrint(&full, "{s}.biases", .{base}), .{ .buffer = codes, .offset = n / 2 + 2 * groups, .bytes = 2 * groups, .dtype = .bf16, .shape = .{ d, k / 64, 1, 1, 1 }, .rank = 2 });
    return rest;
}

/// The residuals into their layers, the weights owning them from here (freed here when that fails).
pub fn attach(w: *wts.Weights, residuals: Residuals) !void {
    var held: usize = 0;
    for (residuals) |r| held += @intFromBool(r != null);
    w.owned.ensureUnusedCapacity(w.allocator, held) catch |e| {
        free(residuals);
        return e;
    };
    for (residuals, 0..) |r, i| if (r) |b| {
        w.owned.appendAssumeCapacity(b);
        switch (w.layers[i]) {
            .moe => |*m| m.slide = b,
            .mamba => |*m| m.slide = b,
            .attention => |*a| a.slide = b,
        }
    };
}

/// Layer i's residual, if its projection was loaded from bf16.
pub fn residual(w: *const wts.Weights, i: usize) ?mtl.Buffer {
    return switch (w.layers[i]) {
        .moe => |m| m.slide,
        .mamba => |m| m.slide,
        .attention => |a| a.slide,
    };
}

pub fn free(residuals: Residuals) void {
    for (residuals) |r| if (r) |b| b.deinit();
}

/// One layer's output projection with the learned change folded in, as bf16 for its shard.
const Fold = struct {
    layer: usize,
    d: usize,
    k: usize,
    tensors: [3]ckpt.Tensor, // codes, scales, biases as loaded
    rest: ?mtl.Buffer, // the residual a bf16 load left, if any
    site: *const adapters.Site,
    ranks: usize,
    out: []u16,

    fn run(f: Fold) !void {
        const n = f.d * f.k;
        const values = try std.heap.page_allocator.alloc(f32, n);
        defer std.heap.page_allocator.free(values);
        host4.dequantize(f.tensors[0].host(u32), f.tensors[1].host(u16), f.tensors[2].host(u16), values);
        if (f.rest) |r| for (values, r.slice(u16, n)) |*v, x| {
            v.* += host4.f32of(x);
        };
        const a = f.site.a.slice(f32, adapters.max_rank * f.k);
        const b = f.site.b.slice(f32, adapters.max_rank * f.d);
        for (0..f.d) |j| {
            const row = values[j * f.k ..][0..f.k];
            for (0..f.ranks) |q| {
                const c = adapters.scale * b[q * f.d + j];
                if (c != 0) subspace.axpy(row, c, a[q * f.k ..][0..f.k]);
            }
        }
        for (f.out, values) |*o, v| o.* = host4.bf16of(v);
    }
};

/// Every output projection the first `ranks` of the change touch, folded in and written into the model's shards.
pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, ck: *const ckpt.Checkpoint, w: *const wts.Weights, c: cfg.Config, sites: *const adapters.Sites, ranks: usize) !usize {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var folds: std.ArrayList(Fold) = .empty;
    var edits: std.ArrayList(shard_edit.Replacement) = .empty;
    var bytes: usize = 0;
    for (sites.list) |*site| {
        const b = site.b.slice(f32, adapters.max_rank * site.out)[0 .. ranks * site.out];
        if (std.mem.allEqual(f32, b, 0)) continue;
        const i = site.layer;
        var name: [160]u8 = undefined;
        const base = try arena.dupe(u8, try moduleName(&name, c, i));
        var tensors: [3]ckpt.Tensor = undefined;
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, p| tensors[p] = try ck.get(try std.fmt.allocPrint(arena, "{s}.{s}", .{ base, part }));
        const d = c.hidden;
        const k = inWidth(c, i);
        const out = try arena.alloc(u16, d * k);
        try folds.append(arena, .{ .layer = i, .d = d, .k = k, .tensors = tensors, .rest = residual(w, i), .site = site, .ranks = ranks, .out = out });
        const shape = try arena.dupe(usize, &.{ d, k });
        const drop = try arena.dupe([]const u8, &.{ try std.fmt.allocPrint(arena, "{s}.scales", .{base}), try std.fmt.allocPrint(arena, "{s}.biases", .{base}) });
        try edits.append(arena, .{ .name = try std.fmt.allocPrint(arena, "{s}.weight", .{base}), .dtype = .bf16, .shape = shape, .bytes = std.mem.sliceAsBytes(out), .drop = drop });
        bytes += out.len * 2;
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
    try shard_edit.bake(gpa, io, dir, edits.items);
    return bytes;
}
