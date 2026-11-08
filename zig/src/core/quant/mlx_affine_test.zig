//! The shared MLX affine module reader: shape checks without storage, and file-backed views of every family's form.
const std = @import("std");
const affine = @import("mlx_affine.zig");
const checkpoint = @import("../checkpoint.zig");
const fixture = @import("../checkpoint_fixture.zig");
const st = @import("../safetensors.zig");
const a = std.testing.allocator;
const io = std.testing.io;
const Error = affine.Error;

/// A tensor as the Metal checkpoint describes one: dtype, rank, shape and a byte count, no storage.
const Meta = struct { dtype: st.DType, rank: usize, shape: [st.max_rank]usize = @splat(1), bytes: usize };

fn meta(dtype: st.DType, dims: []const usize) Meta {
    var t: Meta = .{ .dtype = dtype, .rank = dims.len, .bytes = dtype.size() };
    for (dims, 0..) |n, i| {
        t.shape[i] = n;
        t.bytes *= n;
    }
    return t;
}

fn accept(w: Meta, s: Meta, b: Meta, stack: usize, n: usize, k: usize) !void {
    _ = try affine.check(w, s, b, .{ .stack = stack, .n = n, .k = k });
}

test "affine module: expert axes cannot be replaced by equal element counts" {
    const w = meta(.u32, &.{ 4, 16, 8 });
    const s = meta(.bf16, &.{ 4, 16, 1 });
    try accept(w, s, s, 4, 16, 128);
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(meta(.u32, &.{ 2, 32, 8 }), s, s, 4, 16, 128));
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(w, meta(.bf16, &.{ 2, 32, 1 }), s, 4, 16, 128));
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(w, s, meta(.bf16, &.{ 2, 32, 1 }), 4, 16, 128));
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(w, meta(.bf16, &.{ 64, 1 }), s, 4, 16, 128));
}

test "affine module: invalid rank, bytes and packed widths return the named error without traps" {
    var w = meta(.u32, &.{ 4, 16, 8 });
    const s = meta(.bf16, &.{ 4, 16, 1 });
    for ([_]usize{ 0, 1, 2, 4, std.math.maxInt(usize) }) |rank| {
        w.rank = rank;
        try std.testing.expectError(Error.UnexpectedAffineTensor, accept(w, s, s, 4, 16, 128));
    }
    w = meta(.u32, &.{ 4, 16, 8 });
    w.bytes += 1;
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(w, s, s, 4, 16, 128));
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(meta(.u32, &.{ 4, 16, 4096 }), s, s, 4, 16, 128));
    try std.testing.expectError(Error.UnexpectedAffineTensor, accept(meta(.u32, &.{ 1, 8, 16 }), meta(.bf16, &.{ 8, 1 }), meta(.bf16, &.{ 8, 1 }), 1, 8, 64));
    const dense = try affine.check(meta(.u32, &.{ 8, 16 }), meta(.bf16, &.{ 8, 1 }), meta(.bf16, &.{ 8, 1 }), .{ .n = 8, .k = 64 });
    try std.testing.expectEqual(@as(u8, 8), dense.bits);
    try std.testing.expectEqual(@as(usize, 64), dense.group);
}

test "affine module: a fixed width refuses shapes of another, a table off by one group, and fp16 tables" {
    const w = meta(.u32, &.{ 8, 16 });
    const s = meta(.bf16, &.{ 8, 2 });
    _ = try affine.check(w, s, s, .{ .n = 8, .k = 128, .bits = 4, .group = 64 });
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.check(w, s, s, .{ .n = 8, .k = 128, .bits = 8, .group = 64 }));
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.check(w, s, s, .{ .n = 8, .k = 128, .bits = 4, .group = 32 }));
    const extra = meta(.bf16, &.{ 8, 3 });
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.check(w, extra, extra, .{ .n = 8, .k = 128, .bits = 4, .group = 64 }));
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.check(w, extra, extra, .{ .n = 8, .k = 128 }));
    const half = meta(.f16, &.{ 8, 2 });
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.check(w, half, s, .{ .n = 8, .k = 128 }));
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.check(w, s, half, .{ .n = 8, .k = 128, .bits = 4, .group = 64 }));
}

test "affine module: dims are the stored shapes, a stack of one is rank 2, uneven K is refused" {
    const one = try affine.dims(.{ .n = 6, .k = 192 }, 3, 64);
    try std.testing.expectEqual(@as(usize, 2), one.rank);
    try std.testing.expectEqualSlices(usize, &.{ 6, 18 }, one.words[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 6, 3 }, one.tables[0..2]);
    const stack = try affine.dims(.{ .stack = 5, .n = 6, .k = 128 }, 8, 32);
    try std.testing.expectEqualSlices(usize, &.{ 5, 6, 32 }, stack.words[0..3]);
    try std.testing.expectEqualSlices(usize, &.{ 5, 6, 4 }, stack.tables[0..3]);
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.dims(.{ .n = 6, .k = 96 }, 4, 64));
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.dims(.{ .n = 6, .k = 40 }, 3, 8));
    try std.testing.expectError(Error.UnexpectedAffineTensor, affine.dims(.{ .stack = 0, .n = 6, .k = 64 }, 4, 64));
}

/// One family's module form: its name, stack, [n, k], the width the file stores and the width the family fixes.
const Form = struct { name: []const u8, stack: usize, n: usize, k: usize, bits: u8, group: usize, fixed: bool };

const forms = [_]Form{
    .{ .name = "language_model.model.layers.0.mlp.gate_proj", .stack = 1, .n = 8, .k = 128, .bits = 4, .group = 64, .fixed = true }, // qwen3_5
    .{ .name = "model.layers.1.feed_forward.switch_mlp.up_proj", .stack = 4, .n = 8, .k = 128, .bits = 6, .group = 64, .fixed = true }, // lfm2_moe
    .{ .name = "model.layers.2.self_attn.gate_proj", .stack = 1, .n = 16, .k = 64, .bits = 8, .group = 32, .fixed = true }, // lfm3
    .{ .name = "model.layers.3.mlp.switch_mlp.gate_up_proj", .stack = 8, .n = 16, .k = 256, .bits = 2, .group = 128, .fixed = false }, // minimax_m3
    .{ .name = "model.layers.4.moe.down_proj", .stack = 1, .n = 8, .k = 192, .bits = 3, .group = 64, .fixed = false }, // step5
    .{ .name = "model.layers.5.eh_proj", .stack = 1, .n = 8, .k = 320, .bits = 5, .group = 64, .fixed = false }, // step5 mtp
};

fn payload(arena: std.mem.Allocator, len: usize, seed: u64) ![]u8 {
    const out = try arena.alloc(u8, len);
    var rng: std.Random.DefaultPrng = .init(seed);
    rng.random().bytes(out);
    return out;
}

fn entries(arena: std.mem.Allocator) ![]fixture.Entry {
    var out: std.ArrayList(fixture.Entry) = .empty;
    for (forms, 0..) |f, i| {
        const d = try affine.dims(.{ .stack = f.stack, .n = f.n, .k = f.k }, f.bits, f.group);
        inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, j| {
            const words = j == 0;
            const shape = try arena.dupe(usize, if (words) d.words[0..d.rank] else d.tables[0..d.rank]);
            var len: usize = if (words) 4 else 2;
            for (shape) |n| len *= n;
            const name = try std.fmt.allocPrint(arena, "{s}.{s}", .{ f.name, suffix });
            try out.append(arena, .{ .name = name, .dtype = if (words) "U32" else "BF16", .shape = shape, .bytes = try payload(arena, len, 3 * i + j) });
        }
    }
    return out.toOwnedSlice(arena);
}

test "affine module: file-backed views of each family's form keep the stored bytes, shapes and dtypes" {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const all = try entries(arena.allocator());
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(a, io, tmp.dir, "model.safetensors", all);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var ck = try checkpoint.Checkpoint.openModel(a, io, dir);
    defer ck.close();
    for (forms, 0..) |f, i| {
        const want: affine.Want = if (f.fixed) .{ .stack = f.stack, .n = f.n, .k = f.k, .bits = f.bits, .group = f.group } else .{ .stack = f.stack, .n = f.n, .k = f.k };
        const t = try affine.read(&ck, f.name, want);
        try std.testing.expectEqual(f.bits, t.bits);
        try std.testing.expectEqual(f.group, t.group);
        inline for (.{ "weight", "scales", "biases" }, 0..) |field, j| {
            const tensor = @field(t, field);
            const written = all[3 * i + j];
            try std.testing.expectEqual(std.hash.Wyhash.hash(0, written.bytes), std.hash.Wyhash.hash(0, tensor.bytes));
            try std.testing.expectEqualSlices(usize, written.shape, tensor.shape[0..tensor.rank]);
            try std.testing.expectEqual(@as(st.DType, if (j == 0) .u32 else .bf16), tensor.dtype);
        }
    }
}
