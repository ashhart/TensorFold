//! Host tests verifying multi-threaded fast SDOT, repacked 8x4, and fast experts match 1-thread bit-for-bit.
const std = @import("std");
const g = @import("core").gguf;
const pool_mod = @import("cpu_gguf_pool.zig");
const sdot = @import("cpu_gguf_q4_sdot.zig");
const repack8 = @import("cpu_gguf_repack8.zig");

fn makeMatrix(a: std.mem.Allocator, kind: g.TensorKind, width: usize, n: usize, seed: usize) !struct { bytes: []u8, tensor: g.Tensor } {
    const blk = kind.block();
    const bytes = try a.alloc(u8, n * (width / blk.values) * blk.bytes);
    for (bytes, 0..) |*v, i| v.* = @intCast((i * 41 + seed * 17 + 7) % 256);
    for (0..bytes.len / blk.bytes) |b| {
        const part = bytes[b * blk.bytes ..][0..blk.bytes];
        const at: usize = if (kind == .q6_k) 208 else 0;
        std.mem.writeInt(u16, part[at..][0..2], if ((b + seed) % 2 == 0) 0x3400 else 0xb400, .little);
        if (kind == .q4_k) std.mem.writeInt(u16, part[2..4], 0x3000, .little);
    }
    return .{
        .bytes = bytes,
        .tensor = .{ .name = "fast_pool", .kind = kind, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = bytes.len, .bytes = bytes },
    };
}

test "fast pool SDOT and 8x4 repacked projections match 1-thread bit-for-bit across thread counts and tail rows" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const width: usize = 256;
    const n: usize = 77;
    const x = try a.alloc(f32, 3 * width);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19) % 39)) - 19)) / 11.0;
    const want = try a.alloc(f32, 3 * n);
    defer a.free(want);
    const got = try a.alloc(f32, 3 * n);
    defer a.free(got);

    for ([_]g.TensorKind{ .q4_k, .q6_k, .q5_0 }) |kind| {
        const mat = try makeMatrix(a, kind, width, n, @backingInt(kind));
        defer a.free(mat.bytes);
        const rep = try repack8.repack(a, mat.tensor);
        defer if (rep) |r| r.deinit(a);

        for ([_]usize{ 1, 3 }) |count| {
            try sdot.projectBatch(a, mat.tensor, x[0 .. count * width], want[0 .. count * n], count, true);
            for ([_]usize{ 1, 2, 4 }) |threads| {
                const p = try pool_mod.Pool.create(a, io, threads);
                defer p.destroy();
                p.min_weights = 1;
                try p.projectSdot(a, mat.tensor, x[0 .. count * width], got[0 .. count * n], count, true);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want[0 .. count * n]), std.mem.sliceAsBytes(got[0 .. count * n]));
                if (rep) |r| {
                    try p.projectSdotRepacked(a, r, mat.tensor, x[0 .. count * width], got[0 .. count * n], count, true);
                    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want[0 .. count * n]), std.mem.sliceAsBytes(got[0 .. count * n]));
                }
            }
        }
    }
}

test "fast pool routed experts match 1-thread bit-for-bit across thread counts" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const hidden: usize = 256;
    const inter: usize = 256;
    const gate_m = try makeMatrix(a, .q4_k, hidden, inter, 1);
    defer a.free(gate_m.bytes);
    const up_m = try makeMatrix(a, .q4_k, hidden, inter, 2);
    defer a.free(up_m.bytes);
    const down_m = try makeMatrix(a, .q6_k, inter, hidden, 3);
    defer a.free(down_m.bytes);
    const gate_rep = (try repack8.repack(a, gate_m.tensor)).?;
    defer gate_rep.deinit(a);
    const up_rep = (try repack8.repack(a, up_m.tensor)).?;
    defer up_rep.deinit(a);
    const down_rep = (try repack8.repack(a, down_m.tensor)).?;
    defer down_rep.deinit(a);

    var x: [hidden]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 27)) - 13)) / 8.0;
    var acts: [1]sdot.Activation256 = undefined;
    try sdot.quantize256(&x, &acts, true);

    var want: [4 * hidden]f32 = undefined;
    var got: [4 * hidden]f32 = undefined;
    var list: [4]pool_mod.FastExpert = undefined;
    for (0..4) |i| list[i] = .{
        .gate = gate_m.tensor,
        .up = up_m.tensor,
        .down = down_m.tensor,
        .gate_rep = gate_rep,
        .up_rep = up_rep,
        .down_rep = down_rep,
        .use_sdot_gate = true,
        .use_sdot_up = true,
        .use_sdot_down = true,
        .out = want[i * hidden ..].ptr,
    };

    const p1 = try pool_mod.Pool.create(a, io, 1);
    defer p1.destroy();
    try p1.expertsFast(&list, &x, &acts, hidden, inter, 7.0);

    for (0..4) |i| list[i].out = got[i * hidden ..].ptr;
    const p4 = try pool_mod.Pool.create(a, io, 4);
    defer p4.destroy();
    try p4.expertsFast(&list, &x, &acts, hidden, inter, 7.0);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want), std.mem.sliceAsBytes(&got));
}

test "extractAllowedCpus reads sparse affinity masks in ascending order without hardcoded cores" {
    const mask = [_]usize{ (@as(usize, 1) << 0) | (@as(usize, 1) << 3) | (@as(usize, 1) << 5) | (@as(usize, 1) << 7), (@as(usize, 1) << 2) };
    var out: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 4), pool_mod.extractAllowedCpus(&mask, &out));
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 5, 7 }, &out);
    var all: [8]usize = undefined;
    try std.testing.expectEqual(@as(usize, 5), pool_mod.extractAllowedCpus(&mask, &all));
    try std.testing.expectEqual(@as(usize, @bitSizeOf(usize) + 2), all[4]);
}

test "expertsFastBatch grouped multi-row experts match 1-row expertsFast byte for byte across thread counts" {
    if (!@import("cpu_gguf_sdot_act.zig").has_dotprod) return error.SkipZigTest;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const hidden: usize = 256;
    const inter: usize = 256;
    const count: usize = 6;
    const gate_m = try makeMatrix(a, .q4_k, hidden, inter, 11);
    defer a.free(gate_m.bytes);
    const up_m = try makeMatrix(a, .q4_k, hidden, inter, 22);
    defer a.free(up_m.bytes);
    const down_m = try makeMatrix(a, .q6_k, inter, hidden, 33);
    defer a.free(down_m.bytes);
    const gate_rep = (try repack8.repack(a, gate_m.tensor)).?;
    defer gate_rep.deinit(a);
    const up_rep = (try repack8.repack(a, up_m.tensor)).?;
    defer up_rep.deinit(a);
    const down_rep = (try repack8.repack(a, down_m.tensor)).?;
    defer down_rep.deinit(a);

    var x: [count * hidden]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13) % 31)) - 15)) / 7.0;
    var acts: [count]sdot.Activation256 = undefined;
    for (0..count) |t| try sdot.quantize256(x[t * hidden ..][0..hidden], acts[t .. t + 1], true);

    var want: [count * hidden]f32 = undefined;
    const p1 = try pool_mod.Pool.create(a, io, 1);
    defer p1.destroy();
    for (0..count) |t| {
        const one = [_]pool_mod.FastExpert{.{
            .gate = gate_m.tensor,
            .up = up_m.tensor,
            .down = down_m.tensor,
            .gate_rep = gate_rep,
            .up_rep = up_rep,
            .down_rep = down_rep,
            .use_sdot_gate = true,
            .use_sdot_up = true,
            .use_sdot_down = true,
            .out = want[t * hidden ..].ptr,
        }};
        try p1.expertsFast(&one, x[t * hidden ..][0..hidden], acts[t .. t + 1].ptr, hidden, inter, 7.0);
    }

    var got: [count * hidden]f32 = undefined;
    const grp = [_]pool_mod.FastExpertGroup{
        .{
            .expert = .{
                .gate = gate_m.tensor,
                .up = up_m.tensor,
                .down = down_m.tensor,
                .gate_rep = gate_rep,
                .up_rep = up_rep,
                .down_rep = down_rep,
                .use_sdot_gate = true,
                .use_sdot_up = true,
                .use_sdot_down = true,
                .out = got[0..].ptr,
            },
            .n_rows = 5,
            .rows = .{ 0, 1, 2, 3, 4, 0, 0, 0 },
            .outs = .{ got[0..].ptr, got[hidden..].ptr, got[2 * hidden ..].ptr, got[3 * hidden ..].ptr, got[4 * hidden ..].ptr, undefined, undefined, undefined },
        },
        .{
            .expert = .{
                .gate = gate_m.tensor,
                .up = up_m.tensor,
                .down = down_m.tensor,
                .gate_rep = gate_rep,
                .up_rep = up_rep,
                .down_rep = down_rep,
                .use_sdot_gate = true,
                .use_sdot_up = true,
                .use_sdot_down = true,
                .out = got[5 * hidden ..].ptr,
            },
            .n_rows = 1,
            .rows = .{ 5, 0, 0, 0, 0, 0, 0, 0 },
            .outs = .{ got[5 * hidden ..].ptr, undefined, undefined, undefined, undefined, undefined, undefined, undefined },
        },
    };
    for ([_]usize{ 1, 2, 3, 4, 8 }) |threads| {
        const pool = try pool_mod.Pool.create(a, io, threads);
        defer pool.destroy();
        @memset(&got, 0);
        try pool.expertsFastBatch(&grp, &x, &acts, count, hidden, inter, 7.0);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&want), std.mem.sliceAsBytes(&got));
    }
}
