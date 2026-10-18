//! EXL3 shared prefill and concurrent chained product qualification.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const decoder = @import("exl3_projection_decode.zig");
const fixture = @import("exl3_gpu_fixture.zig");
const k = fixture.k;
const n = fixture.n;
const rows = fixture.rows;
const Owned = fixture.Owned;
const bf16 = fixture.bf16;
const input = fixture.input;
const complete = fixture.complete;
const reference = fixture.reference;

test "shared prefill retains exact row bits and guards across tile boundaries and fallback" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const prefill = p.prefill orelse return error.SharedPrefillUnavailable;
    try std.testing.expect(prefill.simdWidth() == 32 and prefill.maxThreads() >= 128);
    var legacy = p;
    legacy.prefill = null;
    const queue = try device.queue();
    defer queue.deinit();
    const w = try Owned.init(device, 5, .mul1, 0x7368617265);
    defer w.deinit();
    for ([_]usize{ 31, 32, 33, 63, 64, 65 }) |count| {
        const x = try input(device, count);
        defer x.deinit();
        const ids = try device.buffer(count * 4, mtl.ResourceOptions.shared);
        defer ids.deinit();
        for (ids.slice(u32, count), 0..) |*id, i| id.* = @intCast(i % 2);
        for ([_]ex.Output{ .bf16, .f32 }) |kind| {
            const size: usize = if (kind == .bf16) 2 else 4;
            const bytes = count * (n + 8) * size;
            const actual = try device.buffer(bytes + 64, mtl.ResourceOptions.shared);
            defer actual.deinit();
            const original = try device.buffer(bytes + 64, mtl.ResourceOptions.shared);
            defer original.deinit();
            const scratch_bytes = try ex.scratchBytes(k, n, count);
            const scratch = try device.buffer(scratch_bytes + 32, mtl.ResourceOptions.shared);
            defer scratch.deinit();
            for ([_]bool{ false, true }) |filtered| {
                @memset(actual.slice(u8, bytes + 64), 0xa5);
                @memset(original.slice(u8, bytes + 64), 0xa5);
                @memset(scratch.slice(u8, scratch_bytes + 32), 0xcd);
                const r: ex.Rows = .{ .count = count, .x_stride = k + 16, .y_stride = n + 8, .output = kind, .routes = if (filtered) .{ .ids = .{ .buffer = ids } } else null };
                const cb = queue.commandBuffer();
                const e = cb.compute(.concurrent);
                try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = actual, .offset = 32 }, .{ .buffer = scratch, .offset = 16 }, r);
                e.barrier();
                try ex.encode(&legacy, e, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = original, .offset = 32 }, .{ .buffer = scratch, .offset = 16 }, r);
                try complete(cb, e);
                try std.testing.expectEqualSlices(u8, original.slice(u8, bytes + 64), actual.slice(u8, bytes + 64));
                if (!filtered) {
                    @memset(original.slice(u8, bytes + 64), 0xa5);
                    const resumed = queue.commandBuffer();
                    const re = resumed.compute(.concurrent);
                    var first = r;
                    first.count = 31;
                    try ex.encode(&p, re, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = original, .offset = 32 }, .{ .buffer = scratch, .offset = 16 }, first);
                    if (count > 31) {
                        re.barrier();
                        var tail = r;
                        tail.count = count - 31;
                        try ex.encode(&p, re, w.matrix, .{ .buffer = x, .offset = 32 + 31 * (k + 16) * 2 }, .{ .buffer = original, .offset = 32 + 31 * (n + 8) * size }, .{ .buffer = scratch, .offset = 16 }, tail);
                    }
                    try complete(resumed, re);
                    try std.testing.expectEqualSlices(u8, original.slice(u8, bytes + 64), actual.slice(u8, bytes + 64));
                }
                for (actual.slice(u8, bytes + 64)[0..32]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
                for (actual.slice(u8, bytes + 64)[bytes + 32 ..]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
                for (0..count) |row| {
                    const pad = 32 + (row * (n + 8) + n) * size;
                    for (actual.slice(u8, bytes + 64)[pad .. pad + 8 * size]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
                    if (filtered and row % 2 != 0) {
                        const at = 32 + row * (n + 8) * size;
                        for (actual.slice(u8, bytes + 64)[at .. at + n * size]) |v| try std.testing.expectEqual(@as(u8, 0xa5), v);
                    }
                }
                for (scratch.slice(u8, scratch_bytes + 32)[0..16]) |v| try std.testing.expectEqual(@as(u8, 0xcd), v);
                for (scratch.slice(u8, scratch_bytes + 32)[scratch_bytes + 16 ..]) |v| try std.testing.expectEqual(@as(u8, 0xcd), v);
            }
        }
    }
}

test "shared prefill preserves gather and scatter maps across a partial tile" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (p.prefill == null) return error.SharedPrefillUnavailable;
    var legacy = p;
    legacy.prefill = null;
    const queue = try device.queue();
    defer queue.deinit();
    const w = try Owned.init(device, 3, .mul1, 0x676174686572);
    defer w.deinit();
    const x = try input(device, 3);
    defer x.deinit();
    const count = 33;
    const gather = try device.buffer(count * 4 + 16, mtl.ResourceOptions.shared);
    defer gather.deinit();
    const scatter = try device.buffer(count * 4 + 16, mtl.ResourceOptions.shared);
    defer scatter.deinit();
    for (0..count) |i| {
        gather.slice(u32, count + 4)[4 + i] = @intCast((i * 5) % 3);
        scatter.slice(u32, count + 4)[4 + i] = @intCast(count - 1 - i);
    }
    for ([_]ex.Output{ .bf16, .f32 }) |kind| {
        const size: usize = if (kind == .bf16) 2 else 4;
        const bytes = count * (n + 8) * size + 64;
        const actual = try device.buffer(bytes, mtl.ResourceOptions.shared);
        defer actual.deinit();
        const original = try device.buffer(bytes, mtl.ResourceOptions.shared);
        defer original.deinit();
        @memset(actual.slice(u8, bytes), 0xa5);
        @memset(original.slice(u8, bytes), 0xa5);
        const scratch = try device.buffer(try ex.scratchBytes(k, n, count), mtl.ResourceOptions.shared);
        defer scratch.deinit();
        const r: ex.Rows = .{ .count = count, .x_stride = k + 16, .y_stride = n + 8, .output = kind, .input_ids = .{ .buffer = gather, .offset = 16 }, .output_ids = .{ .buffer = scatter, .offset = 16 }, .input_rows = 3, .output_rows = count };
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = actual, .offset = 32 }, .{ .buffer = scratch }, r);
        e.barrier();
        try ex.encode(&legacy, e, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = original, .offset = 32 }, .{ .buffer = scratch }, r);
        try complete(cb, e);
        try std.testing.expectEqualSlices(u8, original.slice(u8, bytes), actual.slice(u8, bytes));
    }
}

/// Four rounds feed the plain product into the next input while routed output stays separate; serial is the control.
fn chainedRounds(queue: mtl.Queue, p: *const ex.Pipelines, a: ex.Matrix, b: ex.Matrix, descriptors: mtl.Buffer, ids: mtl.Buffer, xs: [5]mtl.Buffer, outs: [4]mtl.Buffer, scratch: [2]mtl.Buffer, concurrent: bool) !void {
    const tokens = 64;
    const cb = queue.commandBuffer();
    const e = cb.compute(if (concurrent) .concurrent else .serial);
    ex.useMatrix(e, a);
    ex.useMatrix(e, b);
    for (0..4) |round| {
        const x: ex.View = .{ .buffer = xs[round] };
        const next: ex.View = .{ .buffer = xs[round + 1] };
        const out: ex.View = .{ .buffer = outs[round] };
        const plain_rows: ex.Rows = .{ .count = tokens, .x_stride = k, .y_stride = k };
        const routed_rows: ex.Rows = .{ .count = tokens, .x_stride = k, .y_stride = n, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 1 } };
        const experts: ex.Experts = .{ .descriptors = .{ .buffer = descriptors }, .count = 2, .k = k, .n = n };
        if (concurrent) {
            try ex.encodeJobs(p, e, &.{
                .{ .weights = .{ .experts = experts }, .x = x, .y = out, .scratch = .{ .buffer = scratch[1] }, .rows = routed_rows },
                .{ .weights = .{ .matrix = a }, .x = x, .y = next, .scratch = .{ .buffer = scratch[0] }, .rows = plain_rows },
            });
        } else {
            try ex.encodeIndexed(p, e, experts, x, out, .{ .buffer = scratch[0] }, routed_rows);
            try ex.encode(p, e, a, x, next, .{ .buffer = scratch[0] }, plain_rows);
        }
    }
    try complete(cb, e);
}

test "concurrent product groups keep serial bits across chained rounds" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const tokens = 64;
    const a = try Owned.init(device, 5, .mul1, 61);
    defer a.deinit();
    const b0 = try Owned.init(device, 4, .mul1, 62);
    defer b0.deinit();
    const b1 = try Owned.init(device, 6, .mul1, 63);
    defer b1.deinit();
    const descriptors = try device.buffer(2 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, 2)[0] = try ex.descriptor(b0.matrix);
    descriptors.slice(ex.ExpertDescriptor, 2)[1] = try ex.descriptor(b1.matrix);
    const ids = try device.buffer(tokens * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    for (ids.slice(u32, tokens), 0..) |*v, i| v.* = @intCast((i / 3) % 2);
    var xs: [2][5]mtl.Buffer = undefined;
    var outs: [2][4]mtl.Buffer = undefined;
    var made_x: usize = 0;
    var made_o: usize = 0;
    defer for (0..made_x) |i| xs[i / 5][i % 5].deinit();
    defer for (0..made_o) |i| outs[i / 4][i % 4].deinit();
    for (0..2) |run| {
        for (0..5) |i| {
            xs[run][i] = try device.buffer(tokens * k * 2, mtl.ResourceOptions.shared);
            made_x += 1;
            // The upper half of every round's input stays this pattern; the plain product writes the lower half.
            for (xs[run][i].slice(u16, tokens * k), 0..) |*v, j| v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast((j + 7 * i) % 41)) - 20)) / 8);
        }
        for (0..4) |i| {
            outs[run][i] = try device.buffer(tokens * n * 2, mtl.ResourceOptions.shared);
            made_o += 1;
        }
    }
    var scratch: [2]mtl.Buffer = undefined;
    scratch[0] = try device.buffer(try ex.scratchBytes(k, n, tokens), mtl.ResourceOptions.shared);
    defer scratch[0].deinit();
    scratch[1] = try device.buffer(try ex.scratchBytes(k, n, tokens), mtl.ResourceOptions.shared);
    defer scratch[1].deinit();
    try chainedRounds(queue, &p, a.matrix, b0.matrix, descriptors, ids, xs[0], outs[0], scratch, false);
    const initial = try std.testing.allocator.dupe(u16, xs[0][0].slice(u16, tokens * k));
    defer std.testing.allocator.free(initial);
    for (0..8) |_| {
        for (1..5) |i| for (xs[1][i].slice(u16, tokens * k), 0..) |*v, j| {
            v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast((j + 7 * i) % 41)) - 20)) / 8);
        };
        for (0..4) |i| @memset(outs[1][i].slice(u16, tokens * n), 0);
        try chainedRounds(queue, &p, a.matrix, b0.matrix, descriptors, ids, xs[1], outs[1], scratch, true);
        for (1..5) |i| try std.testing.expectEqualSlices(u16, xs[0][i].slice(u16, tokens * k), xs[1][i].slice(u16, tokens * k));
        for (0..4) |i| try std.testing.expectEqualSlices(u16, outs[0][i].slice(u16, tokens * n), outs[1][i].slice(u16, tokens * n));
    }
    try std.testing.expectEqualSlices(u16, initial, xs[1][0].slice(u16, tokens * k));
}

test "a product group whose jobs share scratch or write another job's input is rejected" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const w = try Owned.init(device, 4, .mul1, 71);
    defer w.deinit();
    const x = try input(device, 2);
    defer x.deinit();
    const y = try device.buffer(2 * k * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const z = try device.buffer(2 * n * 2, mtl.ResourceOptions.shared);
    defer z.deinit();
    const scratch = try device.buffer(2 * try ex.scratchBytes(k, n, 2), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const group_rows: ex.Rows = .{ .count = 2, .x_stride = k + 16, .y_stride = n };
    const second: ex.View = .{ .buffer = scratch, .offset = try ex.scratchBytes(k, n, 2) };
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    // Both jobs on the same scratch.
    try std.testing.expectError(error.AliasedExl3View, ex.encodeJobs(&p, e, &.{
        .{ .weights = .{ .matrix = w.matrix }, .x = .{ .buffer = x, .offset = 32 }, .y = .{ .buffer = z }, .scratch = .{ .buffer = scratch }, .rows = group_rows },
        .{ .weights = .{ .matrix = w.matrix }, .x = .{ .buffer = x, .offset = 32 }, .y = .{ .buffer = y }, .scratch = .{ .buffer = scratch }, .rows = group_rows },
    }));
    // The second job writes the first job's input.
    try std.testing.expectError(error.AliasedExl3View, ex.encodeJobs(&p, e, &.{
        .{ .weights = .{ .matrix = w.matrix }, .x = .{ .buffer = y }, .y = .{ .buffer = z }, .scratch = .{ .buffer = scratch }, .rows = .{ .count = 2, .x_stride = k, .y_stride = n } },
        .{ .weights = .{ .matrix = w.matrix }, .x = .{ .buffer = x, .offset = 32 }, .y = .{ .buffer = y }, .scratch = second, .rows = group_rows },
    }));
    // Disjoint jobs are accepted.
    try ex.encodeJobs(&p, e, &.{
        .{ .weights = .{ .matrix = w.matrix }, .x = .{ .buffer = x, .offset = 32 }, .y = .{ .buffer = z }, .scratch = .{ .buffer = scratch }, .rows = group_rows },
        .{ .weights = .{ .matrix = w.matrix }, .x = .{ .buffer = x, .offset = 32 }, .y = .{ .buffer = y }, .scratch = second, .rows = group_rows },
    });
    try complete(cb, e);
}

test {
    _ = @import("exl3_partition_gpu_test.zig");
}
