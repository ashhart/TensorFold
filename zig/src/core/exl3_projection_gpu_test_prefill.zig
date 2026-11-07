const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const decoder = @import("exl3_projection_decode.zig");
const support = @import("exl3_projection_gpu_test_support.zig");
const k = support.k;
const n = support.n;
const rows = support.rows;
const Owned = support.Owned;
const bf16 = support.bf16;
const input = support.input;
const complete = support.complete;
const reference = support.reference;

const StateWindowInput = extern struct { lo: u32, hi: u32, off: u32, reserved: u32 = 0 };

test "Metal EXL3 mapped windows match independent 64-bit extraction at every offset" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, sources.core_exl3_projection, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, "tf_exl3_state_window_check", false);
    defer pipeline.deinit();
    if (pipeline.maxThreads() < 256) return error.UnsupportedProbeThreads;
    var patterns: [1095][2]u32 = undefined;
    for (0..32) |bit| {
        patterns[bit] = .{ @as(u32, 1) << @intCast(bit), 0 };
        patterns[32 + bit] = .{ 0, @as(u32, 1) << @intCast(bit) };
    }
    const special = [_][2]u32{
        .{ 0, 0 },                   .{ 0xffffffff, 0 },          .{ 0, 0xffffffff },          .{ 0xffffffff, 0xffffffff },
        .{ 0xaaaaaaaa, 0x55555555 }, .{ 0x55555555, 0xaaaaaaaa }, .{ 0x80000001, 0x80000001 },
    };
    for (special, 0..) |p, i| patterns[64 + i] = p;
    var prng = std.Random.DefaultPrng.init(0x7472656c6c6973);
    for (patterns[71..]) |*p| p.* = .{ prng.random().int(u32), prng.random().int(u32) };
    const count = patterns.len * 32;
    const state_inputs = try device.buffer((count + 2) * @sizeOf(StateWindowInput), mtl.ResourceOptions.shared);
    defer state_inputs.deinit();
    const state_outputs = try device.buffer((count + 16) * @sizeOf(u16), mtl.ResourceOptions.shared);
    defer state_outputs.deinit();
    @memset(state_outputs.slice(u16, count + 16), 0xa55a);
    const expected = try std.testing.allocator.alloc(u16, count);
    defer std.testing.allocator.free(expected);
    const records = state_inputs.slice(StateWindowInput, count + 2)[1..][0..count];
    for (0..32) |off| for (patterns, 0..) |p, i| {
        const at = off * patterns.len + i;
        records[at] = .{ .lo = p[0], .hi = p[1], .off = @intCast(off) };
        const joined: u64 = (@as(u64, p[0]) << 32) | p[1];
        expected[at] = @truncate(joined >> @intCast(48 - off));
    };
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    e.setPipeline(pipeline);
    e.setBuffer(state_inputs, 16, 0);
    e.setBuffer(state_outputs, 16, 1);
    e.setValue(@as(u32, count), 2);
    e.dispatchThreads(.{ .width = count }, .{ .width = 256 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.ProbeGpuFailure;
    for (expected, state_outputs.slice(u16, count + 16)[8..][0..count], 0..) |want, got, i| {
        if (want != got) {
            std.debug.print("state mismatch off={d} index={d} expected={d} actual={d}\n", .{ records[i].off, i, want, got });
            return error.MappedStateMismatch;
        }
    }
    for (state_outputs.slice(u16, count + 16)[0..8]) |guard| try std.testing.expectEqual(@as(u16, 0xa55a), guard);
    for (state_outputs.slice(u16, count + 16)[count + 8 ..]) |guard| try std.testing.expectEqual(@as(u16, 0xa55a), guard);
    std.debug.print("mapped_state_cases={d} offsets=32 mismatches=0 guards=unchanged\n", .{count});
}

// Larger and resumed windows must retain the original per-row arithmetic and untouched padding.
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

test "the fused SwiGLU + down rotate writes the two-pass rotated rows bit for bit" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const w = try Owned.init(device, 6, .mul1, 0x53574752);
    defer w.deinit();
    const count = 6;
    const descriptors = try device.buffer(@sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, 1)[0] = try ex.descriptor(w.matrix);
    // Gate and up rows span the sigmoid's linear and saturating ranges; the weights are not one.
    const gate = try device.buffer(count * (k + 8) * 2 + 32, mtl.ResourceOptions.shared);
    defer gate.deinit();
    const up = try device.buffer(count * (k + 8) * 2 + 32, mtl.ResourceOptions.shared);
    defer up.deinit();
    for (0..count) |row| {
        for (0..k) |i| {
            const a: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37 + row * 11) % 29)) - 14)) / 16;
            const b: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 53 + row * 7) % 31)) - 15)) / 8;
            gate.slice(u16, count * (k + 8) + 16)[16 + row * (k + 8) + i] = bf16(a);
            up.slice(u16, count * (k + 8) + 16)[16 + row * (k + 8) + i] = bf16(b);
        }
    }
    const weights = try device.buffer(count * 4 + 16, mtl.ResourceOptions.shared);
    defer weights.deinit();
    for (0..count) |row| weights.slice(f32, count + 4)[4 + row] = 0.25 + 0.1 * @as(f32, @floatFromInt(row));
    const routes = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer routes.deinit();
    @memset(routes.slice(u32, count), 0);
    // A destination permutation: the fused pass scatters a token-major row into its expert tile's slot.
    const destinations = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer destinations.deinit();
    const dest = [_]u32{ 3, 0, 4, 1, 5, 2 };
    @memcpy(destinations.slice(u32, count), &dest);
    const activation = try device.buffer(count * k * 2, mtl.ResourceOptions.shared);
    defer activation.deinit();
    const plain = try device.buffer(count * n * 2, mtl.ResourceOptions.shared);
    defer plain.deinit();
    const fused = try device.buffer(count * n * 2, mtl.ResourceOptions.shared);
    defer fused.deinit();
    const scratch_plain = try device.buffer(try ex.scratchBytes(k, n, count), mtl.ResourceOptions.shared);
    defer scratch_plain.deinit();
    const scratch_fused = try device.buffer(try ex.scratchBytes(k, n, count), mtl.ResourceOptions.shared);
    defer scratch_fused.deinit();
    var opts = mtl.CompileOptions.mlx();
    opts.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_ops, opts);
    defer lib.deinit();
    const swiglu = try mtl.Pipeline.init(device, lib, "dsv41_swiglu", false);
    defer swiglu.deinit();
    const dims = ex.Experts{ .descriptors = .{ .buffer = descriptors }, .count = 1, .k = k, .n = n };
    // The shipped two-pass path: swiglu into the activation buffer, then the down call's own rotate_in.
    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        ex.useMatrix(e, w.matrix);
        e.setPipeline(swiglu);
        e.setBuffer(gate, 16, 0);
        e.setBuffer(up, 16, 1);
        e.setBuffer(weights, 16, 2);
        e.setBuffer(activation, 0, 3);
        e.setValue([6]u32{ count, k, k + 8, k, 1, 0 }, 4);
        e.dispatchThreads(.{ .width = k, .height = count }, .{ .width = 256 });
        e.barrier();
        try ex.encodeIndexed(&p, e, dims, .{ .buffer = activation }, .{ .buffer = plain }, .{ .buffer = scratch_plain }, .{ .count = count, .x_stride = k, .y_stride = n, .routes = .{ .ids = .{ .buffer = routes } } });
        try complete(cb, e);
    }
    // The fused path: one pass writes the rotated rows at their destinations, the down call skips rotate_in.
    {
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        ex.useMatrix(e, w.matrix);
        ex.encodeSwiGLUPreRotate(&p, e, .{ .buffer = gate, .offset = 16 }, .{ .buffer = up, .offset = 16 }, .{ .buffer = weights, .offset = 16 }, .{ .buffer = routes }, .{ .buffer = destinations }, .{ .buffer = descriptors }, .{ .buffer = scratch_fused }, .{ .rows = count, .k = k, .input_stride = k + 8, .output_stride = k, .experts = 1 });
        e.barrier();
        try ex.encodeIndexed(&p, e, dims, .{ .buffer = scratch_fused }, .{ .buffer = fused }, .{ .buffer = scratch_fused }, .{ .count = count, .x_stride = k, .y_stride = n, .routes = .{ .ids = .{ .buffer = routes } }, .prerotated = true });
        try complete(cb, e);
    }
    var inverse: [count]usize = undefined;
    for (dest, 0..) |d, r| inverse[d] = r;
    const plain_rows = plain.slice(u16, count * n);
    const fused_rows = fused.slice(u16, count * n);
    for (0..count) |r| try std.testing.expectEqualSlices(u16, plain_rows[inverse[r] * n ..][0..n], fused_rows[r * n ..][0..n]);
}
