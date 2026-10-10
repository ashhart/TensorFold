//! CPU controls and frozen-kernel GPU comparisons use complete ABI bytes and bounded synthetic vocabularies.
const std = @import("std");
const mtl = @import("metal");
const cpu = @import("bf16_topk");
const gpu = @import("bf16_topk_gpu");
const control_source = @embedFile("bf16_topk_control.metal");
const Mode = enum { random, ties, ascending, descending, zeros, nonfinite };
fn key(raw: u16) u32 {
    const bits: u32 = if (raw & 0x7fff == 0) 0 else raw;
    return if (bits & 0x8000 != 0) ~bits & 0xffff else bits | 0x8000;
}
fn before(a: cpu.Entry, b: cpu.Entry) bool {
    return key(@intCast(@as(u32, @bitCast(a.score)) >> 16)) > key(@intCast(@as(u32, @bitCast(b.score)) >> 16)) or (a.score == b.score and a.token < b.token);
}
fn sift(items: []cpu.Entry) void {
    var at: usize = 0;
    while (2 * at + 1 < items.len) {
        var child = 2 * at + 1;
        if (child + 1 < items.len and before(items[child], items[child + 1])) child += 1;
        if (!before(items[at], items[child])) break;
        std.mem.swap(cpu.Entry, &items[at], &items[child]);
        at = child;
    }
}
fn localMerge(words: []const u16, k: u32) !cpu.Row {
    var heads: [256][16]cpu.Entry = @splat(@splat(.{}));
    var mask: u32 = 0;
    for (0..256) |lane| {
        var token = lane;
        while (token < words.len) : (token += 256) {
            const raw = words[token];
            if (raw & 0x7f80 == 0x7f80) {
                mask |= if (raw & 0x7f != 0) @as(u32, 1) else if (raw & 0x8000 == 0) @as(u32, 2) else 4;
                continue;
            }
            const item = cpu.Entry{ .token = @intCast(token), .score = @bitCast(@as(u32, raw) << 16) };
            var count: usize = 0;
            while (count < k and heads[lane][count].token != cpu.invalid_token) count += 1;
            if (count < k) {
                var at = count;
                heads[lane][at] = item;
                while (at > 0) {
                    const parent = (at - 1) / 2;
                    if (!before(heads[lane][parent], heads[lane][at])) break;
                    std.mem.swap(cpu.Entry, &heads[lane][parent], &heads[lane][at]);
                    at = parent;
                }
            } else if (before(item, heads[lane][0])) {
                heads[lane][0] = item;
                sift(heads[lane][0..count]);
            }
        }
        var count: usize = 0;
        while (count < k and heads[lane][count].token != cpu.invalid_token) count += 1;
        while (count > 1) : (count -= 1) {
            std.mem.swap(cpu.Entry, &heads[lane][0], &heads[lane][count - 1]);
            sift(heads[lane][0 .. count - 1]);
        }
    }
    if (mask != 0) return .{ .nonfinite = mask };
    var out = cpu.Row{};
    var cursors: [256]u32 = @splat(0);
    for (0..k) |rank| {
        var best = cpu.Entry{};
        var owner: usize = 0;
        for (0..256) |lane| {
            if (cursors[lane] >= k) continue;
            const item = heads[lane][cursors[lane]];
            if (item.token != cpu.invalid_token and (best.token == cpu.invalid_token or before(item, best))) {
                best = item;
                owner = lane;
            }
        }
        if (best.token == cpu.invalid_token) return error.EmptyMerge;
        out.entries[rank] = best;
        out.count += 1;
        cursors[owner] += 1;
    }
    return out;
}
fn fill(words: []u16, p: cpu.Params, mode: Mode) void {
    @memset(words, 0x7fc1);
    for (0..p.rows) |row| {
        const x = words[row * p.stride ..][0..p.vocab];
        for (x, 0..) |*v, i| {
            const mixed = @as(u32, @truncate(i)) *% 1664525 +% @as(u32, @intCast(row)) *% 1013904223;
            var raw: u16 = @truncate(mixed >> 8);
            if (raw & 0x7f80 == 0x7f80) raw ^= 0x80;
            v.* = switch (mode) {
                .random, .nonfinite => raw,
                .ties => 0x3f80,
                .ascending => @intCast(i * 0x7f7f / p.vocab),
                .descending => @intCast(0x7f7f - i * 0x7f7f / p.vocab),
                .zeros => if (i % 2 == 0) 0x8000 else 0,
            };
        }
        if (mode == .nonfinite) {
            x[0] = 0x7fc1;
            x[p.vocab / 2] = 0x7f80;
            x[p.vocab - 1] = 0xff80;
        }
    }
}
const Guard = struct {
    buf: mtl.Buffer,
    off: usize,
    bytes: usize,
    fn init(device: mtl.Device, bytes: usize, off: usize) !Guard {
        const b = try device.buffer(off + bytes + 16, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(b.contents()[0 .. off + bytes + 16], 0xa5);
        return .{ .buf = b, .off = off, .bytes = bytes };
    }
    fn ref(g: Guard) gpu.Ref {
        return .{ .buf = g.buf, .off = g.off };
    }
    fn data(g: Guard) []u8 {
        return g.buf.contents()[g.off .. g.off + g.bytes];
    }
    fn check(g: Guard) !void {
        for (g.buf.contents()[0..g.off]) |v| if (v != 0xa5) return error.GuardChanged;
        for (g.buf.contents()[g.off + g.bytes ..][0..16]) |v| if (v != 0xa5) return error.GuardChanged;
    }
};
fn end(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
}
fn milliseconds(queue: mtl.Queue, ops: gpu.Ops, input: Guard, output: Guard, p: cpu.Params) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var active = true;
    errdefer if (active) e.end();
    for (0..8) |_| {
        try ops.encode(e, input.ref(), output.ref(), p);
        e.barrier();
    }
    active = false;
    try end(cb, e);
    return cb.gpuSeconds() * 1e3 / 8;
}
fn cpuCases(a: std.mem.Allocator) !usize {
    var cases: usize = 0;
    for ([_]u32{ 1, 15, 16 }) |rows| for ([_]u32{ 98592, 98624 }) |vocab| for ([_]u32{ 1, 16 }) |k| {
        const p = cpu.Params{ .rows = rows, .vocab = vocab, .k = k, .stride = vocab + 17 };
        const words = try a.alloc(u16, rows * p.stride);
        defer a.free(words);
        for (std.enums.values(Mode)) |mode| {
            fill(words, p, mode);
            for (0..rows) |row| {
                const x = words[row * p.stride ..][0..vocab];
                const oracle = try cpu.oracle(x, k);
                const actual = try localMerge(x, k);
                if (!std.mem.eql(u8, std.mem.asBytes(&oracle), std.mem.asBytes(&actual))) return error.CpuLocalMergeDifference;
            }
            cases += 1;
        }
    };
    return cases;
}
fn gpuCases(a: std.mem.Allocator) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const candidate = try gpu.Ops.init(device);
    defer candidate.deinit();
    const lib = try mtl.Library.fromSource(device, control_source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const old = gpu.Ops{ .pipeline = try mtl.Pipeline.init(device, lib, "tf_bf16_topk", false) };
    defer old.deinit();
    var total: usize = 0;
    for ([_]u32{ 1, 15, 16 }) |rows| for ([_]u32{ 98592, 98624 }) |vocab| for ([_]u32{ 1, 16 }) |k| {
        const p = cpu.Params{ .rows = rows, .vocab = vocab, .k = k, .stride = vocab + 17 };
        const words = try a.alloc(u16, rows * p.stride);
        defer a.free(words);
        const input = try Guard.init(device, words.len * 2, 14);
        defer input.buf.deinit();
        const out = try Guard.init(device, rows * @sizeOf(cpu.Row), 20);
        defer out.buf.deinit();
        const reference = try Guard.init(device, out.bytes, 4);
        defer reference.buf.deinit();
        var expected: [16]cpu.Row = undefined;
        for (std.enums.values(Mode)) |mode| {
            fill(words, p, mode);
            @memcpy(input.data(), std.mem.sliceAsBytes(words));
            for (0..rows) |row| expected[row] = try cpu.oracle(words[row * p.stride ..][0..vocab], k);
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            var active = true;
            errdefer if (active) e.end();
            try candidate.encode(e, input.ref(), out.ref(), p);
            try old.encode(e, input.ref(), reference.ref(), p);
            active = false;
            try end(cb, e);
            const want = std.mem.sliceAsBytes(expected[0..rows]);
            if (!std.mem.eql(u8, out.data(), want) or !std.mem.eql(u8, reference.data(), want)) return error.GpuTopKDifference;
            if (!std.mem.eql(u8, input.data(), std.mem.sliceAsBytes(words))) return error.InputChanged;
            try input.check();
            try out.check();
            try reference.check();
            total += 1;
            if (mode == .nonfinite) continue;
            _ = try milliseconds(queue, candidate, input, out, p);
            _ = try milliseconds(queue, old, input, reference, p);
            var newer: [5]f64 = undefined;
            var older: [5]f64 = undefined;
            for (0..5) |i| {
                if (i % 2 == 0) {
                    newer[i] = try milliseconds(queue, candidate, input, out, p);
                    older[i] = try milliseconds(queue, old, input, reference, p);
                } else {
                    older[i] = try milliseconds(queue, old, input, reference, p);
                    newer[i] = try milliseconds(queue, candidate, input, out, p);
                }
            }
            std.mem.sort(f64, &newer, {}, std.sort.asc(f64));
            std.mem.sort(f64, &older, {}, std.sort.asc(f64));
            std.debug.print("{{\"rows\":{d},\"vocab\":{d},\"k\":{d},\"mode\":\"{s}\",\"candidate_ms\":{d:.6},\"old_ms\":{d:.6},\"speedup\":{d:.3}}}\n", .{ rows, vocab, k, @tagName(mode), newer[2], older[2], older[2] / newer[2] });
        }
    };
    std.debug.print("GPU exact cases {d}, complete Row ABI and guards pass\n", .{total});
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedCpuOrGpu;
    if (std.mem.eql(u8, args[1], "--cpu")) {
        const cases = try cpuCases(init.gpa);
        std.debug.print("CPU local sorted-head merge: {d} cases pass, no GPU device\n", .{cases});
    } else if (std.mem.eql(u8, args[1], "--gpu")) try gpuCases(init.gpa) else return error.ExpectedCpuOrGpu;
}
