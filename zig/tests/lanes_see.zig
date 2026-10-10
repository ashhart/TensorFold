//! Shared context, data masks and lane-local rotary positions on small synthetic buffers only.
const std = @import("std");
const mtl = @import("metal");
const shared = @import("shared_attention");
const kv = @import("shared_kv");
const a = std.heap.page_allocator;
const pad = 16;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
fn data(i: usize, salt: usize) u16 {
    return toBf(@as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + salt * 13) % 29)) - 14)) / 32.0);
}
fn buffer(device: mtl.Device, bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes + 2 * pad, opts);
    @memset(b.contents()[0..b.length()], 0xa5);
    return b;
}
fn ref(b: mtl.Buffer, off: usize) shared.Ref {
    return .{ .buf = b, .off = pad + off };
}
fn values(b: mtl.Buffer, comptime T: type, len: usize) []T {
    return @as([*]T, @ptrCast(@alignCast(b.contents() + pad)))[0..len];
}
fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.debug.print("GPU failed: {s}\n", .{message});
        return error.GpuFailed;
    }
}
fn guards(b: mtl.Buffer) !void {
    for (b.contents()[0..pad]) |x| if (x != 0xa5) return error.GuardOverwrite;
    for (b.contents()[b.length() - pad .. b.length()]) |x| if (x != 0xa5) return error.GuardOverwrite;
}

fn checkRope(s: shared.Shape, input: []const u16, output: []const u16, positions: []const u32, freq: []const f32, heads: usize) !void {
    for (0..positions.len) |r| for (0..heads) |h| for (0..s.head_dim) |d| {
        const at = (r * heads + h) * s.head_dim + d;
        if (d >= s.rotary_dim) {
            if (input[at] != output[at]) return error.RopeTailChanged;
            continue;
        }
        const pair = if (s.rope == .interleaved) d / 2 else d % (s.rotary_dim / 2);
        const first = if (s.rope == .interleaved) pair * 2 else pair;
        const second = if (s.rope == .interleaved) first + 1 else first + s.rotary_dim / 2;
        const base = at - d;
        const x: f64 = fromBf(input[base + first]);
        const y: f64 = fromBf(input[base + second]);
        const angle: f32 = @as(f32, @floatFromInt(positions[r])) * freq[pair];
        const c = @cos(@as(f64, angle));
        const sn = @sin(@as(f64, angle));
        const high = if (s.rope == .interleaved) d % 2 != 0 else d >= s.rotary_dim / 2;
        const want = if (high) x * sn + y * c else x * c - y * sn;
        const got: f64 = fromBf(output[at]);
        // One final bf16 cast; allow its rounding and the fp32 trig/FMA rounding only.
        if (@abs(got - want) > 0.004 * @abs(want) + 0.00001) return error.RopeMismatch;
    };
}

fn cpuAttention(s: shared.Shape, q: []const u16, k: []const u16, v: []const u16, mask: []const u8, keys: usize, row: usize, head: usize, scale: f32, out: []f64) !void {
    const scores = try a.alloc(f64, keys);
    defer a.free(scores);
    const kh = head / (s.query_heads / s.kv_heads);
    var maximum: f64 = -std.math.inf(f64);
    for (0..keys) |key| {
        var score: f64 = 0;
        if (mask[row * keys + key] == 0) {
            scores[key] = -std.math.inf(f64);
            continue;
        }
        for (0..s.head_dim) |d| score += @as(f64, fromBf(q[(row * s.query_heads + head) * s.head_dim + d])) * fromBf(k[(key * s.kv_heads + kh) * s.head_dim + d]);
        scores[key] = score * scale;
        maximum = @max(maximum, scores[key]);
    }
    @memset(out, 0);
    if (maximum == -std.math.inf(f64)) return;
    var denom: f64 = 0;
    for (0..keys) |key| {
        if (mask[row * keys + key] == 0) continue;
        const probability = @exp(scores[key] - maximum);
        denom += probability;
        for (0..s.head_dim) |d| out[d] += probability * @as(f64, fromBf(v[(key * s.kv_heads + kh) * s.head_dim + d]));
    }
    for (out) |*x| x.* /= denom;
}

fn outputValue(b: mtl.Buffer, index: usize, output: shared.Output) f32 {
    return if (output == .f32) values(b, f32, index + 1)[index] else fromBf(values(b, u16, index + 1)[index]);
}

const Counts = struct { ordinary: usize = 0, bf16_values: usize = 0, max_error: f64 = 0, max_fraction: f64 = 0, cases: usize = 0, rows: usize = 0, cpu_values: usize = 0, rope_values: usize = 0, prompt_bytes: usize = 0 };

fn run(device: mtl.Device, queue: mtl.Queue, n: usize, prompt: usize, dims: [4]usize, mode: shared.Rope, output: shared.Output, counts: *Counts) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const s = shared.Shape{ .query_heads = dims[0], .kv_heads = dims[1], .head_dim = dims[2], .rotary_dim = dims[3], .rope = mode, .capacity = prompt + n * 4, .output = output };
    const p = try shared.Attention.init(a, device, s);
    defer p.deinit();
    var c = try kv.Cache.init(a, .{ .lanes = n, .prompt_capacity = prompt, .round_capacity = 4, .kv_heads = s.kv_heads, .head_dim = s.head_dim });
    defer c.deinit();
    const width = s.kv_heads * s.head_dim;
    const qwidth = s.query_heads * s.head_dim;
    const ck = try buffer(device, s.capacity * width * 2);
    defer ck.deinit();
    const cv = try buffer(device, s.capacity * width * 2);
    defer cv.deinit();
    @memset(values(ck, u16, s.capacity * width), 0x7fc1); // hidden slots are NaNs and must not be read
    @memset(values(cv, u16, s.capacity * width), 0x7fc1);
    const max_rows = @max(n, prompt);
    const raw_k = try buffer(device, max_rows * width * 2);
    defer raw_k.deinit();
    const prepared_k = try buffer(device, max_rows * width * 2);
    defer prepared_k.deinit();
    const raw_v = try buffer(device, max_rows * width * 2);
    defer raw_v.deinit();
    const positions = try buffer(device, max_rows * 4);
    defer positions.deinit();
    const frequencies = try buffer(device, @max(s.rotary_dim / 2 * 4, 4));
    defer frequencies.deinit();
    const freq = values(frequencies, f32, @max(s.rotary_dim / 2, 1));
    for (freq, 0..) |*v, d| v.* = 1.0 / @as(f32, @floatFromInt(1 + d * 7));

    if (prompt > 0) {
        const raw = values(raw_k, u16, prompt * width);
        const val = values(raw_v, u16, prompt * width);
        const pos = values(positions, u32, prompt);
        for (raw, 0..) |*v, i| v.* = data(i, 1);
        for (val, 0..) |*v, i| v.* = data(i, 2);
        for (pos, 0..) |*v, i| v.* = @intCast(i);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try p.rotate(e, prompt, s.kv_heads, ref(raw_k, 0), ref(positions, 0), ref(frequencies, 0), ref(prepared_k, 0));
        e.end();
        try finish(cb);
        try checkRope(s, raw, values(prepared_k, u16, prompt * width), pos, freq, s.kv_heads);
        counts.rope_values += prompt * width;
        @memcpy(values(ck, u16, prompt * width), values(prepared_k, u16, prompt * width));
        @memcpy(values(cv, u16, prompt * width), val);
        const ids = try a.alloc(u32, prompt);
        defer a.free(ids);
        for (ids, 0..) |*v, i| v.* = @intCast(10 + i);
        try c.appendPrompt(ids, values(prepared_k, u16, prompt * width), val);
    }
    const prompt_k = try a.dupe(u16, values(ck, u16, prompt * width));
    defer a.free(prompt_k);
    const prompt_v = try a.dupe(u16, values(cv, u16, prompt * width));
    defer a.free(prompt_v);
    var last: kv.Round = undefined;
    for (0..3) |round| {
        const active = if (round == 1 and n > 1) c.shape.activeMask() & ~@as(u16, 2) else c.shape.activeMask();
        const ticket = try c.begin(active);
        var slots: [16]u32 = undefined;
        var owners: [16]usize = undefined;
        var used: usize = 0;
        const pos = values(positions, u32, n);
        for (0..n) |lane| if (active & (@as(u16, 1) << @intCast(lane)) != 0) {
            owners[used] = lane;
            slots[used] = @intCast(try c.slot(ticket, lane));
            pos[used] = ticket.positions[lane];
            for (0..width) |d| {
                values(raw_k, u16, n * width)[used * width + d] = data(d, 3 + round * 17 + lane);
                values(raw_v, u16, n * width)[used * width + d] = data(d, 5 + round * 19 + lane);
            }
            used += 1;
        };
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try p.rotate(e, used, s.kv_heads, ref(raw_k, 0), ref(positions, 0), ref(frequencies, 0), ref(prepared_k, 0));
        e.barrier();
        try p.appendKv(e, slots[0..used], ref(prepared_k, 0), ref(raw_v, 0), ref(ck, 0), ref(cv, 0));
        e.end();
        try finish(cb);
        try checkRope(s, values(raw_k, u16, used * width), values(prepared_k, u16, used * width), pos[0..used], freq, s.kv_heads);
        counts.rope_values += used * width;
        for (0..used) |i| try c.put(ticket, owners[i], @intCast(100 + round * 16 + owners[i]), values(prepared_k, u16, used * width)[i * width ..][0..width], values(raw_v, u16, used * width)[i * width ..][0..width]);
        try std.testing.expectEqualSlices(u16, prompt_k, values(ck, u16, prompt * width));
        try std.testing.expectEqualSlices(u16, prompt_v, values(cv, u16, prompt * width));
        for (0..used) |i| {
            const off = @as(usize, slots[i]) * width;
            try std.testing.expectEqualSlices(u16, values(prepared_k, u16, used * width)[i * width ..][0..width], values(ck, u16, s.capacity * width)[off..][0..width]);
            try std.testing.expectEqualSlices(u16, values(raw_v, u16, used * width)[i * width ..][0..width], values(cv, u16, s.capacity * width)[off..][0..width]);
        }
        if (round < 2) try c.commit(ticket) else last = ticket;
    }
    const view = try c.current(last);
    const raw_q = try buffer(device, n * qwidth * 2);
    defer raw_q.deinit();
    const q = try buffer(device, n * qwidth * 2);
    defer q.deinit();
    const masks = try buffer(device, n * view.keys);
    defer masks.deinit();
    const elem: usize = if (output == .f32) 4 else 2;
    const out = try buffer(device, n * qwidth * elem);
    defer out.deinit();
    const serial = try buffer(device, n * qwidth * elem);
    defer serial.deinit();
    const raw = values(raw_q, u16, n * qwidth);
    for (raw, 0..) |*v, i| v.* = data(i, 73);
    const qp = values(positions, u32, n);
    @memcpy(qp, last.positions[0..n]);
    const mask = values(masks, u8, n * view.keys);
    try c.visibility(view, n, mask);
    if (n > 1) {
        @memset(mask[0..view.keys], 0);
        mask[last.base + n - 1] = 1; // current peer, not own lane
        @memset(mask[(n - 1) * view.keys ..][0..view.keys], 0); // fully hidden row returns exact zero
        for (1..n - 1) |row| {
            mask[row * view.keys + row % view.keys] = 0;
        }
    }
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(s.head_dim)));
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try p.rotate(e, n, s.query_heads, ref(raw_q, 0), ref(positions, 0), ref(frequencies, 0), ref(q, 0));
    e.barrier();
    try p.encode(e, n, view.keys, view.keys, scale, ref(q, 0), ref(ck, 0), ref(cv, 0), ref(masks, 0), ref(out, 0));
    for (0..n) |row| try p.encode(e, 1, view.keys, view.keys, scale, ref(q, row * qwidth * 2), ref(ck, 0), ref(cv, 0), ref(masks, row * view.keys), ref(serial, row * qwidth * elem));
    if (n == 1) try p.ordinary(e, view.keys, scale, ref(q, 0), ref(ck, 0), ref(cv, 0), ref(serial, 0));
    e.end();
    try finish(cb);
    try checkRope(s, raw, values(q, u16, n * qwidth), qp, freq, s.query_heads);
    counts.rope_values += n * qwidth;
    try std.testing.expectEqualSlices(u8, out.contents()[pad..][0 .. n * qwidth * elem], serial.contents()[pad..][0 .. n * qwidth * elem]);
    const expected = try a.alloc(f64, s.head_dim);
    defer a.free(expected);

    for (0..n) |row| for (0..s.query_heads) |head| {
        try cpuAttention(s, values(q, u16, n * qwidth), values(ck, u16, s.capacity * width), values(cv, u16, s.capacity * width), mask, view.keys, row, head, scale, expected);
        for (expected, 0..) |want, d| {
            const actual: f64 = outputValue(out, (row * s.query_heads + head) * s.head_dim + d, output);
            var magnitude: f64 = 0;
            for (0..view.keys) |key| if (mask[row * view.keys + key] != 0) {
                const kh = head / (s.query_heads / s.kv_heads);
                magnitude = @max(magnitude, @abs(@as(f64, fromBf(values(cv, u16, s.capacity * width)[(key * s.kv_heads + kh) * s.head_dim + d]))));
            };
            const h: f64 = @floatFromInt(s.head_dim + 8 * view.keys + 16);
            const gamma = h * 0x1p-24 / (1 - h * 0x1p-24);
            const bound = gamma * magnitude + 0x1p-120;
            const err = @abs(actual - want);
            const allowed = if (output == .f32) bound else bound + 0.004 * @abs(want) + 0.00001;
            if (!std.math.isFinite(actual) or err > allowed) return error.AttentionCpuMismatch;
            if (output == .f32) {
                counts.max_error = @max(counts.max_error, err);
                counts.max_fraction = @max(counts.max_fraction, err / bound);
                counts.cpu_values += 1;
            } else counts.bf16_values += 1;
        }
    };
    if (n > 1) for (0..s.query_heads) |head| for (0..s.head_dim) |d| {
        const kh = head / (s.query_heads / s.kv_heads);
        try std.testing.expectEqual(fromBf(values(cv, u16, s.capacity * width)[((last.base + n - 1) * s.kv_heads + kh) * s.head_dim + d]), outputValue(out, head * s.head_dim + d, output));
        try std.testing.expectEqual(@as(f32, 0), outputValue(out, ((n - 1) * s.query_heads + head) * s.head_dim + d, output));
    };
    for ([_]mtl.Buffer{ ck, cv, raw_k, prepared_k, raw_v, positions, frequencies, raw_q, q, masks, out, serial }) |b| try guards(b);
    if (n == 1) counts.ordinary += 1;
    counts.prompt_bytes += prompt * width * 4;
    counts.rows += n;
    counts.cases += 1;
    std.debug.print("PASS N{d} P{d} QH{d}/KVH{d} D{d} RoPE{d} {s} {s}: data masks, peer-current visibility, serial-row equality, prompt unchanged\n", .{ n, prompt, s.query_heads, s.kv_heads, s.head_dim, s.rotary_dim, @tagName(mode), @tagName(output) });
}

pub fn main(_: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var counts: Counts = .{};
    for ([_][4]usize{ .{ 2, 1, 32, 0 }, .{ 4, 2, 64, 32 }, .{ 8, 2, 128, 128 }, .{ 2, 2, 256, 64 } }) |dims| for ([_]shared.Rope{ .split_half, .interleaved }) |mode| for (1..17) |n| {
        try run(device, queue, n, if (n % 3 == 0) 0 else if (n % 3 == 1) 5 else 33, dims, mode, .f32, &counts);
    };
    for ([_][4]usize{ .{ 2, 1, 32, 0 }, .{ 4, 2, 64, 32 }, .{ 8, 2, 128, 128 }, .{ 2, 2, 256, 64 } }) |dims| for ([_]shared.Rope{ .split_half, .interleaved }) |mode| {
        for ([_]usize{ 0, 1, 33 }) |prompt| try run(device, queue, 1, prompt, dims, mode, .f32, &counts);
        try run(device, queue, 1, 5, dims, mode, .bf16, &counts);
        try run(device, queue, 8, 5, dims, mode, .bf16, &counts);
    };
    std.debug.print("PASS {d} cases, {d} ordinary N=1 comparisons, {d} batch/serial rows, {d} fp32 CPU attention values, {d} bf16 values, {d} RoPE values, {d} prompt bytes preserved, max fp32 error {d:.9}, max bound fraction {d:.6}\n", .{ counts.cases, counts.ordinary, counts.rows, counts.cpu_values, counts.bf16_values, counts.rope_values, counts.prompt_bytes, counts.max_error, counts.max_fraction });
}
