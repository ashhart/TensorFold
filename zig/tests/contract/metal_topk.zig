//! The bf16 top-k under the contract on Metal: bytes equal to its CPU oracle, rows holding NaN or infinities refused.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const cpu = tf.bf16_topk;
const gpu = tf.bf16_topk_gpu;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "a row holding NaN or an infinity returns no entries and the mask of what it held (1 NaN, 2 +Inf, 4 -Inf)";

/// Rows, entries a row (k) and vocabulary; 4,096 is where a row splits into spans and a merge.
pub fn spec() c.Spec {
    return .{
        .name = "metal bf16 top-k tf_bf16_topk",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = 16 }, .{ .size = 16 }, .{ .size = 4096 } },
        .served = &.{ .{ .rows = 16, .n = 16, .k = 248320 }, .{ .rows = 1, .n = 1, .k = 248320 }, .{ .rows = 8, .n = 8, .k = 4095 } },
    };
}

pub const TopK = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    ops: gpu.Ops,
    p: cpu.Params = undefined,
    words: []u16 = &.{},
    x: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !TopK {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .ops = try gpu.Ops.init(device) };
    }

    pub fn deinit(t: *TopK) void {
        t.a.free(t.words);
        t.ops.deinit();
        t.rig.deinit();
    }

    pub fn kernel(t: *TopK) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }

    const vt = g.hooks(TopK);

    pub fn prepare(t: *TopK, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.rig.reset();
        t.p = .{ .rows = @intCast(shape.rows), .vocab = @intCast(shape.k), .k = @intCast(shape.n), .stride = @intCast(shape.k) };
        cpu.check(t.p) catch return false;
        t.a.free(t.words);
        t.words = &.{};
        t.words = try t.a.alloc(u16, shape.rows * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.words) |*w| w.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 128))) / 8 - 8); // eighths: ties
        if (fill == .nonfinite) {
            const k = shape.k;
            const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
            const put = [_]u16{ 0x7fc1, 0x7f80, 0xff80, 0x7f80, 0xff80 };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                t.words[p[0] * k + p[1]] = v;
            };
        }
        t.x = try t.rig.input(std.mem.sliceAsBytes(t.words));
        t.out = try t.rig.output(shape.rows * @sizeOf(cpu.Row));
        return true;
    }

    pub fn run(t: *TopK) !void {
        const pass = t.rig.begin();
        try t.ops.encode(pass.e, .{ .buf = t.x.buf, .off = t.x.off }, .{ .buf = t.out.buf, .off = t.out.off }, t.p);
        try t.rig.finish(pass);
    }

    pub fn count(t: *TopK) usize {
        return t.p.rows * @sizeOf(cpu.Row);
    }

    pub fn reference(_: *TopK, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every row's bytes against the oracle; at the non-finite gate, also the rule on the rows that were planted.
    pub fn check(t: *TopK, at: g.Gate) !?[]const u8 {
        const v = t.p.vocab;
        const got = t.rig.bytes(t.out, t.count());
        var rejected: usize = 0;
        for (0..t.p.rows) |r| {
            const row = t.words[r * v ..][0..v];
            const want = try cpu.oracle(row, t.p.k);
            if (!std.mem.eql(u8, std.mem.asBytes(&want), got[r * @sizeOf(cpu.Row) ..][0..@sizeOf(cpu.Row)]))
                return try std.fmt.bufPrint(&t.msg, "row {d} of {d} (vocab {d}, k {d}) differs from the CPU oracle", .{ r, t.p.rows, v, t.p.k });
            var mask: u32 = 0;
            for (row) |w| mask |= if (w & 0x7f80 != 0x7f80) 0 else if (w & 0x7f != 0) @as(u32, 1) else if (w & 0x8000 == 0) @as(u32, 2) else 4;
            if (want.nonfinite != mask or (mask != 0 and want.count != 0)) return try std.fmt.bufPrint(&t.msg, "row {d}: mask {d} and {d} entries, it held {d}", .{ r, want.nonfinite, want.count, mask });
            if (mask != 0) rejected += 1;
        }
        if (at == .nonfinite and (rejected == 0 or rejected == t.p.rows)) return try std.fmt.bufPrint(&t.msg, "{d} of {d} rows planted: no proof", .{ rejected, t.p.rows });
        return null;
    }
};
