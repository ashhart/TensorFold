//! Production LFM3 operators under guarded repeat, nonfinite, shape and accumulator probes.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const math = @import("lfm3_math.zig");
const AttentionArgs = @import("tensorfold").lfm3.kernels.AttentionArgs;
pub const Kind = math.Kind;

fn depth(_: c.Shape) usize {
    return 32;
}
fn attentionDepth(s: c.Shape) usize {
    return 5 * (s.k + s.rows) + 128;
}
fn intermediateRounds(kind: Kind) f64 {
    return switch (kind) {
        .norm, .gate, .swiglu => 1,
        .post_add, .conv, .qk_full, .qk_sliding => 2,
        else => 0,
    };
}
pub fn name(k: Kind) []const u8 {
    return switch (k) {
        inline else => |kind| "metal LFM3 " ++ @tagName(kind),
    };
}
fn symbol(k: Kind) []const u8 {
    if (math.embedding(k)) return switch (math.bits(k)) {
        4 => "lfm3_embed_b4",
        6 => "lfm3_embed_b6",
        else => "lfm3_embed_b8",
    };
    return switch (k) {
        .qk_full, .qk_sliding => "lfm3_qk",
        .attention_full, .attention_sliding => "lfm3_attn_partial",
        inline else => |kind| "lfm3_" ++ @tagName(kind),
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: Kind,
    pipeline: mtl.Pipeline,
    merge: ?mtl.Pipeline = null,
    data: math.Data = undefined,
    args: [6]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    extra: [3]rig_.Ref = undefined,
    risk: rig_.Ref = undefined,

    /// The production kernel text both LFM3 source files build, in the order the pipelines are compiled.
    pub fn source(a: std.mem.Allocator) ![]u8 {
        return std.mem.concat(a, u8, &.{ sources.lfm3, "\n", sources.lfm3_attention });
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind) !Probe {
        const text = try source(a);
        defer a.free(text);
        return initSource(a, device, queue, kind, text);
    }
    /// The probe over kernel text the caller supplies, which a planted control edits.
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, text: []const u8) !Probe {
        const pipeline = try rig_.pipeline(device, text, symbol(kind));
        errdefer pipeline.deinit();
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .pipeline = pipeline, .merge = if (math.attention(kind)) try rig_.pipeline(device, text, "lfm3_attn_merge") else null };
    }
    pub fn deinit(p: *Probe) void {
        p.arena.deinit();
        p.rig.deinit();
        p.pipeline.deinit();
        if (p.merge) |pipeline| pipeline.deinit();
    }
    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        const reduction = p.kind == .norm or p.kind == .post_add or p.kind == .conv or math.qk(p.kind) or math.attention(p.kind);
        const shapes: []const c.Shape = if (p.kind == .attention_sliding)
            &.{ .{ .rows = 8, .n = 2048, .k = 257 }, .{ .rows = 1, .n = 2048, .k = 4097 }, .{ .rows = 17, .n = 2048, .k = 4350 }, .{ .rows = 8, .n = 2048, .k = 4351 }, .{ .rows = 17, .n = 2048, .k = 8193 }, .{ .rows = 8, .n = 2048, .k = 65536 } }
        else if (math.attention(p.kind))
            &.{ .{ .rows = 8, .n = 2048, .k = 257 }, .{ .rows = 1, .n = 2048, .k = 4097 }, .{ .rows = 17, .n = 2048, .k = 1025 } }
        else if (p.kind == .swiglu)
            &.{ .{ .rows = 8, .n = 7680, .k = 7680 }, .{ .rows = 17, .n = 7680, .k = 7680 } }
        else
            &.{ .{ .rows = 8, .n = 2048, .k = 2048 }, .{ .rows = 17, .n = 2048, .k = 2048 } };
        return .{ .ctx = p, .vt = &vt, .spec = .{ .name = name(p.kind), .accum = if (reduction) .f32 else .none, .depth = if (math.attention(p.kind)) attentionDepth else depth, .ops = 64, .mids = intermediateRounds(p.kind), .out = .bf16, .in_place = p.kind == .post_add or p.kind == .add or p.kind == .gate, .served = shapes, .tiles = .{ .{ .size = 8 }, .{}, .{} }, .sums = if (p.kind == .conv or p.kind == .post_add) .{ .exact = false } else .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "nonlinear BF16 nodes do not guarantee a 100x reduced-accumulator witness" } } };
    }
    fn values(a: std.mem.Allocator, count_: usize, seed: u64) ![]u16 {
        const out = try a.alloc(u16, count_);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (out) |*v| v.* = c.toBf16((random.float(f32) * 2 - 1) * 0.5);
        return out;
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 17) return false;
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
        const a = p.arena.allocator();
        const rows = shape.rows;
        const cap = shape.k + rows;
        const count_ = rows * (if (p.kind == .swiglu) @as(usize, 7680) else 2048);
        var d: math.Data = .{ .kind = p.kind, .rows = rows, .history = shape.k, .x = try values(a, if (p.kind == .conv) rows * 6144 else count_, seed), .w = try values(a, if (p.kind == .conv) 2048 * 7 else if (math.attention(p.kind)) cap * 512 else if (math.qk(p.kind)) rows * 512 else count_, seed +% 1), .z = try values(a, if (p.kind == .conv) 6 * 2048 else if (math.attention(p.kind)) cap * 512 else if (math.qk(p.kind)) rows * 512 else count_, seed +% 2), .gain = try values(a, 2048, seed +% 3), .codes = &.{}, .ids = &.{} };
        for (d.gain) |*v| v.* = c.toBf16(1 + @as(f32, c.fromBf16(v.*)) / 4);
        var words: []u8 = &.{};
        if (math.embedding(p.kind)) {
            d.ids = try a.alloc(u32, rows);
            for (d.ids, 0..) |*id, i| id.* = @intCast(i % 8);
            d.codes = try a.alloc(u8, 8 * 2048);
            words = try a.alloc(u8, @as(usize, 8 * 2048) * math.bits(p.kind) / 8);
            @memset(words, 0);
            for (d.codes, 0..) |*code, i| {
                code.* = @intCast((i *% 37 +% seed) % (@as(usize, 1) << @intCast(math.bits(p.kind))));
                for (0..math.bits(p.kind)) |bit| {
                    const at = i * math.bits(p.kind) + bit;
                    words[at / 8] |= @as(u8, @intCast((code.* >> @as(u3, @intCast(bit))) & 1)) << @as(u3, @intCast(at % 8));
                }
            }
            d.w = try values(a, 8 * 2048 / math.group(p.kind), seed +% 4);
            d.z = try values(a, d.w.len, seed +% 5);
        }
        if (fill == .nonfinite) {
            const plants = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80 };
            for (plants, 0..) |plant, r| {
                if (math.embedding(p.kind)) d.w[r * (2048 / math.group(p.kind))] = plant else if (r < rows) d.x[r * (if (p.kind == .conv) @as(usize, 6144) else if (p.kind == .swiglu) @as(usize, 7680) else @as(usize, 2048))] = plant;
            }
            if (!math.embedding(p.kind) and rows > 3) d.x[3 * (if (p.kind == .conv) @as(usize, 6144) else if (p.kind == .swiglu) @as(usize, 7680) else @as(usize, 2048)) + 1] = 0xff80;
        }
        if (fill == .stress) {
            for (d.x, 0..) |*v, i| v.* = c.toBf16(if (i % 4 < 2) 32 else -32);
            for (d.w, 0..) |*v, i| v.* = c.toBf16(if (i % 4 < 2) 2 else -2);
            if (p.kind == .post_add) {
                @memset(d.gain, c.toBf16(1));
                var squares: f64 = 0;
                for (0..2048) |i| {
                    const value: f32 = @floatFromInt(1 + (i * 37 + 11) % 63);
                    const x = if (i % 2 == 0) -value else value;
                    squares += @as(f64, x) * x;
                    for (0..rows) |r| d.x[r * 2048 + i] = c.toBf16(x);
                }
                const inverse = 1 / @sqrt(squares / 2048 + 1e-5);
                for (d.x, d.z) |x, *z| z.* = c.toBf16(@floatCast(-math.bf(math.val(x) * inverse)));
            }
            if (p.kind == .conv) {
                @memset(d.x, 0);
                @memset(d.w, 0);
                @memset(d.z, 0);
                for (0..rows) |r| {
                    for (0..2048) |ch| d.x[r * 6144 + 2048 + ch] = c.toBf16(1);
                }
                for (0..2048) |ch| {
                    for ([_]f32{ 65536, 255, 254, 1 }, 0..) |bx_value, tap| d.z[tap * 2048 + ch] = c.toBf16(bx_value);
                    for ([_]f32{ 1, -258, 1, 1 }, 0..) |weight, tap| d.w[ch * 7 + tap] = c.toBf16(weight);
                }
            }
        }
        p.data = d;
        for ([_][]const u16{ d.x, d.w, d.z, d.gain }, 0..) |v, i| p.args[i] = try p.rig.input(std.mem.sliceAsBytes(v));
        if (math.embedding(p.kind)) {
            p.args[4] = try p.rig.input(words);
            p.args[5] = try p.rig.input(std.mem.sliceAsBytes(d.ids));
        }
        p.out = try p.rig.output(count_ * 2);
        if (p.kind == .conv) p.extra[0] = try p.rig.output(rows * 6 * 2048 * 2);
        if (math.qk(p.kind)) {
            p.extra[0] = try p.rig.output(rows * 512 * 2);
            p.extra[1] = try p.rig.output(rows * 512 * 2);
        }
        if (math.attention(p.kind)) {
            p.risk = try p.rig.scratch(rows * 16 * 64 * 4);
            p.extra[0] = try p.rig.scratch(rows * 16 * 64 * 4);
            p.extra[1] = try p.rig.scratch(rows * 16 * 64 * 4);
            p.extra[2] = try p.rig.scratch(rows * 16 * 64 * 128 * 4);
        }
        return true;
    }
    fn bind(e: mtl.ComputeEncoder, refs: []const rig_.Ref) void {
        for (refs, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
    }
    pub fn run(p: *Probe) !void {
        const d = p.data;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipeline);
        if (math.embedding(p.kind)) {
            bind(e, &.{ p.args[5], p.args[4], p.args[1], p.args[2], p.out });
            e.setValue(@as(u32, @intCast(math.group(p.kind))), 5);
            e.dispatchThreads(.of(2048, d.rows, 1), .of(256, 1, 1));
        } else switch (p.kind) {
            .norm => {
                bind(e, &.{ p.args[0], p.args[2], p.args[3] });
                e.setValue(@as(f32, 1e-5), 3);
                e.setValue(@as(u32, 0), 4);
                e.setBuffer(p.out.buf, p.out.off, 5);
                e.dispatchGroups(.of(d.rows, 1, 1), .of(256, 1, 1));
            },
            .post_add => {
                @memcpy(p.rig.bytes(p.out, d.rows * 2048 * 2), std.mem.sliceAsBytes(d.z));
                bind(e, &.{ p.args[0], p.args[3], p.out });
                e.setValue(@as(f32, 1e-5), 3);
                e.dispatchGroups(.of(d.rows, 1, 1), .of(256, 1, 1));
            },
            .add, .gate => {
                const initial = if (p.kind == .add) d.z else d.x;
                @memcpy(p.rig.bytes(p.out, initial.len * 2), std.mem.sliceAsBytes(initial));
                if (p.kind == .add) bind(e, &.{ p.args[0], p.out }) else bind(e, &.{ p.out, p.args[1] });
                e.setValue(@as(u32, @intCast(d.rows * 2048)), 2);
                e.dispatchThreads(.of(d.rows * 2048, 1, 1), .of(256, 1, 1));
            },
            .swiglu => {
                bind(e, &.{ p.args[0], p.args[1], p.out });
                e.setValue(@as(u32, @intCast(d.rows * 7680)), 3);
                e.dispatchThreads(.of(d.rows * 7680 / 8, 1, 1), .of(256, 1, 1));
            },
            .conv => {
                bind(e, &.{ p.args[0], p.args[2], p.args[1], p.out, p.extra[0] });
                e.setValue(@as(u32, 0), 5);
                e.dispatchThreads(.of(2048, d.rows, 1), .of(256, 1, 1));
            },
            .qk_full, .qk_sliding => {
                bind(e, &.{ p.args[0], p.args[1], p.args[2], p.args[3], p.args[3] });
                const args: extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32, rotary_pairs: u32 } = .{ .len = 0, .cap = @intCast(d.rows), .eps = 1e-5, .log2_theta = std.math.log2(@as(f32, if (p.kind == .qk_full) 1e6 else 1e4)), .rotary_pairs = if (p.kind == .qk_full) 32 else 64 };
                e.setBytes(std.mem.asBytes(&args), 5);
                for ([_]rig_.Ref{ p.out, p.extra[0], p.extra[1] }, 6..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
                e.dispatchGroups(.of(20, d.rows, 1), .of(32, 1, 1));
            },
            .attention_full, .attention_sliding => {
                for (p.extra, [_]usize{ d.rows * 16 * 64 * 4, d.rows * 16 * 64 * 4, d.rows * 16 * 64 * 128 * 4 }) |ref, len|
                    @memset(p.rig.bytes(ref, len), 0xff);
                p.encodeAttention(e, true);
            },
            else => unreachable,
        }
        try p.rig.finish(pass);
        if (p.kind == .attention_sliding) {
            const bounded = try p.a.dupe(u8, try p.rig.outputBytes());
            defer p.a.free(bounded);
            const baseline = p.rig.begin();
            p.encodeAttention(baseline.e, false);
            try p.rig.finish(baseline);
            if (!std.mem.eql(u8, bounded, try p.rig.outputBytes())) return error.SlidingChunkPruneChangedBytes;
        }
    }
    fn encodeAttention(p: *Probe, e: mtl.ComputeEncoder, prune: bool) void {
        const d = p.data;
        const window: usize = if (p.kind == .attention_sliding) 256 else 0;
        const first = if (prune and window != 0) (d.history + 1 -| window) / 4096 else 0;
        const last = (d.history + d.rows + 4095) / 4096;
        const args = AttentionArgs{ .len = @intCast(d.history), .cap = @intCast(d.history + d.rows), .scale = 1.0 / @sqrt(@as(f32, 128)), .window = @intCast(window), .chunk_base = @intCast(first) };
        e.setPipeline(p.pipeline);
        bind(e, &.{ p.args[0], p.args[1], p.args[2] });
        e.setBytes(std.mem.asBytes(&args), 3);
        for (p.extra, 4..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(p.risk.buf, p.risk.off, 7);
        e.dispatchGroups(.of(16, d.rows, last - first), .of(128, 1, 1));
        e.setPipeline(p.merge.?);
        bind(e, &p.extra);
        e.setBytes(std.mem.asBytes(&args), 3);
        e.setBuffer(p.out.buf, p.out.off, 4);
        for (p.args[0..3], 5..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setBuffer(p.risk.buf, p.risk.off, 8);
        e.dispatchGroups(.of(16, d.rows, 1), .of(32, 1, 1));
    }
    pub fn count(p: *Probe) usize {
        return p.data.rows * (if (p.kind == .conv) @as(usize, 7 * 2048) else if (math.qk(p.kind)) @as(usize, 3072) else if (p.kind == .swiglu) @as(usize, 7680) else @as(usize, 2048));
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try math.evaluate(p.a, p.data, .none, .sequential, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        try math.evaluate(p.a, p.data, acc, order, out, mag);
    }
};
