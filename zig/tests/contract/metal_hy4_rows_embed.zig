//! Core embed_norm.metal's tf_embed_b4_g64 and _b8_g64 at Hy4's width as Head.embed launches them, bit-exact.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const text = @import("metal_hy4_rows_text.zig");

const c = contract.spec;
const g = contract.gates;
const bf = h.fl.bf;

const group = 64;
const streams = h.hy.ihc.streams;

/// Rows: ids (a token's four streams each); n: the hidden width. A decode token first, DFly2's window, a prompt chunk.
const shapes = [_]c.Shape{
    .{ .rows = streams, .n = h.hidden, .k = 1 },
    .{ .rows = h.window_rows * streams, .n = h.hidden, .k = 1 },
    .{ .rows = h.prompt_rows * streams, .n = h.hidden, .k = 1 },
};

pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    bits: u8,
    vocab: usize,
    fault: ?text.Fault,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    ids: []u32 = &.{},
    table: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    in: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn function(bits: u8) []const u8 {
        return if (bits == 4) "tf_embed_b4_g64" else "tf_embed_b8_g64";
    }
    pub fn base(bits: u8) []const u8 {
        return if (bits == 4) "metal hy4 embed_norm tf_embed_b4_g64" else "metal hy4 embed_norm tf_embed_b8_g64";
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, fault: ?text.Fault, vocab: usize) !Embed {
        const src = try text.embedNorm(a, fault);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try text.pipeline(device, src, function(bits)), .bits = bits, .vocab = vocab, .fault = fault };
    }
    pub fn deinit(t: *Embed) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Embed) void {
        inline for (.{ "ids", "table", "scales", "biases" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    pub fn kernel(t: *Embed) g.Kernel {
        t.name = h.fl.named(base(t.bits), if (t.fault) |f| f.drops else null, &t.label);
        // n: a thread every 2 values (4 bits) or 1 (8 bits), 128 threads a group; widths in whole groups of 64
        const tile: usize = if (t.bits == 4) 256 else 128;
        const s: c.Spec = .{ .name = t.name, .accum = .none, .depth = h.fl.noSums, .out = .bf16, .tiles = .{ .{}, .{ .size = tile, .step = group, .min = group }, .{} }, .served = &shapes };
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.fault != null) .shapes else null };
    }
    const vt = blk: {
        var v = g.hooks(Embed);
        v.release = release;
        break :blk v;
    };
    /// The vocabulary's table is the largest host copy in the group: freed with the buffers once its gates are done.
    fn release(ctx: *anyopaque) void {
        const t: *Embed = @ptrCast(@alignCast(ctx));
        t.free();
        t.rig.reset();
    }

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % group != 0 or shape.k != 1) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        const gpr = n / group;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the whole vocabulary's table, so the ids reach its last row
        t.table = try t.a.alloc(u8, t.vocab * n * t.bits / 8);
        rng.bytes(t.table);
        t.scales = try t.a.alloc(u16, t.vocab * gpr);
        t.biases = try t.a.alloc(u16, t.vocab * gpr);
        const top: f32 = @floatFromInt(@as(u32, 1) << @intCast(t.bits));
        for (t.scales, t.biases) |*s, *b| {
            const sc = 0.001 + rng.float(f32) * 0.02;
            s.* = c.toBf16(sc);
            b.* = c.toBf16(-sc * rng.float(f32) * top);
        }
        // each token's id once a stream (metal_model.zig's stage); the first token the last row, the second row 0
        t.ids = try t.a.alloc(u32, shape.rows);
        var token: u32 = 0;
        for (t.ids, 0..) |*v, i| {
            if (i % streams == 0) token = switch (i / streams) {
                0 => @intCast(t.vocab - 1),
                1 => 0,
                else => rng.uintLessThan(u32, @intCast(t.vocab)),
            };
            v.* = token;
        }
        if (fill == .nonfinite and gpr >= 5) {
            const last = (t.vocab - 1) * gpr;
            t.scales[last + 0] = c.toBf16(h.fl.planted[0]);
            t.biases[last + 1] = c.toBf16(h.fl.planted[1]);
            t.biases[2] = c.toBf16(h.fl.planted[2]);
            t.scales[3] = c.toBf16(h.fl.planted[1]);
            t.biases[4] = c.toBf16(h.fl.planted[2]);
        }
        t.in[0] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.in[1] = try t.rig.input(t.table);
        t.in[2] = try t.rig.input(std.mem.sliceAsBytes(t.scales));
        t.in[3] = try t.rig.input(std.mem.sliceAsBytes(t.biases));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    /// metal_model.zig's Head.embed: the width as an int, a thread a packed unit and id, in groups of 128.
    pub fn run(t: *Embed) !void {
        const n = t.shape.n;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.in, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setValue(@as(i32, @intCast(n)), 4);
        p.e.setBuffer(t.out.buf, t.out.off, 5);
        p.e.dispatchThreads(mtl.Size.of(n / @as(usize, if (t.bits == 4) 2 else 1), t.shape.rows, 1), mtl.Size.of(128, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Embed) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Metal evaluates the bfloat `scale * q + bias` in fp32 (the product exact), and the store rounds it once to bf16.
    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const n = t.shape.n;
        for (want, 0..) |*w, i| {
            const id: usize = t.ids[i / n];
            const j = i % n;
            const at = id * n + j;
            const q: u8 = if (t.bits == 8) t.table[at] else (t.table[at / 2] >> @intCast(4 * (j % 2))) & 15;
            const gi = id * (n / group) + j / group;
            w.* = bf(@as(f64, @floatFromInt(q)) * c.fromBf16(t.scales[gi]) + c.fromBf16(t.biases[gi]));
        }
    }

    /// Exact: every value must give float64's bf16 bits at that one point.
    pub fn bounds(_: *Embed, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }
};
