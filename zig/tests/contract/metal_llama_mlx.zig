//! Llama's MLX checkpoints under the contract: the embedding's dequantized rows, the shared affine kernels' q|k|v rows.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const llama = @import("metal_llama.zig");
const controls = @import("metal_llama_controls.zig");
const affine_k = @import("metal_affine.zig");
const c = contract.spec;
const g = contract.gates;

/// glue.metal's ll_affine_args, as forward.zig fills it.
const EArgs = extern struct { rows: u32, width: u32, bits: u32, group: u32 };

const no_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums: a gather of dequantized rows" };

fn none(_: c.Shape) usize {
    return 0;
}

/// Llama 3.2 3B's widths (the family's tied MLX geometry): q|k|v 3,072 + 2 x 1,024 and gate|up 2 x 8,192 over 3,072.
const rows_served = [_]c.Shape{ .{ .rows = 32, .n = 5120, .k = 3072 }, .{ .rows = 5, .n = 16384, .k = 3072 } };
const mm_served = [_]c.Shape{ .{ .rows = 128, .n = 5120, .k = 3072 }, .{ .rows = 65, .n = 16384, .k = 3072 } };

/// The shared kernels at the widths the family admits, three projections a row, and a control each.
pub const forms = blk: {
    var out: [10]affine_k.Reg = undefined;
    var i: usize = 0;
    for ([_]u8{ 4, 8 }) |bits| for ([_]usize{ 32, 64 }) |group| {
        const tag = std.fmt.comptimePrint("q{d} g{d} q|k|v", .{ bits, group });
        out[i] = .{ .v = .{ .kind = .row, .bits = bits, .group = group, .parts = 3, .served = &rows_served }, .name = "llama mlx rows " ++ tag };
        out[i + 1] = .{ .v = .{ .kind = .affine_mm, .bits = bits, .group = group, .parts = 3, .served = &mm_served }, .name = "llama mlx affine_mm " ++ tag };
        i += 2;
    };
    out[8] = .{ .v = .{ .kind = .row, .bits = 4, .group = 64, .parts = 3, .control = true, .served = &rows_served }, .name = "control: llama mlx rows, each of q|k|v written at its own width, not the row's" };
    out[9] = .{ .v = .{ .kind = .affine_mm, .bits = 4, .group = 64, .parts = 3, .control = true, .served = &mm_served }, .name = "control: llama mlx affine_mm, each of q|k|v written at its own width, not the row's" };
    break :blk out;
};

/// ll_embed_affine: R[r, i] = bf16(bf16(scale * q) + bias) of table row ids[r]; the width is the shape's k.
pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    mutation: controls.Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    bits: u8,
    group: usize,
    name: []const u8,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    ids: []u32 = &.{},
    in: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Table rows the ids pick from: enough that the gather's row arithmetic is exercised.
    const table_rows = 8;
    const served = [_]c.Shape{ .{ .rows = 32, .n = 3072, .k = 3072 }, .{ .rows = 2048, .n = 4096, .k = 4096 } };

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, group: usize, mutation: controls.Mutation) !Embed {
        const lib = try controls.glueLibrary(device, a, mutation);
        errdefer lib.deinit();
        const name = try std.fmt.allocPrint(a, "llama glue ll_embed_affine q{d} g{d}", .{ bits, group });
        errdefer a.free(name);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .mutation = mutation, .lib = lib, .pipe = try llama.pipelineOf(lib, device, "ll_embed_affine"), .bits = bits, .group = group, .name = name };
    }

    fn drop(p: *Embed) void {
        p.a.free(p.codes);
        p.a.free(p.scales);
        p.a.free(p.biases);
        p.a.free(p.ids);
        p.codes = &.{};
        p.scales = &.{};
        p.biases = &.{};
        p.ids = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Embed) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
        p.a.free(p.name);
    }

    const vt = g.hooks(Embed);

    pub fn kernel(p: *Embed) g.Kernel {
        const tiles: [3]c.Tile = .{ .{}, .{}, .{ .size = 256, .step = 64, .min = 64 } };
        return controls.mark(.{ .spec = .{ .name = p.name, .accum = .none, .depth = none, .out = .bf16, .tiles = tiles, .served = &served, .sums = no_sums }, .ctx = p, .vt = &vt }, p.mutation);
    }

    pub fn prepare(p: *Embed, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.k % p.group != 0 or s.k * p.bits % 32 != 0) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d6c78);
        const rng = prng.random();
        const w = s.k;
        const groups = w / p.group;
        p.codes = try p.a.alloc(u8, table_rows * w);
        p.scales = try p.a.alloc(u16, table_rows * groups);
        p.biases = try p.a.alloc(u16, table_rows * groups);
        for (p.codes) |*q| q.* = @intCast(rng.uintLessThan(u32, @as(u32, 1) << @intCast(p.bits)));
        for (p.scales) |*v| v.* = c.toBf16(rng.float(f32) * 0.02 + 0.001);
        for (p.biases) |*v| v.* = c.toBf16((rng.float(f32) - 0.66) * 0.15);
        if (fill == .nonfinite) { // a NaN and both infinities: row 0's first scale, row 1's first bias, row 2's last scale
            p.scales[0] = c.toBf16(std.math.nan(f32));
            p.biases[groups] = c.toBf16(std.math.inf(f32));
            p.scales[2 * groups + groups - 1] = c.toBf16(-std.math.inf(f32));
        }
        p.ids = try p.a.alloc(u32, s.rows);
        for (p.ids, 0..) |*id, i| id.* = @intCast((i * 5 + 3) % table_rows);
        const words = try p.a.alloc(u32, table_rows * w * p.bits / 32);
        defer p.a.free(words);
        @memset(words, 0);
        for (p.codes, 0..) |q, i| words[i * p.bits / 32] |= @as(u32, q) << @intCast(i * p.bits % 32);
        p.in = .{ try p.rig.input(std.mem.sliceAsBytes(p.ids)), try p.rig.input(std.mem.sliceAsBytes(words)), try p.rig.input(std.mem.sliceAsBytes(p.scales)), try p.rig.input(std.mem.sliceAsBytes(p.biases)) };
        p.out = try p.rig.output(s.rows * w * 2);
        return true;
    }

    pub fn count(p: *Embed) usize {
        return p.shape.rows * p.shape.k;
    }

    pub fn run(p: *Embed) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.in, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue(EArgs{ .rows = @intCast(s.rows), .width = @intCast(s.k), .bits = p.bits, .group = @intCast(p.group) }, 4);
        e.setBuffer(p.out.buf, p.out.off, 5);
        e.dispatchThreads(mtl.Size.of(s.k, s.rows, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    /// MLX's dequantize on bf16 tables, its two roundings: bf16(scale * q), then bf16 of that plus the bias.
    pub fn reference(p: *Embed, want: []f64, mag: []f64) !void {
        const w = p.shape.k;
        const groups = w / p.group;
        for (0..p.shape.rows) |r| for (0..w) |i| {
            const row = p.ids[r];
            const at = row * groups + i / p.group;
            const q: f64 = @floatFromInt(p.codes[row * w + i]);
            const sq = llama.bf(@as(f64, c.fromBf16(p.scales[at])) * q);
            want[r * w + i] = llama.bf(sq + c.fromBf16(p.biases[at]));
            mag[r * w + i] = @abs(want[r * w + i]);
        };
    }
};
