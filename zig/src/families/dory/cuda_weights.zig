//! Dory's weights on the GPU: NVFP4 projections packed for the lane matmul, scale vectors as torch forms them.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");

const Config = config.Config;
const DeviceBuffer = cuda.DeviceBuffer;
const Weight = cuda.qlinear.Weight;
const bfr = core.bf16_round;

pub const Attn = struct {
    alpha: u64, // bf16 [D]
    q: Weight,
    k: Weight,
    v: Weight,
    g: Weight,
    o: Weight,
    qs: u64, // fp32 [Q]: (sqk f)^2
    sg: u64, // fp32 [Q]: (s_gate f)^2
};

pub const Mlp = struct {
    alpha: u64,
    gate: Weight,
    up: Weight,
    down: Weight,
    su_g: u64, // fp32 [F]: suv_gate sqrt(D)
    su_u: u64,
};

pub const Sub = union(enum) { attn: Attn, mlp: Mlp };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    embed: u64 = 0, // bf16 [V, D]
    head: u64 = 0, // bf16 [V, D]
    logit: u64 = 0, // fp32 [V]: logit_scale / std as torch multiplies
    subs: []Sub = &.{},
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.subs);
        w.* = undefined;
    }

    fn upload(w: *Weights, d: *const cuda.Driver, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(d, bytes);
        errdefer b.free();
        try w.buffers.append(w.gpa, b);
        w.bytes += bytes.len;
        return b.ptr;
    }

    /// A projection in its lane layout: NVFP4 words and scale tiles (padded to 128 rows), or bf16 rows.
    fn proj(w: *Weights, d: *const cuda.Driver, p: names.Proj) !Weight {
        if (p.blocks == null) {
            const n = cuda.modelopt.padded(p.n);
            const out = try w.gpa.alloc(u16, n * p.k);
            defer w.gpa.free(out);
            const src = try w.gpa.alloc(u16, p.n * p.k);
            defer w.gpa.free(src);
            @memcpy(std.mem.sliceAsBytes(src), p.w.bytes);
            cuda.qmmf.packBf16(out, src, p.n, p.k);
            return .{ .bf16 = cuda.modelopt.bf16Weight(try w.upload(d, std.mem.sliceAsBytes(out)), p.n, p.k) };
        }
        const words = try w.gpa.alloc(u32, cuda.nvfp4.padded(p.n) / 64 * (p.k / 64) * 512);
        defer w.gpa.free(words);
        const tiles = try w.gpa.alloc(u8, cuda.nvfp4.padded(p.n) * (p.k / 16));
        defer w.gpa.free(tiles);
        cuda.nvfp4.packWords(words, p.w.bytes, p.n, p.k);
        cuda.nvfp4.packScales(tiles, p.blocks.?.bytes, p.n, p.k);
        const wp = try w.upload(d, std.mem.sliceAsBytes(words));
        const tp = try w.upload(d, tiles);
        return .{ .nvfp4 = cuda.nvfp4.weight(wp, tp, p.scale, p.n, p.k) };
    }

    /// fp32 vector: each bf16 value through `f` (the torch steps D3 of the progress log).
    fn vector(w: *Weights, d: *const cuda.Driver, t: core.checkpoint.Tensor, f: anytype) !u64 {
        const n = t.bytes.len / 2;
        const v = try w.gpa.alloc(f32, n);
        defer w.gpa.free(v);
        for (v, 0..) |*o, i| o.* = f.at(bfr.value(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
        return w.upload(d, std.mem.sliceAsBytes(v));
    }
};

/// (v * factor)^2 in fp32, as dory.py squares sqk and s_gate after their factor.
const Square = struct {
    f: f32,
    fn at(s: Square, v: f32) f32 {
        const t = v * s.f;
        return t * t;
    }
};

/// v * factor in fp32: suv by sqrt(hidden), logit_scale by 1 / std.
const Times = struct {
    f: f32,
    fn at(s: Times, v: f32) f32 {
        return v * s.f;
    }
};

/// config.json, every tensor mapped and checked, then sublayer by sublayer onto the GPU; `loops` overrides the count.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, loops: ?usize) !Weights {
    var c = try Config.read(gpa, io, dir);
    if (c.quant != .nvfp4) return error.UnsupportedDoryCudaQuantization;
    if (loops) |l| c = try c.withLoops(l);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var m = try names.map(gpa, &ck, c);
    defer m.deinit();
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    const f = c.factors();
    w.subs = try gpa.alloc(Sub, c.sublayers);
    w.embed = try w.upload(d, m.embed.bytes);
    for (w.subs, m.subs) |*out, s| out.* = switch (s) {
        .attn => |a| .{ .attn = .{
            .alpha = try w.upload(d, a.alpha.bytes),
            .q = try w.proj(d, a.q),
            .k = try w.proj(d, a.k),
            .v = try w.proj(d, a.v),
            .g = try w.proj(d, a.g),
            .o = try w.proj(d, a.o),
            .qs = try w.vector(d, a.sqk, Square{ .f = f.sqk }),
            .sg = try w.vector(d, a.s_gate, Square{ .f = f.gate }),
        } },
        .mlp => |p| .{ .mlp = .{
            .alpha = try w.upload(d, p.alpha.bytes),
            .gate = try w.proj(d, p.gate),
            .up = try w.proj(d, p.up),
            .down = try w.proj(d, p.down),
            .su_g = try w.vector(d, p.suv_gate, Times{ .f = f.suv }),
            .su_u = try w.vector(d, p.suv_up, Times{ .f = f.suv }),
        } },
    };
    w.head = try w.upload(d, m.head.bytes);
    w.logit = try w.vector(d, m.logit_scale, Times{ .f = f.logit });
    return w;
}

const UploadTest = struct {
    var live: usize = 0;
    fn alloc(p: *u64, _: usize) callconv(.c) cuda.abi.Result {
        p.* = 0x10000;
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        live -= 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "Dory loader: a host ownership-list failure refunds the uploaded device buffer" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = UploadTest.alloc;
    d.api.cuMemFree_v2 = UploadTest.free;
    d.api.cuMemcpyHtoD_v2 = UploadTest.copy;
    d.api.cuStreamSynchronize = UploadTest.sync;
    UploadTest.live = 0;
    // The control's leaked fake buffer is still reclaimed after the assertion, so memory accounting stays valid.
    defer if (UploadTest.live != 0) {
        var leaked: DeviceBuffer = .{ .d = &d, .ptr = 0x10000, .len = 16 };
        leaked.free();
    };
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var w: Weights = .{ .gpa = failing.allocator(), .config = undefined };
    defer w.deinit();
    try std.testing.expectError(error.OutOfMemory, w.upload(&d, &@as([16]u8, @splat(0))));
    try std.testing.expectEqual(@as(usize, 0), UploadTest.live);
    try std.testing.expectEqual(@as(usize, 0), w.buffers.items.len);
    try std.testing.expectEqual(@as(usize, 0), w.bytes);
}

test "Dory loader: host-only BF16 and MXFP8 maps refuse before opening weights or touching a driver" {
    const a = std.testing.allocator;
    const text = try config.pinnedText(a);
    defer a.free(text);
    const at = std.mem.indexOf(u8, text, "\"quantization_config\"").?;
    const end = std.mem.indexOf(u8, text, "\"head_dim\"").?;
    const mx = "\"quantization_config\":{\"quant_method\":\"compressed-tensors\",\"config_groups\":{\"g\":{\"format\":\"mxfp8-quantized\",\"weights\":{\"num_bits\":8,\"type\":\"float\",\"group_size\":32}}}},";
    for ([_][]const u8{ "", mx }) |quant| {
        const candidate = try std.mem.concat(a, u8, &.{ text[0..at], quant, text[end..] });
        defer a.free(candidate);
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = candidate });
        const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        defer a.free(path);
        var d: cuda.Driver = undefined;
        try std.testing.expectError(error.UnsupportedDoryCudaQuantization, load(a, std.testing.io, &d, path, null));
    }
}
