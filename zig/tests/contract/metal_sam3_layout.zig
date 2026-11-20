//! SAM3's data-movement kernels (layout.metal, convolution.metal): each output word against a host index map.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const vt = g.hooks(Probe);

const Case = enum { copy, columns, transpose, tiles_into, tiles_from, channel_fill, conv_pack, deconv_pack, unfold };
const cases = std.enums.values(Case);
const rule = "moves 32-bit words bit for bit (NaN payloads, signs and subnormals kept); writes nothing else";
const fill_rule = "no float inputs: writes 1.0 or 0.0 by bit into its channels and leaves every other word as it was";

const Edit = struct { function: []const u8, from: []const u8, to: []const u8 };
const edits = [_]Edit{
    .{ .function = "sam3_copy", .from = "dst[i] = src[i];", .to = "dst[i] = src[min(i + 1u, count - 1u)];" },
    .{ .function = "sam3_columns", .from = "src[row * shape.y + shape.z + column]", .to = "src[row * shape.w + shape.z + column]" },
    .{ .function = "sam3_transpose", .from = "dst[column * shape.x + row] = src[i];", .to = "dst[row * shape.y + column] = src[i];" },
    .{ .function = "sam3_tiles", .from = "const uint tx = index / across, ty = index % across;", .to = "const uint ty = index / across, tx = index % across;" },
    .{ .function = "sam3_channel_fill", .from = "dst[pixel * p.channels + p.first + bit]", .to = "dst[pixel * p.channels + bit]" },
    .{ .function = "sam3_conv_pack", .from = "packed[i] = raw[(output * inputs + channel) * taps + tap];", .to = "packed[i] = raw[(output * inputs + channel) * taps + (taps - 1u - tap)];" },
    .{ .function = "sam3_deconv_pack", .from = "packed[i] = raw[(channel * outputs + output) * taps + tap];", .to = "packed[i] = raw[(output * inputs + channel) * taps + tap];" },
    .{ .function = "sam3_unfold", .from = "const uint pixel = p.first_row + row;", .to = "const uint pixel = row;" },
};

fn editOf(case: Case) Edit {
    const name = function(case);
    for (edits) |e| if (std.mem.eql(u8, e.function, name)) return e;
    unreachable;
}

fn function(case: Case) []const u8 {
    return switch (case) {
        .copy => "sam3_copy",
        .columns => "sam3_columns",
        .transpose => "sam3_transpose",
        .tiles_into, .tiles_from => "sam3_tiles",
        .channel_fill => "sam3_channel_fill",
        .conv_pack => "sam3_conv_pack",
        .deconv_pack => "sam3_deconv_pack",
        .unfold => "sam3_unfold",
    };
}

fn label(case: Case, control: bool) []const u8 {
    if (control) return switch (case) {
        .copy => "control: sam3 layout copy reads the next word",
        .columns => "control: sam3 layout columns step rows by the count",
        .transpose => "control: sam3 layout transpose copies untransposed",
        .tiles_into => "control: sam3 layout tiles into, enumerated row by row",
        .tiles_from => "control: sam3 layout tiles from, enumerated row by row",
        .channel_fill => "control: sam3 layout channel fill writes from channel 0",
        .conv_pack => "control: sam3 layout conv pack flips the taps",
        .deconv_pack => "control: sam3 layout deconv pack reads a conv layout",
        .unfold => "control: sam3 layout unfold ignores its chunk's first row",
    };
    return switch (case) {
        .copy => "metal sam3 layout copy",
        .columns => "metal sam3 layout columns",
        .transpose => "metal sam3 layout transpose",
        .tiles_into => "metal sam3 layout tiles into",
        .tiles_from => "metal sam3 layout tiles from",
        .channel_fill => "metal sam3 layout channel fill",
        .conv_pack => "metal sam3 layout conv pack",
        .deconv_pack => "metal sam3 layout deconv pack",
        .unfold => "metal sam3 layout unfold",
    };
}

/// layout.metal and convolution.metal as concatenated into the SAM library (no tensor ops: builds on every Mac).
fn kernelText() ![2][]const u8 {
    const all = sources.sam3_baseline_source;
    if (!std.mem.startsWith(u8, all, sources.sam3_deconv)) return error.LayoutTextNotFound;
    const layout_end = std.mem.indexOf(u8, all, sources.sam3_resample) orelse return error.LayoutTextNotFound;
    const conv_start = layout_end + sources.sam3_resample.len;
    const rotary = std.mem.indexOfPos(u8, all, conv_start, "struct Sam3Rotary") orelse return error.LayoutTextNotFound;
    const conv_end = std.mem.lastIndexOf(u8, all[0..rotary], "#include <metal_stdlib>") orelse return error.LayoutTextNotFound;
    const layout = all[sources.sam3_deconv.len..layout_end];
    if (conv_end <= conv_start or std.mem.indexOf(u8, layout, "void sam3_copy(") == null or
        std.mem.indexOf(u8, all[conv_start..conv_end], "void sam3_unfold(") == null) return error.LayoutTextNotFound;
    return .{ layout, all[conv_start..conv_end] };
}

pub fn text(a: std.mem.Allocator, control: ?Edit) ![]u8 {
    const parts = try kernelText();
    const raw = try std.mem.concat(a, u8, &parts);
    const e = control orelse return raw;
    defer a.free(raw);
    if (std.mem.count(u8, raw, e.from) != 1) return error.ControlTextNotFoundOnce;
    return std.mem.replaceOwned(u8, a, raw, e.from, e.to);
}

pub fn limitText(a: std.mem.Allocator, name: []const u8, control: bool) ![]u8 {
    if (!control) return text(a, null);
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    for (edits) |e| if (std.mem.eql(u8, e.function, fun)) return text(a, e);
    return error.UnknownLaunch;
}

/// An unfold's convolution: input side and channels, kernel, stride and padding (the family's ConvShape).
const Geometry = struct { side: u32, in: u32, k: u32, stride: u32, pad: u32 };
const geometries = [_]Geometry{
    .{ .side = 576, .in = 16, .k = 3, .stride = 2, .pad = 1 }, // mask downsampler, second stage
    .{ .side = 1008, .in = 3, .k = 14, .stride = 14, .pad = 0 }, // patch embedding
    .{ .side = 288, .in = 64, .k = 3, .stride = 2, .pad = 1 },
    .{ .side = 144, .in = 256, .k = 3, .stride = 2, .pad = 1 },
    .{ .side = 72, .in = 256, .k = 3, .stride = 1, .pad = 1 }, // 3x3 convolutions of the neck and pixel decoder
    .{ .side = 288, .in = 256, .k = 3, .stride = 1, .pad = 1 },
    .{ .side = 1152, .in = 32, .k = 3, .stride = 2, .pad = 1 }, // mask downsampler, first stage (grouped input)
    .{ .side = 144, .in = 256, .k = 3, .stride = 1, .pad = 1 },
};

fn outSide(m: Geometry) usize {
    return (m.side + 2 * m.pad - m.k) / m.stride + 1;
}

fn depth(_: c.Shape) usize {
    return 0;
}

/// Unfold: rows = chunk rows (convolution.zig's chunkPixels), n = geometry, k = the chunk's first output pixel.
const unfold_served = [_]c.Shape{
    .{ .rows = 16384, .n = 0, .k = 0 },   .{ .rows = 1024, .n = 0, .k = 81920 }, .{ .rows = 5184, .n = 1, .k = 0 },
    .{ .rows = 14528, .n = 2, .k = 0 },   .{ .rows = 6208, .n = 2, .k = 14528 }, .{ .rows = 3584, .n = 3, .k = 0 },
    .{ .rows = 1600, .n = 3, .k = 3584 }, .{ .rows = 3584, .n = 4, .k = 0 },     .{ .rows = 1600, .n = 4, .k = 3584 },
    .{ .rows = 3584, .n = 7, .k = 0 },    .{ .rows = 2816, .n = 7, .k = 17920 }, .{ .rows = 3584, .n = 5, .k = 0 },
    .{ .rows = 512, .n = 5, .k = 82432 }, .{ .rows = 16384, .n = 6, .k = 0 },    .{ .rows = 4096, .n = 6, .k = 327680 },
};

fn served(case: Case) []const c.Shape {
    return switch (case) {
        // a token row, presence and queries, a frame's memory values and its keys (concat and dedicatedCopy)
        .copy => &.{ .{ .rows = 256, .n = 1, .k = 1 }, .{ .rows = 201 * 256, .n = 1, .k = 1 }, .{ .rows = 5184 * 64, .n = 1, .k = 1 }, .{ .rows = 5184 * 256, .n = 1, .k = 1 } },
        // rows, width, first column of one-column reads off 288 x 288 logits (slots, kept queries)
        .columns => &.{ .{ .rows = 82944, .n = 16, .k = 0 }, .{ .rows = 82944, .n = 16, .k = 15 }, .{ .rows = 82944, .n = 200, .k = 199 } },
        .transpose => &.{ .{ .rows = 16, .n = 82944, .k = 1 }, .{ .rows = 1, .n = 82944, .k = 1 } },
        // channels, grid, tile of the trunk's tiled blocks
        .tiles_into, .tiles_from => &.{.{ .rows = 1024, .n = 72, .k = 24 }},
        // pixels, channels, indicator count of the grouped memory input
        .channel_fill => &.{.{ .rows = 1008 * 1008, .n = 32, .k = 16 }},
        // outputs, inputs, taps of the packed weights
        .conv_pack => &.{
            .{ .rows = 1024, .n = 3, .k = 196 }, .{ .rows = 256, .n = 256, .k = 9 },  .{ .rows = 64, .n = 16, .k = 9 },
            .{ .rows = 256, .n = 64, .k = 9 },   .{ .rows = 1024, .n = 256, .k = 9 }, .{ .rows = 16, .n = 32, .k = 9 },
            .{ .rows = 256, .n = 1024, .k = 1 },
        },
        .deconv_pack => &.{ .{ .rows = 512, .n = 1024, .k = 4 }, .{ .rows = 256, .n = 512, .k = 4 }, .{ .rows = 64, .n = 256, .k = 4 }, .{ .rows = 32, .n = 64, .k = 4 } },
        .unfold => &unfold_served,
    };
}

/// Grid row of tile-ordered row r (tiles column by column, row-major inside a tile), from the operator spec's map.
fn gridRow(r: usize, grid: usize, tile: usize) usize {
    const across = grid / tile;
    const index = r / (tile * tile);
    const local = r % (tile * tile);
    return ((index % across) * tile + local / tile) * grid + (index / across) * tile + local % tile;
}

const specials = [_]u32{ 0x7fc00001, 0xffc12345, 0x7f800001, 0x7f800000, 0xff800000, 0x80000000, 0x00000001 };

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    case: Case,
    control: bool,
    shape: c.Shape = undefined,
    bits: u16 = 0,
    words: []u32 = &.{},
    inverse: []u32 = &.{},
    input: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    why: [200]u8 = undefined,

    fn spec(p: *const Probe) c.Spec {
        const on_n = p.case == .transpose;
        return .{
            .name = label(p.case, p.control),
            .accum = .none,
            .depth = depth,
            .out = .f32,
            .nan = .{ .stated = if (p.case == .channel_fill) fill_rule else rule },
            .served = served(p.case),
            .tiles = .{ if (on_n) .{} else .{ .size = 256 }, if (on_n) .{ .size = 256 } else .{}, .{} },
        };
    }

    pub fn reset(p: *Probe) void {
        p.a.free(p.words);
        p.a.free(p.inverse);
        p.words = &.{};
        p.inverse = &.{};
        p.rig.reset();
    }

    fn inputs(p: *const Probe) usize {
        const s = p.shape;
        return switch (p.case) {
            .copy => s.rows,
            .columns, .transpose => s.rows * s.n,
            .tiles_into, .tiles_from => s.n * s.n * s.rows,
            .channel_fill => 0,
            .conv_pack => s.rows * s.n * s.k,
            .deconv_pack => s.rows * s.n * 4,
            .unfold => blk: {
                const m = geometries[s.n];
                break :blk @as(usize, m.side) * m.side * m.in;
            },
        };
    }

    pub fn count(p: *const Probe) usize {
        const s = p.shape;
        return switch (p.case) {
            .copy, .columns => s.rows,
            .transpose, .channel_fill => s.rows * s.n,
            .unfold => s.rows * geometries[s.n].k * geometries[s.n].k * geometries[s.n].in,
            else => p.inputs(),
        };
    }

    fn valid(p: *const Probe, s: c.Shape) bool {
        if (s.rows == 0 or (s.n == 0 and p.case != .unfold)) return false;
        return switch (p.case) {
            .copy => s.n == 1 and s.k == 1,
            .columns => s.k < s.n,
            .transpose => s.k == 1,
            .tiles_into, .tiles_from => s.k > 0 and s.n % s.k == 0,
            .channel_fill => s.k > 0 and s.k <= 16 and s.k <= s.n,
            .conv_pack => s.k > 0,
            .deconv_pack => s.k == 4,
            .unfold => s.n < geometries.len and s.k + s.rows <= outSide(geometries[s.n]) * outSide(geometries[s.n]),
        };
    }

    /// The input word output o copies; null: a constant (unfold's padding, channel fill).
    fn source(p: *const Probe, o: usize) ?usize {
        const s = p.shape;
        switch (p.case) {
            .copy => return o,
            .columns => return o * s.n + s.k,
            .transpose => return (o % s.rows) * s.n + o / s.rows,
            .tiles_into => return gridRow(o / s.rows, s.n, s.k) * s.rows + o % s.rows,
            .tiles_from => return p.inverse[o / s.rows] * s.rows + o % s.rows,
            .channel_fill => return null,
            .conv_pack => {
                const ch = o % s.n;
                const tap = o / s.n % s.k;
                return ((o / (s.n * s.k)) * s.n + ch) * s.k + tap;
            },
            .deconv_pack => {
                const ch = o % s.n;
                const output = o / s.n % s.rows;
                return (ch * s.rows + output) * 4 + o / (s.n * s.rows);
            },
            .unfold => {
                const m = geometries[s.n];
                const width = m.k * m.k * m.in;
                const col = o % width;
                const tap = col / m.in;
                const pixel = s.k + o / width;
                const ow = outSide(m);
                const y = @as(isize, @intCast((pixel / ow) * m.stride + tap / m.k)) - @as(isize, m.pad);
                const x = @as(isize, @intCast((pixel % ow) * m.stride + tap % m.k)) - @as(isize, m.pad);
                if (y < 0 or x < 0 or y >= @as(isize, m.side) or x >= @as(isize, m.side)) return null;
                return (@as(usize, @intCast(y)) * m.side + @as(usize, @intCast(x))) * m.in + col % m.in;
            },
        }
    }

    fn expected(p: *const Probe, o: usize) u32 {
        if (p.case == .channel_fill) {
            const ch = o % p.shape.n;
            const first = p.shape.n - p.shape.k;
            if (ch < first) return 0xffffffff;
            return if ((p.bits >> @intCast(ch - first)) & 1 != 0) 0x3f800000 else 0;
        }
        return if (p.source(o)) |s| p.words[s] else 0;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!p.valid(shape)) return false;
        p.reset();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        p.bits = prng.random().int(u16) | 1;
        p.words = try p.a.alloc(u32, p.inputs());
        prng.random().bytes(std.mem.sliceAsBytes(p.words));
        if (p.case == .tiles_from) {
            p.inverse = try p.a.alloc(u32, shape.n * shape.n);
            for (0..p.inverse.len) |r| p.inverse[gridRow(r, shape.n, shape.k)] = @intCast(r);
        }
        if (fill == .nonfinite) {
            var at: usize = 0;
            var o: usize = 0;
            while (o < p.count() and at < specials.len) : (o += 1) {
                if (p.source(o)) |s| {
                    p.words[s] = specials[at];
                    at += 1;
                }
            }
        }
        if (p.case != .channel_fill) p.input = try p.rig.input(std.mem.sliceAsBytes(p.words));
        p.out = try p.rig.output(p.count() * 4);
        return true;
    }

    pub fn run(p: *Probe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        var at: usize = 0;
        if (p.case != .channel_fill) {
            pass.e.setBuffer(p.input.buf, p.input.off, 0);
            at = 1;
        }
        pass.e.setBuffer(p.out.buf, p.out.off, at);
        const u = struct {
            fn w(x: usize) u32 {
                return @intCast(x);
            }
        }.w;
        var threads = p.count();
        switch (p.case) {
            .copy => pass.e.setValue(u(s.rows), 2),
            .columns => pass.e.setValue([4]u32{ u(s.rows), u(s.n), u(s.k), 1 }, 2),
            .transpose => pass.e.setValue([2]u32{ u(s.rows), u(s.n) }, 2),
            .tiles_into, .tiles_from => pass.e.setValue([4]u32{ u(s.rows), u(s.n), u(s.k), @intFromBool(p.case == .tiles_from) }, 2),
            .channel_fill => {
                pass.e.setValue([5]u32{ u(s.rows), u(s.n), u(s.n - s.k), u(s.k), p.bits }, 1);
                threads = s.rows * s.k;
            },
            .conv_pack => pass.e.setValue([3]u32{ u(s.rows), u(s.n), u(s.k) }, 2),
            .deconv_pack => pass.e.setValue([3]u32{ u(s.rows), u(s.n), 4 }, 2),
            .unfold => {
                const m = geometries[s.n];
                const os = u(outSide(m));
                pass.e.setValue([11]u32{ m.side, m.side, m.in, 1, m.k, os, os, m.stride, m.pad, u(s.k), u(s.rows) }, 2);
            },
        }
        pass.e.dispatchThreads(mtl.Size.of(threads, 1, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = std.mem.bytesAsSlice(u32, try p.rig.outputBytes());
        for (got, 0..) |word, o| {
            const want = p.expected(o);
            if (word != want) return try std.fmt.bufPrint(&p.why, "output {d} is 0x{x:0>8} where its word is 0x{x:0>8}", .{ o, word, want });
        }
        return null;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, o| {
            w.* = @as(f32, @bitCast(p.expected(o)));
            m.* = @abs(w.*);
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: [2 * cases.len]?*Probe = @splat(null),

    pub fn collect(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (&s.probes, 0..) |*entry, i| {
            const case = cases[i / 2];
            const control = i % 2 == 1;
            if (only) |needle| if (std.mem.indexOf(u8, label(case, control), needle) == null) continue;
            const source = try text(s.a, if (control) editOf(case) else null);
            defer s.a.free(source);
            const pipe = try rig_.pipeline(device, source, function(case));
            errdefer pipe.deinit();
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            try kernels.ensureUnusedCapacity(s.a, 1);
            probe.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = device, .queue = queue }, .pipe = pipe, .case = case, .control = control };
            entry.* = probe;
            kernels.appendAssumeCapacity(.{ .spec = probe.spec(), .ctx = probe, .vt = &vt, .control = if (control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes) |maybe| if (maybe) |p| {
            p.reset();
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        };
    }
};
