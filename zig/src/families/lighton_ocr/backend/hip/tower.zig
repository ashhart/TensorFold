//! Pixtral and LightOnOCR's learned projector on HIP, with BF16 rows and losslessly widened FP32 weights.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const cfg = @import("../../config.zig");
const weights = @import("../../weights.zig");
const ops = @import("ops.zig");
const vision = @import("vision.zig");
const CancelCheck = @import("lanes").stream.CancelCheck;
const cancels = @import("hip_serve").cancel;
const Allocator = std.mem.Allocator;

pub const max_grid = 110;
pub const max_patches = max_grid * max_grid;

pub const Shape = struct {
    rows: usize,
    tokens: usize,
    h: usize,
    w: usize,

    pub fn init(c: cfg.Config, gh: u32, gw: u32, pixel_count: usize) !Shape {
        try geometry(c);
        if (gh == 0 or gw == 0 or gh > max_grid or gw > max_grid) return error.ImageGrid;
        const rows = @as(usize, gh) * gw;
        const tokens = @as(usize, gh / 2) * (gw / 2);
        if (tokens == 0 or pixel_count != rows * c.patchWidth()) return error.BadImagePatches;
        return .{ .rows = rows, .tokens = tokens, .h = gh, .w = gw };
    }
};

fn geometry(c: cfg.Config) !void {
    if (c.vision.hidden != 1024 or c.vision.heads != 16 or c.vision.head_dim != 64 or
        c.vision.intermediate != 4096 or c.vision.layers != 24 or c.vision.patch != 14 or
        c.vision.side != 1540 or c.hidden != 1024 or c.merge != 2)
        return error.UnsupportedLightOnGeometry;
}

/// Q storage becomes channel-first unfold storage after the encoder has finished every patch, including odd edges.
pub const Scratch = struct {
    inv: usize,
    pixels: usize,
    x: usize,
    norm: usize,
    q: usize,
    key: usize,
    value: usize,
    qr: usize,
    kr: usize,
    vr: usize,
    attention: usize,
    branch: usize,
    gate: usize,
    up: usize,
    cosine: usize,
    sine: usize,
    bytes: usize,

    pub fn init(c: cfg.Config, rows: usize) !Scratch {
        try geometry(c);
        if (rows == 0 or rows > max_patches) return error.BadImagePatches;
        var at: usize = 0;
        const row_bytes = rows * c.vision.hidden * 2;
        const mid_bytes = rows * c.vision.intermediate * 2;
        const table_bytes = rows * (c.vision.head_dim / 2) * 2;
        return .{
            .inv = take(&at, c.vision.head_dim / 2 * 4),
            .pixels = take(&at, rows * c.patchWidth() * 2),
            .x = take(&at, row_bytes),
            .norm = take(&at, row_bytes),
            .q = take(&at, row_bytes),
            .key = take(&at, row_bytes),
            .value = take(&at, row_bytes),
            .qr = take(&at, row_bytes),
            .kr = take(&at, row_bytes),
            .vr = take(&at, row_bytes),
            .attention = take(&at, row_bytes),
            .branch = take(&at, row_bytes),
            .gate = take(&at, mid_bytes),
            .up = take(&at, mid_bytes),
            .cosine = take(&at, table_bytes),
            .sine = take(&at, table_bytes),
            .bytes = at,
        };
    }

    fn take(at: *usize, bytes: usize) usize {
        const offset = std.mem.alignForward(usize, at.*, 256);
        at.* = std.mem.alignForward(usize, offset + bytes, 256);
        return offset;
    }
};

pub fn scratchBytes(c: cfg.Config) !usize {
    return (try Scratch.init(c, max_patches)).bytes;
}

pub const Tower = struct {
    gpa: Allocator,
    r: *const hip.Runtime,
    s: hip.Stream,
    shared: *const hip.Launcher,
    c: cfg.Config,
    base: u64,
    weight_bytes: usize,
    blocks: []weights.VisionLayer,
    top: struct { patch: u64, pre_norm: u64, projector_norm: u64, merger: u64, project1: u64, project2: u64 },
    functions: ops.Kernels,
    scratch: hip.DeviceBuffer,

    /// The caller owns the FP32 weight arena and shared launcher; all copied metadata and scratch remain tower-owned.
    pub fn init(gpa: Allocator, r: *const hip.Runtime, s: hip.Stream, w: *const weights.Layout, base: u64, shared: *const hip.Launcher) !Tower {
        try geometry(w.c);
        if (r != s.r or base == 0 or base % 4 != 0 or w.vision.len != w.c.vision.layers) return error.BadTowerWeights;
        const weight_bytes = try std.math.mul(usize, w.bytes, 2);
        _ = std.math.add(u64, base, weight_bytes) catch return error.BadTowerWeights;
        const blocks = try gpa.dupe(weights.VisionLayer, w.vision);
        errdefer gpa.free(blocks);
        var functions = try ops.Kernels.load(r, hip.kernels.images[@backingInt(hip.kernels.Group.lighton_ocr)], shared);
        errdefer functions.unload();
        var scratch = try hip.DeviceBuffer.alloc(r, try scratchBytes(w.c));
        errdefer scratch.free();
        var inv: [32]f32 = undefined;
        try ops.inverseFrequencies(&inv, @floatCast(w.c.vision.theta));
        try scratch.upload(0, std.mem.sliceAsBytes(&inv));
        return .{
            .gpa = gpa,
            .r = r,
            .s = s,
            .shared = shared,
            .c = w.c,
            .base = base,
            .weight_bytes = weight_bytes,
            .blocks = blocks,
            .top = .{ .patch = w.patch, .pre_norm = w.pre_norm, .projector_norm = w.projector_norm, .merger = w.merger, .project1 = w.project1, .project2 = w.project2 },
            .functions = functions,
            .scratch = scratch,
        };
    }

    pub fn deinit(t: *Tower) void {
        drain(t.s);
        t.scratch.free();
        t.functions.unload();
        t.gpa.free(t.blocks);
        t.* = undefined;
    }

    /// All image patches enter the encoder; the learned merger alone discards unmatched spatial edges.
    pub fn encode(t: *Tower, pixels: []const f32, gh: u32, gw: u32, out: u64) !void {
        return t.encodeCancelable(pixels, gh, gw, out, null);
    }

    pub fn encodeCancelable(t: *Tower, pixels: []const f32, gh: u32, gw: u32, out: u64, cancel: ?CancelCheck) !void {
        try cancels.poll(cancel);
        const shape = try Shape.init(t.c, gh, gw, pixels.len);
        try outputRange(t.scratch.base(), t.scratch.len, t.base, t.weight_bytes, out, shape.tokens * t.c.hidden * 2);
        const p = try Scratch.init(t.c, shape.rows);
        if (p.bytes > t.scratch.len) return error.TowerScratchTooSmall;
        const host = try t.gpa.alloc(u16, pixels.len);
        defer t.gpa.free(host);
        for (host, pixels) |*dst, value| dst.* = core.bf16_round.bits(value);
        try t.s.synchronize();
        try t.scratch.upload(p.pixels, std.mem.sliceAsBytes(host));
        errdefer drain(t.s);
        try t.functions.tables(t.s, t.ptr(p.inv), t.ptr(p.cosine), t.ptr(p.sine), gh, gw);
        const n = shape.rows;
        const h: usize = t.c.vision.hidden;
        const mid: usize = t.c.vision.intermediate;
        try t.linear(n, h, t.c.patchWidth(), p.pixels, t.top.patch, t.ptr(p.norm));
        try t.rms(p.norm, p.x, t.top.pre_norm, n, t.c.vision.eps);
        for (t.blocks) |b| {
            try cancels.drain(t.s, cancel);
            try t.rms(p.x, p.norm, b.input_norm, n, t.c.vision.eps);
            try t.linear(n, h, h, p.norm, b.q, t.ptr(p.q));
            try t.linear(n, h, h, p.norm, b.k, t.ptr(p.key));
            try t.linear(n, h, h, p.norm, b.v, t.ptr(p.value));
            try t.functions.rotary(t.s, t.ptr(p.q), t.ptr(p.key), t.ptr(p.value), t.ptr(p.cosine), t.ptr(p.sine), t.ptr(p.qr), t.ptr(p.kr), t.ptr(p.vr), n, t.c.vision.heads);
            try t.functions.attention(t.s, t.ptr(p.qr), t.ptr(p.kr), t.ptr(p.vr), t.ptr(p.attention), n, t.c.vision.heads, 0, n, 0.125);
            try t.linear(n, h, h, p.attention, b.o, t.ptr(p.branch));
            try t.functions.residual(t.s, t.ptr(p.branch), t.ptr(p.x), n, h);
            try t.rms(p.x, p.norm, b.post_norm, n, t.c.vision.eps);
            try t.linear(n, mid, h, p.norm, b.gate, t.ptr(p.gate));
            try t.linear(n, mid, h, p.norm, b.up, t.ptr(p.up));
            try t.functions.torchSwiGlu(t.s, t.ptr(p.gate), t.ptr(p.up), t.ptr(p.gate), n, mid);
            try t.linear(n, h, mid, p.gate, b.down, t.ptr(p.branch));
            try t.functions.residual(t.s, t.ptr(p.branch), t.ptr(p.x), n, h);
        }
        try cancels.drain(t.s, cancel);
        try t.rms(p.x, p.norm, t.top.projector_norm, n, t.c.eps);
        try t.functions.merge(t.s, t.ptr(p.norm), t.ptr(p.q), gh, gw, h);
        try t.linear(shape.tokens, h, t.c.mergedWidth(), p.q, t.top.merger, t.ptr(p.branch));
        try t.linear(shape.tokens, t.c.hidden, h, p.branch, t.top.project1, t.ptr(p.norm));
        try t.functions.gelu(t.s, t.ptr(p.norm), t.ptr(p.norm), shape.tokens, t.c.hidden);
        try t.linear(shape.tokens, t.c.hidden, t.c.hidden, p.norm, t.top.project2, out);
        try t.s.synchronize();
        try cancels.poll(cancel);
    }

    fn ptr(t: *const Tower, offset: usize) u64 {
        return t.scratch.base() + offset;
    }

    fn weight(t: *const Tower, offset: u64) u64 {
        return t.base + offset * 2;
    }

    fn rms(t: *Tower, x: usize, y: usize, weight_offset: u64, rows: usize, eps: f32) !void {
        try t.functions.rms(t.s, t.ptr(x), t.weight(weight_offset), t.ptr(y), rows, t.c.vision.hidden, eps);
    }

    fn linear(t: *Tower, rows: usize, outputs: usize, inputs: usize, x: usize, weight_offset: u64, out: u64) !void {
        var unused_arena = hip.Arena.counting();
        const o: hip.ops.Ops = .{ .l = t.shared, .bf16 = true, .stream = t.s.handle, .arena = &unused_arena };
        try o.denseRows(.{ .ptr = t.ptr(x), .kind = .bf16 }, t.weight(weight_offset), .{ .ptr = out, .kind = .bf16 }, rows, outputs, inputs);
    }
};

fn drain(s: hip.Stream) void {
    s.synchronize() catch @panic("LightOnOCR HIP tower work did not drain; live image resources cannot be freed");
}

pub fn outputRange(scratch: u64, scratch_len: usize, base: u64, weight_len: usize, out: u64, bytes: usize) !void {
    if (out == 0 or out % 2 != 0 or bytes == 0) return error.RowShape;
    const end = std.math.add(u64, out, bytes) catch return error.RowShape;
    const scratch_end = std.math.add(u64, scratch, scratch_len) catch return error.RowShape;
    const weight_end = std.math.add(u64, base, weight_len) catch return error.RowShape;
    if ((out < scratch_end and scratch < end) or (out < weight_end and base < end)) return error.Aliasing;
}

test {
    _ = @import("tower_test.zig");
}
