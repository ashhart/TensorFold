//! flashnext-exl3: the raw-pack reader, f64 conversions, error metrics and the one-layer pass rig the gates share.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const fwd = @import("exl3_fwd_cpu.zig");
const hc = @import("exl3_hc.zig");
const moe = @import("exl3_moe.zig");
const router_cpu = @import("exl3_router.zig");
const route_cpu = @import("exl3_route.zig");
const scan = @import("../pack/exl3_scan.zig");
const place = @import("../pack/exl3_place.zig");
const gdn_cpu = @import("exl3_gdn_cpu.zig");
const attn_cpu = @import("exl3_attn_cpu.zig");

const X = flash.exl3;
const wt = X.prompt_wt;
const pl = X.prompt_load;
const batch = X.batch;
const Gpu = check.Gpu;
const st = core.safetensors;

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const topk = slots - 1;
const experts_n = wt.experts_n;

/// "The capital of France is Paris. It is known for the Eiffel Tower." (exl3_fwd_tests.zig's default ids)
pub const default_ids = [_]u32{ 760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13 };

// -- the raw pack (the oracle's view: tensors exactly as stored, no widening) -------------------------------------

pub const RawPack = struct {
    files: std.ArrayList(st.File) = .empty,

    pub fn open(gpu: Gpu, dir: []const u8) !RawPack {
        var p: RawPack = .{};
        var buf: [512]u8 = undefined;
        for (0..7) |i| {
            const path = try std.fmt.bufPrint(&buf, "{s}/model-{d:0>5}-of-00007.safetensors", .{ dir, i + 1 });
            try p.files.append(gpu.gpa, try st.File.open(gpu.gpa, gpu.io, path));
        }
        return p;
    }

    pub fn close(p: *RawPack, gpu: Gpu) void {
        for (p.files.items) |*f| f.close(gpu.io);
        p.files.deinit(gpu.gpa);
    }

    pub fn get(p: *const RawPack, comptime fmt: []const u8, args: anytype) !st.Tensor {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        for (p.files.items) |*f| if (f.get(name)) |t| return t;
        std.debug.print("raw pack: no {s}\n", .{name});
        return error.MissingTensor;
    }
};

pub fn typed(comptime T: type, a: std.mem.Allocator, bytes: []const u8) ![]T {
    const out = try a.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

pub fn valueOf(t: st.Tensor, i: usize) f64 {
    const h = std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little);
    return switch (t.dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(h))),
        .bf16 => @as(f32, @bitCast(@as(u32, h) << 16)),
        else => unreachable,
    };
}

pub fn f64Of(a: std.mem.Allocator, t: st.Tensor) ![]f64 {
    if (t.dtype != .f16 and t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try a.alloc(f64, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = valueOf(t, i);
    return out;
}

pub fn f32Of(a: std.mem.Allocator, t: st.Tensor) ![]f32 {
    if (t.dtype != .f16 and t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try a.alloc(f32, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = @floatCast(valueOf(t, i));
    return out;
}

pub fn f16Of(a: std.mem.Allocator, t: st.Tensor) ![]f16 {
    if (t.dtype != .f16) return error.UnexpectedTensor;
    return typed(f16, a, t.bytes);
}

/// The EXL3 matrix `prefix` as the float64 oracle's QMat (W = diag(suh) H Wq H diag(svh), exl3_format's decoder).
pub fn qmat(a: std.mem.Allocator, raw: *const RawPack, prefix: []const u8, threads: usize) !fwd.QMat {
    const t = try raw.get("{s}.trellis", .{prefix});
    const trellis = try typed(i16, a, t.bytes);
    defer a.free(trellis);
    const suh = try typed(f16, a, (try raw.get("{s}.suh", .{prefix})).bytes);
    defer a.free(suh);
    const svh = try typed(f16, a, (try raw.get("{s}.svh", .{prefix})).bytes);
    defer a.free(svh);
    return fwd.QMat.init(a, @intCast(t.dim(2) / 8), trellis, t.dim(0), t.dim(1), suh, svh, threads);
}

pub fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn bfValue(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

pub const Err = struct { max_abs: f64 = 0, peak: f64 = 0, rel: f64 = 0 };

pub fn errOf(got: anytype, want: []const f64) Err {
    var e: Err = .{};
    for (want, 0..) |w, i| {
        const g: f64 = switch (@TypeOf(got)) {
            []const u16, []u16 => bfValue(got[i]),
            else => @floatCast(got[i]),
        };
        e.max_abs = @max(e.max_abs, @abs(g - w));
        e.peak = @max(e.peak, @abs(w));
    }
    e.rel = if (e.peak > 0) e.max_abs / e.peak else 0;
    return e;
}

pub fn download(comptime T: type, gpu: Gpu, s: cuda.Stream, ptr: u64, n: usize) ![]T {
    try s.synchronize();
    const out = try gpu.gpa.alloc(T, n);
    const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
    try b.download(0, std.mem.sliceAsBytes(out));
    return out;
}

/// A pageable copy into the pass's scratch: the legacy-stream copy is ordered against nothing, so drain both sides.
pub fn uploadTo(gpu: Gpu, s: cuda.Stream, ptr: u64, bytes: []const u8) !void {
    try s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = bytes.len };
    try b.upload(0, bytes);
    try gpu.ctx.synchronize();
}

// -- the pass rig -----------------------------------------------------------------------------------------------------

pub const Pass = batch.Pass(flash.Tri);

pub const Rig = struct {
    gpu: Gpu,
    stream: cuda.Stream,
    glue: flash.Glue,
    k: batch.Kernels,
    s: batch.Scratch,
    store: wt.Store,
    pack_mod: cuda.Module,
    pack_fn: cuda.Function,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    conv: [wt.linear_n]cuda.DeviceBuffer = undefined,
    state: [wt.linear_n]cuda.DeviceBuffer = undefined,
    kc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    vc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    ikc: [wt.attn_n]cuda.DeviceBuffer = undefined,
    pooled: [wt.attn_n]cuda.DeviceBuffer = undefined,
    tail: cuda.DeviceBuffer = undefined,
    history: [2]i64 = .{ 0, 0 },
    scores: cuda.DeviceBuffer = undefined,
    cap: usize,
    /// The ranks' NCCL owner when a check runs two ranks (flashnext_exl3_tp_tests.zig); null: one rank.
    collective: ?*flash.collective.Collective = null,

    pub fn init(r: *Rig, gpu: Gpu, dir: []const u8, rows: usize, cap: usize) !void {
        r.* = .{ .gpu = gpu, .stream = try cuda.Stream.init(gpu.d, true), .glue = undefined, .k = undefined, .s = undefined, .store = undefined, .pack_mod = undefined, .pack_fn = undefined, .cap = cap };
        r.glue = try flash.Glue.load(gpu.d);
        r.k = try batch.Kernels.load(gpu.d);
        r.s = try batch.Scratch.alloc(gpu.gpa, gpu.d, r.stream, rows);
        r.store = try wt.Store.open(gpu.gpa, gpu.io, dir);
        try r.store.detectNorms();
        r.pack_mod = try cuda.Module.load(gpu.d, cuda.kernels.experts_pack);
        r.pack_fn = try r.pack_mod.function(wt.pack_symbol);
        const gdn = flash.gdn;
        const z: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = 0, .len = 0 };
        r.conv = @splat(z);
        r.state = @splat(z);
        r.kc = @splat(z);
        r.vc = @splat(z);
        r.ikc = @splat(z);
        r.pooled = @splat(z);
        // every layer's state, as Engine.newSeq makes them (a check may step any layer)
        for (&r.conv, &r.state) |*c, *x| {
            c.* = try r.zeroed((gdn.taps - 1) * gdn.conv_dim * 2);
            x.* = try r.zeroed(wt.state_n * 4);
        }
        for (&r.kc, &r.vc, &r.ikc, &r.pooled) |*kc, *vc, *ikc, *pooled| {
            kc.* = try r.zeroed(cap * wt.kv_heads * wt.head_dim * 2);
            vc.* = try r.zeroed(cap * wt.kv_heads * wt.head_dim * 2);
            ikc.* = try r.zeroed(cap * wt.index_dim * 2);
            pooled.* = try r.zeroed(flash.prompt.pooledBytes(cap));
        }
        r.tail = try r.zeroed(wt.ple_tail_n * wide * 2);
        r.scores = try r.zeroed(batch.scoreBytes(cap, rows));
    }

    fn zeroed(r: *Rig, bytes: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0, r.stream.handle);
        try r.stream.synchronize();
        return b;
    }

    pub fn reset(r: *Rig) !void {
        for (r.bufs.items) |b| try b.fill8(0, r.stream.handle);
        try r.stream.synchronize();
    }

    pub fn deinit(r: *Rig) void {
        r.stream.synchronize() catch {};
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
        r.pack_mod.unload();
        r.store.deinit();
        r.s.free(r.gpu.gpa);
        r.k.deinit();
        r.glue.deinit();
        r.stream.deinit();
    }

    pub fn pass(r: *Rig, layers: []const pl.Layer) Pass {
        return .{
            .gpa = r.gpu.gpa,
            .io = r.gpu.io,
            .stream = r.stream,
            .tri = .{ .own = &r.glue, .s = r.stream },
            .k = &r.k,
            .s = &r.s,
            .m = .{ .table_w = 0, .table_s = 0, .table_b = 0, .layers = layers, .mixer = undefined, .head = undefined, .ngram = undefined, .collective = r.collective },
            .seq = .{ .conv = &r.conv, .state = &r.state, .kc = &r.kc, .vc = &r.vc, .ikc = &r.ikc, .pooled = &r.pooled, .tail = r.tail, .history = &r.history, .cap = r.cap, .scores = r.scores.ptr },
            .hooks = .{},
        };
    }
};

pub fn bf16Rows(a: std.mem.Allocator, rng: std.Random, n: usize, sigma: f32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Bits(rng.floatNorm(f32) * sigma);
    return v;
}
