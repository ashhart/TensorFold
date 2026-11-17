//! tf-cuda-test glm-pdl-order [MAIN_FATBINS]: GLM kernels launched dependent behind a slow writer read its output.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const q = cuda.qmmf;

const spins: i64 = 20_000_000; // about 8 ms at GB10's clock: a dependent grid starts long before the writer ends
const poison: u8 = 0xff;
const tries = 8;

const Mode = enum { plain, dependent };
const Bytes = []align(16) u8;

const Ctx = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    probe: cuda.Module,
    main_dir: ?[]const u8,
    mains: std.StringHashMap(cuda.Module),

    /// Main's build of `image` (MAIN_FATBINS/<image>.fatbin), loaded once.
    fn main(c: *Ctx, image: []const u8) !cuda.Module {
        if (c.mains.get(image)) |m| return m;
        const a = c.gpu.gpa;
        const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ c.main_dir.?, image });
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(c.gpu.io, path, a, .limited(512 << 20));
        defer a.free(bytes);
        const al = try a.alignedAlloc(u8, .@"8", bytes.len); // cuModuleLoadData wants an aligned image
        defer a.free(al);
        @memcpy(al, bytes);
        const m = try cuda.Module.load(c.gpu.d, al);
        try c.mains.put(image, m);
        return m;
    }
};

/// A kernel's input (`in`, poisoned, then filled from `src` by the slow writer) and output bytes.
const Bufs = struct {
    src: cuda.DeviceBuffer,
    in: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,

    fn init(c: *const Ctx, input: Bytes, out_bytes: usize) !Bufs {
        std.debug.assert(input.len % 16 == 0);
        var src = try cuda.DeviceBuffer.fromHost(c.gpu.d, input);
        errdefer src.free();
        var in = try cuda.DeviceBuffer.alloc(c.gpu.d, input.len);
        errdefer in.free();
        return .{ .src = src, .in = in, .out = try cuda.DeviceBuffer.alloc(c.gpu.d, out_bytes) };
    }

    fn deinit(b: *Bufs) void {
        b.out.free();
        b.in.free();
        b.src.free();
    }
};

/// Launches the kernel under test from `m` (the head's image or main's), plainly or as the production launcher does.
const Launch = *const fn (c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void;

fn slowCopy(c: *const Ctx, b: *const Bufs) !void {
    var a: cuda.Args = .{};
    a.add(b.src.ptr);
    a.add(b.in.ptr);
    a.add(@as(i64, @intCast(b.in.len / 16)));
    a.add(spins);
    try cuda.launch.launch(try c.probe.function("tf_probe_pdl_slow_copy"), .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, c.s, &a);
}

fn slowCopyOn(c: *const Ctx, s: cuda.Stream, src: u64, dst: u64, bytes: usize) !void {
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(i64, @intCast(bytes / 16)));
    a.add(spins);
    try cuda.launch.launch(try c.probe.function("tf_probe_pdl_slow_copy"), .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, s, &a);
}

/// The output after the slow writer then the kernel, from fresh poisoned input and zeroed output.
fn after(c: *Ctx, b: *const Bufs, launch: Launch, mode: Mode, m: cuda.Module, sync_first: bool) ![]u8 {
    try b.in.fill8(poison, c.s.handle); // on the test stream: a null-stream fill is not ordered with a non-blocking one
    try b.out.fill8(0, c.s.handle);
    try c.s.synchronize();
    try slowCopy(c, b);
    if (sync_first) try c.s.synchronize();
    try launch(c, b, mode, m);
    try c.s.synchronize();
    return check.download(c.gpu, b.out);
}

/// One kernel: dependent after the writer equals plain after it, and (with MAIN_FATBINS) main's bytes.
fn case(c: *Ctx, name: []const u8, image: []const u8, head: cuda.Module, b: *const Bufs, launch: Launch) !void {
    const a = c.gpu.gpa;
    const want = try after(c, b, launch, .plain, head, true);
    defer a.free(want);
    const got = try after(c, b, launch, .dependent, head, false);
    defer a.free(got);
    try check.sameBytes(name, got, want);
    const vs_main = c.main_dir != null and !std.mem.eql(u8, image, "probe"); // main's probe image has no such kernel
    if (vs_main) {
        const was = try after(c, b, launch, .plain, try c.main(image), true);
        defer a.free(was);
        try check.sameBytes(name, want, was);
    }
    check.pass("{s}: dependent after a slow writer = plain{s}", .{ name, if (vs_main) " = main" else "" });
}

fn go(f: cuda.Function, cfg: cuda.launch.Config, mode: Mode, c: *const Ctx, a: *cuda.Args) !void {
    if (mode == .dependent) return glm.pdl.launch(f, cfg, c.s, a);
    return cuda.launch.launch(f, cfg, c.s, a);
}

fn visToBf16(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
    const n: i64 = @intCast(b.in.len / 4);
    var a: cuda.Args = .{};
    a.add(b.in.ptr);
    a.add(b.out.ptr);
    a.add(n);
    try go(try m.function("glm_vis_to_bf16"), .{ .grid = .{ .x = @intCast(@divExact(n, 256)) }, .block = .{ .x = 256 } }, mode, c, &a);
}

const copy_rows = 4096; // 8,192 CTAs of 256: many waves behind the writer
const copy_row_bytes = 512;
fn copyRows(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
    var a: cuda.Args = .{};
    a.add(b.in.ptr);
    a.add(@as(i64, copy_row_bytes));
    a.add(b.out.ptr);
    a.add(@as(i64, 2 * copy_row_bytes));
    a.add(@as(c_int, copy_rows));
    a.add(@as(c_int, copy_row_bytes));
    try go(try m.function("tf_glm_copy_rows"), .{ .grid = .{ .x = copy_rows * copy_row_bytes / 256 }, .block = .{ .x = 256 } }, mode, c, &a);
}

const norm_rows = 4;
const norm_out = norm_rows * 4096 * 2; // the normed rows, then each row's 64 group sums (fp32) in the same buffer
var norm_w: u64 = 0;
fn rmsnorm(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
    var a: cuda.Args = .{};
    a.add(b.in.ptr);
    a.add(@as(i64, 4096));
    a.add(norm_w);
    a.add(b.out.ptr);
    a.add(@as(i64, 4096));
    a.add(b.out.ptr + norm_out);
    a.add(@as(f32, 1e-6));
    a.add(@as(c_int, 1));
    try go(try m.function("tf_glm_rmsnorm_4096_f32w"), .{ .grid = .{ .x = norm_rows }, .block = .{ .x = 256 } }, mode, c, &a);
}

const amax_e = 8;
fn amax(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
    var a: cuda.Args = .{};
    a.add(b.in.ptr);
    a.add(@as(i64, @intCast(b.in.len / 2 / amax_e)));
    a.add(b.out.ptr);
    try go(try m.function("tf_glm_nvfp4_amax"), .{ .grid = .{ .x = amax_e }, .block = .{ .x = 1024 } }, mode, c, &a);
}

const swiglu_rows = 4;
const swiglu_f = 2048;
fn swiglu(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
    var a: cuda.Args = .{};
    a.add(b.in.ptr);
    a.add(b.out.ptr);
    a.add(@as(c_int, swiglu_f));
    try go(try m.function("df_swiglu"), .{ .grid = .{ .x = swiglu_rows, .y = (swiglu_f + 1023) / 1024 }, .block = .{ .x = 256 } }, mode, c, &a);
}

const reduce_total = 8192;
const reduce_sk = 4;
fn mmReduce(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
    var a: cuda.Args = .{};
    a.add(b.in.ptr);
    a.add(b.out.ptr);
    a.add(@as(i64, reduce_total));
    a.add(@as(c_int, reduce_sk));
    a.add(@as(c_int, 0));
    try go(try m.function("glm_mm_reduce"), .{ .grid = .{ .x = reduce_total / 256 }, .block = .{ .x = 256 } }, mode, c, &a);
}

/// The qmmf decode rows: a lane whose functions come from `m` (the head's or main's qmmf image), `pdl` when dependent.
const Qmmf = struct {
    var lane: q.Lane = undefined;
    var w: q.Weight = undefined;
    var part: u64 = 0;
    var clusters = false;

    fn run(c: *Ctx, b: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
        var l = lane;
        if (m.handle != lane.mod.handle) try remap(c, &l, m);
        l.clusters = clusters;
        l.pdl = mode == .dependent;
        try l.matmulOut(c.s, b.in.ptr, w.k, 1, w, b.out.ptr, .bf16, part);
    }

    /// The same lane over main's image: each function by its symbol.
    fn remap(c: *const Ctx, l: *q.Lane, m: cuda.Module) !void {
        const d = c.gpu.d;
        for (&l.fns) |*set| for (set) |*f| {
            f.* = try m.function(named(d, f.*));
        };
        for (&l.f32_fns) |*set| for (set) |*f| {
            f.* = try m.function(named(d, f.*));
        };
        for (&l.reduce) |*f| f.* = try m.function(named(d, f.*));
        for (&l.half) |*a| for (a) |*o| for (o) |*f| {
            f.* = try m.function(named(d, f.*));
        };
    }
};

fn bytesOf(comptime T: type, a: std.mem.Allocator, n: usize, seed: u64, gen: *const fn (std.Random) T) !Bytes {
    const bytes = try a.alignedAlloc(u8, .@"16", n * @sizeOf(T));
    var prng = std.Random.DefaultPrng.init(seed);
    for (std.mem.bytesAsSlice(T, bytes)) |*x| x.* = gen(prng.random());
    return bytes;
}

/// A driver-given kernel name as the C string it is.
fn named(d: *const cuda.Driver, f: cuda.Function) [:0]const u8 {
    const n = d.functionName(f.handle);
    return n.ptr[0..n.len :0];
}

fn bf16(r: std.Random) u16 {
    return (r.int(u16) & 0x807f) | (@as(u16, 118 + r.uintLessThan(u16, 16)) << 7); // finite, |x| from 2^-9 to 2^7
}
fn f32v(r: std.Random) f32 {
    return (r.float(f32) - 0.5) * 8;
}
fn word(r: std.Random) u32 {
    return r.int(u32) & 0x7fffffff;
}
fn small(r: std.Random) f32 {
    return (r.float(f32) - 0.5) / 4;
}

/// The planted control: reading before the wait sees the poison at least once in `tries`, and then reads wrong.
fn control(c: *Ctx) !void {
    const a = c.gpu.gpa;
    const n = 1024;
    const input = try bytesOf(u32, a, n, 7, word);
    defer a.free(input);
    var b = try Bufs.init(c, input, n * 4);
    defer b.deinit();
    var seen = try cuda.DeviceBuffer.alloc(c.gpu.d, 16);
    defer seen.free();
    const words = std.mem.bytesAsSlice(u32, input);
    for (0..tries) |t| {
        try b.in.fill8(poison, c.s.handle);
        try b.out.fill8(0, c.s.handle);
        try seen.fill8(0, c.s.handle);
        try c.s.synchronize();
        try slowCopy(c, &b);
        var args: cuda.Args = .{};
        args.add(b.in.ptr);
        args.add(b.out.ptr);
        args.add(seen.ptr);
        args.add(@as(c_int, n));
        try cuda.launch.launch(try c.probe.function("tf_probe_pdl_early_read"), .{ .grid = .{ .x = n / 256 }, .block = .{ .x = 256 }, .pdl = true }, c.s, &args);
        try c.s.synchronize();
        const s = try check.download(c.gpu, seen);
        defer a.free(s);
        if (std.mem.readInt(u32, s[0..4], .little) != 0xffffffff) continue;
        const out = try check.download(c.gpu, b.out);
        defer a.free(out);
        try check.expect(std.mem.readInt(u32, out[0..4], .little) != words[0] *% 2 +% 1, "control: read the poison yet wrote the right value", .{});
        check.pass("control: a read before the wait saw the poison (try {d}) and wrote the wrong value, as planted", .{t + 1});
        return;
    }
    try check.expect(false, "control: overlap not exercised in {d} tries: the dependent grid never started before the writer ended", .{tries});
}

/// A chain through a plain launch, a device copy and an event wait ends where all-plain does.
fn chain(c: *Ctx) !void {
    const a = c.gpu.gpa;
    const n = 4096;
    const input = try bytesOf(u32, a, n, 11, word);
    defer a.free(input);
    var b = try Bufs.init(c, input, n * 4);
    defer b.deinit();
    var mid: [4]cuda.DeviceBuffer = undefined;
    for (&mid) |*m| m.* = try cuda.DeviceBuffer.alloc(c.gpu.d, n * 4);
    defer for (&mid) |*m| m.free();
    var other = try cuda.Stream.init(c.gpu.d, true);
    defer other.deinit();
    var ready = try cuda.Event.init(c.gpu.d, false);
    defer ready.deinit();
    var done = try cuda.Event.init(c.gpu.d, false);
    defer done.deinit();
    const f = try c.probe.function("tf_probe_pdl_enter_read");
    const step = struct {
        fn go_(s: cuda.Stream, fx: cuda.Function, src: u64, dst: u64, pdl: bool) !void {
            var args: cuda.Args = .{};
            args.add(src);
            args.add(dst);
            args.add(@as(c_int, n));
            try cuda.launch.launch(fx, .{ .grid = .{ .x = n / 256 }, .block = .{ .x = 256 }, .pdl = pdl }, s, &args);
        }
    }.go_;
    try b.in.fill8(poison, c.s.handle);
    try c.s.synchronize();
    try slowCopy(c, &b);
    try step(c.s, f, b.in.ptr, mid[0].ptr, true);
    try step(c.s, f, mid[0].ptr, mid[1].ptr, false);
    try ready.record(c.s);
    try other.wait(ready); // a producer on another stream, after a spin of its own: the consumer must wait for it
    try slowCopyOn(c, other, mid[1].ptr, mid[3].ptr, n * 4);
    try step(other, f, mid[3].ptr, mid[2].ptr, false);
    try done.record(other);
    try c.s.wait(done);
    try mid[3].uploadAsync(0, input, c.s.handle); // a host copy between the event wait and the dependent launch
    try step(c.s, f, mid[2].ptr, b.out.ptr, true);
    try c.s.synchronize();
    const out = try check.download(c.gpu, b.out);
    defer a.free(out);
    const words = std.mem.bytesAsSlice(u32, input);
    for (words, std.mem.bytesAsSlice(u32, out), 0..) |x, y, i| {
        const want = (((x *% 2 +% 1) *% 2 +% 1) *% 2 +% 1) *% 2 +% 1;
        try check.expect(y == want, "chain: word {d} is {x}, want {x}", .{ i, y, want });
    }
    check.pass("chain: dependent, plain, another stream's slow producer behind events, a host copy, dependent = the all-plain result", .{});
    // a dependent launch right after a host upload reads the uploaded bytes, not the poison under them
    try mid[0].fill8(poison, c.s.handle);
    try mid[1].fill8(0, c.s.handle);
    try c.s.synchronize();
    try mid[0].uploadAsync(0, input, c.s.handle);
    try step(c.s, f, mid[0].ptr, mid[1].ptr, true);
    try c.s.synchronize();
    const up = try check.download(c.gpu, mid[1]);
    defer a.free(up);
    for (words, std.mem.bytesAsSlice(u32, up), 0..) |x, y, i| {
        try check.expect(y == x *% 2 +% 1, "host upload: word {d} is {x}, want {x}", .{ i, y, x *% 2 +% 1 });
    }
    check.pass("host upload then a dependent launch: the kernel read the uploaded bytes", .{});
}

pub fn run(gpu: check.Gpu, main_dir: ?[]const u8) !void {
    const cap = try gpu.ctx.capability();
    if (cap < 90) {
        check.pass("glm-pdl-order: sm_{d} has no dependent launches; skipped", .{cap});
        return;
    }
    try check.expect(glm.pdl.on(), "glm-pdl-order: TF_GLM_PDL=0 launches GLM's kernels plain; unset it to test dependent launches", .{});
    const a = gpu.gpa;
    var c: Ctx = .{ .gpu = gpu, .s = try cuda.Stream.init(gpu.d, true), .probe = try cuda.Module.load(gpu.d, cuda.kernels.probe), .main_dir = main_dir, .mains = .init(a) };
    defer c.mains.deinit();
    defer c.s.deinit();
    defer c.probe.unload();
    try control(&c);
    try chain(&c);
    {
        const input = try bytesOf(u32, a, 4096, 3, word);
        defer a.free(input);
        var b = try Bufs.init(&c, input, 4096 * 4);
        defer b.deinit();
        try case(&c, "probe tf_probe_pdl_enter_read", "probe", c.probe, &b, struct {
            fn r(cx: *Ctx, bx: *const Bufs, mode: Mode, m: cuda.Module) anyerror!void {
                var args: cuda.Args = .{};
                args.add(bx.in.ptr);
                args.add(bx.out.ptr);
                args.add(@as(c_int, 4096));
                try cuda.launch.launch(try m.function("tf_probe_pdl_enter_read"), .{ .grid = .{ .x = 16 }, .block = .{ .x = 256 }, .pdl = mode == .dependent }, cx.s, &args);
            }
        }.r);
    }
    const images = [_][]const u8{ "glm_vision", "glm_ops", "glm_glue", "glm_quant", "glm_dflash", "glm_mm" };
    var mods: [images.len]cuda.Module = undefined;
    inline for (images, 0..) |name, i| mods[i] = try cuda.Module.load(gpu.d, @field(cuda.kernels, name));
    defer for (&mods) |*m| m.unload();
    const host_w = try bytesOf(f32, a, 4096, 5, small);
    defer a.free(host_w);
    var w = try cuda.DeviceBuffer.fromHost(gpu.d, host_w);
    defer w.free();
    norm_w = w.ptr;
    const Case = struct { name: []const u8, image: usize, input: Bytes, out: usize, run: Launch };
    const cases = [_]Case{
        .{ .name = "glm_vision glm_vis_to_bf16", .image = 0, .input = try bytesOf(f32, a, 1 << 16, 21, f32v), .out = (1 << 16) * 2, .run = visToBf16 },
        .{ .name = "glm_ops tf_glm_copy_rows (multi-wave)", .image = 1, .input = try bytesOf(u32, a, copy_rows * copy_row_bytes / 4, 22, word), .out = copy_rows * copy_row_bytes * 2, .run = copyRows },
        .{ .name = "glm_glue tf_glm_rmsnorm_4096_f32w", .image = 2, .input = try bytesOf(u16, a, norm_rows * 4096, 23, bf16), .out = norm_out + norm_rows * 64 * 4, .run = rmsnorm },
        .{ .name = "glm_quant tf_glm_nvfp4_amax (a weight writer)", .image = 3, .input = try bytesOf(u16, a, amax_e * 8192, 24, bf16), .out = amax_e * 4, .run = amax },
        .{ .name = "glm_dflash df_swiglu", .image = 4, .input = try bytesOf(u16, a, swiglu_rows * 2 * swiglu_f, 25, bf16), .out = swiglu_rows * swiglu_f * 2, .run = swiglu },
        .{ .name = "glm_mm glm_mm_reduce", .image = 5, .input = try bytesOf(f32, a, reduce_sk * reduce_total, 26, f32v), .out = reduce_total * 2, .run = mmReduce },
    };
    defer for (cases) |k| a.free(k.input);
    for (cases) |k| {
        var b = try Bufs.init(&c, k.input, k.out);
        defer b.deinit();
        try case(&c, k.name, images[k.image], mods[k.image], &b, k.run);
    }
    try qmmf(&c);
    var it = c.mains.valueIterator();
    while (it.next()) |m| m.unload();
}

/// The weight-first qmmf decode rows behind the writer (BF16, FP4, MXFP8): K slices and their reduce, a cluster.
fn qmmf(c: *Ctx) !void {
    const a = c.gpu.gpa;
    Qmmf.lane = try q.Lane.load(c.gpu.d, (try c.gpu.ctx.capability()) / 10);
    defer Qmmf.lane.unload();
    var part = try cuda.DeviceBuffer.alloc(c.gpu.d, 8 << 20);
    defer part.free();
    Qmmf.part = part.ptr;
    for ([_]struct { mode: q.Mode, n: usize, k: usize, sk: u32, clusters: bool, what: []const u8 }{
        .{ .mode = .bf16, .n = 2560, .k = 6144, .sk = 4, .clusters = false, .what = "qmmf BF16 decode row, 4 K slices and their reduce" },
        .{ .mode = .bf16, .n = 2560, .k = 6144, .sk = 4, .clusters = true, .what = "qmmf BF16 decode row, a cluster of 4 slices" },
        .{ .mode = .bf16, .n = 2560, .k = 64, .sk = 1, .clusters = false, .what = "qmmf BF16 decode row, one K group (fewer than the prologue's stages)" },
        .{ .mode = .fp4, .n = 2560, .k = 6144, .sk = 4, .clusters = false, .what = "qmmf FP4 decode row, 4 K slices and their reduce" },
        .{ .mode = .fp4, .n = 2560, .k = 6144, .sk = 4, .clusters = true, .what = "qmmf FP4 decode row, a cluster of 4 slices" },
        .{ .mode = .mxfp8, .n = 2560, .k = 6144, .sk = 4, .clusters = false, .what = "qmmf MXFP8 decode row, 4 K slices and their reduce" },
        .{ .mode = .mxfp8, .n = 2560, .k = 6144, .sk = 4, .clusters = true, .what = "qmmf MXFP8 decode row, a cluster of 4 slices" },
    }) |t| {
        const npad = (t.n + 127) / 128 * 128;
        var prng = std.Random.DefaultPrng.init(0x4218 + t.k + @backingInt(t.mode));
        const codes_host = try packed_(a, t.mode, t.n, t.k, npad, prng.random());
        defer a.free(codes_host);
        var codes = try cuda.DeviceBuffer.fromHost(c.gpu.d, codes_host);
        defer codes.free();
        // finite scales near 1: e4m3 0x30..0x3f (FP4, one a 16), e8m0 124..131 (MXFP8, one a 32)
        const scales_host = try a.alloc(u8, if (t.mode == .fp4) npad * t.k / 16 else if (t.mode == .mxfp8) npad * t.k / 32 else 16);
        defer a.free(scales_host);
        for (scales_host) |*v| v.* = if (t.mode == .fp4) 0x30 + prng.random().uintLessThan(u8, 16) else 124 + prng.random().uintLessThan(u8, 8);
        var scales = try cuda.DeviceBuffer.fromHost(c.gpu.d, scales_host);
        defer scales.free();
        const sp = if (t.mode == .bf16) 0 else scales.ptr;
        Qmmf.w = try q.pinned(.{ .mode = t.mode, .codes = codes.ptr, .scales = sp, .n = @intCast(t.n), .k = @intCast(t.k), .npad = @intCast(npad), .scale = 1 }, t.sk);
        Qmmf.clusters = t.clusters;
        const input = try bytesOf(u16, a, t.k, 31 + t.k, bf16);
        defer a.free(input);
        var b = try Bufs.init(c, input, t.n * 2);
        defer b.deinit();
        try case(c, t.what, "qmmf", Qmmf.lane.mod, &b, Qmmf.run);
    }
}

/// A packed weight's bytes: BF16 tiled from random values, FP4 and MXFP8 random finite codes (any layout is a weight).
fn packed_(a: std.mem.Allocator, mode: q.Mode, n: usize, k: usize, npad: usize, r: std.Random) ![]u8 {
    if (mode == .bf16) {
        const host_w = try a.alloc(u16, n * k);
        defer a.free(host_w);
        for (host_w) |*v| v.* = bf16(r);
        const tiled = try a.alloc(u16, npad * k);
        defer a.free(tiled);
        q.packBf16(tiled, host_w, n, k);
        return a.dupe(u8, std.mem.sliceAsBytes(tiled));
    }
    const out = try a.alloc(u8, if (mode == .fp4) npad * k / 2 else npad * k);
    r.bytes(out);
    if (mode == .mxfp8) for (out) |*v| if (v.* & 0x7f == 0x7f) {
        v.* ^= 1; // e4m3's NaN codes
    };
    return out;
}
