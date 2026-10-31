//! tf-idol-generate: Idol end to end on Metal, one picture to a 3DGS PLY: both encoders, the denoiser, the decoders.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const id = tf.idol;
const cfg = id.config;
const vae = id.vae;
const dn = id.dino;
const dc = id.decoder_config;
const gs = id.gaussian;

const usage =
    \\tf-idol-generate SNAPSHOT_DIR VAE_DIR DINO_DIR PICTURE OUT_DIR [--seed N] [--steps N] [--cfg X] [--gaussians N]
    \\                 [--noise FILE] [--features FILE]...
    \\SNAPSHOT_DIR is a speridlabs/idol snapshot, VAE_DIR FLUX.2-klein-4B's vae/ folder, DINO_DIR
    \\timm/vit_huge_plus_patch16_dinov3.lvd1689m. PICTURE is a safetensors file holding the 1,024 x 1,024 picture
    \\IdolPipeline.preprocess returns, in [0, 1]: `image` [1024, 1024, 3] (idol_case.py vae) or `pixels` [3, 1024, 1024]
    \\(idol_case.py dino). Defaults: seed 42, 50 steps, guidance 3, 262,144 Gaussians, as IdolPipeline's.
    \\The denoiser starts from torch.Generator("cpu").manual_seed(seed)'s randn [8192, 16] then [5], as idol_case.py
    \\case draws it (the SDK draws on its device's generator). The VAE posterior's noise and the octree's draws come from
    \\native seeded generators. --noise FILE gives the posterior's noise instead (`noise` [128, 128, 32], as idol_case.py
    \\vae records the SDK's) and, if it holds them, the starting `latent` [8192, 16] and `camera` [5]. --features FILE
    \\(once or twice) gives feature1 [1, 4101, 1280] and feature2 [1, 4101, 128] and skips both encoders (PICTURE: -).
    \\Writes OUT_DIR/generate.ply, camera.json, start.f32 (the starting latent and camera), sample.f32 and
    \\tf-idol-generate.json (each stage's seconds). Each model is freed once its stage is done.
;

const N = cfg.latent_rows * cfg.latent_channels;
const cam_n = cfg.camera_channels;
const f1_n = cfg.cond_rows * cfg.cond_channels;
const f2_n = cfg.cond_rows * cfg.cond2_channels;
const noise_n = vae.latent * vae.latent * id.vae_weights.latent_channels;
const side = vae.size;
const rescale: f64 = 3.0;

/// torch's CPU generator: mt19937 seeded as CPUGeneratorImpl seeds it.
const TorchCpu = struct {
    s: [624]u32,
    i: usize = 624,
    /// normal_distribution<double>'s cached second sample
    spare: ?f64 = null,

    fn init(seed: u64) TorchCpu {
        var g: TorchCpu = .{ .s = undefined };
        g.s[0] = @truncate(seed);
        for (1..624) |j| g.s[j] = 1812433253 *% (g.s[j - 1] ^ (g.s[j - 1] >> 30)) +% @as(u32, @intCast(j));
        return g;
    }

    fn next(g: *TorchCpu) u32 {
        if (g.i >= 624) {
            for (0..624) |k| {
                const y = (g.s[k] & 0x80000000) | (g.s[(k + 1) % 624] & 0x7fffffff);
                g.s[k] = g.s[(k + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            g.i = 0;
        }
        var y = g.s[g.i];
        g.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        return y ^ (y >> 18);
    }

    fn uniform(g: *TorchCpu) f32 {
        return @as(f32, @floatFromInt(g.next() & 0xffffff)) * 0x1p-24;
    }

    /// normal_fill_16: Box-Muller on 8 pairs in place; theta is rounded to float as torch's scalar_t is.
    fn fill16(d: *[16]f32) void {
        for (0..8) |j| {
            const r = @sqrt(-2 * logf(1 - d[j]));
            const theta: f32 = @floatCast(2.0 * std.math.pi * @as(f64, d[j + 8]));
            d[j] = r * @as(f32, @floatCast(@cos(@as(f64, theta))));
            d[j + 8] = r * @as(f32, @floatCast(@sin(@as(f64, theta))));
        }
    }

    /// torch.randn(out.len, generator=g) for float32: normal_fill from 16 values on, else normal_distribution<double>.
    fn randn(g: *TorchCpu, out: []f32) void {
        if (out.len >= 16) {
            for (out) |*v| v.* = g.uniform();
            var i: usize = 0;
            while (i + 16 <= out.len) : (i += 16) fill16(out[i..][0..16]);
            if (out.len % 16 != 0) {
                const tail = out[out.len - 16 ..][0..16];
                for (tail) |*v| v.* = g.uniform();
                fill16(tail);
            }
            return;
        }
        for (out) |*v| v.* = @floatCast(g.normal());
    }

    fn normal(g: *TorchCpu) f64 {
        if (g.spare) |s| {
            g.spare = null;
            return s;
        }
        const u_1 = g.uniform53();
        const u_2 = g.uniform53();
        const r = @sqrt(-2 * std.math.log1p(-u_2));
        const theta = 2 * std.math.pi * u_1;
        g.spare = r * @sin(theta);
        return r * @cos(theta);
    }

    fn uniform53(g: *TorchCpu) f64 {
        const hi: u64 = g.next();
        const lo: u64 = g.next();
        return @as(f64, @floatFromInt(((hi << 32) | lo) & ((1 << 53) - 1))) * 0x1p-53;
    }
};

extern "c" fn logf(x: f32) f32;

const Options = struct {
    seed: u64 = 42,
    steps: usize = 50,
    guidance: f32 = 3.0,
    gaussians: usize = 262144,
    noise: ?[]const u8 = null,
    features: [2]?[]const u8 = .{ null, null },
};

fn parse(args: []const [:0]const u8) !Options {
    var o: Options = .{};
    var i: usize = 6;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--seed")) {
            o.seed = try std.fmt.parseInt(u64, v, 10);
        } else if (std.mem.eql(u8, k, "--steps")) {
            o.steps = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--cfg")) {
            o.guidance = try std.fmt.parseFloat(f32, v);
        } else if (std.mem.eql(u8, k, "--gaussians")) {
            o.gaussians = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--noise")) {
            o.noise = v;
        } else if (std.mem.eql(u8, k, "--features")) {
            if (o.features[0] == null) o.features[0] = v else if (o.features[1] == null) o.features[1] = v else return error.Usage;
        } else return error.Usage;
    }
    if (o.steps == 0) return error.Usage;
    return o;
}

/// Tensor `name` of `f`, float32 with `n` values, or null when absent.
fn maybe(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8, n: usize) !?[]f32 {
    const t = f.get(name) orelse return null;
    if (t.dtype != .f32 or t.numel() != n) {
        std.debug.print("{s} is not {d} float32 values\n", .{ name, n });
        return error.BadTensor;
    }
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

/// The picture as [3, 1024, 1024] (DINOv3's input) and [1024, 1024, 3] (the VAE's).
fn picture(a: std.mem.Allocator, io: std.Io, path: []const u8) ![2][]f32 {
    var f = try tf.safetensors.File.open(a, io, path);
    defer f.close(io);
    const n = 3 * side * side;
    const chw = try a.alloc(f32, n);
    const hwc = try a.alloc(f32, n);
    if (try maybe(a, &f, "pixels", n)) |p| {
        @memcpy(chw, p);
        for (0..side * side) |i| for (0..3) |c| {
            hwc[i * 3 + c] = p[c * side * side + i];
        };
    } else if (try maybe(a, &f, "image", n)) |p| {
        @memcpy(hwc, p);
        for (0..side * side) |i| for (0..3) |c| {
            chw[c * side * side + i] = p[i * 3 + c];
        };
    } else {
        std.debug.print("{s} holds neither `pixels` [3, 1024, 1024] nor `image` [1024, 1024, 3]\n", .{path});
        return error.MissingTensor;
    }
    return .{ chw, hwc };
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

/// Seconds per stage, as printed and written.
const Times = struct {
    load_dino: f64 = 0,
    dino: f64 = 0,
    load_vae: f64 = 0,
    vae: f64 = 0,
    load_denoiser: f64 = 0,
    condition: f64 = 0,
    denoise: f64 = 0,
    load_decoders: f64 = 0,
    octree: f64 = 0,
    gaussian_decoder: f64 = 0,
    activation: f64 = 0,
    ply: f64 = 0,
    total: f64 = 0,
};

/// feature1 and feature2 from the encoders, or from --features files.
fn encode(a: std.mem.Allocator, io: std.Io, device: mtl.Device, args: []const [:0]const u8, o: Options, t: *Times, metal_peak: *u64) ![2][]f32 {
    if (o.features[0]) |first| {
        var f1: ?[]f32 = null;
        var f2: ?[]f32 = null;
        for ([_]?[]const u8{ first, o.features[1] }) |maybe_path| if (maybe_path) |path| {
            var f = try tf.safetensors.File.open(a, io, path);
            defer f.close(io);
            if (try maybe(a, &f, "feature1", f1_n)) |v| f1 = v;
            if (try maybe(a, &f, "feature2", f2_n)) |v| f2 = v;
        };
        if (f1 == null or f2 == null) {
            std.debug.print("--features needs feature1 [1, 4101, 1280] and feature2 [1, 4101, 128]\n", .{});
            return error.MissingTensor;
        }
        std.debug.print("features: given, both encoders skipped\n", .{});
        return .{ f1.?, f2.? };
    }
    const pics = try picture(a, io, args[4]);
    const f1 = try a.alloc(f32, f1_n);
    const f2 = try a.alloc(f32, f2_n);
    {
        var started = mtl.clock.seconds();
        const enc = try dn.Encoder.load(std.heap.page_allocator, io, device, args[3]);
        defer enc.deinit();
        t.load_dino = mtl.clock.seconds() - started;
        started = mtl.clock.seconds();
        const tm = try enc.encode(pics[0], f1);
        t.dino = mtl.clock.seconds() - started;
        metal_peak.* = @max(metal_peak.*, device.allocated());
        std.debug.print("DINOv3: loaded in {d:.2} s; feature1 in {d:.3} s ({d:.3} s on the GPU)\n", .{ t.load_dino, t.dino, tm.gpu });
    }
    const noise = try a.alloc(f32, noise_n);
    var given = false;
    if (o.noise) |path| {
        var f = try tf.safetensors.File.open(a, io, path);
        defer f.close(io);
        if (try maybe(a, &f, "noise", noise_n)) |v| {
            @memcpy(noise, v);
            given = true;
        }
    }
    if (!given) {
        var rng = std.Random.DefaultPrng.init(o.seed +% 0x9e3779b97f4a7c15);
        for (noise) |*v| v.* = rng.random().floatNorm(f32);
    }
    {
        var started = mtl.clock.seconds();
        const enc = try vae.Encoder.load(std.heap.page_allocator, io, device, args[2]);
        defer enc.deinit();
        t.load_vae = mtl.clock.seconds() - started;
        started = mtl.clock.seconds();
        const tm = try enc.encode(pics[1], noise, f2, .{});
        t.vae = mtl.clock.seconds() - started;
        metal_peak.* = @max(metal_peak.*, device.allocated());
        std.debug.print("VAE: loaded in {d:.2} s; feature2 in {d:.3} s ({d:.3} s on the GPU), posterior noise {s}\n", .{ t.load_vae, t.vae, tm.gpu, if (given) "given" else "native" });
    }
    return .{ f1, f2 };
}

/// The guided schedule from the starting noise `x` (latent then camera), in place.
fn denoise(a: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, f: [2][]f32, x: []f32, o: Options, t: *Times, metal_peak: *u64) !void {
    var started = mtl.clock.seconds();
    const model = try id.Model.load(std.heap.page_allocator, io, device, dir);
    defer model.deinit();
    t.load_denoiser = mtl.clock.seconds() - started;
    started = mtl.clock.seconds();
    var conds: [2]id.Condition = undefined;
    conds[0] = try model.condition(f[0], f[1]);
    defer conds[0].deinit();
    @memset(f[0], 0);
    @memset(f[1], 0);
    conds[1] = try model.condition(f[0], f[1]);
    defer conds[1].deinit();
    t.condition = mtl.clock.seconds() - started;
    const ts = try a.alloc(f64, o.steps + 1);
    id.host.schedule(ts, rescale);
    const vc = try a.alloc(f32, N);
    const vu = try a.alloc(f32, N);
    var cc: [cam_n]f32 = undefined;
    var cu: [cam_n]f32 = undefined;
    var gpu: f64 = 0;
    const w = o.guidance;
    started = mtl.clock.seconds();
    for (0..o.steps) |s| {
        const time: f32 = @floatCast(1000 * ts[s]);
        const cam: *const [cam_n]f32 = x[N..][0..cam_n];
        const p = try model.forward(&conds[0], x[0..N], cam, time, vc, &cc);
        const q = try model.forward(&conds[1], x[0..N], cam, time, vu, &cu);
        const by: f32 = @floatCast(ts[s] - ts[s + 1]);
        for (x[0..N], vc, vu) |*v, pc, pu| v.* -= ((1 + w) * pc - w * pu) * by;
        for (x[N..], cc, cu) |*v, pc, pu| v.* -= ((1 + w) * pc - w * pu) * by;
        gpu += p.gpu + q.gpu;
    }
    t.denoise = mtl.clock.seconds() - started;
    metal_peak.* = @max(metal_peak.*, device.allocated());
    std.debug.print("denoiser: loaded in {d:.2} s; condition {d:.3} s; {d} steps in {d:.2} s ({d:.2} s on the GPU)\n", .{ t.load_denoiser, t.condition, o.steps, t.denoise, gpu });
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    const o = (if (args.len >= 6) parse(args) else error.Usage) catch {
        std.debug.print("{s}\n", .{usage});
        return 2;
    };
    const out_dir = args[5];
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    const begun = mtl.clock.seconds();
    var t: Times = .{};
    var metal_peak: u64 = 0;

    // the starting noise first: it does not depend on the picture
    const x = try a.alloc(f32, N + cam_n);
    var gen = TorchCpu.init(o.seed);
    gen.randn(x[0..N]);
    gen.randn(x[N..]);
    var start_given = false;
    if (o.noise) |path| {
        var f = try tf.safetensors.File.open(a, io, path);
        defer f.close(io);
        const lat = try maybe(a, &f, "latent", N);
        const cam = try maybe(a, &f, "camera", cam_n);
        if (lat != null and cam != null) {
            @memcpy(x[0..N], lat.?);
            @memcpy(x[N..], cam.?);
            start_given = true;
        }
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/start.f32", .{out_dir}), .data = std.mem.sliceAsBytes(x) });

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    std.debug.print("{s}: seed {d}, {d} steps, guidance {d}, {d} Gaussians; starting noise {s}\n", .{ device.name(), o.seed, o.steps, o.guidance, o.gaussians, if (start_given) "given" else "torch CPU-seeded" });
    const features = try encode(a, io, device, args, o, &t, &metal_peak);
    try denoise(a, io, device, args[1], features, x, o, &t, &metal_peak);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/sample.f32", .{out_dir}), .data = std.mem.sliceAsBytes(x) });

    var started = mtl.clock.seconds();
    const d = try id.Decoder.load(std.heap.page_allocator, io, device, args[1], try dc.anchors(o.gaussians));
    t.load_decoders = mtl.clock.seconds() - started;
    var draws: id.octree.Draws = .{ .native = std.Random.DefaultPrng.init(o.seed) };
    var r = try id.decode.decode(std.heap.page_allocator, d, x[0..N], o.gaussians, &draws, null);
    defer r.deinit(std.heap.page_allocator);
    metal_peak = @max(metal_peak, device.allocated());
    d.deinit();
    t.octree = r.time.octree;
    t.gaussian_decoder = r.time.gaussian;
    t.activation = r.time.activation;
    std.debug.print("decoders: loaded in {d:.2} s; octree {d:.3} s, Gaussian decoder {d:.3} s, activation {d:.3} s\n", .{ t.load_decoders, t.octree, t.gaussian_decoder, t.activation });

    started = mtl.clock.seconds();
    const bytes = try gs.ply(a, r.gaussians);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/generate.ply", .{out_dir}), .data = bytes });
    const c = gs.camera(x[N..][0..cam_n].*);
    const cam_json = try std.fmt.allocPrint(a, "{{\"yaw\": {d}, \"pitch\": {d}, \"radius\": {d}, \"fov\": {d}}}\n", .{ c.yaw, c.pitch, c.radius, c.fov });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/camera.json", .{out_dir}), .data = cam_json });
    t.ply = mtl.clock.seconds() - started;
    t.total = mtl.clock.seconds() - begun;

    const load = t.load_dino + t.load_vae + t.load_denoiser + t.load_decoders;
    std.debug.print("PLY: {d} Gaussians, {d} bytes in {d:.3} s; camera {s}", .{ r.gaussians.count(), bytes.len, t.ply, cam_json });
    std.debug.print("total {d:.2} s: load {d:.2}, dino {d:.3}, vae {d:.3}, condition {d:.3}, denoise {d:.2}, octree {d:.3}, Gaussian decoder {d:.3} (+ activation {d:.3}), PLY {d:.3}; Metal peak {d:.2} GB\n", .{ t.total, load, t.dino, t.vae, t.condition, t.denoise, t.octree, t.gaussian_decoder, t.activation, t.ply, gb(metal_peak) });
    var out: std.ArrayList(u8) = .empty;
    try print(&out, a, "{{\"device\": \"{s}\", \"seed\": {d}, \"steps\": {d}, \"cfg\": {d}, \"gaussians\": {d}, \"start_noise\": \"{s}\", \"vae_noise\": \"{s}\", \"features_given\": {}", .{ device.name(), o.seed, o.steps, o.guidance, r.gaussians.count(), if (start_given) "given" else "torch_cpu", if (o.features[0] != null) "unused" else if (o.noise != null) "given" else "native", o.features[0] != null });
    try print(&out, a, ", \"load_s\": {d:.4}", .{load});
    inline for (@typeInfo(Times).@"struct".field_names) |name| try print(&out, a, ", \"{s}_s\": {d:.4}", .{ name, @field(t, name) });
    try print(&out, a, ", \"metal_peak_allocated_gb\": {d:.3}, \"camera\": {s}}}\n", .{ gb(metal_peak), std.mem.trimEnd(u8, cam_json, "\n") });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-generate.json", .{out_dir}), .data = out.items });
    return 0;
}
