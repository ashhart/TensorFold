//! TFLINGM1/TFLINGA1 dump parsing and Metal-vs-python mixer check.
const std = @import("std");
const mtl = @import("metal");
const mla = @import("mla");
const c = @import("ling_mla_gpu_common.zig");

const dump_dir = "/tmp/ling-mla-dumps";
pub const dump_mixer_path = dump_dir ++ "/mla_tiny_mixer.bin";
pub const dump_a1_t4_path = dump_dir ++ "/mla_tiny_t4_pos0.bin";
pub const dump_a1_t2_path = dump_dir ++ "/mla_tiny_t2_pos30000.bin";
const dump_py = "zig/src/families/ling/oracle/mla_dump.py";

fn takeU32(b: []const u8, off: *usize) u32 {
    const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
    off.* += 4;
    return v;
}

fn takeF32s(alloc: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f32 {
    const x = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 4]);
    off.* += n * 4;
    return x;
}

fn takeF64s(alloc: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f64 {
    const x = try alloc.alloc(f64, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 8]);
    off.* += n * 8;
    return x;
}

const DumpCase = struct {
    T: u32,
    pos0: u32,
    q: []f32,
    kv: []f32,
    g: []f32,
    y_pre: []f64,
    y64: []f64,
    ybf: []f64,
};

const Dump = struct {
    hidden: u32,
    layer: u32,
    heads: u32,
    w: []f32,
    kvb: []f32,
    Wo: []f32,
    cases: []DumpCase,

    fn load(alloc: std.mem.Allocator, bytes: []const u8) !Dump {
        try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGM1"));
        var off: usize = 8;
        const hidden = takeU32(bytes, &off);
        const layer = takeU32(bytes, &off);
        const heads = takeU32(bytes, &off);
        const n_cases = takeU32(bytes, &off);
        const w = try takeF32s(alloc, bytes, &off, mla.lat);
        errdefer alloc.free(w);
        const kvb = try takeF32s(alloc, bytes, &off, heads * (mla.nope + mla.vd) * mla.lat);
        errdefer alloc.free(kvb);
        const Wo = try takeF32s(alloc, bytes, &off, hidden * heads * mla.vd);
        errdefer alloc.free(Wo);
        const cases = try alloc.alloc(DumpCase, n_cases);
        errdefer alloc.free(cases);
        for (cases) |*case| {
            const T = takeU32(bytes, &off);
            const pos0 = takeU32(bytes, &off);
            const q = try takeF32s(alloc, bytes, &off, T * heads * mla.qd);
            const kv = try takeF32s(alloc, bytes, &off, T * mla.key);
            const g = try takeF32s(alloc, bytes, &off, T * heads);
            const y_pre = try takeF64s(alloc, bytes, &off, T * heads * mla.vd);
            const y64 = try takeF64s(alloc, bytes, &off, T * hidden);
            const ybf = try takeF64s(alloc, bytes, &off, T * hidden);
            case.* = .{ .T = T, .pos0 = pos0, .q = q, .kv = kv, .g = g, .y_pre = y_pre, .y64 = y64, .ybf = ybf };
        }
        try std.testing.expectEqual(bytes.len, off);
        return .{ .hidden = hidden, .layer = layer, .heads = heads, .w = w, .kvb = kvb, .Wo = Wo, .cases = cases };
    }

    fn deinit(d: Dump, alloc: std.mem.Allocator) void {
        for (d.cases) |case| {
            alloc.free(case.q);
            alloc.free(case.kv);
            alloc.free(case.g);
            alloc.free(case.y_pre);
            alloc.free(case.y64);
            alloc.free(case.ybf);
        }
        alloc.free(d.cases);
        alloc.free(d.w);
        alloc.free(d.kvb);
        alloc.free(d.Wo);
    }
};

fn parseA1(alloc: std.mem.Allocator, bytes: []const u8) !struct { T: u32, hidden: u32, layer: u32, pos0: u32, y64: []f64, ybf: []f64 } {
    try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGA1"));
    var off: usize = 8;
    const T = takeU32(bytes, &off);
    const hidden = takeU32(bytes, &off);
    const layer = takeU32(bytes, &off);
    const pos0 = takeU32(bytes, &off);
    const n = T * hidden;
    const y64 = try takeF64s(alloc, bytes, &off, n);
    errdefer alloc.free(y64);
    const ybf = try takeF64s(alloc, bytes, &off, n);
    errdefer alloc.free(ybf);
    try std.testing.expectEqual(bytes.len, off);
    return .{ .T = T, .hidden = hidden, .layer = layer, .pos0 = pos0, .y64 = y64, .ybf = ybf };
}

fn dupeRound(alloc: std.mem.Allocator, src: []const f32) ![]f32 {
    const d = try alloc.dupe(f32, src);
    c.roundSlice(d);
    return d;
}

fn fileOk(path: []const u8) bool {
    std.Io.Dir.cwd().access(std.testing.io, path, .{}) catch return false;
    return true;
}

fn hasTorchAndSafetensors(py: []const u8) bool {
    const a = std.testing.allocator;
    const r = std.process.run(a, std.testing.io, .{
        .argv = &.{ py, "-c", "import torch, safetensors.torch" },
        .stdout_limit = .limited(256),
        .stderr_limit = .limited(4096),
    }) catch return false;
    defer a.free(r.stdout);
    defer a.free(r.stderr);
    return r.term.success();
}

fn findPython() ?[]const u8 {
    if (std.c.getenv("TF_LING_PYTHON")) |p| {
        const s = std.mem.span(p);
        if (hasTorchAndSafetensors(s)) return s;
    }
    if (hasTorchAndSafetensors("python3")) return "python3";
    return null;
}

fn ensureDumps() !void {
    if (fileOk(dump_mixer_path) and fileOk(dump_a1_t4_path) and fileOk(dump_a1_t2_path)) return;
    if (!fileOk(dump_py)) {
        std.debug.print("skip: {s} not in cwd; TFLINGA1 dumps need ling_forward.py at test time\n", .{dump_py});
        return error.SkipZigTest;
    }
    const py = findPython() orelse {
        std.debug.print("skip: no Python with both torch and safetensors (set TF_LING_PYTHON or put such a python3 on PATH); TFLINGA1 dumps come from ling_forward.py at test time\n", .{});
        return error.SkipZigTest;
    };
    const a = std.testing.allocator;
    const r = try std.process.run(a, std.testing.io, .{
        .argv = &.{ py, dump_py, dump_dir },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(65536),
    });
    defer a.free(r.stdout);
    defer a.free(r.stderr);
    if (!r.term.success()) {
        std.debug.print("mla_dump.py failed: {s}{s}\n", .{ r.stdout, r.stderr });
        return error.DumpGen;
    }
    if (!fileOk(dump_mixer_path) or !fileOk(dump_a1_t4_path) or !fileOk(dump_a1_t2_path)) return error.DumpGen;
}

fn readDump(path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, std.testing.allocator, .limited(32 << 20));
}

pub fn checkDumpFile(a1_path: []const u8, case_i: usize, tag: []const u8, gpu: bool) !void {
    try ensureDumps();
    const a = std.testing.allocator;
    const mixer_bytes = try readDump(dump_mixer_path);
    defer a.free(mixer_bytes);
    const a1_bytes = try readDump(a1_path);
    defer a.free(a1_bytes);
    try checkDump(mixer_bytes, a1_bytes, case_i, tag, gpu);
}

fn checkDump(mixer_bytes: []const u8, a1_bytes: []const u8, case_i: usize, tag: []const u8, gpu: bool) !void {
    const a = std.testing.allocator;
    const dump = try Dump.load(a, mixer_bytes);
    defer dump.deinit(a);
    const a1 = try parseA1(a, a1_bytes);
    defer a.free(a1.y64);
    defer a.free(a1.ybf);
    const case = dump.cases[case_i];
    try std.testing.expectEqual(a1.T, case.T);
    try std.testing.expectEqual(a1.hidden, dump.hidden);
    try std.testing.expectEqual(a1.layer, dump.layer);
    try std.testing.expectEqual(a1.pos0, case.pos0);
    try std.testing.expect(std.mem.eql(f64, a1.y64, case.y64));

    const s = mla.Shape{ .heads = dump.heads, .eps = mla.eps_tiny, .theta = mla.theta_tiny };
    const width = s.outWidth();
    const n_mix = case.T * width;
    const mix64 = try a.alloc(f64, n_mix);
    defer a.free(mix64);
    try mla.mixF64(a, mix64, case.q, case.kv, dump.w, dump.kvb, case.g, s, case.T, case.pos0);
    const d_pre = c.dist(mix64, case.y_pre);
    std.debug.print("{s} mixF64-to-python mixer rel L2 {e} max abs {e}\n", .{ tag, d_pre.rel_l2, d_pre.max_abs });
    const layer_mix = try a.alloc(f64, case.T * dump.hidden);
    defer a.free(layer_mix);
    c.oProj(layer_mix, mix64, dump.Wo, case.T, dump.hidden, width);
    const d_layer = c.dist(layer_mix, a1.y64);
    std.debug.print("{s} mixF64+o_proj-to-TFLINGA1 rel L2 {e} max abs {e}\n", .{ tag, d_layer.rel_l2, d_layer.max_abs });
    if (!gpu) return;

    const mix_bf = try a.alloc(f64, n_mix);
    defer a.free(mix_bf);
    const q_m = try dupeRound(a, case.q);
    defer a.free(q_m);
    const kv_m = try dupeRound(a, case.kv);
    defer a.free(kv_m);
    const w_m = try dupeRound(a, dump.w);
    defer a.free(w_m);
    const kvb_m = try dupeRound(a, dump.kvb);
    defer a.free(kvb_m);
    const g_m = try dupeRound(a, case.g);
    defer a.free(g_m);
    try mla.mixBf16(a, mix_bf, q_m, kv_m, w_m, kvb_m, g_m, s, case.T, case.pos0);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var gpu_dev = try c.Gpu.init();
    defer gpu_dev.deinit();
    var l = try c.Launch.open(gpu_dev, s, case.T, q_m, kv_m, w_m, kvb_m, g_m);
    defer l.deinit();
    try l.dispatchMixer(gpu_dev, 0, case.T);
    const metal_mix = try a.alloc(f64, n_mix);
    defer a.free(metal_mix);
    c.bitsToF64(l.y(), metal_mix);
    const d_metal_pre = c.dist(metal_mix, case.y_pre);
    std.debug.print("{s} Metal-to-python mixer rel L2 {e} max abs {e}\n", .{ tag, d_metal_pre.rel_l2, d_metal_pre.max_abs });

    const layer_bf = try a.alloc(f64, case.T * dump.hidden);
    defer a.free(layer_bf);
    const layer_metal = try a.alloc(f64, case.T * dump.hidden);
    defer a.free(layer_metal);
    c.oProj(layer_bf, mix_bf, dump.Wo, case.T, dump.hidden, width);
    c.oProj(layer_metal, metal_mix, dump.Wo, case.T, dump.hidden, width);
    var tbuf: [96]u8 = undefined;
    const layer_tag = try std.fmt.bufPrint(&tbuf, "{s} TFLINGA1", .{tag});
    try c.gateNamedF64(layer_tag, layer_metal, a1.y64, layer_bf);
}
