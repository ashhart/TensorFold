//! Qwen's int8 prompt projections through head norms into FP8 RoPE: PDL on must equal the all-plain launch bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");

const Gpu = check.Gpu;
const Rig = quant_ref.RigT;
const qlinear = cuda.qlinear;
const bf = quant_ref.helpers.bfOf;
const bf16 = quant_ref.helpers.bf16Of;

const rows = 170; // the reporter's 170-SM RTX 5090: enough live heads to expose predecessor races
const hd = 256;
const qh = 24;
const kvh = 4;
const kdim = 256;
const qwidth = qh * 2 * hd; // fused Q and gate rows
const kwidth = kvh * hd;
const repeats = 16;

const Matrix = struct { codes: []u8, scales: []u16, biases: []u16, n: usize, k: usize };
const Geo = extern struct { hid: c_int, nk: c_int, nv: c_int, qh: c_int, kvh: c_int, nl: c_int };
const Ancestor = extern struct { bits: u64, depth: u32, chain: u32 };
const Seg = extern struct {
    rec: u64 = 0,
    conv: u64 = 0,
    kv: u64,
    tap: u64 = 0,
    log: u64 = 0,
    pt: u64,
    len: c_int,
    end: c_int,
    row0: c_int,
    head0: c_int,
    cap: c_int,
    rows: c_int,
    flags: c_int = 0,
    head1: c_int = 0,
    ps: c_int,
    ptok: c_int,
};

const Prompt = struct {
    x: u64,
    q: qlinear.Affine4,
    out: u64,
    ws: u64,
    flag: u64,
};

fn matrix(r: *Rig, n: usize, k: usize) !Matrix {
    const codes = try r.a.alloc(u8, n * k);
    for (codes) |*code| code.* = r.rng.uintLessThan(u8, 16);
    const scales = try r.a.alloc(u16, n * k / 64);
    const biases = try r.a.alloc(u16, n * k / 64);
    for (scales, biases) |*scale, *bias| {
        const v = (r.rng.float(f32) * 0.9 + 0.1) * 0.0025;
        scale.* = bf16(v);
        bias.* = bf16(-@as(f32, @floatCast(bf(scale.*))) * 8 * (0.8 + 0.4 * r.rng.float(f32)));
    }
    return .{ .codes = codes, .scales = scales, .biases = biases, .n = n, .k = k };
}

/// MLX words: eight codes a word, low code first.
fn words(r: *Rig, m: Matrix) ![]u32 {
    const out = try r.a.alloc(u32, m.n * m.k / 8);
    @memset(out, 0);
    for (m.codes, 0..) |code, i| out[i / 8] |= @as(u32, code) << @intCast(4 * (i % 8));
    return out;
}

fn prompt(r: *Rig, a4: *const qlinear.Affine4Kernels, x: u64, n: usize) !Prompt {
    const m = try matrix(r, n, kdim);
    const layout = qlinear.Affine4.layout(n, kdim);
    const q: qlinear.Affine4 = .{
        .w = try r.zeros(layout.words),
        .s = try r.zeros(layout.scales),
        .b = try r.zeros(layout.scales),
        .n = n,
        .k = kdim,
        .npad = layout.npad,
    };
    try a4.pack(
        r.s,
        try r.dev(std.mem.sliceAsBytes(try words(r, m))),
        try r.dev(std.mem.sliceAsBytes(m.scales)),
        try r.dev(std.mem.sliceAsBytes(m.biases)),
        q,
    );
    return .{
        .x = x,
        .q = q,
        .out = try r.zeros(rows * n * 2),
        .ws = try r.zeros(cuda.w4a8.Kernels.scratch(rows, kdim)),
        .flag = try r.zeros(@sizeOf(u32)),
    };
}

fn norm(f: cuda.Function, s: cuda.Stream, x: u64, w: u64, out: u64, heads: usize, stride: usize, hstride: usize, pdl: bool) !void {
    var args: cuda.Args = .{};
    inline for (.{ x, w, out, @as(c_int, @intCast(heads)), @as(c_int, @intCast(stride)), @as(c_int, @intCast(hstride)), @as(f32, 1e-6) }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows * heads) }, .block = .{ .x = 32 }, .pdl = pdl }, s, &args);
}

fn fp8Rope(f: cuda.Function, s: cuda.Stream, qn: u64, kn: u64, vr: u64, segs: u64, rowseg: u64, rowanc: u64, pdl: bool) !void {
    const geo: Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = qh, .kvh = kvh, .nl = 0 };
    var args: cuda.Args = .{};
    inline for (.{ geo, qn, kn, vr, segs, rowseg, rowanc, @as(c_int, 0), @as(u64, 0), @as(u64, 0), @as(u64, 0) }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = rows, .y = qh + kvh }, .block = .{ .x = hd }, .pdl = pdl }, s, &args);
}

/// After int8 prompt rows, the fixed arm attributes only the two norms; Engine.runPdl keeps rope8 plain.
fn arm(r: *Rig, kk: cuda.w4a8.Kernels, prompts: *const [3]Prompt, qnorm: cuda.Function, rope: cuda.Function, qn: u64, kn: u64, kv: u64, segs: u64, rowseg: u64, rowanc: u64, wq: u64, wk: u64, pdl: bool) !void {
    for ([_]usize{ qn, kn, kv }, [_]usize{ rows * qh * hd * 2, rows * kvh * hd * 2, 2 * kvh * rows * (hd + 4) }) |ptr, len| {
        try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = ptr, .len = len }).fill8(0xff, r.s.handle);
    }
    for (prompts) |p| {
        var exceptional: u32 = undefined;
        try kk.prompt(r.s, p.x, p.q, p.out, rows, p.ws, p.flag, &exceptional);
        try check.expect(exceptional == 0, "Qwen PDL stress: finite inputs must take the int8 path", .{});
    }
    try norm(qnorm, r.s, prompts[0].out, wq, qn, qh, qwidth, 2 * hd, pdl);
    try norm(qnorm, r.s, prompts[1].out, wk, kn, kvh, kwidth, hd, pdl);
    // cuda_engine.zig's runPdl drops this attribute: qwen35_kv8.cu's rope8 has no qwen_pdl::enter().
    try fp8Rope(rope, r.s, qn, kn, prompts[2].out, segs, rowseg, rowanc, false);
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x51a9);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }

    const images = [_][]const u8{
        cuda.kernels.qmm_group,
        cuda.kernels.qmm_prefill,
        cuda.kernels.lane_gemv,
        cuda.kernels.affine4_pack,
        cuda.kernels.w4a8_prefill,
        cuda.kernels.qwen35,
        cuda.kernels.qwen35_kv8,
    };
    var mods: [images.len]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*mod| mod.unload();
    for (images, &mods) |image, *mod| {
        mod.* = try cuda.Module.load(gpu.d, image);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, false);
    defer a4.deinit();
    const kk = try cuda.w4a8.Kernels.resolve(mods[4]);
    const qnorm = try mods[5].function("tf_q35_head_norm");
    const rope = try mods[6].function("tf_q35_rope8");

    const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(rows * kdim, 2)));
    const prompts = [3]Prompt{
        try prompt(&r, &a4, x, qwidth),
        try prompt(&r, &a4, x, kwidth),
        try prompt(&r, &a4, x, kwidth),
    };
    const wq = try r.dev(std.mem.sliceAsBytes(try r.bfs(hd, 0.1)));
    const wk = try r.dev(std.mem.sliceAsBytes(try r.bfs(hd, 0.1)));
    const qn = try r.zeros(rows * qh * hd * 2);
    const kn = try r.zeros(rows * kvh * hd * 2);
    const kv_bytes = 2 * kvh * rows * (hd + 4);
    const kv = try r.zeros(kv_bytes);
    const pt = try r.dev(std.mem.sliceAsBytes(&[_]i32{0}));
    const seg: Seg = .{ .kv = kv, .pt = pt, .len = 0, .end = 0, .row0 = 0, .head0 = 0, .cap = rows, .rows = rows, .ps = kv_bytes, .ptok = rows };
    const rowseg_host = try r.a.alloc(i32, rows);
    @memset(rowseg_host, 0);
    const rowanc_host = try r.a.alloc(Ancestor, rows);
    for (rowanc_host, 0..) |*anc, row| anc.* = .{ .bits = 0, .depth = @intCast(row), .chain = 1 };
    const segs = try r.dev(std.mem.sliceAsBytes(&[_]Seg{seg}));
    const rowseg = try r.dev(std.mem.sliceAsBytes(rowseg_host));
    const rowanc = try r.dev(std.mem.sliceAsBytes(rowanc_host));

    var baseline_q: ?[]const u8 = null;
    var baseline_kv: ?[]const u8 = null;
    for (0..repeats) |rep| {
        for ([_]bool{ false, true }) |pdl| {
            try arm(&r, kk, &prompts, qnorm, rope, qn, kn, kv, segs, rowseg, rowanc, wq, wk, pdl);
            const got_q = try r.back(qn, rows * qh * hd * 2);
            const got_kv = try r.back(kv, kv_bytes);
            if (!pdl) {
                if (rep == 0) {
                    baseline_q = try r.a.dupe(u8, got_q);
                    baseline_kv = try r.a.dupe(u8, got_kv);
                }
                try check.sameBytes("Qwen PDL stress: all-plain Q repeats byte for byte", got_q, baseline_q.?);
                try check.sameBytes("Qwen PDL stress: all-plain FP8 K/V repeats byte for byte", got_kv, baseline_kv.?);
            } else {
                try check.sameBytes("Qwen PDL stress: PDL-on Q == all-plain Q", got_q, baseline_q.?);
                try check.sameBytes("Qwen PDL stress: PDL-on FP8 K/V == all-plain FP8 K/V", got_kv, baseline_kv.?);
            }
        }
    }
    std.debug.print("PASS Qwen int8 prompt -> head norms -> FP8 RoPE: 170 rows, 24 Q/4 KV heads, {d} PDL-on/off pairs byte for byte\n", .{repeats});
}
