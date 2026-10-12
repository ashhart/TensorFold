//! MiMo's EXL3 launches against the Python engine's bytes (oracle/mimo_exl3_fixtures.py), same weights and inputs.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Gpu = check.Gpu;
const exl3 = cuda.exl3;
const fmt = core.exl3_format;

const Mat = struct {
    trellis: []i16, // [tk, tn, 8 * k2]
    suh: []f16,
    svh: []f16,
    tk: usize,
    tn: usize,
    k2: u32,

    fn deinit(m: Mat, a: std.mem.Allocator) void {
        a.free(m.trellis);
        a.free(m.suh);
        a.free(m.svh);
    }
};

fn load(a: std.mem.Allocator, ck: *core.Checkpoint, prefix: []const u8) !Mat {
    var nb: [256]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix}));
    const tk, const tn, const w = .{ t.shape[0], t.shape[1], t.shape[2] };
    const tr = try a.alloc(i16, t.numel());
    @memcpy(std.mem.sliceAsBytes(tr), t.bytes);
    return .{ .trellis = tr, .suh = try scales(a, ck, prefix, "suh", tk * 16), .svh = try scales(a, ck, prefix, "svh", tn * 16), .tk = tk, .tn = tn, .k2 = @intCast(w / 8) };
}

fn scales(a: std.mem.Allocator, ck: *core.Checkpoint, prefix: []const u8, part: []const u8, size: usize) ![]f16 {
    var nb: [256]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, part }));
    const out = try a.alloc(f16, size);
    if (t.dtype == .f16 and t.numel() == size) {
        @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    } else if (t.dtype == .i16 and t.numel() * 16 == size) {
        const w = try a.alloc(i16, t.numel());
        defer a.free(w);
        @memcpy(std.mem.sliceAsBytes(w), t.bytes);
        fmt.unpackSigns(w, out);
    } else return error.BadScales;
    return out;
}

/// weights.py cut(): outputs [lo, hi) (axis n: trellis columns, svh) or inputs [lo, hi) (axis k: trellis rows, suh).
fn cut(a: std.mem.Allocator, m: Mat, lo: usize, hi: usize, axis_n: bool) !Mat {
    const w = 8 * @as(usize, m.k2);
    if (axis_n) {
        const tn = (hi - lo) / 16;
        const tr = try a.alloc(i16, m.tk * tn * w);
        for (0..m.tk) |kt| @memcpy(tr[kt * tn * w ..][0 .. tn * w], m.trellis[(kt * m.tn + lo / 16) * w ..][0 .. tn * w]);
        return .{ .trellis = tr, .suh = try a.dupe(f16, m.suh), .svh = try a.dupe(f16, m.svh[lo..hi]), .tk = m.tk, .tn = tn, .k2 = m.k2 };
    }
    const tk = (hi - lo) / 16;
    return .{ .trellis = try a.dupe(i16, m.trellis[(lo / 16) * m.tn * w ..][0 .. tk * m.tn * w]), .suh = try a.dupe(f16, m.suh[lo..hi]), .svh = try a.dupe(f16, m.svh), .tk = tk, .tn = m.tn, .k2 = m.k2 };
}

/// linear.strips: [K/16, N/16, W] int32 words -> [N/128, K/16, 8, W].
fn strips(a: std.mem.Allocator, m: Mat) ![]u32 {
    const w32 = 4 * @as(usize, m.k2);
    const words: []align(2) const u32 = std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(m.trellis));
    const out = try a.alloc(u32, words.len);
    for (0..m.tk) |kt| for (0..m.tn / 8) |nb8| for (0..8) |j| {
        @memcpy(out[((nb8 * m.tk + kt) * 8 + j) * w32 ..][0..w32], words[((kt * m.tn) + nb8 * 8 + j) * w32 ..][0..w32]);
    };
    return out;
}

fn readFile(gpu: Gpu, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 31));
}

fn dtypeOf(s: []const u8) !exl3.DType {
    if (std.mem.eql(u8, s, "f16")) return .f16;
    if (std.mem.eql(u8, s, "bf16")) return .bf16;
    if (std.mem.eql(u8, s, "f32")) return .f32;
    return error.BadDType;
}

pub fn run(gpu: Gpu, model_dir: []const u8, fixtures: []const u8) !void {
    const a = gpu.gpa;
    var ck = try core.Checkpoint.openModel(a, gpu.io, model_dir);
    defer ck.close();
    var kern = try exl3.Kernels.load(gpu.d);
    defer kern.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();

    var dir = try std.Io.Dir.cwd().openDir(gpu.io, fixtures, .{ .iterate = true });
    defer dir.close(gpu.io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(gpu.io)) |e| if (e.kind == .directory) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);

    var passed: usize = 0;
    for (names.items) |name| {
        const case_dir = try std.fs.path.join(a, &.{ fixtures, name });
        defer a.free(case_dir);
        if (std.mem.startsWith(u8, name, "experts-")) {
            try routedCase(gpu, &ck, &kern, s, case_dir, name);
        } else {
            try linearCase(gpu, &ck, &kern, s, case_dir, name);
        }
        passed += 1;
    }
    try check.expect(passed > 0, "no fixtures under {s}", .{fixtures});
    check.pass("mimo-exl3-fixtures: {d} cases byte-equal to the Python engine", .{passed});
}

fn linearCase(gpu: Gpu, ck: *core.Checkpoint, kern: *const exl3.Kernels, s: cuda.Stream, dir: []const u8, name: []const u8) !void {
    const a = gpu.gpa;
    const meta_text = try readFile(gpu, dir, "meta.json");
    defer a.free(meta_text);
    const meta = try std.json.parseFromSlice(struct { prefix: []const u8, axis: []const u8, lo: usize, hi: usize, m: usize, x: []const u8, y: []const u8, k: usize, n: usize, k2: u32, split: [2]usize }, a, meta_text, .{});
    defer meta.deinit();
    const v = meta.value;
    const full = try load(a, ck, v.prefix);
    defer full.deinit(a);
    const mat = try cut(a, full, v.lo, v.hi, std.mem.eql(u8, v.axis, "n"));
    defer mat.deinit(a);
    const k, const n = .{ mat.tk * 16, mat.tn * 16 };
    try check.expect(k == v.k and n == v.n and mat.k2 == v.k2, "{s}: shape {d}x{d} K2 {d}, Python {d}x{d} K2 {d}", .{ name, k, n, mat.k2, v.k, v.n, v.k2 });
    const split = exl3.plan(k, n);
    try check.expect(split[0] == v.split[0] and split[1] == v.split[1], "{s}: plan {any}, Python {any}", .{ name, split, v.split });
    const st = try strips(a, mat);
    defer a.free(st);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(st));
    defer dw.free();
    var suh = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mat.suh));
    defer suh.free();
    var svh = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(mat.svh));
    defer svh.free();
    var counters = try cuda.DeviceBuffer.alloc(gpu.d, 8 * (n / 128) * 4);
    defer counters.free();
    try counters.fill32(0, null);
    const lin: exl3.Linear = .{ .words = dw.ptr, .suh = suh.ptr, .svh = svh.ptr, .counters = counters.ptr, .k = k, .n = n, .k2 = mat.k2, .split = split };
    const x = try readFile(gpu, dir, "x.bin");
    defer a.free(x);
    const want = try readFile(gpu, dir, "y.bin");
    defer a.free(want);
    const xd, const yd = .{ try dtypeOf(v.x), try dtypeOf(v.y) };
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
    defer dx.free();
    var xh = try cuda.DeviceBuffer.alloc(gpu.d, v.m * k * 2);
    defer xh.free();
    var y = try cuda.DeviceBuffer.alloc(gpu.d, want.len);
    defer y.free();
    var z = try cuda.DeviceBuffer.alloc(gpu.d, split[0] * v.m * n * 4);
    defer z.free();
    try kern.linear(s, lin, dx.ptr, xd, xh.ptr, y.ptr, yd, if (split[0] > 1) z.ptr else 0, v.m);
    try s.synchronize();
    const got = try check.download(gpu, y);
    defer a.free(got);
    try check.sameBytes(name, got, want);
    check.pass("{s}: {d}x{d} K2 {d} (SK {d}, WK {d}) byte-equal to the Python engine", .{ name, k, n, mat.k2, split[0], split[1] });
}

fn routedCase(gpu: Gpu, ck: *core.Checkpoint, kern: *const exl3.Kernels, s: cuda.Stream, dir: []const u8, name: []const u8) !void {
    const a = gpu.gpa;
    const meta_text = try readFile(gpu, dir, "meta.json");
    defer a.free(meta_text);
    const meta = try std.json.parseFromSlice(struct { layer: usize, width: usize, experts: usize, slots: usize, rows: usize, cfg_gu: [4]u32, cfg_d: [4]u32, k2_gu: [2]u32, k2_d: [2]u32 }, a, meta_text, .{});
    defer meta.deinit();
    const v = meta.value;
    const E, const ni = .{ v.experts, v.width };
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(a);
    }
    const ptrs = try a.alloc(u64, 3 * E);
    defer a.free(ptrs);
    const k2s = try a.alloc(i32, 3 * E);
    defer a.free(k2s);
    var d: usize = 0;
    var suh_g: std.ArrayList(f16) = .empty;
    var suh_u: std.ArrayList(f16) = .empty;
    var svh_g: std.ArrayList(f16) = .empty;
    var svh_u: std.ArrayList(f16) = .empty;
    var suh_d: std.ArrayList(f16) = .empty;
    var svh_d: std.ArrayList(f16) = .empty;
    defer inline for (.{ &suh_g, &suh_u, &svh_g, &svh_u, &suh_d, &svh_d }) |l| l.deinit(a);
    var lo_gu: u32 = 99;
    var hi_gu: u32 = 0;
    var lo_d: u32 = 99;
    var hi_d: u32 = 0;
    var nb: [256]u8 = undefined;
    for (0..E) |e| {
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, j| {
            const full = try load(a, ck, try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.{d}.{s}", .{ v.layer, e, proj }));
            defer full.deinit(a);
            const m = try cut(a, full, 0, ni, j != 2);
            defer m.deinit(a);
            d = if (j == 2) m.tn * 16 else m.tk * 16;
            try bufs.append(a, try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(m.trellis)));
            ptrs[j * E + e] = bufs.items[bufs.items.len - 1].ptr;
            k2s[j * E + e] = @intCast(m.k2);
            if (j == 2) {
                lo_d = @min(lo_d, m.k2);
                hi_d = @max(hi_d, m.k2);
                try suh_d.appendSlice(a, m.suh);
                try svh_d.appendSlice(a, m.svh);
            } else {
                lo_gu = @min(lo_gu, m.k2);
                hi_gu = @max(hi_gu, m.k2);
                try (if (j == 0) &suh_g else &suh_u).appendSlice(a, m.suh);
                try (if (j == 0) &svh_g else &svh_u).appendSlice(a, m.svh);
            }
        }
    }
    try check.expect(lo_gu == v.k2_gu[0] and hi_gu == v.k2_gu[1] and lo_d == v.k2_d[0] and hi_d == v.k2_d[1], "{s}: K2 ranges differ from Python's", .{name});
    const cfg_gu = try exl3.defaultConfig(d, ni, true);
    const cfg_d = try exl3.defaultConfig(ni, d, false);
    try check.expect(std.mem.eql(u32, &cfg_gu, &v.cfg_gu) and std.mem.eql(u32, &cfg_d, &v.cfg_d), "{s}: tile settings differ from Python's", .{name});

    const up = struct {
        fn f(g: Gpu, list: *std.ArrayList(cuda.DeviceBuffer), bytes: []const u8) !u64 {
            try list.append(g.gpa, try cuda.DeviceBuffer.fromHost(g.d, bytes));
            return list.items[list.items.len - 1].ptr;
        }
    }.f;
    const ex: exl3.Experts = .{
        .gate_ptr = try up(gpu, &bufs, std.mem.sliceAsBytes(ptrs[0..E])),
        .up_ptr = try up(gpu, &bufs, std.mem.sliceAsBytes(ptrs[E .. 2 * E])),
        .down_ptr = try up(gpu, &bufs, std.mem.sliceAsBytes(ptrs[2 * E ..])),
        .gate_k2 = try up(gpu, &bufs, std.mem.sliceAsBytes(k2s[0..E])),
        .up_k2 = try up(gpu, &bufs, std.mem.sliceAsBytes(k2s[E .. 2 * E])),
        .down_k2 = try up(gpu, &bufs, std.mem.sliceAsBytes(k2s[2 * E ..])),
        .suh_g = try up(gpu, &bufs, std.mem.sliceAsBytes(suh_g.items)),
        .suh_u = try up(gpu, &bufs, std.mem.sliceAsBytes(suh_u.items)),
        .svh_g = try up(gpu, &bufs, std.mem.sliceAsBytes(svh_g.items)),
        .svh_u = try up(gpu, &bufs, std.mem.sliceAsBytes(svh_u.items)),
        .suh_d = try up(gpu, &bufs, std.mem.sliceAsBytes(suh_d.items)),
        .svh_d = try up(gpu, &bufs, std.mem.sliceAsBytes(svh_d.items)),
        .count = E,
        .dims = d,
        .width = ni,
        .k2_gu = .{ lo_gu, hi_gu },
        .k2_d = .{ lo_d, hi_d },
    };
    const R = v.rows;
    const sz = exl3.Scratch.bytes(ex, R, v.slots, cfg_gu, cfg_d);
    var sc: exl3.Scratch = .{ .xg = 0, .xu = 0, .xd = 0, .z = 0, .y = 0, .ids = 0, .count = 0, .members = 0, .rows = R, .slots = v.slots, .cfg_gu = cfg_gu, .cfg_d = cfg_d };
    inline for (.{ .{ "xg", sz.xg }, .{ "xu", sz.xg }, .{ "xd", sz.xd }, .{ "z", sz.z }, .{ "y", sz.y }, .{ "ids", sz.ids }, .{ "count", 4 }, .{ "members", sz.members } }) |f| {
        var b = try cuda.DeviceBuffer.alloc(gpu.d, f[1]);
        try b.fill8(0, null);
        try bufs.append(a, b);
        @field(sc, f[0]) = b.ptr;
    }
    try bufs.items[bufs.items.len - 1].fill32(0xFFFFFFFF, null); // members: -1 (torch.full(-1))

    const x = try readFile(gpu, dir, "x.bin");
    defer a.free(x);
    const pick = try readFile(gpu, dir, "pick.bin");
    defer a.free(pick);
    const wts = try readFile(gpu, dir, "wts.bin");
    defer a.free(wts);
    const want = try readFile(gpu, dir, "out.bin");
    defer a.free(want);
    const dx = try up(gpu, &bufs, x);
    const dp = try up(gpu, &bufs, pick);
    const dwt = try up(gpu, &bufs, wts);
    var out = try cuda.DeviceBuffer.alloc(gpu.d, want.len);
    defer out.free();
    try kern.routed(s, dx, false, d, dp, dwt, ex, sc, out.ptr, R, std.math.inf(f32), 1, true);
    try s.synchronize();
    const got = try check.download(gpu, out);
    defer a.free(got);
    try check.sameBytes(name, got, want);
    check.pass("{s}: routed experts, {d} rows x {d} slots of {d} (gate/up K2 {d}-{d}, down {d}-{d}) byte-equal to the Python engine", .{ name, R, v.slots, E, lo_gu, hi_gu, lo_d, hi_d });
}
