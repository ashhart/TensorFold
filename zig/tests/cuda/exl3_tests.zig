//! EXL3: the decode-row linear against float64 references, row invariance, the engine's fixture bytes, bandwidth.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3fmt = @import("core").exl3_format;
const exl3 = cuda.exl3;
const Gpu = check.Gpu;
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;

const rows = [_]usize{ 1, 3, 8, 16, 17, 64, 128 }; // every row count in the f64 checks (128 covers the rest)

const Cfg = struct { id: []const u8, halves: u32, codebook: exl3fmt.Codebook, k: usize, n: usize };

const f64_cfgs = [_]Cfg{
    .{ .id = "exl3.linear.f64.mul1.4", .halves = 8, .codebook = .mul1, .k = 4096, .n = 1024 },
    .{ .id = "exl3.linear.f64.mul1.4", .halves = 2, .codebook = .mul1, .k = 512, .n = 512 }, // 1 bit
    .{ .id = "exl3.linear.f64.mul1.4", .halves = 4, .codebook = .mul1, .k = 512, .n = 512 }, // 2 bits
    .{ .id = "exl3.linear.f64.mul1.5", .halves = 10, .codebook = .mul1, .k = 1024, .n = 4096 },
    .{ .id = "exl3.linear.f64.mul1.5", .halves = 6, .codebook = .mul1, .k = 512, .n = 512 }, // 3 bits
    .{ .id = "exl3.linear.f64.mul1.6", .halves = 12, .codebook = .mul1, .k = 2048, .n = 8192 }, // 8 warps a block
    .{ .id = "exl3.linear.f64.mul1.7", .halves = 14, .codebook = .mul1, .k = 512, .n = 1024 },
    .{ .id = "exl3.linear.f64.mul1.8", .halves = 16, .codebook = .mul1, .k = 2048, .n = 4096 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 8, .codebook = .mcg, .k = 1024, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 2, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 4, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 6, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 10, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 12, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 14, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.mcg", .halves = 16, .codebook = .mcg, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 8, .codebook = .inst3, .k = 1024, .n = 1024 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 2, .codebook = .inst3, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 4, .codebook = .inst3, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 6, .codebook = .inst3, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 10, .codebook = .inst3, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 12, .codebook = .inst3, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 14, .codebook = .inst3, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.inst3", .halves = 16, .codebook = .inst3, .k = 512, .n = 512 },
    // the mul1 half steps, 1.5 to 6.5 bits, under one check
    .{ .id = "exl3.linear.f64.halfstep", .halves = 3, .codebook = .mul1, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.halfstep", .halves = 5, .codebook = .mul1, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.halfstep", .halves = 7, .codebook = .mul1, .k = 512, .n = 512 },
    .{ .id = "exl3.linear.f64.halfstep", .halves = 9, .codebook = .mul1, .k = 1024, .n = 1024 },
    .{ .id = "exl3.linear.f64.halfstep", .halves = 11, .codebook = .mul1, .k = 2048, .n = 1024 },
    .{ .id = "exl3.linear.f64.halfstep", .halves = 13, .codebook = .mul1, .k = 512, .n = 512 },
};

/// One check's outcome: a failure prints its FAIL line (TestFailed ones already printed theirs) and fails the run.
pub const Run = struct {
    failed: bool = false,

    pub fn fail(self: *Run, e: anyerror, id: []const u8) void {
        if (e != error.TestFailed) std.debug.print("FAIL {s}: {t}\n", .{ id, e });
        self.failed = true;
    }
};

pub fn run(gpu: Gpu, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();

    var state = Run{};
    var last: ?[]const u8 = null;
    var ok = true;
    for (f64_cfgs) |c| {
        if (last) |prev| {
            if (!std.mem.eql(u8, prev, c.id)) {
                if (ok) check.pass("{s}", .{prev});
                ok = true;
            }
        }
        f64Linear(&r, &lin, c) catch |e| {
            state.fail(e, c.id);
            ok = false;
        };
        last = c.id;
    }
    if (last) |prev| {
        if (ok) check.pass("{s}", .{prev});
    }
    {
        var rows_ok = true;
        rowsInvariant(&r, &lin) catch |e| {
            state.fail(e, "exl3.linear.rows");
            rows_ok = false;
        };
        if (rows_ok) check.pass("exl3.linear.rows", .{});
    }
    pythonChecks(&r, &lin, dir, &state) catch |e| {
        state.fail(e, "exl3.linear.python");
    };
    try @import("exl3_dense_tests.zig").run(&r, &state, dir);
    try @import("exl3_experts_tests.zig").run(&r, &state, dir);
    try @import("exl3_layer_tests.zig").run(&r, &state, dir);
    if (state.failed) return error.TestFailed;
    try bench(&r, &lin);
}

/// A zeroed device buffer (counters must read zero, unlike zeros()'s 0xff canaries).
pub fn zeroed(r: *Rig, bytes: usize) !u64 {
    const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(bytes, 16));
    try r.bufs.append(r.gpu.gpa, b);
    try b.fill8(0, r.s.handle);
    return b.ptr;
}

/// One codebook and width at one shape: random trellis, scales and 128 rows of x, then every row count of `rows`.
fn f64Linear(r: *Rig, lin: *const exl3.Linear, c: Cfg) !void {
    const width = exl3fmt.Width{ .halves = @intCast(c.halves) };
    const cb: exl3.Codebook = switch (c.codebook) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
    };
    const k = c.k;
    const n = c.n;
    const trellis = try r.a.alloc(i16, (k / 16) * (n / 16) * width.tileWords());
    for (trellis) |*v| v.* = r.rng.int(i16);
    const suh = try r.a.alloc(u16, k);
    const svh = try r.a.alloc(u16, n);
    for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
    for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
    const x = try r.bfs(128 * k, 1.0);
    const want = try reference(r, width, c.codebook, trellis, k, n, x, suh, svh);
    const st = exl3.strides(k, n, c.halves, false);
    const w: exl3.Weight = .{
        .trellis = try r.dev(std.mem.sliceAsBytes(trellis)),
        .suh = try r.dev(std.mem.sliceAsBytes(suh)),
        .svh = try r.dev(std.mem.sliceAsBytes(svh)),
        .bias = 0,
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = c.halves,
        .codebook = cb,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const counters = try zeroed(r, exl3.Linear.counterBytes(w));
    for (rows) |m| {
        const dy = try r.zeros(m * n * 2);
        const xh = try r.zeros(m * k * 2);
        const z: u64 = if (exl3.plan(k, n, 192, 8).sk > 1) try r.zeros(exl3.Linear.zBytes(w, m)) else 0;
        try lin.matmul(r.s, w, dx, .bf16, m, dy, .bf16, xh, z, counters);
        try closeOut(r, c.id, dy, m * n, want.want, want.tol);
    }
}

/// The Python engine's f64 reference: Wq fp16 from the trellis, fp16 xh, the y sums, and each element's |t| sum.
fn reference(r: *Rig, width: exl3fmt.Width, cb: exl3fmt.Codebook, trellis: []const i16, k: usize, n: usize, x: []const u16, suh: []const u16, svh: []const u16) !struct { want: []f64, tol: []f64 } {
    const m = 128;
    const wq = try r.a.alloc(f16, k * n);
    exl3fmt.unpack(width, cb, trellis, k / 16, n / 16, wq);
    const xs = try r.a.alloc(f64, m * k);
    for (0..m) |row| {
        for (0..k) |i| xs[row * k + i] = bf(x[row * k + i]) * @as(f64, @floatCast(@as(f16, @bitCast(suh[i]))));
    }
    exl3fmt.rotate(xs, m, k, 1);
    const xh = try r.a.alloc(f16, m * k);
    for (xh, xs) |*o, v| o.* = @floatCast(v);
    const ypre = try r.a.alloc(f64, m * n); // [n][m], rotated along n
    @memset(ypre, 0);
    const s1 = try r.a.alloc(f64, m * n); // each output's sum of |xh * wq| over k
    @memset(s1, 0);
    var xv: [128]f64 = undefined;
    var xa: [128]f64 = undefined;
    for (0..k) |ki| {
        for (0..m) |row| {
            xv[row] = @floatCast(xh[row * k + ki]);
            xa[row] = @abs(xv[row]);
        }
        for (0..n) |col| {
            const wv: f64 = @floatCast(wq[ki * n + col]);
            const wa = @abs(wv);
            const base = col * m;
            for (0..m) |row| {
                ypre[base + row] += xv[row] * wv;
                s1[base + row] += xa[row] * wa;
            }
        }
    }
    exl3fmt.rotate(ypre, n, m, 0);
    const want = try r.a.alloc(f64, m * n);
    const tol = try r.a.alloc(f64, m * n);
    for (0..n) |col| {
        const scale: f64 = @floatCast(@as(f16, @bitCast(svh[col])));
        for (0..m) |row| {
            want[row * n + col] = ypre[col * m + row] * scale;
            // bf16 output ulp (2^-7 of the value) plus fp32 accumulation and fp16 xh cases, 2^-11 of the |t| sum
            tol[row * n + col] = @abs(want[row * n + col]) / 128 + s1[row * n + col] / 2048 + 1e-30;
        }
    }
    return .{ .want = want, .tol = tol };
}

/// bf16 outputs within the f64 reference's tolerance, with the exact FAIL line.
pub fn closeOut(r: *Rig, id: []const u8, ptr: u64, len: usize, want: []const f64, tol: []const f64) !void {
    const raw = try r.back(ptr, len * 2);
    var most: f64 = 0;
    var at: usize = 0;
    for (want[0..len], tol[0..len], 0..) |wv, t, i| {
        const g = bf(std.mem.bytesToValue(u16, raw[2 * i ..][0..2]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT {s}: largest difference {d:.3} of its tolerance (element {d})\n", .{ id, most, at });
    if (most > 1) {
        std.debug.print("FAIL {s}: largest difference {d:.3} of its tolerance\n", .{ id, most });
        return error.TestFailed;
    }
}

/// The first m rows of a 128-row call byte-equal every narrower call, at a split shape and a single-split one.
fn rowsInvariant(r: *Rig, lin: *const exl3.Linear) !void {
    const shapes = [_][2]usize{ .{ 4096, 1024 }, .{ 512, 4096 } };
    for (shapes) |sh| {
        const k = sh[0];
        const n = sh[1];
        const width: exl3fmt.Width = .{ .halves = 8 };
        const trellis = try r.a.alloc(i16, (k / 16) * (n / 16) * width.tileWords());
        for (trellis) |*v| v.* = r.rng.int(i16);
        const suh = try r.a.alloc(u16, k);
        const svh = try r.a.alloc(u16, n);
        for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
        for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
        const st = exl3.strides(k, n, 8, false);
        const w: exl3.Weight = .{
            .trellis = try r.dev(std.mem.sliceAsBytes(trellis)),
            .suh = try r.dev(std.mem.sliceAsBytes(suh)),
            .svh = try r.dev(std.mem.sliceAsBytes(svh)),
            .bias = 0,
            .k = @intCast(k),
            .n = @intCast(n),
            .halves = 8,
            .codebook = .mul1,
            .stride_k = st.k,
            .stride_nb = st.nb,
        };
        const x = try r.bfs(128 * k, 1.0);
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const counters = try zeroed(r, exl3.Linear.counterBytes(w));
        const y128 = try r.zeros(128 * n * 2);
        {
            const xh = try r.zeros(128 * k * 2);
            const z: u64 = if (exl3.plan(k, n, 192, 8).sk > 1) try r.zeros(exl3.Linear.zBytes(w, 128)) else 0;
            try lin.matmul(r.s, w, dx, .bf16, 128, y128, .bf16, xh, z, counters);
        }
        const got128 = try r.back(y128, 128 * n * 2);
        for ([_]usize{ 1, 2, 3, 16, 17, 64 }) |m| {
            const ym = try r.zeros(m * n * 2);
            const xh = try r.zeros(m * k * 2);
            const z: u64 = if (exl3.plan(k, n, 192, 8).sk > 1) try r.zeros(exl3.Linear.zBytes(w, m)) else 0;
            try lin.matmul(r.s, w, dx, .bf16, m, ym, .bf16, xh, z, counters);
            const got = try r.back(ym, m * n * 2);
            const id = try std.fmt.allocPrint(r.a, "exl3.linear.rows k={d} n={d} m={d}", .{ k, n, m });
            try check.sameBytes(id, got, got128[0 .. m * n * 2]);
        }
    }
}

/// The Python engine's recorded outputs, bit for bit (exl3_oracle.py's manifest and raw files).
fn pythonChecks(r: *Rig, lin: *const exl3.Linear, dir: []const u8, state: *Run) !void {
    if (std.mem.eql(u8, dir, "none")) {
        for (4..9) |bits| std.debug.print("SKIP exl3.linear.python.{d}\n", .{bits});
        return;
    }
    const manifest_path = try std.fs.path.join(r.a, &.{ dir, "manifest.json" });
    const text = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, manifest_path, r.a, .limited(1 << 24)) catch |e| {
        for (4..9) |bits| {
            std.debug.print("FAIL exl3.linear.python.{d}: {t}\n", .{ bits, e });
            state.failed = true;
        }
        return;
    };
    var parsed = try std.json.parseFromSlice(std.json.Value, r.a, text, .{});
    var seen = [_]bool{ false, false, false, false, false, false, false, false, false };
    for (parsed.value.object.get("linears").?.array.items) |item| {
        const o = item.object;
        const bits: u32 = bitsOf(o.get("bits").?);
        seen[bits] = true;
        const id = try std.fmt.allocPrint(r.a, "exl3.linear.python.{d}", .{bits});
        var ok = true;
        pythonLinear(r, lin, dir, o, id) catch |e| {
            state.fail(e, id);
            ok = false;
        };
        if (ok) check.pass("{s}", .{id});
    }
    for (4..9) |bits| {
        if (!seen[bits]) std.debug.print("SKIP exl3.linear.python.{d}\n", .{bits});
    }
}

fn bitsOf(v: std.json.Value) u32 {
    return switch (v) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => unreachable,
    };
}

fn pythonLinear(r: *Rig, lin: *const exl3.Linear, dir: []const u8, o: std.json.ObjectMap, id: []const u8) !void {
    const k: usize = @intCast(o.get("k").?.integer);
    const n: usize = @intCast(o.get("n").?.integer);
    const bits: u32 = bitsOf(o.get("bits").?);
    const cb: exl3.Codebook = if (std.mem.eql(u8, o.get("codebook").?.string, "mcg"))
        .mcg
    else if (std.mem.eql(u8, o.get("codebook").?.string, "inst3"))
        .inst3
    else
        .mul1;
    const bias = o.get("bias").?;
    const st = exl3.strides(k, n, 2 * bits, false);
    const w: exl3.Weight = .{
        .trellis = try r.dev(try readFile(r, dir, o.get("trellis").?.string)),
        .suh = try r.dev(try readFile(r, dir, o.get("suh").?.string)),
        .svh = try r.dev(try readFile(r, dir, o.get("svh").?.string)),
        .bias = if (std.meta.activeTag(bias) == .null) 0 else try r.dev(try readFile(r, dir, bias.string)),
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = 2 * bits,
        .codebook = cb,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    const xmap = o.get("x").?.object;
    const ymap = o.get("y").?.object;
    const counters = try zeroed(r, exl3.Linear.counterBytes(w));
    for (o.get("rows").?.array.items) |rv| {
        const m: usize = @intCast(rv.integer);
        const key = try std.fmt.allocPrint(r.a, "{d}", .{m});
        const x = try readFile(r, dir, xmap.get(key).?.string);
        const want = try readFile(r, dir, ymap.get(key).?.string);
        const dy = try r.zeros(m * n * 2);
        const xh = try r.zeros(m * k * 2);
        const z: u64 = if (exl3.plan(k, n, 192, 8).sk > 1) try r.zeros(exl3.Linear.zBytes(w, m)) else 0;
        try lin.matmul(r.s, w, try r.dev(x), .bf16, m, dy, .bf16, xh, z, counters);
        const got = try r.back(dy, want.len);
        try check.sameBytes(id, got, want);
    }
}

pub fn readFile(r: *Rig, dir: []const u8, name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, name }), r.a, .limited(1 << 30));
}

/// Trellis bytes a second: one row of a 4-bit 4096x32768 layer (64 MiB, past L2), decode-dominated.
fn bench(r: *Rig, lin: *const exl3.Linear) !void {
    const k = 4096;
    const n = 32768;
    const bytes = k * n / 2;
    const st = exl3.strides(k, n, 8, false);
    const suh = try r.a.alloc(u16, k);
    const svh = try r.a.alloc(u16, n);
    @memset(suh, 0x3C00); // 1.0
    @memset(svh, 0x3C00);
    const x = try r.bfs(k, 1.0);
    const w: exl3.Weight = .{
        .trellis = try r.zeros(bytes),
        .suh = try r.dev(std.mem.sliceAsBytes(suh)),
        .svh = try r.dev(std.mem.sliceAsBytes(svh)),
        .bias = 0,
        .k = k,
        .n = n,
        .halves = 8,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dy = try r.zeros(n * 2);
    const xh = try r.zeros(k * 2);
    const z = try r.zeros(exl3.Linear.zBytes(w, 1));
    const counters = try zeroed(r, exl3.Linear.counterBytes(w));
    for (0..3) |_| try lin.matmul(r.s, w, dx, .bf16, 1, dy, .bf16, xh, z, counters);
    try r.s.synchronize();
    const reps = 200;
    const t0 = check.now(r.gpu.io);
    for (0..reps) |_| try lin.matmul(r.s, w, dx, .bf16, 1, dy, .bf16, xh, z, counters);
    try r.s.synchronize();
    const t1 = check.now(r.gpu.io);
    const sec = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const gb = @as(f64, @floatFromInt(bytes * reps)) / sec / 1e9;
    std.debug.print("RESULT exl3.bench.decode {d:.1}\n", .{gb});
}
