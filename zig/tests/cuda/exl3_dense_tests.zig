//! EXL3 dense prompt path: qlinear's dispatch, float64 references, chunk invariance, the engine's prompt bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3fmt = @import("core").exl3_format;
const exl3_load = @import("core").exl3_load;
const exl3 = cuda.exl3;
const qlinear = cuda.qlinear;
const exl3_tests = @import("exl3_tests.zig");
const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    var dense = try exl3.Dense.load(r.gpu.d);
    defer dense.unload();
    const ql: qlinear.Linear = .{ .exl3 = &dense };
    {
        var ok = true;
        qlinearCheck(r, &ql, &dense.lin) catch |e| {
            state.fail(e, "exl3.qlinear");
            ok = false;
        };
        if (ok) check.pass("exl3.qlinear", .{});
    }
    {
        var ok = true;
        promptF64(r, &dense.lin) catch |e| {
            state.fail(e, "exl3.prompt.linear.f64");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.linear.f64", .{});
    }
    {
        var ok = true;
        chunks(r, &dense) catch |e| {
            state.fail(e, "exl3.prompt.linear.chunks");
            ok = false;
        };
        if (ok) check.pass("exl3.prompt.linear.chunks", .{});
    }
    if (pythonPrompt(r, &dense, dir) catch |e| blk: {
        state.fail(e, "exl3.prompt.python");
        break :blk false;
    }) check.pass("exl3.prompt.python", .{});
    if (loadRedsnow(r, &dense, dir) catch |e| blk: {
        state.fail(e, "exl3.load.redsnow");
        break :blk false;
    }) check.pass("exl3.load.redsnow", .{});
}

/// Through qlinear.Linear's decode and prompt dispatch, 1, 17 and 300 rows byte-equal the direct block loop.
fn qlinearCheck(r: *Rig, ql: *const qlinear.Linear, lin: *const exl3.Linear) !void {
    const k = 1024;
    const n = 512;
    const trellis, const suh, const svh = try tensors(r, k, n, .{ .halves = 8 });
    const w = try weight(r, trellis, suh, svh, k, n, 8, .mul1);
    const x = try r.bfs(300 * k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const counters = try exl3_tests.zeroed(r, exl3.Linear.counterBytes(w));
    const xh = try exl3_tests.zeroed(r, 128 * k * 2);
    const z: u64 = if (exl3.plan(k, n, 192, 8).sk > 1) try exl3_tests.zeroed(r, exl3.Linear.zBytes(w, 128)) else 0;
    for ([_]usize{ 1, 17, 300 }) |m| {
        const want = try r.zeros(m * n * 2);
        var off: usize = 0;
        while (off < m) : (off += 128) {
            try lin.matmul(r.s, w, dx + off * k * 2, .bf16, @min(m - off, 128), want + off * n * 2, .bf16, xh, z, counters);
        }
        const wantraw = try r.back(want, m * n * 2);
        const Kind = enum { decode, prompt };
        for ([_]Kind{ .decode, .prompt }) |kind| {
            const got = try r.zeros(m * n * 2);
            switch (kind) {
                .decode => try ql.decode(r.s, .{ .exl3 = w }, .{ .x = dx }, got, m),
                .prompt => try ql.prompt(r.s, .{ .exl3 = w }, .{ .x = dx }, got, m),
            }
            const raw = try r.back(got, m * n * 2);
            const id = try std.fmt.allocPrint(r.a, "exl3.qlinear.{s} m={d}", .{ @tagName(kind), m });
            try check.sameBytes(id, raw, wantraw);
        }
    }
}

/// 300 and 1000 prompt rows against the float64 reference (exl3_tests' reference with m a parameter).
fn promptF64(r: *Rig, lin: *const exl3.Linear) !void {
    const cfgs = [_]struct { halves: u32, codebook: exl3fmt.Codebook, k: usize, n: usize }{
        .{ .halves = 8, .codebook = .mul1, .k = 512, .n = 512 },
        .{ .halves = 10, .codebook = .mul1, .k = 1024, .n = 512 },
    };
    for (cfgs) |c| {
        const width: exl3fmt.Width = .{ .halves = @intCast(c.halves) };
        const trellis, const suh, const svh = try tensors(r, c.k, c.n, width);
        const w = try weight(r, trellis, suh, svh, c.k, c.n, c.halves, cbOf(@tagName(c.codebook)));
        const counters = try exl3_tests.zeroed(r, exl3.Linear.counterBytes(w));
        const xh = try exl3_tests.zeroed(r, 128 * c.k * 2);
        const z: u64 = if (exl3.plan(c.k, c.n, 192, 8).sk > 1) try exl3_tests.zeroed(r, exl3.Linear.zBytes(w, 128)) else 0;
        for ([_]usize{ 300, 1000 }) |m| {
            const x = try r.bfs(m * c.k, 1.0);
            const dx = try r.dev(std.mem.sliceAsBytes(x));
            const ref = try promptReference(r, width, c.codebook, trellis, c.k, c.n, x, suh, svh, m);
            const id = try std.fmt.allocPrint(r.a, "exl3.prompt.linear.f64 k={d} n={d} m={d}", .{ c.k, c.n, m });
            const dy = try r.zeros(m * c.n * 2);
            var off: usize = 0;
            while (off < m) : (off += 128) {
                try lin.matmul(r.s, w, dx + off * c.k * 2, .bf16, @min(m - off, 128), dy + off * c.n * 2, .bf16, xh, z, counters);
            }
            try exl3_tests.closeOut(r, id, dy, m * c.n, ref.want, ref.tol);
        }
    }
}

/// A row's bytes are identical whether the prompt runs as one call, 128-row calls, a split at 37, or 2048 chunks.
fn chunks(r: *Rig, de: *exl3.Dense) !void {
    const k = 512;
    const n = 512;
    const rows = 2185; // 17 whole 128-blocks and a 9-row tail
    const trellis, const suh, const svh = try tensors(r, k, n, .{ .halves = 8 });
    const w = try weight(r, trellis, suh, svh, k, n, 8, .mul1);
    const x = try r.bfs(rows * k, 1.0);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const y0 = try r.zeros(rows * n * 2);
    try de.prompt(r.s, w, dx, .bf16, rows, y0, .bf16);
    const want = try r.back(y0, rows * n * 2);
    var b128: [17]usize = undefined;
    for (&b128, 1..) |*v, i| v.* = i * 128;
    for ([_][]const usize{ &.{}, &b128, &.{37}, &.{2048} }) |bounds| {
        const y = try r.zeros(rows * n * 2);
        var prev: usize = 0;
        for (bounds) |b| {
            try de.prompt(r.s, w, dx + prev * k * 2, .bf16, b - prev, y + prev * n * 2, .bf16);
            prev = b;
        }
        try de.prompt(r.s, w, dx + prev * k * 2, .bf16, rows - prev, y + prev * n * 2, .bf16);
        const got = try r.back(y, rows * n * 2);
        const id = try std.fmt.allocPrint(r.a, "exl3.prompt.linear.chunks {d} splits", .{bounds.len});
        try check.sameBytes(id, got, want);
    }
}

/// The Python engine's recorded prompt bytes for one real dense projection, bit for bit.
fn pythonPrompt(r: *Rig, de: *exl3.Dense, dir: []const u8) !bool {
    if (std.mem.eql(u8, dir, "none")) {
        std.debug.print("SKIP exl3.prompt.python\n", .{});
        return false;
    }
    const d = (try denseEntry(r, dir)) orelse {
        std.debug.print("SKIP exl3.prompt.python: no dense entry\n", .{});
        return false;
    };
    const k: usize = @intCast(d.get("k").?.integer);
    const n: usize = @intCast(d.get("n").?.integer);
    const bits: u32 = bitsOf(d.get("bits").?);
    const m: usize = @intCast(d.get("rows").?.integer);
    const bias = d.get("bias").?;
    const st = exl3.strides(k, n, 2 * bits, false);
    const w: exl3.Weight = .{
        .trellis = try r.dev(try exl3_tests.readFile(r, dir, d.get("trellis").?.string)),
        .suh = try r.dev(try exl3_tests.readFile(r, dir, d.get("suh").?.string)),
        .svh = try r.dev(try exl3_tests.readFile(r, dir, d.get("svh").?.string)),
        .bias = if (std.meta.activeTag(bias) == .null) 0 else try r.dev(try exl3_tests.readFile(r, dir, bias.string)),
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = 2 * bits,
        .codebook = cbOf(d.get("codebook").?.string),
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    const x = try exl3_tests.readFile(r, dir, d.get("x").?.string);
    const want = try exl3_tests.readFile(r, dir, d.get("y").?.string);
    const dy = try r.zeros(m * n * 2);
    try de.prompt(r.s, w, try r.dev(x), .bf16, m, dy, .bf16);
    const got = try r.back(dy, want.len);
    try check.sameBytes("exl3.prompt.python", got, want);
    return true;
}

/// The checkpoint loader over real RED-SNOW: every EXL3 group admitted, one layer's prompt vs the fixture bytes.
fn loadRedsnow(r: *Rig, de: *exl3.Dense, dir: []const u8) !bool {
    if (std.mem.eql(u8, dir, "none")) {
        std.debug.print("SKIP exl3.load.redsnow\n", .{});
        return false;
    }
    const o = try manifest(r, dir);
    const model = o.get("model") orelse return error.TestFailed; // the fixture names its model folder
    const d = (try denseEntry(r, dir)) orelse {
        std.debug.print("SKIP exl3.load.redsnow: no dense entry\n", .{});
        return false;
    };
    var s = try exl3_load.scan(r.gpu.gpa, r.gpu.io, model.string);
    defer s.deinit();
    const sum = s.summary;
    // the inspect tool's counts: every group suh/svh mul1, the whole-bits histogram, the experts' downs
    try check.expect(sum.groups == 37408, "exl3.load.redsnow: {d} groups, expected 37408", .{sum.groups});
    try check.expect(sum.mul1 == 37408, "exl3.load.redsnow: {d} mul1, expected 37408", .{sum.mul1});
    try check.expect(sum.suh_svh == 37408, "exl3.load.redsnow: {d} suh/svh, expected 37408", .{sum.suh_svh});
    const hist = [_]usize{ 15425, 11801, 10054, 43, 85 }; // whole bits 4 to 8
    const downs = [_]usize{ 10283, 1448, 653, 0, 0 };
    for (hist, downs, 4..) |want, down, bits| {
        try check.expect(sum.bits[2 * bits] == want, "exl3.load.redsnow: {d} groups at {d} bits, expected {d}", .{ sum.bits[2 * bits], bits, want });
        try check.expect(sum.experts_down[2 * bits] == down, "exl3.load.redsnow: {d} expert down projections at {d} bits, expected {d}", .{ sum.experts_down[2 * bits], bits, down });
    }
    try check.expect(s.bad == 0, "exl3.load.redsnow: {d} tensors refused", .{s.bad});
    const gi = s.find(d.get("name").?.string) orelse return error.TestFailed; // the fixture's layer
    const g = s.groups[gi];
    const p = try s.parts(r.a, gi);
    const st = exl3.strides(g.k, g.n, g.halves, false);
    const w: exl3.Weight = .{
        .trellis = try r.dev(p.trellis.bytes),
        .suh = try r.dev(p.suh.bytes),
        .svh = try r.dev(p.svh.bytes),
        .bias = if (p.bias) |b| try r.dev(b.bytes) else 0,
        .k = @intCast(g.k),
        .n = @intCast(g.n),
        .halves = g.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    try check.expect(g.halves == 2 * bitsOf(d.get("bits").?) and g.k == @as(usize, @intCast(d.get("k").?.integer)) and
        g.n == @as(usize, @intCast(d.get("n").?.integer)), "exl3.load.redsnow: the loaded layer is not the fixture's", .{});
    const m: usize = @intCast(d.get("rows").?.integer);
    const x = try exl3_tests.readFile(r, dir, d.get("x").?.string);
    const want = try exl3_tests.readFile(r, dir, d.get("y").?.string);
    const dy = try r.zeros(m * g.n * 2);
    try de.prompt(r.s, w, try r.dev(x), .bf16, m, dy, .bf16);
    const got = try r.back(dy, want.len);
    try check.sameBytes("exl3.load.redsnow", got, want);
    return true;
}

/// The manifest's "dense" entry, null when an older fixture dir has none.
fn denseEntry(r: *Rig, dir: []const u8) !?std.json.ObjectMap {
    const o = try manifest(r, dir);
    return if (o.get("dense")) |v| v.object else null;
}

fn manifest(r: *Rig, dir: []const u8) !std.json.ObjectMap {
    const text = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, try std.fs.path.join(r.a, &.{ dir, "manifest.json" }), r.a, .limited(1 << 24));
    const parsed = try std.json.parseFromSlice(std.json.Value, r.a, text, .{});
    return parsed.value.object;
}

fn bitsOf(v: std.json.Value) u32 {
    return switch (v) {
        .integer => |i| @intCast(i),
        .float => |f| @intFromFloat(f),
        else => unreachable,
    };
}

fn cbOf(name: []const u8) exl3.Codebook {
    if (std.mem.eql(u8, name, "mcg")) return .mcg;
    if (std.mem.eql(u8, name, "inst3")) return .inst3;
    return .mul1;
}

/// Random trellis, scales at one shape (the same layout as exl3_tests' f64 cfgs).
fn tensors(r: *Rig, k: usize, n: usize, width: exl3fmt.Width) !struct { []i16, []u16, []u16 } {
    const trellis = try r.a.alloc(i16, (k / 16) * (n / 16) * width.tileWords());
    for (trellis) |*v| v.* = r.rng.int(i16);
    const suh = try r.a.alloc(u16, k);
    const svh = try r.a.alloc(u16, n);
    for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
    for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
    return .{ trellis, suh, svh };
}

fn weight(r: *Rig, trellis: []const i16, suh: []const u16, svh: []const u16, k: usize, n: usize, halves: u32, cb: exl3.Codebook) !exl3.Weight {
    const st = exl3.strides(k, n, halves, false);
    return .{
        .trellis = try r.dev(std.mem.sliceAsBytes(trellis)),
        .suh = try r.dev(std.mem.sliceAsBytes(suh)),
        .svh = try r.dev(std.mem.sliceAsBytes(svh)),
        .bias = 0,
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = halves,
        .codebook = cb,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

/// The Python engine's f64 reference, with the row count a parameter.
fn promptReference(r: *Rig, width: exl3fmt.Width, cb: exl3fmt.Codebook, trellis: []const i16, k: usize, n: usize, x: []const u16, suh: []const u16, svh: []const u16, m: usize) !struct { want: []f64, tol: []f64 } {
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
    var xv: [1000]f64 = undefined;
    var xa: [1000]f64 = undefined;
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
