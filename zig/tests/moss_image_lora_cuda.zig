//! tf-moss-image-lora: runtime LoRA overlays on every projection class, real weights, against an FP64 reference.
const std = @import("std");
const cuda = @import("cuda");
const mi = @import("moss_image");

const usage =
    \\tf-moss-image-lora TRANSFORMER_DIR WORK_DIR [LORA_DIR NAME]
    \\Writes a rank-16 f32 adapter on every projection class to WORK_DIR (seeded, its change about 5% of each output),
    \\then runs each class's projections on the same seeded bf16 rows: alone, and with that adapter (and LORA_DIR's
    \\NAME, if given, on the classes it covers) as a forward runs them. FP64 computes x Wᵀ (+ bias) + scale (x Aᵀ) Bᵀ
    \\from the same bf16 rows and the adapter file's own values. Per class it prints native error, torch's unmerged
    \\order (each product rounded to bf16, as PEFT runs it) and plain bf16 rounding of the FP64 sum, as relative RMS,
    \\worst column and worst element in bf16 steps. A step is bf16's at the element or at its column's RMS, whichever
    \\is larger (sums near zero cancel). Then an f32 and an f16 adapter whose A values cancel (bits below bf16's step,
    \\and f32 values past bf16's largest) run on rows built to cancel them. Fails when a LoRA output is more than 0.75
    \\of a step from FP64, or worse than torch's order in RMS or worst column, when a cancellation case would pass
    \\with A rounded to bf16 (it would no longer catch that), or when a projection with no LoRA picked, after LoRA
    \\requests, differs in any byte from the same one before any LoRA.
;

const rows = 256;
const synthetic = "gate-synthetic";
const synthetic_rank = 16;
const synthetic_alpha = 8;
const synthetic_strength = 0.75;

const Class = struct { name: []const u8, modules: []const []const u8 };
const classes = [_]Class{
    .{ .name = "block q", .modules = &.{ "layers.0.proj.q", "layers.20.proj.q", "layers.41.proj.q" } },
    .{ .name = "block k", .modules = &.{ "layers.0.proj.k", "layers.20.proj.k", "layers.41.proj.k" } },
    .{ .name = "block v", .modules = &.{ "layers.0.proj.v", "layers.20.proj.v", "layers.41.proj.v" } },
    .{ .name = "block o", .modules = &.{ "layers.0.proj.o", "layers.20.proj.o", "layers.41.proj.o" } },
    .{ .name = "block ffn_up_gate", .modules = &.{ "layers.0.proj.ffn_up_gate", "layers.20.proj.ffn_up_gate", "layers.41.proj.ffn_up_gate" } },
    .{ .name = "block ffn_down", .modules = &.{ "layers.0.proj.ffn_down", "layers.20.proj.ffn_down", "layers.41.proj.ffn_down" } },
    .{ .name = "refiner qkv", .modules = &.{ "text_embedder.blocks.0.qkv", "text_embedder.blocks.1.qkv" } },
    .{ .name = "refiner proj", .modules = &.{ "text_embedder.blocks.0.proj", "text_embedder.blocks.1.proj" } },
    .{ .name = "refiner ffn_up", .modules = &.{ "text_embedder.blocks.0.ffn_up", "text_embedder.blocks.1.ffn_up" } },
    .{ .name = "refiner ffn_gate", .modules = &.{ "text_embedder.blocks.0.ffn_gate", "text_embedder.blocks.1.ffn_gate" } },
    .{ .name = "refiner ffn_down", .modules = &.{ "text_embedder.blocks.0.ffn_down", "text_embedder.blocks.1.ffn_down" } },
    .{ .name = "text_embedder.proj", .modules = &.{"text_embedder.proj"} },
    .{ .name = "video_embedder", .modules = &.{"video_embedder"} },
    .{ .name = "final_linear_video", .modules = &.{"final_linear_video"} },
};

fn bf(v: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
}

/// bf16's step at |r| (normal range).
fn step(r: f64) f64 {
    const e = std.math.frexp(@max(@abs(r), 1e-30)).exponent;
    return std.math.ldexp(@as(f64, 1), e - 8);
}

/// r rounded to the nearest bf16, ties to even, from FP64 directly.
fn round16(r: f64) f64 {
    const s = step(r);
    const q = r / s;
    const f = @floor(q);
    const d = q - f;
    const n = if (d > 0.5) f + 1 else if (d < 0.5) f else if (@mod(f, 2) == 0) f else f + 1;
    return n * s;
}

fn toBf16(v: f64) u16 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// One class's error against FP64: squared sums, the worst column's relative RMS and the worst element in bf16 steps.
const Stats = struct {
    err2: f64 = 0,
    ref2: f64 = 0,
    worst_column: f64 = 0,
    worst_steps: f64 = 0,
    off: usize = 0,
    count: usize = 0,

    fn rel(s: Stats) f64 {
        return @sqrt(s.err2 / s.ref2);
    }

    /// `got` [rows, n] against `ref` [rows, n].
    fn add(s: *Stats, gpa: std.mem.Allocator, got: []const f64, ref: []const f64, n: usize) !void {
        const col_e = try gpa.alloc(f64, n);
        defer gpa.free(col_e);
        const col_r = try gpa.alloc(f64, n);
        defer gpa.free(col_r);
        @memset(col_e, 0);
        @memset(col_r, 0);
        for (got, ref, 0..) |g, r, i| {
            const e = g - r;
            s.err2 += e * e;
            s.ref2 += r * r;
            col_e[i % n] += e * e;
            col_r[i % n] += r * r;
            if (g != round16(r)) s.off += 1;
            s.count += 1;
        }
        for (col_e, col_r) |e, r| if (r > 0) {
            s.worst_column = @max(s.worst_column, @sqrt(e / r));
        };
        // a step is bf16's at the element or at its column's RMS, whichever is larger: sums near zero cancel
        const m: f64 = @floatFromInt(got.len / n);
        for (got, ref, 0..) |g, r, i| s.worst_steps = @max(s.worst_steps, @abs(g - r) / @max(step(r), step(@sqrt(col_r[i % n] / m))));
    }
};

const Factors = struct { a: []const f64, b: []const f64, rank: usize, scale: f64 };

/// The FP64 sums of rows [lo, hi): base x Wᵀ + bias, and with every overlay; torch's order beside them.
const Work = struct {
    x: []const f64,
    w: []const u16,
    bias: ?[]const u16,
    overlays: []const Factors,
    n: usize,
    k: usize,
    base: []f64,
    full: []f64,
    torch: []f64,

    fn run(job: *const Work, lo: usize, hi: usize) void {
        var low: [256]f64 = undefined;
        for (lo..hi) |r| {
            const xr = job.x[r * job.k ..][0..job.k];
            for (0..job.n) |j| {
                const wr = job.w[j * job.k ..][0..job.k];
                var acc: f64 = 0;
                for (xr, wr) |xv, wv| acc += xv * bf(wv);
                if (job.bias) |b| acc += bf(b[j]);
                job.base[r * job.n + j] = acc;
                job.full[r * job.n + j] = acc;
                job.torch[r * job.n + j] = round16(acc);
            }
            for (job.overlays) |o| {
                for (0..o.rank) |q| {
                    var acc: f64 = 0;
                    for (xr, o.a[q * job.k ..][0..job.k]) |xv, av| acc += xv * av;
                    low[q] = acc;
                }
                for (0..job.n) |j| {
                    var exact: f64 = 0;
                    var rounded: f64 = 0;
                    for (low[0..o.rank], o.b[j * o.rank ..][0..o.rank]) |l, bv| {
                        exact += l * bv;
                        rounded += round16(l) * bv;
                    }
                    job.full[r * job.n + j] += o.scale * exact;
                    // PEFT in bf16: lora_B(lora_A(x)) rounded, times scaling rounded, added to the rounded base
                    const t = &job.torch[r * job.n + j];
                    t.* = round16(t.* + round16(round16(rounded) * o.scale));
                }
            }
        }
    }
};

fn fp64(job: *const Work) !void {
    const threads: usize = @min(std.Thread.getCpuCount() catch 4, 32);
    var pool: [32]std.Thread = undefined;
    const per = (rows + threads - 1) / threads;
    var started: usize = 0;
    defer for (pool[0..started]) |t| t.join();
    while (started < threads and started * per < rows) : (started += 1) {
        const lo = started * per;
        pool[started] = try std.Thread.spawn(.{}, Work.run, .{ job, lo, @min(rows, lo + per) });
    }
}

/// A safetensors file being written: tensors appended in order, then `save`.
const Writer = struct {
    gpa: std.mem.Allocator,
    header: std.ArrayList(u8) = .empty,
    data: std.ArrayList(u8) = .empty,

    fn deinit(w: *Writer) void {
        w.header.deinit(w.gpa);
        w.data.deinit(w.gpa);
    }

    /// `name` [rows, cols] (or a scalar when `cols` is 0) as F32 or F16 from `values`, which that type holds exactly.
    fn put(w: *Writer, name: []const u8, comptime T: type, height: usize, cols: usize, values: []const f64) !void {
        const start = w.data.items.len;
        for (values) |v| {
            const x: T = @floatCast(v);
            if (@as(f64, x) != v) return error.NotExact;
            var bytes: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(if (T == f32) u32 else u16, &bytes, @bitCast(x), .little);
            try w.data.appendSlice(w.gpa, &bytes);
        }
        const dtype = if (T == f32) "F32" else "F16";
        try w.header.appendSlice(w.gpa, if (w.header.items.len == 0) "{" else ",");
        if (cols == 0)
            try w.header.print(w.gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[],\"data_offsets\":[{d},{d}]}}", .{ name, dtype, start, w.data.items.len })
        else
            try w.header.print(w.gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ name, dtype, height, cols, start, w.data.items.len });
    }

    fn save(w: *Writer, io: std.Io, path: []const u8) !void {
        try w.header.appendSlice(w.gpa, "}");
        while (w.header.items.len % 8 != 0) try w.header.append(w.gpa, ' ');
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, w.header.items.len, .little);
        const file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, &len);
        try file.writeStreamingAll(io, w.header.items);
        try file.writeStreamingAll(io, w.data.items);
    }
};

/// The seeded synthetic adapter (kohya, f32 using all 24 bits): each module of `classes`, rank 16, alpha 8, ~5%.
fn writeSynthetic(gpa: std.mem.Allocator, io: std.Io, model: *mi.Model, path: []const u8) !void {
    var out: Writer = .{ .gpa = gpa };
    defer out.deinit();
    var prng = std.Random.DefaultPrng.init(0x4c6f5241);
    const rnd = prng.random();
    const eff = @as(f64, synthetic_alpha) / synthetic_rank * synthetic_strength;
    var buf: [128]u8 = undefined;
    for (classes) |c| for (c.modules) |m| {
        const t = model.targets.get(m) orelse return error.UnknownProjection;
        const w: []align(1) const u16 = std.mem.bytesAsSlice(u16, t.bytes);
        var w2: f64 = 0;
        for (w) |v| w2 += bf(v) * bf(v);
        const out_rms = @sqrt(w2 / @as(f64, @floatFromInt(t.n)));
        const sigma_b = 0.05 * out_rms / (@sqrt(@as(f64, synthetic_rank)) * eff);
        const parts = .{ .{ "lora_down.weight", synthetic_rank, t.k, 1 / @sqrt(@as(f64, @floatFromInt(t.k))) }, .{ "lora_up.weight", t.n, synthetic_rank, sigma_b } };
        inline for (parts) |p| {
            const values = try gpa.alloc(f64, p[1] * p[2]);
            defer gpa.free(values);
            for (values) |*v| v.* = @as(f32, @floatCast(rnd.floatNorm(f64) * p[3]));
            try out.put(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ m, p[0] }), f32, p[1], p[2], values);
        }
        try out.put(try std.fmt.bufPrint(&buf, "{s}.alpha", .{m}), f32, 0, 0, &.{synthetic_alpha});
    };
    try out.save(io, path);
}

/// The cancellation cases' projection and rank; their adapters target it alone.
const cancel_module = "layers.0.proj.q";
const cancel_rank = 2;

/// A cancellation adapter's A [2, k], B [n, 2] (scale 1): each A row pairs two columns cancelling to bits bf16 drops.
fn cancelFactors(gpa: std.mem.Allocator, f16_case: bool, n: usize, k: usize, a: *[]f64, b: *[]f64) !void {
    a.* = try gpa.alloc(f64, cancel_rank * k);
    @memset(a.*, 0);
    // f32: its largest against a neighbour (both inf in bf16), leaving 255 steps at 2^104; 1 + 2^-10 against -1
    const big: f64 = @as(f32, @bitCast(@as(u32, 0x7F7FFFFF)));
    const near: f64 = @as(f32, @bitCast(@as(u32, 0x7F7FFF00)));
    // f16: 1 + 2^-10 against -1, 0.5 + 2^-11 against -0.5; bf16 rounding leaves 0 for each
    const pairs: [2][2]f64 = if (f16_case) .{ .{ 1 + 0x1p-10, -1 }, .{ 0.5 + 0x1p-11, -0.5 } } else .{ .{ big, -near }, .{ 1 + 0x1p-10, -1 } };
    a.*[0] = pairs[0][0];
    a.*[1] = pairs[0][1];
    a.*[k + 2] = pairs[1][0];
    a.*[k + 3] = pairs[1][1];
    const gaps = [2]f64{ pairs[0][0] + pairs[0][1], pairs[1][0] + pairs[1][1] };
    b.* = try gpa.alloc(f64, n * cancel_rank);
    var prng = std.Random.DefaultPrng.init(if (f16_case) 0x663136 else 0x663332);
    for (0..n) |j| for (0..cancel_rank) |q| {
        // an f32 near 1 / gap, times a seeded normal
        b.*[j * cancel_rank + q] = @as(f32, @floatCast(prng.random().floatNorm(f64) / gaps[q]));
    };
}

fn cancelRows(gpa: std.mem.Allocator, k: usize, xh: *[]u16, xs: *[]f64) !void {
    const cs = [_]f64{ 0.75, -0.5, 0.25, -0.625, 0.375, -0.125, 0.5, -0.75 };
    xs.* = try gpa.alloc(f64, rows * k);
    xh.* = try gpa.alloc(u16, rows * k);
    @memset(xs.*, 0);
    @memset(xh.*, 0);
    for (0..rows) |r| for (0..4) |j| {
        xs.*[r * k + j] = cs[r % cs.len];
        xh.*[r * k + j] = toBf16(cs[r % cs.len]);
    };
}

/// Module `m`'s seeded rows [rows, k], bf16 and widened.
fn seeded(gpa: std.mem.Allocator, m: []const u8, k: usize, xh: *[]u16, xs: *[]f64) !void {
    var prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(7, m));
    xs.* = try gpa.alloc(f64, rows * k);
    xh.* = try gpa.alloc(u16, rows * k);
    for (xs.*, xh.*) |*xv, *hv| {
        hv.* = toBf16(prng.random().floatNorm(f64));
        xv.* = bf(hv.*);
    }
}

/// `module` on its seeded rows as a forward runs it now, bf16 [rows, n].
fn projected(gpa: std.mem.Allocator, d: *const cuda.Driver, model: *mi.Model, module: []const u8) ![]u16 {
    const t = model.targets.get(module).?;
    var xh: []u16 = undefined;
    var xs: []f64 = undefined;
    try seeded(gpa, module, t.k, &xh, &xs);
    defer gpa.free(xh);
    defer gpa.free(xs);
    var x = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(xh));
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(d, rows * t.n * 2);
    defer y.free();
    try model.project(module, x.ptr, y.ptr, rows);
    const out = try gpa.alloc(u16, rows * t.n);
    try y.download(0, std.mem.sliceAsBytes(out));
    return out;
}

fn widen(gpa: std.mem.Allocator, buf: cuda.DeviceBuffer, count: usize) ![]f64 {
    const raw = try gpa.alloc(u16, count);
    defer gpa.free(raw);
    try buf.download(0, std.mem.sliceAsBytes(raw));
    const out = try gpa.alloc(f64, count);
    for (out, raw) |*o, v| o.* = bf(v);
    return out;
}

const Row = struct { class: []const u8, adapter: []const u8, parts: usize = 0, native: Stats = .{}, torch: Stats = .{}, rounding: Stats = .{} };

/// A cancellation case: native against FP64 from the values written, and what A rounded to bf16 would have given.
const Cancel = struct { name: []const u8, native: Stats = .{}, rounded_a: Stats = .{}, parts: usize = 0 };

fn cancelCase(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, model: *mi.Model, dir: []const u8, f16_case: bool) !Cancel {
    const name = if (f16_case) "gate-cancel-f16" else "gate-cancel-f32";
    var result: Cancel = .{ .name = name };
    const t = model.targets.get(cancel_module).?;
    var fa: []f64 = undefined;
    var fb: []f64 = undefined;
    try cancelFactors(gpa, f16_case, t.n, t.k, &fa, &fb);
    defer gpa.free(fa);
    defer gpa.free(fb);
    var out: Writer = .{ .gpa = gpa };
    defer out.deinit();
    if (f16_case) try out.put(cancel_module ++ ".lora_down.weight", f16, cancel_rank, t.k, fa) else try out.put(cancel_module ++ ".lora_down.weight", f32, cancel_rank, t.k, fa);
    try out.put(cancel_module ++ ".lora_up.weight", f32, t.n, cancel_rank, fb);
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.safetensors", .{ dir, name });
    defer gpa.free(path);
    try out.save(io, path);

    var xh: []u16 = undefined;
    var xs: []f64 = undefined;
    try cancelRows(gpa, t.k, &xh, &xs);
    defer gpa.free(xh);
    defer gpa.free(xs);
    var x = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(xh));
    defer x.free();
    var y = try cuda.DeviceBuffer.alloc(d, rows * t.n * 2);
    defer y.free();
    try model.useLoras(io, dir, &.{.{ .name = name, .scale = 1 }});
    result.parts = model.lora.?.factor(name, cancel_module).?.parts;
    try model.project(cancel_module, x.ptr, y.ptr, rows);
    const got = try widen(gpa, y, rows * t.n);
    defer gpa.free(got);

    const base = try gpa.alloc(f64, rows * t.n);
    defer gpa.free(base);
    const full = try gpa.alloc(f64, rows * t.n);
    defer gpa.free(full);
    const torch = try gpa.alloc(f64, rows * t.n);
    defer gpa.free(torch);
    const w: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, t.bytes));
    const exact = [_]Factors{.{ .a = fa, .b = fb, .rank = cancel_rank, .scale = 1 }};
    var job: Work = .{ .x = xs, .w = w, .bias = try biasOf(model, t), .overlays = &exact, .n = t.n, .k = t.k, .base = base, .full = full, .torch = torch };
    try fp64(&job);
    try result.native.add(gpa, got, full, t.n);
    // the same with A rounded to bf16 as the upload used to: what the case has to tell apart from FP64
    const ra = try gpa.alloc(f64, fa.len);
    defer gpa.free(ra);
    for (ra, fa) |*r, v| r.* = bf(toBf16(v));
    const full_ra = try gpa.alloc(f64, rows * t.n);
    defer gpa.free(full_ra);
    const rounded = [_]Factors{.{ .a = ra, .b = fb, .rank = cancel_rank, .scale = 1 }};
    job.overlays = &rounded;
    job.full = full_ra;
    try fp64(&job);
    for (full_ra) |*v| if (std.math.isFinite(v.*)) {
        v.* = round16(v.*);
    };
    try result.rounded_a.add(gpa, full_ra, full, t.n);
    return result;
}

/// Zero-factor adapters whose scale leaves fp32 (alpha / rank, or times strength) are refused; in range, main's bytes.
fn scaleCases(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, model: *mi.Model, dir: []const u8, main_bytes: []const u16) !usize {
    const t = model.targets.get(cancel_module).?;
    const za = try gpa.alloc(f64, cancel_rank * t.k);
    defer gpa.free(za);
    @memset(za, 0);
    const zb = try gpa.alloc(f64, t.n * cancel_rank);
    defer gpa.free(zb);
    @memset(zb, 0);
    var buf: [512]u8 = undefined;
    const peft = "gate-scale-alpha";
    try std.Io.Dir.cwd().createDirPath(io, try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, peft }));
    {
        var out: Writer = .{ .gpa = gpa };
        defer out.deinit();
        try out.put("base_model.model." ++ cancel_module ++ ".lora_A.weight", f32, cancel_rank, t.k, za);
        try out.put("base_model.model." ++ cancel_module ++ ".lora_B.weight", f32, t.n, cancel_rank, zb);
        try out.save(io, try std.fmt.bufPrint(&buf, "{s}/{s}/adapter_model.safetensors", .{ dir, peft }));
        const cfg = try std.Io.Dir.cwd().createFile(io, try std.fmt.bufPrint(&buf, "{s}/{s}/adapter_config.json", .{ dir, peft }), .{});
        defer cfg.close(io);
        try cfg.writeStreamingAll(io, std.fmt.comptimePrint("{{\"peft_type\":\"LORA\",\"r\":{d},\"lora_alpha\":1e39}}", .{cancel_rank}));
    }
    const kohya = "gate-scale-strength";
    {
        var out: Writer = .{ .gpa = gpa };
        defer out.deinit();
        try out.put(cancel_module ++ ".lora_down.weight", f32, cancel_rank, t.k, za);
        try out.put(cancel_module ++ ".lora_up.weight", f32, t.n, cancel_rank, zb);
        try out.put(cancel_module ++ ".alpha", f32, 0, 0, &.{std.math.floatMax(f32)});
        try out.save(io, try std.fmt.bufPrint(&buf, "{s}/{s}.safetensors", .{ dir, kohya }));
    }
    var bad: usize = 0;
    for ([_]struct { []const u8, f32 }{ .{ peft, 1 }, .{ kohya, 4 } }) |c| {
        const r = model.useLoras(io, dir, &.{.{ .name = c[0], .scale = c[1] }});
        const ok = if (r) |_| false else |e| e == error.LoraScaleOverflow;
        std.debug.print("{s} at strength {d}: {s}\n", .{ c[0], c[1], if (ok) "refused (LoraScaleOverflow)" else "NOT refused" });
        if (!ok) bad += 1;
    }
    try model.useLoras(io, dir, &.{.{ .name = kohya, .scale = 1 }});
    const now = try projected(gpa, d, model, cancel_module);
    defer gpa.free(now);
    const same = std.mem.eql(u16, main_bytes, now);
    std.debug.print("{s} at strength 1 (scale {e}, zero factors): {s}\n", .{ kohya, model.lora.?.factor(kohya, cancel_module).?.scale, if (same) "main's bytes" else "DIFFERS from main's bytes" });
    if (!same) bad += 1;
    return bad;
}

/// The projection's bias as stored (bf16), if it has one.
fn biasOf(model: *mi.Model, t: anytype) !?[]const u16 {
    if (t.bias == 0) return null;
    var buf: [112]u8 = undefined;
    const bt = try model.w.ck.get(try std.fmt.bufPrint(&buf, "{s}.bias", .{t.module}));
    return @alignCast(std.mem.bytesAsSlice(u16, bt.bytes));
}

/// Whether `s` is within 0.75 of a bf16 step everywhere, every value finite (@max passes NaN over).
fn within(s: Stats) bool {
    return s.worst_steps <= 0.75 and std.math.isFinite(s.rel()) and std.math.isFinite(s.worst_column);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = std.heap.page_allocator;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 5) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const model = try mi.Model.load(gpa, io, &ctx, args[1], mi.encoder_config.max_rows, 1024, 1024);
    defer model.deinit();
    // main's path, before any LoRA bank exists: each class's projections, byte for byte, to compare after the LoRAs
    var before: std.StringHashMapUnmanaged([]u16) = .empty;
    for (classes) |c| for (c.modules) |m| try before.put(a, m, try projected(a, &driver, model, m));
    const synth_path = try std.fmt.allocPrint(a, "{s}/{s}.safetensors", .{ args[2], synthetic });
    try writeSynthetic(gpa, io, model, synth_path);

    const Set = struct { dir: []const u8, name: []const u8, strength: f32 };
    var sets: std.ArrayList(Set) = .empty;
    try sets.append(a, .{ .dir = args[2], .name = synthetic, .strength = synthetic_strength });
    if (args.len == 5) try sets.append(a, .{ .dir = args[3], .name = args[4], .strength = 1 });

    var table: std.ArrayList(Row) = .empty;
    var base_stats: [classes.len]Stats = @splat(.{});
    var failed = false;
    for (sets.items, 0..) |set, si| {
        // a fresh bank per folder: it reads one folder for its life
        if (model.lora) |b| b.deinit();
        model.lora = null;
        try model.useLoras(io, set.dir, &.{.{ .name = set.name, .scale = set.strength }});
        // the reference's factors: the adapter file's own values, read on the host, never what the device holds
        var file = try model.lora.?.read(set.name, &model.targets);
        defer file.deinit();
        for (classes, 0..) |c, ci| {
            var row: Row = .{ .class = c.name, .adapter = set.name };
            var covered = false;
            for (c.modules) |m| {
                const t = model.targets.get(m).?;
                const f = model.lora.?.factor(set.name, t.module) orelse continue;
                covered = true;
                if (f.rank > 256) return error.RankAbove256;
                var xh: []u16 = undefined;
                var xs: []f64 = undefined;
                try seeded(gpa, m, t.k, &xh, &xs);
                defer gpa.free(xh);
                defer gpa.free(xs);
                var x = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(xh));
                defer x.free();
                var y = try cuda.DeviceBuffer.alloc(&driver, rows * t.n * 2);
                defer y.free();
                const o = for (file.overlays) |ov| {
                    if (std.mem.eql(u8, ov.module, t.module)) break ov;
                } else return error.OverlayNotInFile;
                const fa = try gpa.alloc(f64, o.a.len);
                defer gpa.free(fa);
                for (fa, o.a) |*d, v| d.* = v;
                const fb = try gpa.alloc(f64, o.b.len);
                defer gpa.free(fb);
                for (fb, o.b) |*d, v| d.* = v;
                const bias = try biasOf(model, t);
                const base = try gpa.alloc(f64, rows * t.n);
                defer gpa.free(base);
                const full = try gpa.alloc(f64, rows * t.n);
                defer gpa.free(full);
                const torch = try gpa.alloc(f64, rows * t.n);
                defer gpa.free(torch);
                const job: Work = .{ .x = xs, .w = @alignCast(std.mem.bytesAsSlice(u16, t.bytes)), .bias = bias, .overlays = &.{.{ .a = fa, .b = fb, .rank = f.rank, .scale = f.exact_scale * set.strength }}, .n = t.n, .k = t.k, .base = base, .full = full, .torch = torch };
                try fp64(&job);

                try model.project(t.module, x.ptr, y.ptr, rows);
                const got = try widen(gpa, y, rows * t.n);
                defer gpa.free(got);
                row.parts = @max(row.parts, f.parts);
                try row.native.add(gpa, got, full, t.n);
                try row.torch.add(gpa, torch, full, t.n);
                const rounded = try gpa.alloc(f64, full.len);
                defer gpa.free(rounded);
                for (rounded, full) |*r, v| r.* = round16(v);
                try row.rounding.add(gpa, rounded, full, t.n);
                if (si == 0) {
                    // the same rows with no LoRA picked: the path main runs
                    try model.useLoras(io, set.dir, &.{});
                    try model.project(t.module, x.ptr, y.ptr, rows);
                    const plain = try widen(gpa, y, rows * t.n);
                    defer gpa.free(plain);
                    try base_stats[ci].add(gpa, plain, base, t.n);
                    try model.useLoras(io, set.dir, &.{.{ .name = set.name, .scale = set.strength }});
                }
            }
            if (!covered) continue;
            const bad = !within(row.native) or row.native.rel() > row.torch.rel() or row.native.worst_column > row.torch.worst_column;
            failed = failed or bad;
            try table.append(a, row);
        }
    }
    // the cancellation cases, from WORK_DIR: a fresh bank for that folder
    if (model.lora) |b| b.deinit();
    model.lora = null;
    var cancels: [2]Cancel = undefined;
    for (&cancels, [_]bool{ false, true }) |*c, f16_case| {
        c.* = try cancelCase(gpa, io, &driver, model, args[2], f16_case);
        // native must sit at FP64's rounding, and A rounded to bf16 must not, or the case no longer catches it
        failed = failed or !within(c.native) or within(c.rounded_a);
    }
    const scale_bad = try scaleCases(gpa, io, &driver, model, args[2], before.get(cancel_module).?);
    failed = failed or scale_bad > 0;
    // every LoRA run, then none picked: each projection must give main's bytes again
    try model.useLoras(io, args[2], &.{});
    var differ: usize = 0;
    for (classes) |c| for (c.modules) |m| {
        if (!std.mem.eql(u16, before.get(m).?, try projected(a, &driver, model, m))) {
            std.debug.print("{s}: no LoRA picked after LoRA requests differs from no LoRA bank at all\n", .{m});
            differ += 1;
        }
    };
    failed = failed or differ > 0;
    std.debug.print("{d} seeded bf16 rows per projection; error against FP64 x Wᵀ (+ bias) + scale (x Aᵀ) Bᵀ\n", .{rows});
    std.debug.print("{s:<20} {s:<16} {s:>5} {s:>11} {s:>11} {s:>11}   {s:>11} {s:>11} {s:>11}   {s:>6} {s:>6}   {s:>8} {s:>8}\n", .{ "class", "adapter", "A pcs", "rms native", "rms torch", "rms round", "col native", "col torch", "col round", "steps", "torch", "off", "off torch" });
    for (table.items) |r| {
        std.debug.print("{s:<20} {s:<16} {d:>5} {e:>11.3} {e:>11.3} {e:>11.3}   {e:>11.3} {e:>11.3} {e:>11.3}   {d:>6.3} {d:>6.3}   {d:>7.3}% {d:>7.3}%\n", .{ r.class, r.adapter, r.parts, r.native.rel(), r.torch.rel(), r.rounding.rel(), r.native.worst_column, r.torch.worst_column, r.rounding.worst_column, r.native.worst_steps, r.torch.worst_steps, 100 * @as(f64, @floatFromInt(r.native.off)) / @as(f64, @floatFromInt(r.native.count)), 100 * @as(f64, @floatFromInt(r.torch.off)) / @as(f64, @floatFromInt(r.torch.count)) });
    }
    std.debug.print("cancellation ({s}, rank {d}, rows cancelling bits bf16 can't hold): native against FP64 from the values written, and FP64 with A rounded to bf16\n", .{ cancel_module, cancel_rank });
    for (cancels) |c| {
        std.debug.print("{s:<20} A pieces {d}  native: rms {e:.3}  worst {d:.3} steps   A rounded to bf16: rms {e:.3}  worst {d:.3} steps{s}\n", .{ c.name, c.parts, c.native.rel(), c.native.worst_steps, c.rounded_a.rel(), c.rounded_a.worst_steps, if (std.math.isFinite(c.rounded_a.rel())) "" else " (not finite)" });
    }
    std.debug.print("no LoRA (main's path) against FP64 x Wᵀ (+ bias):\n", .{});
    for (classes, base_stats) |c, s| if (s.count > 0) {
        std.debug.print("{s:<20} rms {e:.3}  worst column {e:.3}  worst {d:.3} steps  off {d:.3}%\n", .{ c.name, s.rel(), s.worst_column, s.worst_steps, 100 * @as(f64, @floatFromInt(s.off)) / @as(f64, @floatFromInt(s.count)) });
    };
    std.debug.print("no LoRA picked after LoRA requests, against no LoRA bank: {d} of {d} projections differ in any byte\n", .{ differ, before.count() });
    std.debug.print("{s}\n", .{if (failed) "FAIL" else "pass: every LoRA output within 0.75 of a bf16 step of FP64, none worse than torch's order, the cancellation cases exact where rounding A to bf16 is not, main's bytes with none"});
    return if (failed) 1 else 0;
}
