//! GLM-5.3-Flash's eh_proj on the tensor units (glm_rows_bf16_nax) under the contract.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const tf = @import("tensorfold");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;

fn convertRow(d: Dense, slice: usize, row: usize, values: []f64) void {
    for (values, slice * (d.shape.k / d.slices)..) |*v, i| v.* = c.fromBf16(d.x[row * d.shape.k + i]);
}

/// One converted input row per reference call; products and both FP64 reduction orders stay unchanged.
pub const RowReferenceCache = struct {
    values: ?[]f64 = null,
    row_index: ?usize = null,
    slice_index: ?usize = null,

    pub fn invalidate(self: *RowReferenceCache) void {
        self.row_index = null;
        self.slice_index = null;
    }

    pub fn deinit(self: *RowReferenceCache, a: std.mem.Allocator) void {
        if (self.values) |values| a.free(values);
        self.* = .{};
    }

    fn load(self: *RowReferenceCache, a: std.mem.Allocator, d: Dense, slice: usize, row: usize) ![]const f64 {
        const per = d.shape.k / d.slices;
        if (self.values == null or self.values.?.len != per) {
            if (self.values) |values| a.free(values);
            self.values = null;
            self.invalidate();
            self.values = try a.alloc(f64, per);
        }
        if (self.row_index != row or self.slice_index != slice) {
            convertRow(d, slice, row, self.values.?);
            self.row_index = row;
            self.slice_index = slice;
        }
        return self.values.?;
    }

    pub fn reference(self: *RowReferenceCache, a: std.mem.Allocator, d: Dense, want: []f64, mag: []f64) !void {
        self.invalidate();
        if (d.shape.rows > 1) return parallelReference(a, d, want, mag, 8, .normal);
        return self.referenceSerial(a, d, want, mag);
    }

    pub fn referenceSerial(self: *RowReferenceCache, a: std.mem.Allocator, d: Dense, want: []f64, mag: []f64) !void {
        self.invalidate();
        const per = d.shape.k / d.slices;
        const terms = try a.alloc(f64, per);
        defer a.free(terms);
        for (0..d.slices) |slice| for (0..d.shape.rows) |row| {
            const x = try self.load(a, d, slice, row);
            for (0..d.shape.n) |column| {
                for (terms, x, slice * per..) |*term, value, i| term.* = value * d.w[column * d.shape.k + i];
                var sum: f64 = 0;
                var magnitude: f64 = 0;
                for (terms) |term| {
                    sum += term;
                    magnitude += @abs(term);
                }
                const at = (slice * d.shape.rows + row) * d.shape.n + column;
                want[at] = sum;
                mag[at] = magnitude;
            }
        };
    }
};

const ReferenceJob = struct {
    d: Dense,
    first: usize,
    end: usize,
    converted: []f64,
    terms: []f64,
    want: []f64,
    mag: []f64,
    fn run(job: *@This()) void {
        const d = job.d;
        const per = d.shape.k / d.slices;
        for (0..d.slices) |slice| for (job.first..job.end) |row| {
            // The row conversion remains outside the output-column loop.
            convertRow(d, slice, row, job.converted);
            for (0..d.shape.n) |column| {
                for (job.terms, job.converted, slice * per..) |*term, value, i|
                    term.* = value * d.w[column * d.shape.k + i];
                var sum: f64 = 0;
                var magnitude: f64 = 0;
                for (job.terms) |term| {
                    sum += term;
                    magnitude += @abs(term);
                }
                const at = (slice * d.shape.rows + row) * d.shape.n + column;
                job.want[at] = sum;
                job.mag[at] = magnitude;
            }
        };
    }
};
const SpawnMode = enum { normal, partial_failure, all_failure };

fn parallelReference(a: std.mem.Allocator, d: Dense, want: []f64, mag: []f64, requested: usize, mode: SpawnMode) !void {
    const count = @min(@min(@max(requested, 1), 8), d.shape.rows);
    if (count == 0) return;
    const per = d.shape.k / d.slices;
    const scratch = try a.alloc(f64, count * per * 2);
    defer a.free(scratch);
    var jobs: [8]ReferenceJob = undefined;
    var threads: [8]?std.Thread = @splat(null);
    for (0..count) |worker| {
        jobs[worker] = .{
            .d = d,
            .first = worker * d.shape.rows / count,
            .end = (worker + 1) * d.shape.rows / count,
            .converted = scratch[worker * per * 2 ..][0..per],
            .terms = scratch[worker * per * 2 + per ..][0..per],
            .want = want,
            .mag = mag,
        };
    }
    // Thread creation failure changes scheduling only; already-started jobs still join.
    for (1..count) |worker| {
        threads[worker] = (if (mode == .all_failure or (mode == .partial_failure and worker % 2 == 0))
            error.ReferenceThreadUnavailable
        else
            std.Thread.spawn(.{}, ReferenceJob.run, .{&jobs[worker]})) catch null;
        if (threads[worker] == null) jobs[worker].run();
    }
    jobs[0].run();
    for (threads[0..count]) |thread| if (thread) |t| t.join();
}

/// One 8-wide MMA step a k-tile (products round entering the fp32 dot); tail guard: k/8 steps at k = 8192.
fn depth(s: c.Shape) usize {
    return s.k / 8 + 8;
}

/// The engine's prompt chunks: full heights (16384 on the largest Macs) + tail (absorb = remaining - 1, any count).
const served = [_]c.Shape{
    .{ .rows = 8, .n = 4096, .k = 8192 },
    .{ .rows = 2048, .n = 4096, .k = 8192 },
    .{ .rows = 4096, .n = 4096, .k = 8192 },
    .{ .rows = 6144, .n = 4096, .k = 8192 },
    .{ .rows = 8192, .n = 4096, .k = 8192 },
    .{ .rows = 16384, .n = 4096, .k = 8192 },
    .{ .rows = 1, .n = 4096, .k = 8192 },
};

pub fn spec() c.Spec {
    return .{
        .name = "metal glm eh_proj tensor tile glm_rows_bf16_nax",
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        // Rows: the launcher's ceil-div grid takes any positive count; n moves in 64-column tiles; k untiled.
        .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 64, .step = 64, .min = 64 }, .{} },
        .served = &served,
    };
}

/// GlmRowDims as glm_rows.metal binds it at buffer 4.
const RowDims = extern struct {
    k: i32,
    n: i32,
    group: i32,
    rows: i32,
    heads: i32 = 1,
    ldx: i32,
    hsx: i32 = 0,
    ldo: i32,
    hso: i32 = 0,
    hsw: i32 = 0,
};

pub const Nax = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    d: ?Dense = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    reference_cache: RowReferenceCache = .{},

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Nax {
        // The engine's compilation unit: wide.zig's gather_source (projection, glue math, rows, gather).
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipe = try rig_.pipeline(device, tf.glm.wide.gather_source, "glm_rows_bf16_nax"),
        };
    }

    pub fn deinit(self: *Nax) void {
        self.reference_cache.deinit(self.a);
        if (self.d) |d| d.deinit();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn kernel(self: *Nax) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Nax);

    pub fn prepare(self: *Nax, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.reference_cache.invalidate();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % 64 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const d = try Dense.init(self.a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        // W as bf16 bits in [-1, 1) or gate 4's mirrored integers; mirrored stress halves cancel (partials under 2^24).
        const bits = try self.a.alloc(u16, shape.n * shape.k);
        defer self.a.free(bits);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6d);
        const rng = prng.random();
        const k = shape.k;
        for (bits) |*b| b.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        if (fill == .stress) for (0..shape.n) |j| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
            bits[j * k + i] = c.toBf16(v);
            bits[j * k + k / 2 + i] = c.toBf16(-v);
        };
        for (d.w, bits) |*wv, b| wv.* = c.fromBf16(b);
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(d.outputs() * c.size(spec().out));
        return true;
    }

    pub fn run(self: *Nax) !void {
        const d = self.d.?;
        const p = self.rig.begin();
        const e = p.e;
        e.setPipeline(self.pipe);
        e.setBuffer(self.x.buf, self.x.off, 0);
        e.setBuffer(self.w.buf, self.w.off, 1);
        e.setValue(RowDims{
            .k = @intCast(d.shape.k),
            .n = @intCast(d.shape.n),
            .group = 0,
            .rows = @intCast(d.shape.rows),
            .ldx = @intCast(d.shape.k),
            .ldo = @intCast(d.shape.n),
        }, 4);
        e.setBuffer(self.out.buf, self.out.off, 5);
        e.dispatchGroups(
            mtl.Size.of(@intCast((d.shape.n + 63) / 64), @intCast((d.shape.rows + 7) / 8), 1),
            mtl.Size.of(8 * 32, 1, 1),
        );
        try self.rig.finish(p);
    }

    pub fn count(self: *Nax) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Nax, want: []f64, mag: []f64) !void {
        return self.reference_cache.reference(self.a, self.d.?, want, mag);
    }

    pub fn referenceUncached(self: *Nax, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Nax, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};

fn checkReferenceCache(cache: *RowReferenceCache, a: std.mem.Allocator, d: Dense) !void {
    const n = d.outputs();
    const buffers = try a.alloc(f64, n * 4);
    defer a.free(buffers);
    const want = buffers[0..n];
    const mag = buffers[n .. 2 * n];
    const cached_want = buffers[2 * n .. 3 * n];
    const cached_mag = buffers[3 * n ..];
    try d.reference(want, mag);
    try cache.reference(a, d, cached_want, cached_mag);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(cached_want));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mag), std.mem.sliceAsBytes(cached_mag));
}

test "GLM eh reference cache preserves FP64 bytes in changing shape and fill orders" {
    const a = std.testing.allocator;
    var cache: RowReferenceCache = .{};
    defer cache.deinit(a);
    const shapes = [_]c.Shape{
        .{ .rows = 1, .n = 3, .k = 32 },
        .{ .rows = 5, .n = 64, .k = 128 },
        .{ .rows = 3, .n = 8, .k = 512 },
        .{ .rows = 4, .n = 64, .k = 8192 },
    };
    for ([_]usize{ 3, 0, 2, 1, 0, 3 }, 0..) |index, turn| {
        for ([_]g.Fill{ .random, .nonfinite, .stress }) |fill| {
            const d = try Dense.init(a, shapes[index], if (index == 2) 2 else 1);
            defer d.deinit();
            const seed = 0x6d + turn;
            d.fillX(fill, seed, 2);
            var random = std.Random.DefaultPrng.init(seed ^ 0x6d);
            const rng = random.random();
            for (d.w) |*weight| weight.* = rng.float(f64) * 2 - 1;
            if (fill == .stress) for (0..d.shape.n) |column| for (0..d.shape.k / 2) |i| {
                const weight: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                d.w[column * d.shape.k + i] = weight;
                d.w[column * d.shape.k + d.shape.k / 2 + i] = -weight;
            };
            try checkReferenceCache(&cache, a, d);
            cache.invalidate();
            try checkReferenceCache(&cache, a, d);
        }
    }
}

test "GLM eh reference cache invalidates a one-row fixture after input mutation" {
    const a = std.testing.allocator;
    var cache: RowReferenceCache = .{};
    defer cache.deinit(a);
    const d = try Dense.init(a, .{ .rows = 1, .n = 2, .k = 64 }, 1);
    defer d.deinit();
    @memset(d.x, c.toBf16(1));
    @memset(d.w, 1);
    var want: [2]f64 = undefined;
    var mag: [2]f64 = undefined;
    try cache.reference(a, d, &want, &mag);
    try std.testing.expectEqual(@as(f64, 64), want[0]);
    d.x[0] = c.toBf16(8);
    try cache.reference(a, d, &want, &mag);
    try std.testing.expectEqual(@as(f64, 71), want[0]);
    try checkReferenceCache(&cache, a, d);
}

pub fn checkParallelReference() !void {
    const a = std.testing.allocator;
    var cache: RowReferenceCache = .{};
    defer cache.deinit(a);
    const shapes = [_]c.Shape{
        .{ .rows = 1, .n = 3, .k = 32 },
        .{ .rows = 3, .n = 8, .k = 512 },
        .{ .rows = 9, .n = 8, .k = 8192 },
    };
    for ([_]usize{ 2, 0, 1, 2 }, 0..) |index, turn| {
        const d = try Dense.init(a, shapes[index], if (index == 1) 2 else 1);
        defer d.deinit();
        const n = d.outputs();
        const values = try a.alloc(f64, 4 * n);
        defer a.free(values);
        const want = values[0..n];
        const mag = values[n .. 2 * n];
        const got = values[2 * n .. 3 * n];
        const got_mag = values[3 * n ..];
        for ([_]g.Fill{ .random, .stress, .nonfinite }) |fill| {
            d.fillX(fill, 517 + turn, 2);
            var random = std.Random.DefaultPrng.init(517 ^ turn);
            const rng = random.random();
            for (d.w) |*weight| weight.* = rng.float(f64) * 2 - 1;
            if (fill == .stress) for (0..d.shape.n) |column| for (0..d.shape.k / 2) |i| {
                const weight: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                d.w[column * d.shape.k + i] = weight;
                d.w[column * d.shape.k + d.shape.k / 2 + i] = -weight;
            };
            for (0..2) |mutation| {
                if (mutation == 1) d.x[0] = c.toBf16(7);
                try cache.referenceSerial(a, d, want, mag);
                for ([_]usize{ 1, 2, 4, 8 }) |workers| {
                    for ([_]SpawnMode{ .normal, .partial_failure, .all_failure }) |mode| {
                        @memset(got, -517);
                        @memset(got_mag, -517);
                        try parallelReference(a, d, got, got_mag, workers, mode);
                        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
                        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mag), std.mem.sliceAsBytes(got_mag));
                    }
                }
            }
        }
    }
    const d = try Dense.init(a, .{ .rows = 3, .n = 8, .k = 512 }, 1);
    defer d.deinit();
    var want: [24]f64 = @splat(-517);
    var mag: [24]f64 = @splat(-519);
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, parallelReference(failing.allocator(), d, &want, &mag, 8, .normal));
    for (want) |value| try std.testing.expectEqual(@as(f64, -517), value);
    for (mag) |value| try std.testing.expectEqual(@as(f64, -519), value);
}

test "GLM eh parallel reference preserves FP64 bytes and finishes thread failure partitions" {
    try checkParallelReference();
}
