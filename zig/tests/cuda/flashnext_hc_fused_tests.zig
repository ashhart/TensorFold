//! BF16 writeback+norm against the original launches, with guarded synthetic/checkpoint weights and event timing.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const guarded = @import("contract_rig.zig");
const bf16 = @import("flashnext_rig.zig").bf16;
const fused = flash.hc_fused;
const weights = fused.weights;
const d = 2560;
const streams = 4;
const wide = d * streams;
const low = 320;
const ndn = low + streams;
const pss_n = d / 256 * streams;
const most = 4218;
const modes = .{ .sums, .moe, .slices };

const Context = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    glue: *const flash.Glue,
    lane: cuda.qmmf.Lane,
    kernels: fused.Kernels,
};

const Span = struct {
    base: u64,
    ptr: u64,
    pitch: usize,
    rows: usize,

    fn output(r: *guarded.Rig, rows: usize, pitch: usize) !Span {
        const base = try r.output((rows + 2) * pitch);
        return .{ .base = base, .ptr = base + pitch, .pitch = pitch, .rows = rows };
    }

    fn edge(self: Span, r: *guarded.Rig) !void {
        for (try r.bytes(self.base, self.pitch)) |b| try check.expect(b == 0xff, "HC prefix row unchanged", .{});
        for (try r.bytes(self.ptr + self.rows * self.pitch, self.pitch)) |b| try check.expect(b == 0xff, "HC suffix row unchanged", .{});
    }
};

const Chain = struct {
    h: Span,
    pss: Span,
    norm: Span,
    proj: Span,
    act: Span,
    xs: Span,
    inj: Span,
    up: Span,
    mixed: Span,
    xsm: Span,

    fn init(r: *guarded.Rig, rows: usize) !Chain {
        return .{
            .h = try Span.output(r, rows, wide * 2),
            .pss = try Span.output(r, rows, pss_n * 4),
            .norm = try Span.output(r, rows, wide * 2),
            .proj = try Span.output(r, rows, ndn * 4),
            .act = try Span.output(r, rows, low * 2),
            .xs = try Span.output(r, rows, low / 32 * 4),
            .inj = try Span.output(r, rows, streams * 2),
            .up = try Span.output(r, rows, wide * 2),
            .mixed = try Span.output(r, rows, d * 2),
            .xsm = try Span.output(r, rows, d / 32 * 4),
        };
    }

    fn edges(self: Chain, r: *guarded.Rig) !void {
        inline for (@typeInfo(Chain).@"struct".field_names) |name| try @field(self, name).edge(r);
    }
};

const Fixture = struct {
    ctx: Context,
    r: *guarded.Rig,
    rows: usize,
    initial: u64,
    gamma: u64,
    wb: fused.Writeback,
    down: flash.projection.Projection,
    up: flash.projection.Projection,
    part: u64,
    old: Chain,
    new: Chain,
    owns_weights: bool = true,

    fn init(ctx: Context, r: *guarded.Rig, a: std.mem.Allocator, rows: usize, seed: u64, mode: anytype) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Generate weights first: their bytes do not depend on the test's row count.
        const dw = try bfs(a, rng, ndn * wide, 0.02);
        const uw = try bfs(a, rng, wide * low, 0.05);
        var down = try flash.projection.dense(a, ctx.gpu.d, tensor(dw, ndn, wide));
        errdefer down.free();
        var up = try flash.projection.dense(a, ctx.gpu.d, tensor(uw, wide, low));
        errdefer up.free();
        const gamma = try floats(a, rng, wide, 0.5, 1.5);
        const h = try bfs(a, rng, rows * wide, 1);
        const inj = try bfs(a, rng, (rows + 2) * streams, 2);
        if (rows > 0) {
            h[0] = 0x3b34;
            inj[streams] = 0x3c9a;
        }
        var wb: fused.Writeback = .{
            .h = 0,
            .pss = 0,
            .mode = mode,
            .inj = try r.input(std.mem.sliceAsBytes(inj)) + streams * 2,
            .stride = (rows + 2) * d,
            .slices = 8,
        };
        switch (wb.mode) {
            .sums => {},
            .slices => {
                const branch = try floats(a, rng, 8 * wb.stride, -0.25, 0.25);
                for (0..8) |sk| branch[sk * wb.stride + d] = if (sk == 0) @bitCast(@as(u32, 0xbe16) << 16) else 0;
                wb.branch = try r.input(std.mem.sliceAsBytes(branch)) + d * 4;
            },
            .moe => {
                const y = try floats(a, rng, (rows + 2) * 11 * d, -0.25, 0.25);
                const wts = try floats(a, rng, (rows + 2) * 11, -0.5, 0.5);
                for (0..11) |k| y[(11 + k) * d] = 0;
                y[11 * d] = @bitCast(@as(u32, 0xbe16) << 16);
                wts[11] = 1;
                wb.y = try r.input(std.mem.sliceAsBytes(y)) + 11 * d * 4;
                wb.wts = try r.input(std.mem.sliceAsBytes(wts)) + 11 * 4;
            },
        }
        return .{
            .ctx = ctx,
            .r = r,
            .rows = rows,
            .initial = try r.input(std.mem.sliceAsBytes(h)),
            .gamma = try r.input(std.mem.sliceAsBytes(gamma)),
            .wb = wb,
            .down = down,
            .up = up,
            .part = try r.scratch(8 * rows * ndn * 4),
            .old = try Chain.init(r, rows),
            .new = try Chain.init(r, rows),
        };
    }

    fn deinit(self: *Fixture) void {
        self.r.s.synchronize() catch {};
        if (self.owns_weights) {
            self.up.free();
            self.down.free();
        }
    }

    fn useWeights(self: *Fixture, hc: *const weights.HcW) !void {
        try check.expect(hc.down.native != null and hc.up.native != null, "checkpoint HC weights are BF16", .{});
        try check.expect(hc.down.native.?.mode == .bf16 and hc.up.native.?.mode == .bf16, "checkpoint HC projections use BF16 mode", .{});
        self.down.free();
        self.up.free();
        self.down = hc.down;
        self.up = hc.up;
        self.gamma = hc.scale.ptr;
        self.owns_weights = false;
    }

    fn reset(self: *Fixture, chain: Chain) !void {
        try self.ctx.gpu.d.check(self.ctx.gpu.d.api.cuMemcpyDtoDAsync_v2(chain.h.ptr, self.initial, self.rows * wide * 2, self.ctx.s.handle), "HC reset");
    }

    fn write(self: *Fixture, chain: Chain, new: bool) !void {
        var wb = self.wb;
        wb.h = chain.h.ptr;
        wb.pss = chain.pss.ptr;
        if (new) return self.ctx.kernels.writeback(self.ctx.s, wb, chain.norm.ptr, self.gamma, self.rows);
        try self.ctx.glue.hcWriteback(self.ctx.s, wb.h, wb.h, wb.pss, wb.branch, wb.inj, wb.y, wb.wts, wb.stride, self.rows, d, streams, @backingInt(wb.mode), 10, 11, wb.slices);
        try self.ctx.glue.hcNormalize(self.ctx.s, wb.h, wb.pss, self.gamma, chain.norm.ptr, 1e-6, self.rows, d, streams);
    }

    fn front(self: *Fixture, chain: Chain) !void {
        try self.ctx.lane.matmulF32(self.ctx.s, chain.norm.ptr, wide, self.rows, self.down.native.?, chain.proj.ptr, self.part);
        try self.ctx.glue.hcReduceAct(self.ctx.s, chain.proj.ptr, chain.act.ptr, chain.xs.ptr, chain.inj.ptr, 1, self.rows, streams, low, ndn, 1);
    }

    fn mix(self: *Fixture, chain: Chain, new: bool) !void {
        try self.ctx.lane.matmul(self.ctx.s, chain.act.ptr, low, self.rows, self.up.native.?, chain.up.ptr, self.part);
        if (new and self.rows >= 512) {
            try self.ctx.kernels.mix(self.ctx.s, chain.up.ptr, chain.norm.ptr, chain.mixed.ptr, chain.xsm.ptr, self.rows);
        } else try self.ctx.glue.hcDenseMix(self.ctx.s, chain.up.ptr, chain.norm.ptr, chain.mixed.ptr, chain.xsm.ptr, self.rows, d, streams);
    }

    fn runChain(self: *Fixture, out: Chain, new: bool) !void {
        try self.write(out, new);
        try self.front(out);
        try self.mix(out, new);
    }

    fn capturedStreams(self: *Fixture) !void {
        const a = self.ctx.gpu.gpa;
        const branch = try a.alloc(f32, self.rows * d);
        defer a.free(branch);
        @memset(branch, 0.25);
        const branch32 = try self.r.input(std.mem.sliceAsBytes(branch));
        const gates = try a.alloc(u16, self.rows * streams);
        defer a.free(gates);
        @memset(gates, bf16(1));
        const inj = try self.r.input(std.mem.sliceAsBytes(gates));
        const old = try Span.output(self.r, self.rows, wide * 2);
        const new = try Span.output(self.r, self.rows, wide * 2);
        for ([_]Chain{ self.old, self.new }, [_]Span{ old, new }) |chain, captured| {
            const before = try a.dupe(u8, try self.r.bytes(chain.h.ptr, wide * 2));
            defer a.free(before);
            var scratch: flash.batch_state.Scratch = undefined;
            scratch.h = chain.h.ptr;
            scratch.pss = chain.pss.ptr;
            scratch.inj = inj;
            scratch.moe_f32 = branch32;
            scratch.dummy = 0;
            var pass: flash.prompt.batch.Pass(flash.Tri) = .{
                .gpa = a,
                .io = self.ctx.gpu.io,
                .stream = self.ctx.s,
                .tri = .{ .own = self.ctx.glue, .s = self.ctx.s },
                .k = undefined,
                .s = &scratch,
                .m = undefined,
                .seq = undefined,
                .hooks = .{ .logits = .none, .streams = captured.ptr },
                .rows = self.rows,
                .pending = 0,
                .pending_f32 = true,
            };
            try pass.head(null);
            try check.expect(pass.pending == null and !pass.pending_f32, "MTP capture consumes the final pending branch", .{});
            try same(self.r, "MTP captures all streams after the final writeback", captured.ptr, chain.h.ptr, self.rows * wide * 2);
            try check.expect(std.mem.indexOfDiff(u8, before, try self.r.bytes(captured.ptr, wide * 2)) != null, "MTP capture witness changes H at the final writeback", .{});
            try captured.edge(self.r);
        }
        try same(self.r, "MTP captured streams: fused equals separate", old.ptr, new.ptr, self.rows * wide * 2);
        try self.compare();
    }

    fn compare(self: *Fixture) !void {
        inline for (.{ "h", "pss", "norm", "proj", "act", "xs", "inj", "up", "mixed", "xsm" }) |name| {
            const old: Span = @field(self.old, name);
            const new: Span = @field(self.new, name);
            try same(self.r, "HC fused " ++ name, old.ptr, new.ptr, old.rows * old.pitch);
        }
        try self.old.edges(self.r);
        try self.new.edges(self.r);
        try check.expect(try self.r.intact(), "HC allocation guards intact", .{});
    }

    fn oneRows(self: *Fixture) !void {
        // A seam witness for every row, including row-tile tails and nonzero row addresses.
        for (0..self.rows) |row| {
            var solo = self.old;
            inline for (@typeInfo(Chain).@"struct".field_names) |name| {
                @field(solo, name).ptr += row * @field(solo, name).pitch;
            }
            var wb = self.wb;
            wb.h = solo.h.ptr;
            wb.pss = solo.pss.ptr;
            wb.inj += row * streams * 2;
            wb.branch += row * d * 4;
            wb.y += row * 11 * d * 4;
            wb.wts += row * 11 * 4;
            try self.ctx.gpu.d.check(self.ctx.gpu.d.api.cuMemcpyDtoDAsync_v2(solo.h.ptr, self.initial + row * wide * 2, wide * 2, self.ctx.s.handle), "HC solo reset");
            try self.ctx.glue.hcWriteback(self.ctx.s, wb.h, wb.h, wb.pss, wb.branch, wb.inj, wb.y, wb.wts, wb.stride, 1, d, streams, @backingInt(wb.mode), 10, 11, wb.slices);
            try self.ctx.glue.hcNormalize(self.ctx.s, wb.h, wb.pss, self.gamma, solo.norm.ptr, 1e-6, 1, d, streams);
            try self.ctx.lane.matmulF32(self.ctx.s, solo.norm.ptr, wide, 1, self.down.native.?, solo.proj.ptr, self.part);
            try self.ctx.glue.hcReduceAct(self.ctx.s, solo.proj.ptr, solo.act.ptr, solo.xs.ptr, solo.inj.ptr, 1, 1, streams, low, ndn, 1);
            try self.ctx.lane.matmul(self.ctx.s, solo.act.ptr, low, 1, self.up.native.?, solo.up.ptr, self.part);
            try self.ctx.glue.hcDenseMix(self.ctx.s, solo.up.ptr, solo.norm.ptr, solo.mixed.ptr, solo.xsm.ptr, 1, d, streams);
        }
        try self.compare();
    }
};

fn bfs(a: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*v| v.* = bf16((rng.float(f32) * 2 - 1) * scale);
    return out;
}

fn floats(a: std.mem.Allocator, rng: std.Random, n: usize, lo: f32, hi: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*v| v.* = lo + (hi - lo) * rng.float(f32);
    return out;
}

fn tensor(values: []const u16, n: usize, k: usize) core.safetensors.Tensor {
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(values) };
}

fn same(r: *guarded.Rig, what: []const u8, old: u64, new: u64, n: usize) !void {
    const bytes = try r.gpu.gpa.dupe(u8, try r.bytes(new, n));
    defer r.gpu.gpa.free(bytes);
    try check.sameBytes(what, bytes, try r.bytes(old, n));
}

pub fn run(gpu: check.Gpu) !void {
    return driver(gpu, null, false, most);
}

pub fn runCheckpoint(gpu: check.Gpu, model: []const u8) !void {
    return driver(gpu, model, false, most);
}

pub fn benchRows(value: ?[]const u8) !usize {
    const text = value orelse return most;
    const rows = std.fmt.parseInt(usize, text, 10) catch return error.BadHcBenchRows;
    if (rows == 0 or rows > flash.batch_state.max_chunk) return error.BadHcBenchRows;
    return rows;
}

pub fn bench(gpu: check.Gpu, rows: usize) !void {
    return driver(gpu, null, true, rows);
}

pub fn benchCheckpoint(gpu: check.Gpu, model: []const u8, rows: usize) !void {
    return driver(gpu, model, true, rows);
}

fn driver(gpu: check.Gpu, model: ?[]const u8, timing: bool, bench_rows: usize) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    const ctx: Context = .{ .gpu = gpu, .s = stream, .glue = &glue, .lane = lane, .kernels = try fused.Kernels.resolve(glue.hc) };
    var store: ?weights.Store = if (model) |path| try weights.Store.open(gpu.gpa, gpu.io, path) else null;
    defer if (store) |*s| s.deinit();
    if (store) |*s| try s.detectNorms();
    for ([_][]const u8{ "attn_hc", "mlp_hc" }, 0..) |label, seed| {
        var loaded: ?weights.HcW = if (store) |*s| blk: {
            var buf: [160]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&buf, "language_model.model.layers.0.{s}_hyper_connection", .{if (seed == 0) "attn" else "mlp"});
            break :blk try weights.loadHc(gpu.gpa, @constCast(gpu.d), s, prefix, true);
        } else null;
        defer if (loaded) |*hc| hc.free();
        const actual: ?*const weights.HcW = if (loaded) |*hc| hc else null;
        if (timing) {
            inline for (modes) |mode| try benchOne(ctx, label, seed, actual, bench_rows, mode);
        } else {
            for ([_]usize{ 1, 16, 64, 65, 1536, most }) |rows| inline for (modes) |mode| {
                try testOne(ctx, label, seed, rows, mode, actual);
            };
        }
    }
}

fn testOne(ctx: Context, label: []const u8, seed: usize, rows: usize, mode: anytype, actual: ?*const weights.HcW) !void {
    var arena = std.heap.ArenaAllocator.init(ctx.gpu.gpa);
    defer arena.deinit();
    var r: guarded.Rig = .{ .gpu = ctx.gpu, .s = ctx.s };
    defer r.deinit();
    var fixture = try Fixture.init(ctx, &r, arena.allocator(), rows, 0x48434600 + seed, mode);
    defer fixture.deinit();
    if (actual) |hc| try fixture.useWeights(hc);
    try fixture.reset(fixture.old);
    try fixture.reset(fixture.new);
    try fixture.runChain(fixture.old, false);
    try fixture.runChain(fixture.new, true);
    try fixture.compare();
    // Repeat with the same inputs, retaining the original old-chain byte oracle.
    try fixture.reset(fixture.new);
    try fixture.runChain(fixture.new, true);
    try fixture.compare();
    try fixture.oneRows();
    try fixture.capturedStreams();
    try check.expect(try r.intact(), "HC guards after solo rows", .{});
    check.pass("HC fused {s} {s}, M={d}, mode={s}: old/new, repeat, solo rows and MTP captured streams equal", .{ if (actual != null) "checkpoint layer0" else "synthetic", label, rows, @tagName(fixture.wb.mode) });
}

fn time(f: *Fixture, new: bool, begin: cuda.Event, end: cuda.Event) !f64 {
    const chain = f.old;
    try f.reset(chain);
    // Stream ordering keeps the reset and input preparation outside the measured event interval.
    try begin.record(f.ctx.s);
    try f.write(chain, new);
    try end.record(f.ctx.s);
    try end.synchronize();
    return @as(f64, try cuda.Event.elapsedMs(begin, end));
}

fn benchOne(ctx: Context, label: []const u8, seed: usize, actual: ?*const weights.HcW, rows: usize, mode: anytype) !void {
    var begin = try cuda.Event.init(ctx.gpu.d, true);
    defer begin.deinit();
    var end = try cuda.Event.init(ctx.gpu.d, true);
    defer end.deinit();
    var arena = std.heap.ArenaAllocator.init(ctx.gpu.gpa);
    defer arena.deinit();
    var r: guarded.Rig = .{ .gpu = ctx.gpu, .s = ctx.s };
    defer r.deinit();
    var fixture = try Fixture.init(ctx, &r, arena.allocator(), rows, 0x48434600 + seed, mode);
    defer fixture.deinit();
    if (actual) |hc| try fixture.useWeights(hc);
    fixture.wb.slices = if (mode == .slices) 2 else 1;
    if (mode == .slices) {
        fixture.wb.stride = rows * d;
        const zero: cuda.DeviceBuffer = .{ .d = ctx.gpu.d, .ptr = fixture.wb.branch + fixture.wb.stride * 4, .len = rows * d * 4 };
        try zero.fill8(0, ctx.s.handle);
    }
    try fixture.reset(fixture.old);
    try fixture.reset(fixture.new);
    try fixture.runChain(fixture.old, false);
    try fixture.runChain(fixture.new, true);
    try fixture.compare();
    {
        var ms: [2][9]f64 = undefined;
        for (0..11) |round| {
            for (0..2) |order| {
                const which = (round + order) % 2;
                const elapsed = try time(&fixture, which == 1, begin, end);
                if (round >= 2) ms[which][round - 2] = elapsed;
            }
        }
        const old = check.median(&ms[0]);
        const new = check.median(&ms[1]);
        std.debug.print("BENCH HC {s} {s} M={d} mode={s} slices={d} writeback_norm: old={d:.4} new={d:.4} saved={d:.4} ms\n", .{ if (actual != null) "checkpoint layer0" else "synthetic", label, rows, @tagName(fixture.wb.mode), fixture.wb.slices, old, new, old - new });
    }
    try fixture.reset(fixture.old);
    try fixture.reset(fixture.new);
    try fixture.runChain(fixture.old, false);
    try fixture.runChain(fixture.new, true);
    try fixture.compare();
}
