//! EXL3 dot bodies checked against independent float64 tiles and the replaced reduction on identical half rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ex = tf.exl3_projection;
const c = contract.spec;
const g = contract.gates;
const Fixture = @import("exl3_projection_ref.zig").For(ex.format);

const Kind = enum { scalar, generic, static, lut, dense, t1, t2, t4, dense_t1, dense_t2, dense_t4, multi2, multi4, ku2 };
const Fault = enum(u32) { clip = 1, input_write = 2, tail = 3, half_sum = 4 };
const FaultArgs = extern struct { k: u32, n: u32, rows: u32, stress: u32, fault: u32, base: f32 };
const Geometry = struct { k: usize, n: usize, halves: u16 };
const geometries = [_]Geometry{
    .{ .k = 512, .n = 128, .halves = 16 },
    .{ .k = 1280, .n = 4096, .halves = 10 },
    .{ .k = 1280, .n = 32768, .halves = 8 },
    .{ .k = 1280, .n = 32768, .halves = 10 },
    .{ .k = 2304, .n = 5120, .halves = 4 },
    .{ .k = 2304, .n = 5120, .halves = 6 },
    .{ .k = 2304, .n = 5120, .halves = 8 },
    .{ .k = 2304, .n = 5120, .halves = 10 },
    .{ .k = 4096, .n = 1024, .halves = 8 },
    .{ .k = 4096, .n = 1024, .halves = 10 },
    .{ .k = 5120, .n = 512, .halves = 8 },
    .{ .k = 5120, .n = 512, .halves = 10 },
    .{ .k = 5120, .n = 512, .halves = 12 },
    .{ .k = 5120, .n = 1280, .halves = 8 },
    .{ .k = 5120, .n = 1280, .halves = 10 },
    .{ .k = 5120, .n = 1280, .halves = 12 },
    .{ .k = 5120, .n = 2304, .halves = 4 },
    .{ .k = 5120, .n = 2304, .halves = 6 },
    .{ .k = 5120, .n = 2304, .halves = 8 },
    .{ .k = 5120, .n = 2304, .halves = 10 },
    .{ .k = 5120, .n = 129280, .halves = 12 },
    .{ .k = 6144, .n = 25600, .halves = 8 },
    .{ .k = 6144, .n = 25600, .halves = 10 },
    .{ .k = 8192, .n = 5120, .halves = 8 },
    .{ .k = 8192, .n = 5120, .halves = 10 },
    .{ .k = 15360, .n = 5120, .halves = 8 },
};

fn routed(kind: Kind) bool {
    return switch (kind) {
        .generic, .static, .lut, .t1, .t2, .t4 => true,
        else => false,
    };
}

fn supports(kind: Kind, halves: u16, book: ex.format.Codebook) bool {
    if (kind == .ku2) return halves == 6;
    if (kind == .static or kind == .lut)
        return halves == 2 or halves == 4 or halves == 6 or halves == 8 or halves == 10 or halves == 12;
    return switch (kind) {
        .dense_t1, .dense_t2, .dense_t4 => book == .mul1,
        else => true,
    };
}

fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 21;
}

fn promptDepth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

const Probe = struct {
    suite: *Suite,
    rig: rig_.Rig,
    spec: c.Spec,
    kind: Kind,
    halves: u16,
    book: ex.format.Codebook,
    fault: ?Fault,
    f: ?Fixture = null,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    table: rig_.Ref = undefined,
    routes: rig_.Ref = undefined,
    descriptors: rig_.Ref = undefined,
    decoded: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    partials: rig_.Ref = undefined,
    args: ex.Args = undefined,
    why: [256]u8 = undefined,

    fn drop(p: *Probe) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
        p.rig.seen.deinit(p.rig.a);
        p.rig.seen = .empty;
        p.rig.outs.deinit(p.rig.a);
        p.rig.outs = .empty;
    }

    fn release(ctx: *anyopaque) void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.drop();
    }

    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        return p.rig.bytes(p.out, p.count() * 4);
    }

    const vt = blk: {
        var hooks = g.hooks(Probe);
        hooks.release = release;
        hooks.output = output;
        break :blk hooks;
    };

    fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = p.spec, .ctx = p, .vt = &vt, .control = if (p.fault) |fault| switch (fault) {
            .clip => .nonfinite,
            .input_write => .repeat,
            .tail => .shapes,
            .half_sum => .sums,
        } else null };
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const max_rows: usize = if (p.kind == .ku2) 16384 else if (routed(p.kind)) 64 else 8;
        if (shape.rows == 0 or shape.rows > max_rows or shape.k < 128 or shape.n < 128 or
            shape.k % 128 != 0 or shape.n % 128 != 0) return false;
        const f = try Fixture.init(p.rig.a, shape, p.halves, p.book, fill, seed);
        p.f = f;
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(std.mem.sliceAsBytes(f.words));
        if (p.kind != .scalar) {
            const table = p.suite.pipelines.codebook_table orelse return error.NoExl3CodebookTable;
            p.table = try p.rig.input(table.contents()[0 .. 3 * 65536 * 2]);
        }
        const ids = try p.rig.a.alloc(u32, shape.rows);
        defer p.rig.a.free(ids);
        @memset(ids, 0);
        p.routes = try p.rig.input(std.mem.sliceAsBytes(ids));
        const ks = shape.n / 16 * f.width.tileWords() * 2;
        const ns = f.width.tileWords() * 2;
        const desc = ex.ExpertDescriptor{
            .trellis = try ex.address(.{ .buffer = p.w.buf, .offset = p.w.off }),
            .suh = 0,
            .svh = 0,
            .bias = 0,
            .tile_k_stride = ks,
            .tile_n_stride = ns,
            .halves = p.halves,
            .codebook = @intCast(@backingInt(p.book)),
            .suh_signs = 0,
            .svh_signs = 0,
            .has_bias = 0,
            .reserved = @intCast(shape.n),
        };
        p.descriptors = try p.rig.input(std.mem.asBytes(&desc));
        p.out = try p.rig.output(shape.rows * shape.n * 4);
        p.base = try p.rig.scratch(shape.rows * shape.n * 4);
        if (p.kind == .scalar) p.partials = try p.rig.scratch(16 * shape.rows * shape.n * 4);
        if (p.fault != null) p.decoded = try p.rig.input(std.mem.sliceAsBytes(f.decoded));
        p.args = std.mem.zeroes(ex.Args);
        p.args.k = @intCast(shape.k);
        p.args.n = @intCast(shape.n);
        p.args.rows = @intCast(shape.rows);
        p.args.halves = p.halves;
        p.args.codebook = @intCast(@backingInt(p.book));
        p.args.output_f32 = 1;
        p.args.repeat = 1;
        p.args.indexed = @intFromBool(routed(p.kind));
        p.args.experts = 1;
        p.args.splits = 1;
        p.args.x_stride = shape.k;
        p.args.y_stride = shape.n;
        p.args.tile_k_stride = ks;
        p.args.tile_n_stride = ns;
        return true;
    }

    fn encode(p: *Probe, e: mtl.ComputeEncoder, pipeline: mtl.Pipeline, target: rig_.Ref, baseline: bool) !void {
        var args = p.args;
        if (p.kind == .scalar) {
            args.indexed = if (baseline) 0 else 1;
            args.splits = 16;
            e.setPipeline(pipeline);
            e.setBuffer(p.x.buf, p.x.off, 0);
            e.setBuffer(p.w.buf, p.w.off, 1);
            e.setBuffer(target.buf, target.off, 2);
            e.setValue(args, 3);
            e.setBuffer(p.routes.buf, p.routes.off, 4);
            e.setBuffer(p.descriptors.buf, p.descriptors.off, 5);
            e.setBuffer(p.partials.buf, p.partials.off, 6);
            e.useResource(p.w.buf, mtl.ResourceUsage.read);
            e.dispatchGroups(.{ .width = (args.n + 15) / 16, .height = 16, .depth = if (baseline) (args.rows + 7) / 8 else args.rows }, .{ .width = 32, .height = 1, .depth = 1 });
            e.setPipeline(p.suite.pipelines.reduce.?);
            e.setBuffer(p.partials.buf, p.partials.off, 0);
            e.setBuffer(target.buf, target.off, 1);
            e.setValue([2]u32{ 16, @intCast(p.count()) }, 2);
            e.dispatchThreads(.{ .width = p.count(), .height = 1, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
            return;
        }
        if (baseline and p.kind != .ku2) args.indexed = 1;
        const threads: usize = if (p.kind == .ku2) 256 else 32 * p.suite.pipelines.fma_warps;
        const cols: usize = if (baseline and p.kind != .ku2) 16 else switch (p.kind) {
            .t2, .dense_t2 => 32,
            .t4, .dense_t4, .ku2 => 64,
            else => 16,
        };
        const rows: usize = if (baseline and p.kind != .ku2) 1 else switch (p.kind) {
            .multi2 => 2,
            .multi4 => 4,
            .ku2 => 128,
            else => 1,
        };
        if (pipeline.simdWidth() != 32 or pipeline.maxThreads() < threads) return error.UnqualifiedExl3Launch;
        e.setPipeline(pipeline);
        e.setBuffer(p.x.buf, p.x.off, 0);
        e.setBuffer(p.w.buf, p.w.off, 1);
        e.setBuffer(target.buf, target.off, 2);
        e.setValue(args, 3);
        e.setBuffer(p.routes.buf, p.routes.off, 4);
        e.setBuffer(p.descriptors.buf, p.descriptors.off, 5);
        e.setBuffer(p.base.buf, p.base.off, 6);
        e.setBuffer(p.table.buf, p.table.off, 7);
        e.useResource(p.w.buf, mtl.ResourceUsage.read);
        e.dispatchGroups(.{ .width = (args.n + cols - 1) / cols, .height = (args.rows + rows - 1) / rows, .depth = 1 }, .{ .width = threads, .height = 1, .depth = 1 });
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        if (p.fault) |fault| {
            const f = p.f.?;
            const args = FaultArgs{ .k = p.args.k, .n = p.args.n, .rows = p.args.rows, .stress = @intFromBool(f.fill == .stress), .fault = @intCast(@backingInt(fault)), .base = ex.format.codebookValue(p.book, 0) };
            pass.e.setPipeline(p.suite.fault);
            pass.e.setBuffer(p.x.buf, p.x.off, 0);
            pass.e.setBuffer(p.decoded.buf, p.decoded.off, 1);
            pass.e.setBuffer(p.out.buf, p.out.off, 2);
            pass.e.setValue(args, 3);
            pass.e.dispatchThreads(.{ .width = p.count(), .height = 1, .depth = 1 }, .{ .width = 64, .height = 1, .depth = 1 });
        } else try p.encode(pass.e, p.suite.pipeline(p.kind), p.out, false);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.f.?.shape.rows * p.f.?.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.f.?.reference(want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }

    pub fn check(p: *Probe, at: g.Gate) !?[]const u8 {
        const n = p.count();
        const want = try p.rig.a.alloc(f64, n);
        defer p.rig.a.free(want);
        const mag = try p.rig.a.alloc(f64, n);
        defer p.rig.a.free(mag);
        try p.reference(want, mag);
        const actual = p.rig.bytes(p.out, n * 4);
        if (at == .nonfinite) {
            var planted: usize = 0;
            for (want, 0..) |w, i| {
                if (c.classOf(w) != .finite) planted += 1;
                if (c.classOf(c.read(.f32, actual, i)) != c.classOf(w)) return "output class differs from independently decoded float64";
            }
            if (planted == 0 or planted == n) return "nonfinite plants do not distinguish finite and nonfinite outputs";
            return null;
        }
        if (at != .shapes) return null;
        var old: ?[]const u8 = null;
        if (p.fault == null) {
            const pass = p.rig.begin();
            const baseline = if (p.kind == .scalar) p.suite.pipelines.matmul else if (p.kind == .ku2) p.suite.ku1.? else p.suite.original.?;
            try p.encode(pass.e, baseline, p.base, true);
            try p.rig.finish(pass);
            old = p.rig.bytes(p.base, n * 4);
        }
        var worst: f64 = 0;
        var prior: f64 = 0;
        var column: usize = 0;
        var squared: f64 = 0;
        var prior_squared: f64 = 0;
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(.f32, actual, i);
            const err = @abs(got - w);
            if (!(err <= c.bound(p.spec, p.f.?.shape, w, m)))
                return try std.fmt.bufPrint(&p.why, "output {d} leaves its independent float64 bound", .{i});
            if (err > worst) {
                worst = err;
                column = i % p.args.n;
            }
            squared += err * err;
            if (old) |bytes| {
                const prev = @abs(c.read(.f32, bytes, i) - w);
                prior = @max(prior, prev);
                prior_squared += prev * prev;
            }
        }
        if (old) |bytes| {
            if (!std.mem.eql(u8, actual, bytes)) return try std.fmt.bufPrint(&p.why, "bytes differ from the replaced reduction at {d}; FP64 worst {e} versus {e}", .{ std.mem.indexOfDiff(u8, actual, bytes).?, worst, prior });
        }
        std.debug.print("FP64 {s} {d}x{d}x{d}: worst={e} column={d} prior={e} squared={e} prior_squared={e}\n", .{ p.spec.name, p.args.rows, p.args.n, p.args.k, worst, column, prior, squared, prior_squared });
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    pipelines: ex.Pipelines,
    original: ?mtl.Pipeline,
    ku1: ?mtl.Pipeline,
    fault: mtl.Pipeline,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn matches(only: ?[]const u8) bool {
        const text = only orelse return true;
        var buffer: [128]u8 = undefined;
        for (std.enums.values(Kind)) |kind| {
            for ([_]u16{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 }) |halves| {
                for (std.enums.values(ex.format.Codebook)) |book| {
                    if ((halves % 2 != 0 and book != .mul1) or !supports(kind, halves, book)) continue;
                    const name = std.fmt.bufPrint(&buffer, "exl3-metal {t} {t} h{d}", .{ kind, book, halves }) catch unreachable;
                    if (std.mem.indexOf(u8, name, text) != null) return true;
                    for (std.enums.values(Fault)) |fault| {
                        const control = std.fmt.bufPrint(&buffer, "control: exl3-metal {t} {t} h{d} {t}", .{ kind, book, halves, fault }) catch unreachable;
                        if (std.mem.indexOf(u8, control, text) != null) return true;
                    }
                }
            }
        }
        return false;
    }

    fn pipeline(s: *const Suite, kind: Kind) mtl.Pipeline {
        return (switch (kind) {
            .scalar => s.pipelines.matmul,
            .generic => s.pipelines.decode_fma,
            .static => s.pipelines.decode_fma_static,
            .lut => s.pipelines.decode_fma_static_lut,
            .dense => s.pipelines.decode_fma_dense,
            .t1 => s.pipelines.decode_fma_t[0],
            .t2 => s.pipelines.decode_fma_t[1],
            .t4 => s.pipelines.decode_fma_t[2],
            .dense_t1 => s.pipelines.decode_fma_t_dense[0],
            .dense_t2 => s.pipelines.decode_fma_t_dense[1],
            .dense_t4 => s.pipelines.decode_fma_t_dense[2],
            .multi2 => s.pipelines.decode_fma_dense_multi2,
            .multi4 => s.pipelines.decode_fma_dense_multi4,
            .ku2 => s.pipelines.prefill_nax_dense_db,
        }) orelse unreachable;
    }

    fn add(s: *Suite, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), kind: Kind, halves: u16, book: ex.format.Codebook, fault: ?Fault, only: ?[]const u8) !void {
        const name = if (fault) |f| try std.fmt.allocPrint(s.a, "control: exl3-metal {t} {t} h{d} {t}", .{ kind, book, halves, f }) else try std.fmt.allocPrint(s.a, "exl3-metal {t} {t} h{d}", .{ kind, book, halves });
        errdefer s.a.free(name);
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) {
            s.a.free(name);
            return;
        };
        try s.probes.ensureUnusedCapacity(s.a, 1);
        try kernels.ensureUnusedCapacity(s.a, 1);
        var shapes: std.ArrayList(c.Shape) = .empty;
        errdefer shapes.deinit(s.a);
        if (fault != null) {
            try shapes.append(s.a, .{ .k = 8192, .n = 128, .rows = 8 });
        } else {
            const rows: []const usize = if (kind == .ku2) &.{ 64, 127, 128, 129, 1024, 4096, 8192, 16384 } else if (routed(kind)) &.{ 1, 2, 3, 4, 7, 8, 16, 32, 63, 64 } else &.{ 1, 2, 3, 4, 5, 6, 7, 8 };
            for (rows) |r| try shapes.append(s.a, .{ .k = 8192, .n = 128, .rows = r });
            for (geometries) |shape| if (shape.halves == halves) {
                for (rows) |r| {
                    if (kind == .ku2 and shape.n == 129280 and r > 1024) continue;
                    try shapes.append(s.a, .{ .k = shape.k, .n = shape.n, .rows = r });
                }
            };
        }
        const served = try shapes.toOwnedSlice(s.a);
        errdefer s.a.free(served);
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        p.* = .{
            .suite = s,
            .rig = .{ .a = s.a, .device = device, .queue = queue },
            .kind = kind,
            .halves = halves,
            .book = book,
            .fault = fault,
            .spec = .{
                .name = name,
                .accum = .f32,
                .depth = if (kind == .ku2 or kind == .scalar) promptDepth else depth,
                .out = .f32,
                .nan = .{ .stated = "finite, NaN and signed infinity classes match independently decoded float64 rows" },
                .tiles = .{ .{ .size = if (kind == .ku2) 128 else 4 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 128, .step = 128, .min = 128 } },
                .served = served,
            },
        };
        s.probes.appendAssumeCapacity(p);
        kernels.appendAssumeCapacity(p.kernel());
    }

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        if (device.tensorUnits() and ex.knobs.fma_warps != 16) return error.UnqualifiedExl3ContractWarpCount;
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        var pipelines = try ex.Pipelines.loadWith(device, a);
        errdefer pipelines.deinit();
        if (pipelines.reduce == null) return error.NoExl3ProjectionPipeline;
        var original: ?mtl.Pipeline = null;
        errdefer if (original) |p| p.deinit();
        var ku1: ?mtl.Pipeline = null;
        errdefer if (ku1) |p| p.deinit();
        if (device.tensorUnits()) {
            if (pipelines.decode_fma == null or pipelines.decode_fma_static == null or pipelines.decode_fma_static_lut == null or
                pipelines.decode_fma_dense == null or pipelines.decode_fma_dense_multi2 == null or pipelines.decode_fma_dense_multi4 == null or
                pipelines.prefill_nax_dense_db == null) return error.NoExl3ProjectionPipeline;
            for (pipelines.decode_fma_t) |p| if (p == null) return error.NoExl3ProjectionPipeline;
            for (pipelines.decode_fma_t_dense) |p| if (p == null) return error.NoExl3ProjectionPipeline;
            const src = try tf.frags.source(device, a, @import("kernel_sources").core_exl3_projection_nax);
            defer a.free(src);
            const original_src = try std.fmt.allocPrint(a, "#define TF_DECODE_TABLE 1\n#define TF_FMA_WARPS 16\n{s}\n{s}", .{ src, @embedFile("metal_exl3_projection_reference.metal") });
            defer a.free(original_src);
            original = try rig_.pipeline(device, original_src, "tf_exl3_contract_fma_reference");
            const ku1_src = try std.mem.replaceOwned(u8, a, src, "constexpr constant uint TF_KU = 2;", "constexpr constant uint TF_KU = 1;");
            defer a.free(ku1_src);
            ku1 = try rig_.pipeline(device, ku1_src, "tf_exl3_matmul_prefill_nax31");
        }
        const fault = try rig_.pipeline(device, @embedFile("metal_exl3_projection_controls.metal"), "exl3_projection_contract_fault");
        errdefer fault.deinit();
        s.* = .{ .a = a, .pipelines = pipelines, .original = original, .ku1 = ku1, .fault = fault };
        errdefer {
            for (s.probes.items) |p| {
                a.free(p.spec.name);
                a.free(p.spec.served);
                a.destroy(p);
            }
            s.probes.deinit(a);
        }
        for (std.enums.values(Kind)) |kind| {
            if (kind != .scalar and !device.tensorUnits()) continue;
            for ([_]u16{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 }) |halves| {
                for (std.enums.values(ex.format.Codebook)) |book| {
                    if ((halves % 2 != 0 and book != .mul1) or !supports(kind, halves, book)) continue;
                    try s.add(device, queue, kernels, kind, halves, book, null, only);
                    for (std.enums.values(Fault)) |f| try s.add(device, queue, kernels, kind, halves, book, f, only);
                }
            }
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.drop();
            p.rig.deinit();
            s.a.free(p.spec.name);
            s.a.free(p.spec.served);
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.fault.deinit();
        if (s.ku1) |p| p.deinit();
        if (s.original) |p| p.deinit();
        s.pipelines.deinit();
        s.a.destroy(s);
    }
};
