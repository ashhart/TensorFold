//! GLM MTP prep and 4-bit/64 greedy head+top1 kernels under gates 1-4 and planted controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const glm_k = tf.glm.kernels;
const MtpPrepArgs = glm_k.MtpPrepArgs;
const HeadTop1Args = glm_k.HeadTop1Args;

pub const Kind = enum { prep4, prep8, head };
const Variant = struct { kind: Kind, ctl: u8 = 0 };

const prep_served = [_]c.Shape{
    .{ .rows = 4, .n = 1024, .k = 512 },
    .{ .rows = 1, .n = 8192, .k = 4096 },
    .{ .rows = 8, .n = 2048, .k = 1024 },
};
const head_served = [_]c.Shape{
    .{ .rows = 1, .n = 256, .k = 512 },
    .{ .rows = 1, .n = 154880, .k = 4096 },
    .{ .rows = 1, .n = 512, .k = 1024 },
};

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn entryName(v: Variant) []const u8 {
    return switch (v.ctl) {
        0 => switch (v.kind) {
            .prep4 => "glm mtp glm_mtp_prep_4b",
            .prep8 => "glm mtp glm_mtp_prep_8b",
            .head => "glm mtp glm_head_top1_4b_one",
        },
        1 => switch (v.kind) {
            .prep4 => "control: glm mtp glm_mtp_prep_4b writes into H",
            .prep8 => "control: glm mtp glm_mtp_prep_8b writes into H",
            .head => "control: glm mtp glm_head_top1_4b_one writes into X",
        },
        else => switch (v.kind) {
            .prep4 => "control: glm mtp glm_mtp_prep_4b drops tail cols",
            .prep8 => "control: glm mtp glm_mtp_prep_8b drops tail cols",
            .head => "control: glm mtp glm_head_top1_4b_one drops tail vocab",
        },
    };
}

fn spec(v: Variant) c.Spec {
    return switch (v.kind) {
        .prep4, .prep8 => .{
            .name = entryName(v),
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .replaces = "embed + 2x glm_rms",
            .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 4096, .step = 512, .min = 512 } },
            .served = &prep_served,
        },
        .head => .{
            .name = entryName(v),
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "qmv + glm_argmax",
            .nan = .{ .stated = "NaN and -Inf logits are skipped; +Inf wins in index order" },
            .tiles = .{ .{}, .{ .size = 64, .step = 64, .min = 64 }, .{ .size = 512, .step = 512, .min = 512 } },
            .served = &head_served,
        },
    };
}

const GlmRows = extern struct { rows: i32, width: i32, x_stride: i32, y_stride: i32, eps: f32 };
const vocab_prep: usize = 32;

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    pipe: mtl.Pipeline,
    reduce_pipe: ?mtl.Pipeline = null,
    embed_pipe: ?mtl.Pipeline = null,
    rms_pipe: mtl.Pipeline,
    argmax_pipe: mtl.Pipeline,
    qmv_pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    aff: ?contract.affine.Affine = null,
    ids: []u32 = &.{},
    table: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    we: []u16 = &.{},
    wh: []u16 = &.{},
    h: []u16 = &.{},
    refs: [8]rig_.Ref = undefined,
    out_ref: rig_.Ref = undefined,
    base_ref: rig_.Ref = undefined,
    pairs_ref: rig_.Ref = undefined,
    logits_ref: rig_.Ref = undefined,
    deq_ref: rig_.Ref = undefined,
    msg: [120]u8 = undefined,

    fn drop(p: *Probe) void {
        if (p.aff) |f| f.deinit();
        p.aff = null;
        p.a.free(p.ids);
        p.a.free(p.table);
        p.a.free(p.scales);
        p.a.free(p.biases);
        p.a.free(p.we);
        p.a.free(p.wh);
        p.a.free(p.h);
        p.ids = &.{};
        p.table = &.{};
        p.scales = &.{};
        p.biases = &.{};
        p.we = &.{};
        p.wh = &.{};
        p.h = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.pipe.deinit();
        if (p.reduce_pipe) |r| r.deinit();
        if (p.embed_pipe) |em| em.deinit();
        p.rms_pipe.deinit();
        p.argmax_pipe.deinit();
        p.qmv_pipe.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        const ctl: ?g.Gate = switch (p.v.ctl) {
            1 => .repeat,
            2 => .shapes,
            else => null,
        };
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = ctl };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.rows == 0 or shape.rows > 16) return false;
        switch (p.v.kind) {
            .prep4, .prep8 => {
                if (shape.k < 64 or shape.k > 4096 or shape.k % 64 != 0 or shape.n != 2 * shape.k) return false;
                try p.preparePrep(shape, fill, seed);
            },
            .head => {
                if ((shape.rows != 1 and fill != .nonfinite) or shape.n == 0 or shape.n % 4 != 0 or shape.k < 512 or shape.k % 512 != 0) return false;
                try p.prepareHead(shape, fill, seed);
            },
        }
        p.shape = shape;
        return true;
    }

    fn preparePrep(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        const bits: usize = if (p.v.kind == .prep4) 4 else 8;
        const w = shape.k;
        const ng = w / 64;
        const row_bytes = w * bits / 8;
        p.ids = try p.a.alloc(u32, shape.rows);
        p.table = try p.a.alloc(u8, vocab_prep * row_bytes);
        p.scales = try p.a.alloc(u16, vocab_prep * ng);
        p.biases = try p.a.alloc(u16, vocab_prep * ng);
        p.we = try p.a.alloc(u16, w);
        p.wh = try p.a.alloc(u16, w);
        p.h = try p.a.alloc(u16, shape.rows * w);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (p.ids) |*id| id.* = rng.uintLessThan(u32, vocab_prep);
        rng.bytes(p.table);
        for (p.scales) |*s| s.* = c.toBf16((rng.float(f32) * 0.5) + 0.1);
        for (p.biases) |*b| b.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.5);
        for (p.we) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 1.5);
        for (p.wh) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 1.5);
        for (p.h) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2.0);
        if (fill == .nonfinite) {
            p.h[0] = 0x7fc0;
            if (shape.rows > 1) p.h[w] = 0x7f80;
            if (shape.rows > 2) p.h[2 * w] = 0xff80;
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.ids));
        p.refs[1] = try p.rig.input(p.table);
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(p.scales));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(p.biases));
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(p.we));
        p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(p.wh));
        p.refs[6] = try p.rig.input(std.mem.sliceAsBytes(p.h));
        p.out_ref = try p.rig.output(shape.rows * 2 * w * 2);
        p.base_ref = try p.rig.scratch(shape.rows * 2 * w * 2);
        p.deq_ref = try p.rig.scratch(shape.rows * w * 2);
    }

    fn prepareHead(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !void {
        const f = try contract.affine.Affine.init(p.a, shape, 4, 64);
        try f.fill(fill, seed);
        p.aff = f;
        const words = try p.a.alloc(u32, shape.n * shape.k / 8);
        defer p.a.free(words);
        f.words(words);
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(words));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(f.scales));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(f.biases));
        const pairs: usize = (shape.n + 7) / 8;
        p.pairs_ref = try p.rig.scratch(pairs * 8);
        p.out_ref = try p.rig.output(shape.rows * 4);
        p.logits_ref = try p.rig.scratch(shape.rows * shape.n * 2);
        p.base_ref = try p.rig.scratch(shape.rows * 4);
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        switch (p.v.kind) {
            .prep4, .prep8 => {
                const args = MtpPrepArgs{ .rows = @intCast(p.shape.rows), .width = @intCast(p.shape.k), .eps = 1e-6 };
                pass.e.setPipeline(p.pipe);
                for (0..7) |i| pass.e.setBuffer(p.refs[i].buf, p.refs[i].off, i);
                pass.e.setBuffer(p.out_ref.buf, p.out_ref.off, 7);
                pass.e.setValue(args, 8);
                const tg = ((p.shape.k + 3) / 4 + 31) / 32 * 32;
                pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(tg, 1, 1));
            },
            .head => {
                const pairs: u32 = @intCast((p.shape.n + 7) / 8);
                const args = HeadTop1Args{ .k = @intCast(p.shape.k), .n = @intCast(p.shape.n), .pairs = pairs };
                for (0..p.shape.rows) |r| {
                    pass.e.setPipeline(p.pipe);
                    pass.e.setBuffer(p.refs[0].buf, p.refs[0].off + r * p.shape.k * 2, 0);
                    for (1..4) |i| pass.e.setBuffer(p.refs[i].buf, p.refs[i].off, i);
                    pass.e.setValue(args, 4);
                    pass.e.setBuffer(p.pairs_ref.buf, p.pairs_ref.off, 5);
                    pass.e.dispatchGroups(mtl.Size.of(pairs, 1, 1), mtl.Size.of(128, 1, 1));
                    pass.e.barrier();
                    pass.e.setPipeline(p.reduce_pipe.?);
                    pass.e.setBuffer(p.pairs_ref.buf, p.pairs_ref.off, 0);
                    pass.e.setBuffer(p.out_ref.buf, p.out_ref.off + r * 4, 1);
                    pass.e.setValue(args, 2);
                    pass.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
                    if (r + 1 < p.shape.rows) pass.e.barrier();
                }
            },
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return switch (p.v.kind) {
            .prep4, .prep8 => p.shape.rows * 2 * p.shape.k,
            .head => p.shape.rows * 4,
        };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.v.kind == .head) {
            @memset(want, 0);
            @memset(mag, 0);
            return;
        }
        const bits: usize = if (p.v.kind == .prep4) 4 else 8;
        const w = p.shape.k;
        const ng = w / 64;
        const row_bytes = w * bits / 8;
        for (0..p.shape.rows) |r| {
            const id = p.ids[r];
            var sq_e: f64 = 0;
            var sq_h: f64 = 0;
            for (0..w) |col| {
                const sc: f64 = c.fromBf16(p.scales[id * ng + col / 64]);
                const bi: f64 = c.fromBf16(p.biases[id * ng + col / 64]);
                const byte = p.table[id * row_bytes + (if (bits == 4) col / 2 else col)];
                const q: u8 = if (bits == 4) (if (col % 2 == 0) byte & 0xf else byte >> 4) else byte;
                const ve = sc * @as(f64, @floatFromInt(q)) + bi;
                const vh: f64 = c.fromBf16(p.h[r * w + col]);
                sq_e += ve * ve;
                sq_h += vh * vh;
            }
            const inv_e = 1.0 / @sqrt(sq_e / @as(f64, @floatFromInt(w)) + 1e-6);
            const inv_h = 1.0 / @sqrt(sq_h / @as(f64, @floatFromInt(w)) + 1e-6);
            for (0..w) |col| {
                const sc: f64 = c.fromBf16(p.scales[id * ng + col / 64]);
                const bi: f64 = c.fromBf16(p.biases[id * ng + col / 64]);
                const byte = p.table[id * row_bytes + (if (bits == 4) col / 2 else col)];
                const q: u8 = if (bits == 4) (if (col % 2 == 0) byte & 0xf else byte >> 4) else byte;
                const ve = sc * @as(f64, @floatFromInt(q)) + bi;
                const vh: f64 = c.fromBf16(p.h[r * w + col]);
                want[r * 2 * w + col] = ve * inv_e * c.fromBf16(p.we[col]);
                want[r * 2 * w + w + col] = vh * inv_h * c.fromBf16(p.wh[col]);
            }
        }
        @memset(mag, 0);
    }

    pub fn check(p: *Probe, at: g.Gate) !?[]const u8 {
        if (p.v.kind != .head or at != .nonfinite) return null;
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        if (!std.mem.eql(u8, got, want)) return "argmax on nonfinite logits disagrees with baseline qmv + glm_argmax";
        return null;
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        switch (p.v.kind) {
            .prep4, .prep8 => return p.replacePrep(),
            .head => return p.replaceHead(),
        }
    }

    fn replacePrep(p: *Probe) ![]const u8 {
        const w = p.shape.k;
        const param: i32 = @intCast(w);
        const pass = p.rig.begin();
        pass.e.setPipeline(p.embed_pipe.?);
        for (0..4) |i| pass.e.setBuffer(p.refs[i].buf, p.refs[i].off, i);
        pass.e.setValue(param, 4);
        pass.e.setBuffer(p.deq_ref.buf, p.deq_ref.off, 5);
        const grid_x = if (p.v.kind == .prep4) w / 2 else w;
        pass.e.dispatchThreads(mtl.Size.of(grid_x, p.shape.rows, 1), mtl.Size.of(256, 1, 1));
        pass.e.barrier();
        const tg = ((w + 3) / 4 + 31) / 32 * 32;
        pass.e.setPipeline(p.rms_pipe);
        pass.e.setBuffer(p.deq_ref.buf, p.deq_ref.off, 0);
        pass.e.setBuffer(p.refs[4].buf, p.refs[4].off, 1);
        pass.e.setBuffer(p.base_ref.buf, p.base_ref.off, 2);
        pass.e.setValue(GlmRows{ .rows = @intCast(p.shape.rows), .width = @intCast(w), .x_stride = @intCast(w), .y_stride = @intCast(2 * w), .eps = 1e-6 }, 3);
        pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(tg, 1, 1));
        pass.e.barrier();
        pass.e.setBuffer(p.refs[6].buf, p.refs[6].off, 0);
        pass.e.setBuffer(p.refs[5].buf, p.refs[5].off, 1);
        pass.e.setBuffer(p.base_ref.buf, p.base_ref.off + w * 2, 2);
        pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(tg, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base_ref, p.shape.rows * 2 * w * 2);
    }

    fn replaceHead(p: *Probe) ![]const u8 {
        if (p.shape.k == 4096 and p.shape.n == 154880) {
            const pass = p.rig.begin();
            pass.e.setPipeline(p.qmv_pipe);
            for (0..4) |i| pass.e.setBuffer(p.refs[i].buf, p.refs[i].off, i);
            pass.e.setBuffer(p.logits_ref.buf, p.logits_ref.off, 4);
            pass.e.dispatchGroups(mtl.Size.of(1, p.shape.n / 4, 1), mtl.Size.of(32, p.shape.rows, 1));
            pass.e.barrier();
            pass.e.setPipeline(p.argmax_pipe);
            pass.e.setBuffer(p.logits_ref.buf, p.logits_ref.off, 0);
            pass.e.setBuffer(p.base_ref.buf, p.base_ref.off, 1);
            pass.e.setValue(@as(u32, @intCast(p.shape.n)), 2);
            pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
            try p.rig.finish(pass);
            return p.rig.bytes(p.base_ref, p.shape.rows * 4);
        }
        const f = p.aff.?;
        const logits = std.mem.bytesAsSlice(u16, p.rig.bytes(p.logits_ref, p.shape.rows * p.shape.n * 2));
        const ng = p.shape.k / 64;
        for (0..p.shape.rows) |r| {
            const xr = f.x[r * p.shape.k ..][0..p.shape.k];
            for (0..p.shape.n) |j| {
                var lane_acc: [32]f32 = @splat(0);
                var k0: usize = 0;
                while (k0 < p.shape.k) : (k0 += 512) {
                    for (0..32) |lane| {
                        const base = k0 + lane * 16;
                        if (base >= p.shape.k) continue;
                        var xsum: f32 = 0;
                        var i: usize = 0;
                        while (i < 16) : (i += 4) {
                            const s0 = c.toBf16(c.fromBf16(xr[base + i]) + c.fromBf16(xr[base + i + 1]));
                            const s1 = c.toBf16(c.fromBf16(s0) + c.fromBf16(xr[base + i + 2]));
                            const s2 = c.toBf16(c.fromBf16(s1) + c.fromBf16(xr[base + i + 3]));
                            xsum += c.fromBf16(s2);
                        }
                        var accum: f32 = 0;
                        for (0..16) |d| accum += c.fromBf16(xr[base + d]) * @as(f32, @floatFromInt(f.codes[j * p.shape.k + base + d]));
                        const gidx = j * ng + (k0 / 64) + lane / 4;
                        lane_acc[lane] += c.fromBf16(f.scales[gidx]) * accum + xsum * c.fromBf16(f.biases[gidx]);
                    }
                }
                var sum: f32 = 0;
                for (lane_acc) |v| sum += v;
                logits[r * p.shape.n + j] = c.toBf16(sum);
            }
        }
        const pass = p.rig.begin();
        pass.e.setPipeline(p.argmax_pipe);
        pass.e.setBuffer(p.logits_ref.buf, p.logits_ref.off, 0);
        pass.e.setBuffer(p.base_ref.buf, p.base_ref.off, 1);
        pass.e.setValue(@as(u32, @intCast(p.shape.n)), 2);
        pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
        return p.rig.bytes(p.base_ref, p.shape.rows * 4);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    lib: mtl.Library,
    glue_lib: mtl.Library,
    embed_lib: mtl.Library,
    qmv_lib: mtl.Library,
    probes: std.ArrayList(Probe) = .empty,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        errdefer a.destroy(s);
        s.* = .{
            .a = a,
            .lib = try mtl.Library.fromSource(device, sources.glm_mtp_opt, mtl.CompileOptions.mlx()),
            .glue_lib = try mtl.Library.fromSource(device, sources.glm_glue, mtl.CompileOptions.mlx()),
            .embed_lib = try mtl.Library.fromSource(device, sources.ops_embed_norm, mtl.CompileOptions.mlx()),
            .qmv_lib = try mtl.Library.fromSource(device, glmSource("qmv_4096_154880").source, mtl.CompileOptions.mlx()),
        };
        const variants = [_]Variant{
            .{ .kind = .prep4 }, .{ .kind = .prep4, .ctl = 1 }, .{ .kind = .prep4, .ctl = 2 },
            .{ .kind = .prep8 }, .{ .kind = .prep8, .ctl = 1 }, .{ .kind = .prep8, .ctl = 2 },
            .{ .kind = .head },  .{ .kind = .head, .ctl = 1 },  .{ .kind = .head, .ctl = 2 },
        };
        for (variants) |v| {
            if (only) |needle| if (std.mem.indexOf(u8, entryName(v), needle) == null) continue;
            const pipe_lib = if (v.ctl == 0) s.lib else blk: {
                const def = if (v.ctl == 1) "#define TF_CTL_GLM_MTP 1\n" else "#define TF_CTL_GLM_MTP 2\n";
                const src = try std.mem.concat(a, u8, &.{ def, sources.glm_mtp_opt });
                defer a.free(src);
                break :blk try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
            };
            defer if (v.ctl != 0) pipe_lib.deinit();
            const fn_name = switch (v.kind) {
                .prep4 => "glm_mtp_prep_4b",
                .prep8 => "glm_mtp_prep_8b",
                .head => "glm_head_top1_4b_one",
            };
            try s.probes.append(a, .{
                .a = a,
                .rig = .{ .a = a, .device = device, .queue = queue },
                .v = v,
                .pipe = try mtl.Pipeline.init(device, pipe_lib, fn_name, false),
                .reduce_pipe = if (v.kind == .head) try mtl.Pipeline.init(device, pipe_lib, "glm_reduce_top1", false) else null,
                .embed_pipe = switch (v.kind) {
                    .prep4 => try mtl.Pipeline.init(device, s.embed_lib, "tf_embed_b4_g64", false),
                    .prep8 => try mtl.Pipeline.init(device, s.embed_lib, "tf_embed_b8_g64", false),
                    .head => null,
                },
                .rms_pipe = try mtl.Pipeline.init(device, s.glue_lib, "glm_rms", false),
                .argmax_pipe = try mtl.Pipeline.init(device, s.glue_lib, "glm_argmax", false),
                .qmv_pipe = try mtl.Pipeline.init(device, s.qmv_lib, glmSource("qmv_4096_154880").functions[0], false),
            });
        }
        for (s.probes.items) |*p| try out.append(a, p.kernel());
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |*p| p.deinit();
        s.probes.deinit(s.a);
        s.qmv_lib.deinit();
        s.embed_lib.deinit();
        s.glue_lib.deinit();
        s.lib.deinit();
        s.a.destroy(s);
    }
};

fn glmSource(key: []const u8) sources.glm.Kernel {
    for (sources.glm.all) |k| if (std.mem.eql(u8, k.key, key)) return k;
    unreachable;
}
