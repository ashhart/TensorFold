//! Inkling 2 Mini's f32 GPU samplers held to lanes.gpu_full.draw and gpu_rule.topCands.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const rc = @import("metal_ricursive.zig");
const c = contract.spec;
const g = contract.gates;
const full = tf.lanes.gpu_full;
const gpu_rule = tf.lanes.gpu_rule;
const Sampling = tf.lanes.Sampling;

const full_rule = "each row's draw equals lanes' gpu_full.draw on f32 logits (main's legacy serial sampler)";
const slice_rule = "each slice yields its sorted top-k (key, id) candidates matching gpu_rule.topCands";

pub const Kind = enum { full, full_ids, slices, topk_rule, topk_rule_ids, rule_1024, rule_1024_ids };

const full_settings = [_]Sampling{
    .{ .seed = 21, .temperature = 1.0, .top_k = 0, .top_p = 0.9 },
    .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9 },
    .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9999995 },
    .{ .seed = 24, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.01 },
    .{ .seed = 22, .temperature = 1.0, .top_k = 3000, .top_p = 0.5 },
};

const topk_settings = [_]Sampling{
    .{ .seed = 31, .temperature = 1.0, .top_k = 1, .top_p = 1.0 },
    .{ .seed = 18508, .temperature = 1.0, .top_k = 20, .top_p = 0.9 },
    .{ .seed = 18508, .temperature = 1.0, .top_k = 40, .top_p = 0.9999995 },
    .{ .seed = 34, .temperature = 1.0, .top_k = 64, .top_p = 1.0, .min_p = 0.01 },
};

const rule1024_settings = [_]Sampling{
    .{ .seed = 41, .temperature = 1.0, .top_k = 65, .top_p = 0.95 },
    .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 0.9 },
    .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 0.9999995 },
    .{ .seed = 44, .temperature = 0.8, .top_k = 1024, .top_p = 1.0, .min_p = 0.01 },
};

const position: u32 = 40000;
const boundary_pos: u32 = 42;
const slices = 32;
const slice_threads = 512;
const kmax = 64;

fn label(k: Kind) []const u8 {
    return switch (k) {
        .full => "metal inkling2 sample tf_sample_full_f32",
        .full_ids => "metal inkling2 sample tf_sample_full_ids_f32",
        .slices => "metal inkling2 sample tf_topk_slices_f32",
        .topk_rule => "metal inkling2 sample tf_ink2_topk_rule_f32",
        .topk_rule_ids => "metal inkling2 sample tf_ink2_topk_rule_ids_f32",
        .rule_1024 => "metal inkling2 sample tf_ink2_rule_1024_f32",
        .rule_1024_ids => "metal inkling2 sample tf_ink2_rule_1024_ids_f32",
    };
}

fn controlLabel(k: Kind) []const u8 {
    return switch (k) {
        .full => "control: metal inkling2 sample tf_sample_full_f32",
        .full_ids => "control: metal inkling2 sample tf_sample_full_ids_f32",
        .slices => "control: metal inkling2 sample tf_topk_slices_f32",
        .topk_rule => "control: metal inkling2 sample tf_ink2_topk_rule_f32",
        .topk_rule_ids => "control: metal inkling2 sample tf_ink2_topk_rule_ids_f32",
        .rule_1024 => "control: metal inkling2 sample tf_ink2_rule_1024_f32",
        .rule_1024_ids => "control: metal inkling2 sample tf_ink2_rule_1024_ids_f32",
    };
}

fn fnName(k: Kind) [:0]const u8 {
    return switch (k) {
        .full => "tf_sample_full_f32",
        .full_ids => "tf_sample_full_ids_f32",
        .slices => "tf_topk_slices_f32",
        .topk_rule => "tf_ink2_topk_rule_f32",
        .topk_rule_ids => "tf_ink2_topk_rule_ids_f32",
        .rule_1024 => "tf_ink2_rule_1024_f32",
        .rule_1024_ids => "tf_ink2_rule_1024_ids_f32",
    };
}

fn usesInk2Source(k: Kind) bool {
    return switch (k) {
        .topk_rule, .topk_rule_ids, .rule_1024, .rule_1024_ids => true,
        else => false,
    };
}

fn controlSource(a: std.mem.Allocator, k: Kind) ![]u8 {
    const Edit = struct { from: []const u8, to: []const u8 };
    const base = if (usesInk2Source(k)) sources.inkling2_sample else sources.nemotron_sample;
    const e: Edit = switch (k) {
        .full => .{
            .from = "return s > bs || (s == bs && (k > bk || (k == bk && i < bi)));",
            .to = "return s < bs || (s == bs && (k > bk || (k == bk && i < bi)));",
        },
        .full_ids => .{
            .from =
            \\    for (uint g = 0; g < NSG; g++) {
            \\      if (beats(s.fsh[g], s.ush[g], s.ush2[g], bs, bk, bi)) { bs = s.fsh[g]; bk = s.ush[g]; bi = s.ush2[g]; }
            \\    }
            \\    const uint pick = min(bi, r.V - 1u);
            \\    TOK[row] = MAP ? IDS[pick] : pick;
            ,
            .to =
            \\    for (uint g = 0; g < NSG; g++) {
            \\      if (beats(s.fsh[g], s.ush[g], s.ush2[g], bs, bk, bi)) { bs = s.fsh[g]; bk = s.ush[g]; bi = s.ush2[g]; }
            \\    }
            \\    const uint pick = min(bi, r.V - 1u);
            \\    TOK[row] = pick;
            ,
        },
        .slices => .{
            .from = "keys[q] = i < e ? tf_key(float(raw[q]) * inv_t) : END.key;",
            .to = "keys[q] = i < e ? tf_key(-float(raw[q]) * inv_t) : END.key;",
        },
        .topk_rule => .{
            .from = "ck[n] = hk;",
            .to = "ck[n] = INK2_END_KEY;",
        },
        .rule_1024 => .{
            .from = "if (k > prefix) {",
            .to = "if (k < prefix) {",
        },
        .topk_rule_ids, .rule_1024_ids => .{
            .from = "TOK[row] = MAP ? IDS[pick] : pick;",
            .to = "TOK[row] = pick;",
        },
    };
    if (std.mem.count(u8, base, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, base, e.from, e.to);
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: Kind,
    pipe: mtl.Pipeline,
    slices_pipe: ?mtl.Pipeline = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    logits: []f32 = &.{},
    map: []u32 = &.{},
    in: [7]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    out2: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: Kind, control: bool) !Probe {
        const base = if (usesInk2Source(k)) sources.inkling2_sample else sources.nemotron_sample;
        const text = if (control) try controlSource(a, k) else try a.dupe(u8, base);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, fnName(k), false);
        errdefer pipe.deinit();
        var sp: ?mtl.Pipeline = null;
        if (k == .topk_rule or k == .topk_rule_ids) {
            const slib = try mtl.Library.fromSource(device, sources.nemotron_sample, mtl.CompileOptions.mlx());
            defer slib.deinit();
            sp = try mtl.Pipeline.init(device, slib, "tf_topk_slices_f32", false);
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = k, .pipe = pipe, .slices_pipe = sp };
    }

    pub fn deinit(p: *Probe) void {
        p.a.free(p.logits);
        p.a.free(p.map);
        p.rig.deinit();
        if (p.slices_pipe) |sp| sp.deinit();
        p.pipe.deinit();
    }

    pub fn spec(p: *const Probe) c.Spec {
        const tg: usize = switch (p.kind) {
            .slices => slice_threads,
            .topk_rule, .topk_rule_ids => 64,
            else => 1024,
        };
        const served = &[_]c.Shape{ .{ .rows = 1, .n = 1, .k = 200058 }, .{ .rows = 4, .n = 1, .k = 4096 }, .{ .rows = 8, .n = 1, .k = 200058 } };
        return .{
            .name = label(p.kind),
            .accum = .none,
            .depth = rc.none,
            .out = .raw,
            .nan = .{ .stated = if (p.kind == .slices) slice_rule else full_rule },
            .tiles = .{ .{}, .{}, .{ .size = tg, .step = 1, .min = 1 } },
            .served = served,
            .sums = rc.no_sums,
        };
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = p.spec(), .ctx = p, .vt = &vt };
    }

    fn settingsFor(k: Kind) []const Sampling {
        return switch (k) {
            .full, .full_ids => &full_settings,
            .slices, .topk_rule, .topk_rule_ids => &topk_settings,
            .rule_1024, .rule_1024_ids => &rule1024_settings,
        };
    }

    fn rowPos(r: usize, s: Sampling) u32 {
        return if (s.seed == 18508) boundary_pos else position + @as(u32, @intCast(r));
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.rows > 8 or s.k < 128 or s.k > 262144 or s.n != 1) return false;
        p.a.free(p.logits);
        p.a.free(p.map);
        p.logits = &.{};
        p.map = &.{};
        p.rig.reset();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.logits = try p.a.alloc(f32, s.rows * s.k);
        for (p.logits, 0..) |*l, i| {
            const peak = (i / s.k) % 4 == 2 and i % s.k < 2000;
            l.* = if (peak) 0.0 else -3.0 * rng.float(f32);
        }
        if (s.rows > 2) {
            p.logits[2 * s.k + 0] = -0.0;
            p.logits[2 * s.k + 1] = 0.0;
        }
        const list = settingsFor(p.kind);
        if (fill != .nonfinite and s.rows >= 4 and s.k >= 4096) {
            // Row 1: uniform24 == 1.0 (+inf) vs raceUniform.
            const r1 = p.logits[1 * s.k ..][0..s.k];
            @memset(r1, -50.0);
            r1[3122] = 0.0;
            r1[1035] = -3.7;
            @memset(r1[1040..1058], -3.7);
            // Row 2: candidate-order norm vs 1024-lane norm across top_p = 0.9999995.
            const r2 = p.logits[2 * s.k ..][0..s.k];
            @memset(r2, -50.0);
            r2[9] = 0.0;
            r2[1035] = -16.7;
            @memset(r2[1056..1088], -16.7);
        }
        if (fill == .nonfinite) {
            const pts = [_][2]usize{ .{ 0, s.k / 3 }, .{ 1, s.k / 2 }, .{ 2, s.k - 1 } };
            const vals = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (pts, vals) |pt, v| if (pt[0] < s.rows) {
                p.logits[pt[0] * s.k + pt[1]] = v;
            };
        }
        p.map = try p.a.alloc(u32, s.k);
        for (p.map, 0..) |*m, i| m.* = @intCast(2 * i + 1);
        var launch: gpu_rule.Launch(8) = .{};
        for (0..s.rows) |r| {
            const samp = list[r % list.len];
            launch.put(samp, rowPos(r, samp));
        }
        const padded = launch.padded();
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(p.logits));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(launch.seeds[0 .. 2 * padded]));
        p.in[2] = try p.rig.input(std.mem.sliceAsBytes(launch.positions[0..padded]));
        p.in[3] = try p.rig.input(std.mem.sliceAsBytes(launch.cfgs[0 .. 4 * padded]));
        p.in[4] = try p.rig.input(std.mem.sliceAsBytes(launch.caps[0..padded]));
        p.in[5] = try p.rig.input(std.mem.sliceAsBytes(p.map));
        p.out = try p.rig.output(if (p.kind == .slices) s.rows * slices * kmax * 4 else s.rows * 4);
        p.out2 = if (p.kind == .slices) try p.rig.output(s.rows * slices * kmax * 4) else if (p.slices_pipe != null) try p.rig.scratch(s.rows * slices * kmax * 4) else p.out;
        p.in[6] = if (p.slices_pipe != null) try p.rig.scratch(s.rows * slices * kmax * 4) else p.out;
        return true;
    }

    pub fn count(p: *Probe) usize {
        return if (p.kind == .slices) p.shape.rows * slices * kmax * 4 else p.shape.rows * 4;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const v: u32 = @intCast(p.shape.k);
        if (p.kind == .slices) {
            e.setPipeline(p.pipe);
            e.setBuffer(p.in[0].buf, p.in[0].off, 0);
            e.setBuffer(p.in[3].buf, p.in[3].off, 3);
            e.setBuffer(p.in[4].buf, p.in[4].off, 4);
            e.setValue(v, 7);
            e.setBuffer(p.out.buf, p.out.off, 8);
            e.setBuffer(p.out2.buf, p.out2.off, 9);
            e.dispatchGroups(mtl.Size.of(slices, p.shape.rows, 1), mtl.Size.of(slice_threads, 1, 1));
            try p.rig.finish(pass);
            return;
        }
        if (p.slices_pipe) |sp| {
            e.setPipeline(sp);
            e.setBuffer(p.in[0].buf, p.in[0].off, 0);
            e.setBuffer(p.in[3].buf, p.in[3].off, 3);
            e.setBuffer(p.in[4].buf, p.in[4].off, 4);
            e.setValue(v, 7);
            e.setBuffer(p.out2.buf, p.out2.off, 8);
            e.setBuffer(p.in[6].buf, p.in[6].off, 9);
            e.dispatchGroups(mtl.Size.of(slices, p.shape.rows, 1), mtl.Size.of(slice_threads, 1, 1));
        } else {
            e.setBuffer(p.in[0].buf, p.in[0].off, 0);
            e.setBuffer(p.in[3].buf, p.in[3].off, 3);
            e.setBuffer(p.in[4].buf, p.in[4].off, 4);
            e.setValue(v, 7);
        }
        e.setPipeline(p.pipe);
        e.setBuffer(p.in[1].buf, p.in[1].off, 1);
        e.setBuffer(p.in[2].buf, p.in[2].off, 2);
        e.setBuffer(p.out.buf, p.out.off, 5);
        const mapped = p.kind == .full_ids or p.kind == .topk_rule_ids or p.kind == .rule_1024_ids;
        e.setBuffer(if (mapped) p.in[5].buf else p.out.buf, if (mapped) p.in[5].off else p.out.off, 6);
        if (p.slices_pipe != null) {
            e.setBuffer(p.out2.buf, p.out2.off, 8);
            e.setBuffer(p.in[6].buf, p.in[6].off, 9);
        }
        const tg: usize = if (p.kind == .topk_rule or p.kind == .topk_rule_ids) 64 else 1024;
        e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(tg, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const k = p.shape.k;
        const list = settingsFor(p.kind);
        if (p.kind == .slices) {
            const got_k = std.mem.bytesAsSlice(u32, p.rig.bytes(p.out, p.shape.rows * slices * kmax * 4));
            const got_i = std.mem.bytesAsSlice(u32, p.rig.bytes(p.out2, p.shape.rows * slices * kmax * 4));
            const span = (k + slices - 1) / slices;
            var host: [kmax]gpu_rule.Cand = undefined;
            for (0..p.shape.rows) |r| {
                const s = list[r % list.len];
                const inv_t = gpu_rule.rule(s).inv_t;
                const row = p.logits[r * k ..][0..k];
                for (0..slices) |sg| {
                    const b = @min(sg * span, k);
                    const e = @min(b + span, k);
                    const base = (r * slices + sg) * kmax;
                    if (b == e) continue;
                    const take = @min(s.top_k, e - b);
                    _ = gpu_rule.topCands(row[b..e], inv_t, host[0..take]);
                    for (0..take) |j| {
                        const wk = gpu_rule.key(host[j].v);
                        const wi: u32 = @intCast(b + host[j].i);
                        if (got_k[base + j] != wk or got_i[base + j] != wi)
                            return try std.fmt.bufPrint(&p.msg, "row {d} slice {d} rank {d}: ({x},{d}) vs ({x},{d})", .{ r, sg, j, got_k[base + j], got_i[base + j], wk, wi });
                    }
                }
            }
            return null;
        }
        const mapped = p.kind == .full_ids or p.kind == .topk_rule_ids or p.kind == .rule_1024_ids;
        const ids = std.mem.bytesAsSlice(u32, p.rig.bytes(p.out, p.shape.rows * 4));
        for (0..p.shape.rows) |r| {
            const s = list[r % list.len];
            const row = p.logits[r * k ..][0..k];
            const want = try full.draw(p.a, row, s, rowPos(r, s), if (mapped) p.map else null);
            if (ids[r] != want) return try std.fmt.bufPrint(&p.msg, "row {d}: drew {d} where host draws {d}", .{ r, ids[r], want });
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, k: Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controlLabel(k) else label(k);
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        probe.* = try Probe.init(s.a, s.device, s.queue, k, control);
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var kern = probe.kernel();
        if (control) {
            kern.spec.name = name;
            kern.control = .shapes;
        }
        try s.kernels.append(s.a, kern);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Kind)) |k| {
            try s.add(k, false, only);
            try s.add(k, true, only);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
