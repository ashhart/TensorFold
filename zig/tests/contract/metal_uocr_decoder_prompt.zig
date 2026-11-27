//! Grouped prompt GEMMs against the original per-expert dispatches, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_uocr_decoder_common.zig");

const c = contract.spec;
const g = contract.gates;
const plan = tf.uocr.prompt_plan;
const H = plan.hidden;
const E = plan.expert_ffn;
const Mutation = k.Mutation;

const served = [_]c.Shape{
    .{ .rows = 1, .n = 2 * E + H, .k = 1 },
    .{ .rows = 6, .n = 2 * E + H, .k = 1 },
    .{ .rows = 8, .n = 2 * E + H, .k = 1 },
    .{ .rows = 64, .n = 2 * E + H, .k = 1 },
    .{ .rows = 65, .n = 2 * E + H, .k = 1 },
};

const Expert = extern struct { gate: u64, up: u64, down: u64 };

pub const Grouped = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    grouped: mtl.Pipeline,
    full: mtl.Pipeline,
    partial: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    start: [plan.experts + 1]usize = @splat(0),
    groups: [3 * plan.experts]plan.Group = undefined,
    table: [plan.experts]Expert = undefined,
    made: plan.Plan = .{ .groups = 0, .threadgroups_y = 0 },
    xe_host: []u16 = &.{},
    ae_host: []u16 = &.{},
    gu_host: []u16 = &.{},
    down_host: []u16 = &.{},
    baseline: []u8 = &.{},
    xe: rig_.Ref = undefined,
    ae: rig_.Ref = undefined,
    gu: rig_.Ref = undefined,
    down: rig_.Ref = undefined,
    descriptors: rig_.Ref = undefined,
    experts: rig_.Ref = undefined,
    outputs: [3]rig_.Ref = undefined,
    plain: [3]rig_.Ref = undefined,

    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr decoder ud_grouped_prompt",
            .accum = .none,
            .depth = k.noSums,
            .out = .bf16,
            .replaces = "per-expert qwv_gemm_bf16_nt dispatches",
            .nan = .{ .stated = "grouped and per-expert BF16 bytes match, including non-finite outputs" },
            .served = &served,
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Grouped {
        const grouped = try k.pipeline(a, device, m, "ud_grouped_prompt");
        errdefer grouped.deinit();
        const full = try k.pipeline(a, device, .none, "qwv_gemm_bf16_nt_11");
        errdefer full.deinit();
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .grouped = grouped,
            .full = full,
            .partial = try k.pipeline(a, device, .none, "qwv_gemm_bf16_nt_01"),
            .m = m,
        };
    }

    pub fn deinit(t: *Grouped) void {
        t.freeHost();
        t.rig.deinit();
        t.grouped.deinit();
        t.full.deinit();
        t.partial.deinit();
    }

    fn freeHost(t: *Grouped) void {
        inline for (.{ "xe_host", "ae_host", "gu_host", "down_host", "baseline" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    pub fn kernel(t: *Grouped) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Grouped);

    pub fn prepare(t: *Grouped, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != 2 * E + H or shape.k != 1) return false;
        if (shape.rows != 1 and shape.rows != 6 and shape.rows != 8 and shape.rows != 64 and shape.rows != 65) return false;
        t.rig.reset();
        t.freeHost();
        t.shape = shape;
        @memset(&t.start, 0);
        for (t.start[1..]) |*v| v.* = shape.rows;
        if (shape.rows == 6) {
            for (0..6) |e| t.start[e + 1] = e + 1;
            for (t.start[7..]) |*v| v.* = 6;
        } else if (shape.rows == 8) {
            t.start[1] = 3;
            t.start[2] = 8;
        }
        t.made = plan.make(&t.start, &t.groups);
        const rows = shape.rows;
        t.xe_host = try t.a.alloc(u16, rows * H);
        t.ae_host = try t.a.alloc(u16, rows * E);
        t.gu_host = try t.a.alloc(u16, 6 * E * H);
        t.down_host = try t.a.alloc(u16, 6 * H * E);
        t.baseline = try t.a.alloc(u8, rows * (2 * E + H) * 2);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        k.fillBf16(rng, t.xe_host, 0.25);
        k.fillBf16(rng, t.ae_host, 0.25);
        k.fillBf16(rng, t.gu_host, 0.25);
        k.fillBf16(rng, t.down_host, 0.25);
        if (fill == .nonfinite) {
            t.xe_host[0] = c.toBf16(k.nan32);
            t.ae_host[0] = c.toBf16(k.inf32);
        }
        t.xe = try t.rig.input(std.mem.sliceAsBytes(t.xe_host));
        t.ae = try t.rig.input(std.mem.sliceAsBytes(t.ae_host));
        t.gu = try t.rig.input(std.mem.sliceAsBytes(t.gu_host));
        t.down = try t.rig.input(std.mem.sliceAsBytes(t.down_host));
        t.descriptors = try t.rig.input(std.mem.sliceAsBytes(&t.groups));
        for (0..plan.experts) |e| {
            const at = (e % 6) * E * H * 2;
            t.table[e] = .{
                .gate = t.gu.buf.gpuAddress() + t.gu.off + at,
                .up = t.gu.buf.gpuAddress() + t.gu.off + at,
                .down = t.down.buf.gpuAddress() + t.down.off + at,
            };
        }
        t.experts = try t.rig.input(std.mem.sliceAsBytes(&t.table));
        for (0..3) |projection| {
            const n = if (projection == 2) H else E;
            t.outputs[projection] = try t.rig.output(rows * n * 2);
            t.plain[projection] = try t.rig.scratch(rows * n * 2);
        }
        return true;
    }

    pub fn run(t: *Grouped) !void {
        const p = t.rig.begin();
        for (0..3) |projection| {
            const n = if (projection == 2) H else E;
            p.e.setPipeline(t.grouped);
            const x = if (projection == 2) t.ae else t.xe;
            p.e.setBuffer(x.buf, x.off, 0);
            p.e.setBuffer(t.experts.buf, t.experts.off, 1);
            const off = t.descriptors.off + projection * plan.experts * @sizeOf(plan.Group);
            p.e.setBuffer(t.descriptors.buf, off, 2);
            p.e.setBuffer(t.outputs[projection].buf, t.outputs[projection].off, 3);
            p.e.setValue(@as(u32, @intCast(t.made.groups)), 4);
            p.e.dispatchThreads(mtl.Size.of((n / 128) * 4 * 32, t.made.threadgroups_y, 2), mtl.Size.of(32, 4, 2));
            p.e.barrier();
        }
        try t.rig.finish(p);
    }

    pub fn replaced(t: *Grouped) ![]const u8 {
        const p = t.rig.begin();
        for (0..3) |projection| {
            const n = if (projection == 2) H else E;
            const k_width = if (projection == 2) E else H;
            const x = if (projection == 2) t.ae else t.xe;
            const w = if (projection == 2) t.down else t.gu;
            for (t.groups[projection * plan.experts ..][0..t.made.groups]) |group| {
                const tm: usize = @intCast(group.p[7]);
                p.e.setPipeline(if (@mod(group.p[0], 64) == 0) t.full else t.partial);
                p.e.setBuffer(x.buf, x.off + group.row0 * k_width * 2, 0);
                p.e.setBuffer(w.buf, w.off + (group.expert % 6) * n * k_width * 2, 1);
                p.e.setBytes(std.mem.asBytes(&group.p), 2);
                p.e.setBuffer(t.plain[projection].buf, t.plain[projection].off + group.row0 * n * 2, 3);
                p.e.dispatchThreads(mtl.Size.of((n / 128) * 4 * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
            }
        }
        try t.rig.finish(p);
        var at: usize = 0;
        for (0..3) |projection| {
            const n = if (projection == 2) H else E;
            const bytes = t.rig.bytes(t.plain[projection], t.shape.rows * n * 2);
            @memcpy(t.baseline[at..][0..bytes.len], bytes);
            at += bytes.len;
        }
        return t.baseline;
    }

    pub fn count(t: *Grouped) usize {
        return t.shape.rows * (2 * E + H);
    }

    pub fn reference(_: *Grouped, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(t: *Grouped, _: c.Gate) !?[]const u8 {
        const got = try t.rig.outputBytes();
        const want = try t.replaced();
        return if (std.mem.eql(u8, got, want)) null else "grouped output bytes differ from the per-expert baseline";
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    cases: std.ArrayList(*Grouped) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Mutation{ .none, .grouped_wrong_expert }) |m| {
            const x = try s.a.create(Grouped);
            x.* = try Grouped.init(s.a, s.device, s.queue, m);
            const kern = x.kernel();
            if (!k.wanted(only, kern.spec.name)) {
                x.deinit();
                s.a.destroy(x);
                continue;
            }
            try s.cases.append(s.a, x);
            try s.kernels.append(s.a, kern);
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.cases.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.cases.deinit(s.a);
    }
};

pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !?k.Texts {
    if (!std.mem.eql(u8, name, "uocr decoder ud_grouped_prompt")) return null;
    return try @import("metal_uocr.zig").towerTexts(a, device);
}
