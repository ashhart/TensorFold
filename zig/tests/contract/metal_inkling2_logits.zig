//! Inkling 2 Mini coalesced multi-row logits kernels: every output equals tf_inkling2_logits, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const glue = @import("tensorfold").inkling2_metal_host.glue;
const controls = @import("metal_inkling2_logits_controls.zig");

const base_name = "metal inkling2 logits tf_inkling2_logits_coalesced";

fn label(control: ?controls.Mutation) []const u8 {
    return if (control) |m| m.name() else base_name;
}

fn depth(s: c.Shape) usize {
    return s.k;
}

const hidden = glue.hidden;
const max_rows = 256;
const max_vocab = 200058;

const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 33, .k = hidden },
    .{ .rows = 4, .n = 64, .k = hidden },
    .{ .rows = 5, .n = 31, .k = hidden },
    .{ .rows = 8, .n = 65, .k = hidden },
    .{ .rows = 9, .n = 96, .k = hidden },
    .{ .rows = 16, .n = 35, .k = hidden },
    .{ .rows = 17, .n = 64, .k = hidden },
    .{ .rows = 32, .n = 129, .k = hidden },
    .{ .rows = 33, .n = 48, .k = hidden },
    .{ .rows = 256, .n = 33, .k = hidden },
};

pub fn spec(control: ?controls.Mutation) c.Spec {
    return .{
        .name = label(control),
        .accum = .none,
        .depth = depth,
        .out = .f32,
        .replaces = "the scalar logits kernel tf_inkling2_logits on exact bf16 inputs and rows",
        .nan = .{ .stated = "every output bit matches tf_inkling2_logits including NaN, Inf and signed zero" },
        // Vocab tiles are 32 wide and row tiles are up to 32 tall across r4/r8/r16/r32.
        .tiles = .{ .{ .size = 32 }, .{ .size = 32 }, .{} },
        .served = &shapes,
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "tf_inkling2_logits's serial fp32 dot product is the baseline, bit for bit" },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: glue.Pipelines,
    control: ?controls.Mutation,
    shape: c.Shape = undefined,
    seed: u64 = 0,
    h: []u16 = &.{},
    unembed: []u16 = &.{},
    refs: [3]rig_.Ref = undefined,
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: ?controls.Mutation) !Probe {
        const text = try controls.apply(a, control orelse .none);
        defer a.free(text);
        const pipes = try glue.Pipelines.loadSource(device, text);
        errdefer {
            var owned = pipes;
            owned.deinit();
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .control = control };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.h);
        p.a.free(p.unembed);
        p.h = &.{};
        p.unembed = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.control), .ctx = p, .vt = &vt, .control = if (p.control) |m| m.gate() else null };
    }
    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != hidden or shape.rows == 0 or shape.rows > max_rows or shape.n == 0 or shape.n > max_vocab) return false;
        p.clear();
        p.shape = shape;
        p.seed = seed;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.h = try p.a.alloc(u16, shape.rows * hidden);
        p.unembed = try p.a.alloc(u16, shape.n * hidden);
        for (p.h) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (p.unembed) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        // Cancel large aligned pairs inside chunk 0 and across chunks 1/2 to catch pair or k_step reordering.
        for (0..shape.rows) |r| {
            p.h[r * hidden + 0] = c.toBf16(1);
            p.h[r * hidden + 2] = c.toBf16(1);
            p.h[r * hidden + 3] = c.toBf16(1);
            p.h[r * hidden + 66] = c.toBf16(1);
            p.h[r * hidden + 128] = c.toBf16(1);
        }
        for (0..shape.n) |v| {
            p.unembed[v * hidden + 0] = c.toBf16(0.25);
            p.unembed[v * hidden + 2] = c.toBf16(262144);
            p.unembed[v * hidden + 3] = c.toBf16(-262144);
            p.unembed[v * hidden + 66] = c.toBf16(262144);
            p.unembed[v * hidden + 128] = c.toBf16(-262144);
        }
        if (fill == .nonfinite) {
            if (shape.rows == 1) {
                p.h[15] = c.toBf16(1);
                p.h[25] = 0x8000;
                p.h[hidden - 1] = c.toBf16(1);
            } else {
                p.h[0] = c.toBf16(std.math.nan(f32));
                p.h[hidden + 10] = c.toBf16(std.math.inf(f32));
                if (shape.rows > 2) p.h[2 * hidden + 20] = c.toBf16(-std.math.inf(f32));
                if (shape.rows > 3) @memset(p.h[3 * hidden ..][0..hidden], 0x8000);
                // Keep tail rows finite in h so unembed nonfinites and finite lanes both run on partial tiles.
                if (shape.rows > 4) {
                    p.h[(shape.rows - 1) * hidden + 15] = c.toBf16(1);
                    p.h[(shape.rows - 1) * hidden + hidden - 1] = c.toBf16(1);
                }
            }
            @memset(p.unembed[0..hidden], 0x3f80);
            p.unembed[hidden / 2] = c.toBf16(std.math.nan(f32));
            if (shape.n > 2) p.unembed[2 * hidden + 15] = c.toBf16(std.math.inf(f32));
            if (shape.n > 4) {
                @memset(p.unembed[3 * hidden ..][0..hidden], 0x8000);
                p.unembed[4 * hidden + 25] = c.toBf16(std.math.inf(f32));
            }
            if (shape.n > 5) p.unembed[shape.n * hidden - 1] = c.toBf16(-std.math.inf(f32));
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.h));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.unembed));
        p.refs[2] = try p.rig.output(p.count() * 4);
        p.baseline = try p.rig.scratch(p.count() * 4);
        return true;
    }

    fn hr(ref: rig_.Ref) glue.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        glue.logits(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), @intCast(p.shape.rows), @intCast(p.shape.n));
        try p.rig.finish(pass);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        const pass = p.rig.begin();
        glue.logitsBaseline(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.baseline), @intCast(p.shape.rows), @intCast(p.shape.n));
        try p.rig.finish(pass);
        return p.rig.bytes(p.baseline, p.count() * 4);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn checkCurrent(p: *Probe) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        if (!std.mem.eql(u8, got, want)) return "an output bit differs from tf_inkling2_logits";
        if (!try p.rig.intact()) return "a run wrote outside its outputs";
        return null;
    }

    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        if (try p.checkCurrent()) |why| return why;
        if (gate == .nonfinite) {
            const base_seed = p.seed;
            // gates.nonfinite clamps served[0] to 5..8 (r8); also cover r4 (1), r16 (9), and r32 (17).
            for ([_]c.Shape{
                .{ .rows = 1, .n = 33, .k = hidden },
                .{ .rows = 9, .n = 35, .k = hidden },
                .{ .rows = 17, .n = 65, .k = hidden },
            }) |extra| {
                if (!try p.prepare(extra, .nonfinite, base_seed +% extra.rows)) return "refused a served nonfinite shape";
                try p.run();
                if (try p.checkCurrent()) |why| return why;
            }
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

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }

    pub fn registerProbe(
        a: std.mem.Allocator,
        probes: *std.ArrayList(*Probe),
        kernels: *std.ArrayList(g.Kernel),
        probe: *Probe,
    ) !void {
        try contract.register(a, probes, kernels, probe, probe.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]?controls.Mutation{ null, .bf16_round, .pair_assoc, .chunk_order }) |control| {
            if (only) |needle| if (std.mem.indexOf(u8, label(control), needle) == null) continue;
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = try Probe.init(s.a, s.device, s.queue, control);
            errdefer probe.deinit();
            try registerProbe(s.a, &s.probes, s.kernels, probe);
        }
    }
};

test "Suite.registerProbe handles FailingAllocator OOM with errdefer cleanup and zero leaks" {
    const base = std.testing.allocator;
    for ([_]usize{ 0, 1, std.math.maxInt(usize) }) |fail_index| {
        var probes: std.ArrayList(*Probe) = .empty;
        var kernels: std.ArrayList(g.Kernel) = .empty;
        defer {
            for (probes.items) |p| base.destroy(p);
            probes.deinit(base);
            kernels.deinit(base);
        }
        const probe = try base.create(Probe);
        var freed = false;
        errdefer if (!freed) base.destroy(probe);
        probe.* = .{ .a = base, .rig = undefined, .pipes = undefined, .control = null };
        var failing = std.testing.FailingAllocator.init(base, .{ .fail_index = fail_index });
        const result = Suite.registerProbe(failing.allocator(), &probes, &kernels, probe);
        if (fail_index < 2) {
            try std.testing.expectError(error.OutOfMemory, result);
            try std.testing.expectEqual(@as(usize, 0), probes.items.len);
            try std.testing.expectEqual(@as(usize, 0), kernels.items.len);
            freed = true;
            base.destroy(probe);
        } else {
            try result;
            freed = true;
            try std.testing.expectEqual(@as(usize, 1), probes.items.len);
            try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
        }
    }
}
