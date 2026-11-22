//! Mini DSpark's greedy chain: the step is the recurrent kernel on a token read from a buffer, the pick a first max.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_dspark_host;
const controls = @import("metal_inkling2_dspark_chain_controls.zig");
const sources = @import("inkling2_metal");

const RANK = host.rank;
const RNN = host.rnn_width;

fn noSums(_: c.Shape) usize {
    return 1;
}
fn hr(b: rig_.Ref) host.Ref {
    return .{ .buf = b.buf, .off = b.off };
}

/// The chain step against the registered recurrent kernel on the same inputs: every byte of its three outputs.
pub const Step = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    chain: host.Chain,
    pipes: host.Pipelines,
    control: ?g.Gate,
    refs: [5]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    twin: [3]rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    const name = "metal inkling2 dspark chain_step tf_ink2_dspark_chain_step";
    /// The slot of the held tokens the probe's step reads, and the token there; the slot after holds another.
    const step = 3;
    const token = 1;
    const sizes = [3]usize{ RNN * 2, RANK * 2, RANK * 2 };

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Step {
        const text = try controls.apply(a, if (control) .step_token else .none);
        defer a.free(text);
        var chain = try host.Chain.load(device, text);
        errdefer chain.deinit();
        return .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .chain = chain, .pipes = try host.Pipelines.load(device), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Step) void {
        t.rig.deinit();
        t.arena.deinit();
        t.pipes.deinit();
        t.chain.deinit();
    }
    pub fn kernel(t: *Step) g.Kernel {
        return .{ .ctx = t, .vt = &vt, .control = t.control, .spec = .{
            .name = if (t.control != null) controls.Mutation.step_token.name() else name,
            .accum = .none,
            .depth = noSums,
            .out = .bf16,
            .nan = .{ .stated = "its bytes are tf_ink2_dspark_recurrent's on the token the buffer holds, whatever their classes" },
            .served = &.{.{ .rows = 8, .n = RNN, .k = 0 }},
        } };
    }
    const vt = g.hooks(Step);

    fn bf16(t: *Step, n: usize, rng: std.Random) ![]u16 {
        const bits = try t.arena.allocator().alloc(u16, n);
        for (bits) |*b| b.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
        return bits;
    }

    pub fn prepare(t: *Step, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows != 8 or shape.n != RNN) return false;
        t.rig.reset();
        _ = t.arena.reset(.retain_capacity);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const state = try t.bf16(RANK, rng);
        if (fill == .nonfinite) state[0] = c.toBf16(std.math.nan(f32));
        const kappa = try t.arena.allocator().alloc(f32, RNN);
        for (kappa) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
        // The Markov rows of tokens 0 and 1 differ, so the token read decides the outputs.
        const held = [8]u32{ 0, 0, 0, token, 0, 0, 0, 0 };
        t.refs[0] = try t.rig.input(std.mem.sliceAsBytes(state));
        t.refs[1] = try t.rig.input(std.mem.sliceAsBytes(kappa));
        t.refs[2] = try t.rig.input(std.mem.sliceAsBytes(try t.bf16(RNN * 5120, rng)));
        t.refs[3] = try t.rig.input(std.mem.sliceAsBytes(try t.bf16(2 * RANK, rng)));
        t.refs[4] = try t.rig.input(std.mem.sliceAsBytes(&held));
        for (&t.outs, &t.twin, sizes) |*out, *twin, bytes| {
            out.* = try t.rig.output(bytes);
            twin.* = try t.rig.scratch(bytes);
        }
        return true;
    }

    pub fn run(t: *Step) !void {
        const pass = t.rig.begin();
        host.chainStep(&t.chain, pass.e, hr(t.refs[0]), hr(t.refs[1]), hr(t.refs[2]), hr(t.refs[3]), hr(t.outs[0]), hr(t.outs[1]), hr(t.outs[2]), hr(t.refs[4]), step);
        try t.rig.finish(pass);
    }

    pub fn count(_: *Step) usize {
        return RNN + 2 * RANK;
    }
    pub fn reference(_: *Step, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Step, _: g.Gate) !?[]const u8 {
        const pass = t.rig.begin();
        host.recurrentStep(&t.pipes, pass.e, hr(t.refs[0]), hr(t.refs[1]), hr(t.refs[2]), hr(t.refs[3]), hr(t.twin[0]), hr(t.twin[1]), hr(t.twin[2]), token);
        try t.rig.finish(pass);
        for (t.outs, t.twin, sizes, 0..) |out, twin, bytes, i| {
            const got = t.rig.bytes(out, bytes);
            const want = t.rig.bytes(twin, bytes);
            if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&t.msg, "output {d} differs from the recurrent kernel's from byte {d}", .{ i, at });
        }
        return null;
    }
};

/// The pick against a scan of the same logits: the first of equal maxima, a NaN only at id 0, no other slot written.
pub const Pick = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    chain: host.Chain,
    control: ?g.Gate,
    logits: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    const name = "metal inkling2 dspark chain_pick tf_ink2_dspark_chain_pick";
    const step = 2;

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Pick {
        const text = try controls.apply(a, if (control) .pick_tie else .none);
        defer a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .chain = try host.Chain.load(device, text), .control = if (control) .shapes else null };
    }
    pub fn deinit(t: *Pick) void {
        t.a.free(t.logits);
        t.rig.deinit();
        t.chain.deinit();
    }
    pub fn kernel(t: *Pick) g.Kernel {
        return .{ .ctx = t, .vt = &vt, .control = t.control, .spec = .{
            .name = if (t.control != null) controls.Mutation.pick_tie.name() else name,
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "a NaN is never the pick unless logit 0 is one, and then the pick is id 0; else ties go to the lower id" },
            .served = &.{.{ .rows = 1, .n = host.vocab, .k = 0 }},
        } };
    }
    const vt = g.hooks(Pick);

    pub fn prepare(t: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != host.vocab) return false;
        t.rig.reset();
        t.a.free(t.logits);
        t.logits = try t.a.alloc(u16, host.vocab);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Few distinct values, so the greatest is tied many times within a simdgroup and across them.
        for (t.logits) |*v| v.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 64))) / 4 - 8);
        t.logits[host.vocab - 1] = t.logits[host.vocab - 2];
        if (fill == .nonfinite) {
            t.logits[host.vocab / 3] = c.toBf16(std.math.nan(f32));
            t.logits[host.vocab / 2] = c.toBf16(-std.math.inf(f32));
            t.logits[host.vocab - 1] = c.toBf16(std.math.nan(f32));
            // An odd seed puts a NaN first, where the rule picks id 0; an even one an infinity late in the row.
            if (seed % 2 == 1) t.logits[0] = c.toBf16(std.math.nan(f32)) else t.logits[host.vocab - 7] = c.toBf16(std.math.inf(f32));
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.logits));
        t.out = try t.rig.output(8 * @sizeOf(u32));
        return true;
    }

    pub fn run(t: *Pick) !void {
        const pass = t.rig.begin();
        host.chainPick(&t.chain, pass.e, hr(t.in), hr(t.out), step);
        try t.rig.finish(pass);
    }

    pub fn count(_: *Pick) usize {
        return 8 * @sizeOf(u32);
    }
    pub fn reference(_: *Pick, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Pick, _: g.Gate) !?[]const u8 {
        var at: u32 = 0;
        if (!std.math.isNan(c.fromBf16(t.logits[0]))) {
            var best = -std.math.inf(f32);
            at = std.math.maxInt(u32);
            for (t.logits, 0..) |bits, i| {
                const v = c.fromBf16(bits);
                if (!std.math.isNan(v) and (v > best or at == std.math.maxInt(u32))) {
                    best = v;
                    at = @intCast(i);
                }
            }
        }
        const held = std.mem.bytesAsSlice(u32, t.rig.bytes(t.out, 8 * @sizeOf(u32)));
        for (held, 0..) |got, slot| {
            const want: u32 = if (slot == step + 1) at else 0xffff_ffff;
            if (got != want) return try std.fmt.bufPrint(&t.msg, "held[{d}] is {d} where the rule leaves {d}", .{ slot, got, want });
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    steps: [2]Step = undefined,
    picks: [2]Pick = undefined,
    step_count: usize = 0,
    pick_count: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.steps[0..s.step_count]) |*t| t.deinit();
        for (s.picks[0..s.pick_count]) |*t| t.deinit();
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |control| {
            const step_label = if (control) controls.Mutation.step_token.name() else Step.name;
            if (only == null or std.mem.indexOf(u8, step_label, only.?) != null) {
                s.steps[s.step_count] = try Step.init(s.a, s.device, s.queue, control);
                s.step_count += 1;
                try s.kernels.append(s.a, s.steps[s.step_count - 1].kernel());
            }
            const pick_label = if (control) controls.Mutation.pick_tie.name() else Pick.name;
            if (only == null or std.mem.indexOf(u8, pick_label, only.?) != null) {
                s.picks[s.pick_count] = try Pick.init(s.a, s.device, s.queue, control);
                s.pick_count += 1;
                try s.kernels.append(s.a, s.picks[s.pick_count - 1].kernel());
            }
        }
    }
};
