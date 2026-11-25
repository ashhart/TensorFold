//! Mini DSpark's ten real Metal launches with guarded buffers, FP64 references and red controls.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_dspark_controls.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_dspark_host;
const sources = @import("inkling2_metal");

const H = host.hidden;
const Q = host.query_width;
const KV = host.kv_width;
const RANK = host.rank;
const RNN = host.rnn_width;
const INTER = host.intermediate;

fn depth(s: c.Shape) usize {
    return s.k + 16;
}
fn bf(v: anytype) f64 {
    return host.roundBf16(v);
}
fn rounded(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}
fn rms(values: []const f32, start: usize, width: usize) f32 {
    var sum: f32 = 0;
    for (0..width) |d| {
        const x = values[start + d];
        sum += x * x;
    }
    return 1.0 / @sqrt(sum / @as(f32, @floatFromInt(width)) + 1e-6);
}
fn rope(d: usize, position: usize, x: f32, other: f32) f64 {
    const frequency = std.math.pow(f32, 1e6, -@as(f32, @floatFromInt(d % 64)) / 64.0);
    const angle = @as(f32, @floatFromInt(position)) * frequency;
    const rotated: f32 = if (d < 64) x * @cos(angle) - other * @sin(angle) else other * @sin(angle) + x * @cos(angle);
    return bf(rotated);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: host.Kind,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: [6]c.Shape = undefined,
    served_len: usize = 0,
    f: [8][]f32 = @splat(&.{}),
    phi: []u16 = &.{},
    refs: [8]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    aux: [4]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: host.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.inkling2_dspark);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: host.Kind, text: []const u8) !Probe {
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .pipes = try host.Pipelines.loadSource(device, text) };
        switch (kind) {
            .init_rows => self.setServed(&.{.{ .rows = 8, .n = H, .k = 0 }}),
            .fuse_taps => self.setServed(&.{ .{ .rows = 5, .n = H, .k = 0 }, .{ .rows = 1, .n = H, .k = 0 }, .{ .rows = 3, .n = H, .k = 0 }, .{ .rows = 7, .n = H, .k = 0 } }),
            .block_qkv => self.setServed(&.{ .{ .rows = 8, .n = Q, .k = 17 }, .{ .rows = 8, .n = Q, .k = 67 } }),
            .context_kv => self.setServed(&.{ .{ .rows = 5, .n = KV, .k = 17 }, .{ .rows = 1, .n = KV, .k = 31 }, .{ .rows = 3, .n = KV, .k = 47 }, .{ .rows = 8, .n = KV, .k = 67 } }),
            .attention => self.setServed(&.{ .{ .rows = 8, .n = 0, .k = 0 }, .{ .rows = 8, .n = 1, .k = 0 }, .{ .rows = 8, .n = 3, .k = 0 }, .{ .rows = 8, .n = 3, .k = 5 }, .{ .rows = 8, .n = 1025, .k = 1 }, .{ .rows = 8, .n = 1025, .k = 7 } }),
            .residual_norm => self.setServed(&.{ .{ .rows = 8, .n = H, .k = 1 }, .{ .rows = 8, .n = H, .k = 0 } }),
            .swiglu => self.setServed(&.{.{ .rows = 8, .n = INTER, .k = 0 }}),
            .kappa => self.setServed(&.{ .{ .rows = 5, .n = RNN, .k = 0 }, .{ .rows = 1, .n = RNN, .k = 0 }, .{ .rows = 3, .n = RNN, .k = 0 }, .{ .rows = 7, .n = RNN, .k = 0 } }),
            .recurrent => self.setServed(&.{.{ .rows = 8, .n = RNN, .k = 0 }}),
            .base => self.setServed(&.{ .{ .rows = 5, .n = 5, .k = 0 }, .{ .rows = 1, .n = 1, .k = 0 }, .{ .rows = 3, .n = 17, .k = 0 }, .{ .rows = 7, .n = 129, .k = 0 } }),
            .logits => self.setServed(&.{.{ .rows = 8, .n = host.vocab, .k = 0 }}),
            .logits_q4 => unreachable,
        }
        return self;
    }
    fn setServed(self: *Probe, shapes: []const c.Shape) void {
        @memcpy(self.served[0..shapes.len], shapes);
        self.served_len = shapes.len;
    }
    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .spec = .{
            .name = self.kind.label(),
            .accum = .none,
            .depth = depth,
            .ops = 0,
            .mids = 0,
            .out = if (self.kind == .kappa) .f32 else .bf16,
            .served = self.served[0..self.served_len],
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "explicit serial FP32 reference and required BF16 nodes are checked directly" },
        } };
    }
    pub fn count(self: *Probe) usize {
        return switch (self.kind) {
            .init_rows => 8 * H,
            .fuse_taps => (4 * self.shape.rows + 1) * H,
            .block_qkv => 8 * (Q + 2 * KV),
            .context_kv => 2 * self.shape.rows * KV,
            .attention => 8 * Q,
            .residual_norm => 2 * 8 * H,
            .swiglu => 8 * INTER,
            .kappa => self.shape.rows * RNN,
            .recurrent => RNN + 2 * RANK,
            .base => self.shape.rows * self.shape.n,
            .logits => host.vocab,
            .logits_q4 => unreachable,
        };
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
        return values;
    }
    fn input(self: *Probe, values: []const f32, is_bf: bool) !rig_.Ref {
        if (!is_bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }
    fn prepInputs(self: *Probe, lens: []const usize, nf: bool, rng: std.Random) !void {
        for (lens, 0..) |n, j| {
            self.f[j] = try self.floats(n, rng);
            if (j == 0 and nf) self.f[0][0] = std.math.nan(f32);
            rounded(self.f[j]);
            self.refs[j] = try self.input(self.f[j], true);
        }
    }
    fn output(self: *Probe, slot: usize, elements: usize, elem: c.Elem) !void {
        self.outs[slot] = try self.rig.output(elements * c.size(elem));
    }
    fn rnnInput(self: *Probe, raw: []const f32, col0: usize, cols: usize) !rig_.Ref {
        const bits = try self.arena.allocator().alloc(u16, cols * RNN);
        for (0..RNN) |ch| for (0..cols) |d| {
            bits[d * RNN + ch] = c.toBf16(raw[ch * 5120 + col0 + d]);
        };
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }
    fn first(self: *const Probe) usize {
        const from = self.shape.k / 2;
        const local = self.shape.k % 2 != 0;
        return @max(from, if (local and self.shape.n > 1024) self.shape.n - 1024 else 0);
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        var admitted = false;
        for (self.served[0..self.served_len]) |served| if (std.meta.eql(shape, served)) {
            admitted = true;
            break;
        };
        if (!admitted) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.f = @splat(&.{});
        self.phi = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const rows = shape.rows;
        switch (self.kind) {
            .init_rows => {
                try self.prepInputs(&.{ H, H }, nf, rng);
                try self.output(0, 8 * H, .bf16);
            },
            .fuse_taps => {
                self.f[0] = try self.floats(rows * 4 * H, rng);
                self.f[1] = try self.floats(16, rng);
                self.f[2] = try self.floats(4 * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                @memcpy(self.f[1][0..4], &[_]f32{ 0, 0.5, -0.25, 2 });
                for (self.f[0][0 .. 4 * H]) |*v| v.* *= 400;
                rounded(self.f[0]);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, 4 * rows * H, .bf16);
                try self.output(1, H, .bf16);
            },
            .block_qkv => {
                try self.prepInputs(&.{ 8 * Q, 8 * KV, 8 * KV, 128, 128 }, nf, rng);
                try self.output(0, 8 * Q, .bf16);
                try self.output(1, 8 * KV, .bf16);
                try self.output(2, 8 * KV, .bf16);
            },
            .context_kv => {
                try self.prepInputs(&.{ rows * KV, rows * KV, 128 }, nf, rng);
                try self.output(0, rows * KV, .bf16);
                try self.output(1, rows * KV, .bf16);
            },
            .attention => {
                const ring = @max(shape.n, 1);
                try self.prepInputs(&.{ 8 * Q, 8 * KV, 8 * KV, ring * KV, ring * KV }, nf, rng);
                try self.output(0, 8 * Q, .bf16);
            },
            .residual_norm => {
                try self.prepInputs(&.{ 8 * H, 8 * H, H }, nf, rng);
                if (!nf and shape.k == 0) {
                    @memset(self.f[0][0..H], 0);
                    @memset(self.f[1][0..H], 0);
                    self.f[0][0] = 1;
                    self.f[1][0] = 0.00390625;
                    self.f[2][0] = 1;
                    for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                }
                try self.output(0, 8 * H, .bf16);
                try self.output(1, 8 * H, .bf16);
            },
            .swiglu => {
                try self.prepInputs(&.{ 8 * INTER, 8 * INTER }, nf, rng);
                try self.output(0, 8 * INTER, .bf16);
            },
            .kappa => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(H, rng);
                self.f[2] = try self.floats(RNN * 5120, rng);
                self.f[3] = try self.floats(RNN, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..4]) |values| rounded(values);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.rnnInput(self.f[2], 2 * RANK, 2 * H);
                self.refs[3] = try self.input(self.f[3], true);
                try self.output(0, rows * RNN, .f32);
            },
            .recurrent => {
                self.f[0] = try self.floats(RANK, rng);
                self.f[1] = try self.floats(RNN, rng);
                self.f[2] = try self.floats(RNN * 5120, rng);
                self.f[3] = try self.floats(2 * RANK, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for ([_]usize{ 0, 2, 3 }) |j| rounded(self.f[j]);
                @memset(self.f[0], 99.5);
                self.f[1][0] = c.fromBf16(c.toBf16(0.6));
                self.f[1][RANK] = c.fromBf16(c.toBf16(0.8));
                self.f[1][2 * RANK] = c.fromBf16(c.toBf16(0.3));
                for ([_]usize{ 0, RANK, 2 * RANK }) |channel| @memset(self.f[2][channel * 5120 ..][0..512], 0);
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], false);
                self.refs[2] = try self.rnnInput(self.f[2], 0, 2 * RANK);
                self.refs[3] = try self.input(self.f[3], true);
                const dummy_held = [_]u32{ 1, 0, 0, 0, 0, 0, 0, 0 };
                self.refs[4] = try self.rig.input(std.mem.sliceAsBytes(&dummy_held));
                try self.output(0, RNN, .bf16);
                try self.output(1, RANK, .bf16);
                try self.output(2, RANK, .bf16);
            },
            .logits => {
                self.f[0] = try self.floats(host.vocab, rng);
                self.f[1] = try self.floats(host.vocab, rng);
                self.f[2] = try self.floats(RANK, rng);
                for (self.f[0]) |*v| v.* = rng.float(f32) * 120 - 60;
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                self.f[0][0] = 40;
                self.f[0][1] = 1000;
                self.f[1][0] = 0.125;
                if (nf) self.f[0][0] = std.math.nan(f32);
                self.f[2][0] = c.fromBf16(c.toBf16(0.0001));
                self.phi = try self.arena.allocator().alloc(u16, host.vocab * RANK);
                for (self.phi) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
                @memset(self.phi[0..RANK], 0);
                self.phi[0] = c.toBf16(1);
                const phi_t = try self.arena.allocator().alloc(u16, host.vocab * RANK);
                for (0..host.vocab) |v| for (0..RANK) |d| {
                    phi_t[d * host.vocab + v] = self.phi[v * RANK + d];
                };
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(phi_t));
                self.refs[2] = try self.input(self.f[1], true);
                self.refs[3] = try self.input(self.f[2], true);
                try self.output(0, host.vocab, .bf16);
                self.aux[0] = try self.rig.scratch(((host.vocab + 255) / 256) * @sizeOf(host.GroupStat));
                self.aux[1] = try self.rig.scratch(8 * 4);
                self.aux[2] = try self.rig.scratch(8 * 4);
                self.aux[3] = try self.rig.scratch(4);
            },
            .base => {
                self.f[0] = try self.floats(shape.n * H, rng);
                self.f[1] = try self.floats(rows * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[0]);
                rounded(self.f[1]);
                if (!nf) {
                    @memset(self.f[0][0..H], 1);
                    @memset(self.f[1][0..H], 1);
                }
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                try self.output(0, rows * shape.n, .bf16);
            },
            .logits_q4 => unreachable,
        }
        return true;
    }

    fn hostRef(ref: rig_.Ref) host.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }
    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const e = pass.e;
        const R = self.refs;
        const O = self.outs;
        const p = &self.pipes;
        switch (self.kind) {
            .init_rows => host.initializeRows(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(O[0]), 1),
            .fuse_taps => host.fuseTaps(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), hostRef(O[1]), @intCast(self.shape.rows), @intCast(self.shape.rows)),
            .block_qkv => host.blockQkv(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(R[4]), hostRef(O[0]), hostRef(O[1]), hostRef(O[2]), &.{@intCast(self.shape.k)}),
            .context_kv => host.commitContext(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), hostRef(O[1]), @intCast(self.shape.k), @intCast(self.shape.rows), @intCast(self.shape.rows)),
            .attention => host.attend(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), &.{hostRef(R[3])}, &.{hostRef(R[4])}, &.{@intCast(self.shape.n)}, &.{@intCast(@max(self.shape.n, 1))}, &.{@intCast(self.shape.k / 2)}, self.shape.k % 2 != 0),
            .residual_norm => host.residualNorm(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), hostRef(O[1]), 8, self.shape.k != 0),
            .swiglu => host.swiglu(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(O[0]), 8 * INTER),
            .kappa => host.kappa(p, e, hostRef(R[0]), &.{hostRef(R[1])}, hostRef(R[2]), hostRef(R[3]), hostRef(O[0]), @intCast(self.shape.rows), @intCast(self.shape.rows), 0),
            .recurrent => host.recurrentStep(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(O[0]), hostRef(O[1]), hostRef(O[2]), hostRef(R[4]), 1, 1, 0, false),
            .base => host.base(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(O[0]), @intCast(self.shape.rows), @intCast(self.shape.n)),
            .logits => {
                host.finalLogits(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(O[0]), hostRef(self.aux[0]), hostRef(self.aux[1]), hostRef(self.aux[2]), hostRef(self.aux[3]), .{ .count = host.vocab, .phi_stride = host.vocab, .base_stride = 0, .streams = 1, .step = 0, .mode = 0, .flags = 0 });
                host.finalLogits(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(O[0]), hostRef(self.aux[0]), hostRef(self.aux[1]), hostRef(self.aux[2]), hostRef(self.aux[3]), .{ .count = host.vocab, .phi_stride = host.vocab, .base_stride = 0, .streams = 1, .step = 0, .mode = 1, .flags = 0 });
            },
            .logits_q4 => unreachable,
        }
        try self.rig.finish(pass);
    }

    fn fusionReference(self: *Probe, want: []f64) void {
        const rows = self.shape.rows;
        for (0..4) |fusion| {
            var scores: [4]f32 = undefined;
            var peak: f32 = -std.math.inf(f32);
            for (0..4) |j| {
                scores[j] = self.f[1][fusion * 4 + j];
                peak = @max(peak, scores[j]);
            }
            var den: f32 = 0;
            for (0..4) |j| {
                scores[j] = @exp(scores[j] - peak);
                den += scores[j];
            }
            var alpha: [4]f32 = undefined;
            for (0..4) |j| alpha[j] = @floatCast(bf(scores[j] / den));
            for (0..rows) |row| {
                var square: f32 = 0;
                for (0..H) |d| {
                    var mixed: f32 = 0;
                    for (0..4) |j| mixed += alpha[j] * self.f[0][(row * 4 + j) * H + d];
                    square += mixed * mixed;
                }
                const scale: f32 = 1.0 / @sqrt(square / H + 1e-6);
                for (0..H) |d| {
                    var mixed: f32 = 0;
                    for (0..4) |j| mixed += alpha[j] * self.f[0][(row * 4 + j) * H + d];
                    const out = bf(mixed * scale * self.f[2][fusion * H + d]);
                    want[(fusion * rows + row) * H + d] = out;
                    if (fusion == 3 and row + 1 == rows) want[4 * rows * H + d] = out;
                }
            }
        }
    }
    fn qkvReference(self: *Probe, want: []f64) void {
        for (0..8) |row| for (0..24) |h| for (0..128) |d| {
            const at = row * Q + h * 128 + d;
            const other_at = row * Q + h * 128 + (if (d < 64) d + 64 else d - 64);
            const scale = rms(self.f[0], row * Q + h * 128, 128);
            const x: f32 = @floatCast(bf(self.f[0][at] * scale * self.f[3][d]));
            const other: f32 = @floatCast(bf(self.f[0][other_at] * scale * self.f[3][if (d < 64) d + 64 else d - 64]));
            want[at] = rope(d, self.shape.k + row, x, other);
        };
        const offset = 8 * Q;
        for (0..8) |row| for (0..8) |h| for (0..128) |d| {
            const at = row * KV + h * 128 + d;
            const other_at = row * KV + h * 128 + (if (d < 64) d + 64 else d - 64);
            const scale = rms(self.f[1], row * KV + h * 128, 128);
            const x: f32 = @floatCast(bf(self.f[1][at] * scale * self.f[4][d]));
            const other: f32 = @floatCast(bf(self.f[1][other_at] * scale * self.f[4][if (d < 64) d + 64 else d - 64]));
            want[offset + at] = rope(d, self.shape.k + row, x, other);
            want[offset + 8 * KV + at] = self.f[2][at];
        };
    }
    fn contextReference(self: *Probe, want: []f64) void {
        const rows = self.shape.rows;
        for (0..rows) |row| for (0..8) |h| for (0..128) |d| {
            const src = row * KV + h * 128 + d;
            const dst = ((self.shape.k + row) % rows) * KV + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const other_src = row * KV + h * 128 + partner;
            const scale = rms(self.f[0], row * KV + h * 128, 128);
            const x: f32 = @floatCast(bf(self.f[0][src] * scale * self.f[2][d]));
            const other: f32 = @floatCast(bf(self.f[0][other_src] * scale * self.f[2][partner]));
            want[dst] = rope(d, self.shape.k + row, x, other);
            want[rows * KV + dst] = self.f[1][src];
        };
    }
    fn attentionReference(self: *Probe, want: []f64) !void {
        const committed = self.shape.n;
        const first_kept = self.first();
        const retained = committed - first_kept;
        const score_count = retained + 8;
        const ring = @max(committed, 1);
        const scores = try self.a.alloc(f32, score_count);
        defer self.a.free(scores);
        for (0..8) |row| for (0..24) |h| {
            const kv = h / 3;
            const q_at = row * Q + h * 128;
            var peak: f32 = -std.math.inf(f32);
            for (0..score_count) |j| {
                var dot: f32 = 0;
                const k_at = if (j < retained) ((first_kept + j) % ring) * KV + kv * 128 else (j - retained) * KV + kv * 128;
                const keys = if (j < retained) self.f[3] else self.f[1];
                for (0..128) |d| dot += self.f[0][q_at + d] * keys[k_at + d];
                scores[j] = dot * @as(f32, 0x1.6a09e6p-4);
                peak = @max(peak, scores[j]);
            }
            var den: f32 = 0;
            for (scores) |s| den += @exp(s - peak);
            for (0..128) |d| {
                var sum: f32 = 0;
                for (0..score_count) |j| {
                    const v_at = if (j < retained) ((first_kept + j) % ring) * KV + kv * 128 + d else (j - retained) * KV + kv * 128 + d;
                    const values = if (j < retained) self.f[4] else self.f[2];
                    sum += (@exp(scores[j] - peak) / den) * values[v_at];
                }
                want[q_at + d] = bf(sum);
            }
        };
    }
    fn recurrentReference(self: *Probe, want: []f64) void {
        for (0..RANK) |i| {
            var z: [3]f32 = undefined;
            for (0..3) |part| {
                const channel = part * RANK + i;
                var sum: f32 = 0;
                for (0..RANK) |d| sum += self.f[0][d] * self.f[2][channel * 5120 + d];
                for (0..RANK) |d| sum += self.f[3][RANK + d] * self.f[2][channel * 5120 + 256 + d];
                sum += self.f[1][channel];
                z[part] = @floatCast(bf(sum));
                want[channel] = z[part];
            }
            const gamma: f32 = @floatCast(bf(1.0 / (1.0 + @exp(-z[0]))));
            const candidate: f32 = @floatCast(bf(std.math.tanh(z[1])));
            const keep: f32 = @floatCast(bf(1.0 - gamma));
            const left: f32 = @floatCast(bf(gamma * self.f[0][i]));
            const right: f32 = @floatCast(bf(keep * candidate));
            want[RNN + i] = bf(left + right);
            want[RNN + RANK + i] = bf(std.math.tanh(z[2]));
        }
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .init_rows => for (0..8) |row| {
                for (0..H) |d| want[row * H + d] = self.f[if (row == 0) 0 else 1][d];
            },
            .fuse_taps => self.fusionReference(want),
            .block_qkv => self.qkvReference(want),
            .context_kv => self.contextReference(want),
            .attention => try self.attentionReference(want),
            .residual_norm => for (0..8) |row| {
                var square: f32 = 0;
                for (0..H) |d| {
                    const at = row * H + d;
                    const sigma: f32 = if (self.shape.k != 0) self.f[0][at] else self.f[0][at] + self.f[1][at];
                    square += sigma * sigma;
                }
                const scale: f32 = 1.0 / @sqrt(square / H + 1e-6);
                for (0..H) |d| {
                    const at = row * H + d;
                    const sigma: f32 = if (self.shape.k != 0) self.f[0][at] else self.f[0][at] + self.f[1][at];
                    want[at] = bf(sigma);
                    want[8 * H + at] = bf(sigma * scale * self.f[2][d]);
                }
            },
            .swiglu => for (want, 0..) |*w, i| {
                const gate = self.f[0][i];
                w.* = bf((gate / (1.0 + @exp(-gate))) * self.f[1][i]);
            },
            .kappa => for (0..self.shape.rows) |row| for (0..RNN) |channel| {
                var sum: f32 = 0;
                for (0..H) |d| sum += self.f[0][row * H + d] * self.f[2][channel * 5120 + 512 + d];
                for (0..H) |d| sum += self.f[1][d] * self.f[2][channel * 5120 + 2816 + d];
                sum += self.f[3][channel];
                want[row * RNN + channel] = sum;
            },
            .recurrent => self.recurrentReference(want),
            .base => for (0..self.shape.rows) |row| for (0..self.shape.n) |id| {
                var dot: f32 = 0;
                for (0..H) |d| dot += self.f[0][id * H + d] * self.f[1][row * H + d];
                want[row * self.shape.n + id] = bf(dot);
            },
            .logits => for (0..host.vocab) |i| {
                var dot: f32 = 0;
                for (0..RANK) |d| dot += c.fromBf16(self.phi[i * RANK + d]) * self.f[2][d];
                const bias: f32 = @floatCast(bf(self.f[1][i] + dot));
                const logit: f32 = @floatCast(bf(self.f[0][i] + bias));
                want[i] = bf(30.0 * std.math.tanh(logit / 30.0));
            },
            .logits_q4 => unreachable,
        }
        for (mag, want) |*m, w| m.* = 1 + @abs(w);
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return "numeric hooks must be used";
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        const bytes = try self.rig.outputBytes();
        for (want, 0..) |w, i| {
            if (self.kind == .kappa) {
                const got = c.read(.f32, bytes, i);
                if (gate == .nonfinite and c.classOf(got) != c.classOf(w)) return "kappa floating-point class differs";
                if (gate == .shapes and std.math.isFinite(w) and @abs(got - w) > 2e-5 * @max(1, @abs(w))) return "FP32 kappa dot or column split differs";
            } else {
                const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
                const expected = c.toBf16(@floatCast(w));
                if (std.math.isNan(w)) {
                    if (!std.math.isNan(c.fromBf16(bits))) return "BF16 NaN was not propagated";
                } else if (bits != expected) {
                    if (self.kind == .residual_norm and gate == .shapes and self.shape.k == 0 and i == 8 * H) return "BF16 endpoint differs from the explicit FP32/FP64 reference";
                    const relative: f64 = switch (self.kind) {
                        .fuse_taps, .block_qkv, .context_kv, .residual_norm, .swiglu, .base, .logits => 0.02,
                        .attention => 0.03,
                        else => 0,
                    };
                    if (@abs(@as(f64, c.fromBf16(bits)) - w) > @max(1e-5, relative * @abs(w))) return "BF16 endpoint differs from the explicit FP32/FP64 reference";
                }
            }
        }
        if (self.kind == .logits and gate == .shapes) {
            if (!(want[1] <= 30 and want[1] > 29.9)) return "the planted large logit does not prove softcap saturation";
            var want_best: usize = 0;
            for (want, 0..) |w, idx| if (w > want[want_best]) {
                want_best = idx;
            };
            var exp_sum: f64 = 0;
            for (want) |w| exp_sum += @exp(w - want[want_best]);
            const got_tok = std.mem.readInt(u32, self.rig.bytes(self.aux[1], 8)[4..8], .little);
            const got_prob: f32 = @bitCast(std.mem.readInt(u32, self.rig.bytes(self.aux[2], 4)[0..4], .little));
            if (got_tok != want_best or @abs(@as(f64, got_prob) - 1.0 / exp_sum) > 0.02) return "softmax mode=1 mismatch";
        }
        if (self.kind == .base and gate == .shapes and self.f[0][0] == 1 and want[0] != 2304) return "raw base projection was softened instead of rounded once";
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, kind: host.Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controls.name(@fromBackingInt(@intCast(@backingInt(kind)))) else kind.label();
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        if (control) {
            const text = try controls.source(s.a, sources.inkling2_dspark, @fromBackingInt(@intCast(@backingInt(kind))));
            defer s.a.free(text);
            probe.* = try Probe.initSource(s.a, s.device, s.queue, kind, text);
        } else probe.* = try Probe.init(s.a, s.device, s.queue, kind);
        errdefer probe.deinit();
        var k = probe.kernel();
        if (control) {
            k.spec.name = name;
            k.control = .shapes;
        }
        try controls.registerProbe(Probe, s.a, &s.probes, s.kernels, probe, k);
    }
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(host.Kind)) |kind| {
            if (kind == .logits_q4) continue;
            try s.add(kind, false, only);
            try s.add(kind, true, only);
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
