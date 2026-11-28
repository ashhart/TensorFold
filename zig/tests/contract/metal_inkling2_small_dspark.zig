//! Small DSpark's seven Small-geometry Metal launches with guarded buffers, FP64 references and red controls.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_small_dspark_controls.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_small_dspark_host;
const sources = @import("inkling2_metal");

const H = host.hidden;
const Q = host.query_width;
const KV = host.kv_width;
const RANK = host.rank;
const RNN = host.rnn_width;
const TAPS = host.taps_count;
const FUSIONS = host.fusions_count;
const RNN_COLS = 2 * RANK + 2 * H;

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
    refs: [8]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: host.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.inkling2_small_dspark);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: host.Kind, text: []const u8) !Probe {
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .pipes = try host.Pipelines.loadSource(device, text) };
        switch (kind) {
            .init_rows => self.setServed(&.{.{ .rows = 8, .n = H, .k = 0 }}),
            .fuse_taps => self.setServed(&.{ .{ .rows = 5, .n = H, .k = 0 }, .{ .rows = 1, .n = H, .k = 0 }, .{ .rows = 3, .n = H, .k = 0 }, .{ .rows = 7, .n = H, .k = 0 } }),
            .block_qkv => self.setServed(&.{ .{ .rows = 8, .n = Q, .k = 17 }, .{ .rows = 8, .n = Q, .k = 67 } }),
            .attention => self.setServed(&.{ .{ .rows = 8, .n = 0, .k = 0 }, .{ .rows = 8, .n = 1, .k = 0 }, .{ .rows = 8, .n = 3, .k = 0 }, .{ .rows = 8, .n = 3, .k = 5 }, .{ .rows = 8, .n = 1025, .k = 1 }, .{ .rows = 8, .n = 1025, .k = 7 } }),
            .residual_norm => self.setServed(&.{ .{ .rows = 8, .n = H, .k = 1 }, .{ .rows = 8, .n = H, .k = 0 } }),
            .kappa => self.setServed(&.{ .{ .rows = 5, .n = RNN, .k = 0 }, .{ .rows = 1, .n = RNN, .k = 0 }, .{ .rows = 3, .n = RNN, .k = 0 }, .{ .rows = 7, .n = RNN, .k = 0 } }),
            .base => self.setServed(&.{ .{ .rows = 5, .n = 5, .k = 0 }, .{ .rows = 1, .n = 1, .k = 0 }, .{ .rows = 3, .n = 17, .k = 0 }, .{ .rows = 7, .n = 129, .k = 0 } }),
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
            .fuse_taps => (FUSIONS * self.shape.rows + 1) * H,
            .block_qkv => 8 * (Q + 2 * KV),
            .attention => 8 * Q,
            .residual_norm => 2 * 8 * H,
            .kappa => self.shape.rows * RNN,
            .base => self.shape.rows * self.shape.n,
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
            bits[d * RNN + ch] = c.toBf16(raw[ch * RNN_COLS + col0 + d]);
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
                self.f[0] = try self.floats(rows * TAPS * H, rng);
                self.f[1] = try self.floats(FUSIONS * TAPS, rng);
                self.f[2] = try self.floats(FUSIONS * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |values| rounded(values);
                @memcpy(self.f[1][0..TAPS], &[_]f32{ 0, 0.5, -0.25, 2, -0.5, 0.25, -1, 1.5 });
                for (self.f[0][0 .. TAPS * H]) |*v| v.* *= 400;
                rounded(self.f[0]);
                for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                try self.output(0, FUSIONS * rows * H, .bf16);
                try self.output(1, H, .bf16);
            },
            .block_qkv => {
                try self.prepInputs(&.{ 8 * Q, 8 * KV, 8 * KV, 128, 128 }, nf, rng);
                try self.output(0, 8 * Q, .bf16);
                try self.output(1, 8 * KV, .bf16);
                try self.output(2, 8 * KV, .bf16);
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
                    self.f[2][0] = 0.9375;
                    for (0..3) |j| self.refs[j] = try self.input(self.f[j], true);
                }
                try self.output(0, 8 * H, .bf16);
                try self.output(1, 8 * H, .bf16);
            },
            .kappa => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(H, rng);
                self.f[2] = try self.floats(RNN * RNN_COLS, rng);
                self.f[3] = try self.floats(RNN, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..4]) |values| rounded(values);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.rnnInput(self.f[2], 2 * RANK, 2 * H);
                self.refs[3] = try self.input(self.f[3], true);
                try self.output(0, rows * RNN, .f32);
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
            .attention => host.attend(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), &.{hostRef(R[3])}, &.{hostRef(R[4])}, &.{@intCast(self.shape.n)}, &.{@intCast(@max(self.shape.n, 1))}, &.{@intCast(self.shape.k / 2)}, self.shape.k % 2 != 0),
            .residual_norm => host.residualNorm(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), hostRef(O[1]), 8, self.shape.k != 0),
            .kappa => host.kappa(p, e, hostRef(R[0]), &.{hostRef(R[1])}, hostRef(R[2]), hostRef(R[3]), hostRef(O[0]), @intCast(self.shape.rows), @intCast(self.shape.rows), 0),
            .base => host.base(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(O[0]), @intCast(self.shape.rows), @intCast(self.shape.n)),
        }
        try self.rig.finish(pass);
    }

    fn fusionReference(self: *Probe, want: []f64) void {
        const rows = self.shape.rows;
        for (0..FUSIONS) |fusion| {
            var scores: [TAPS]f32 = undefined;
            var peak: f32 = -std.math.inf(f32);
            for (0..TAPS) |j| {
                scores[j] = self.f[1][fusion * TAPS + j];
                peak = @max(peak, scores[j]);
            }
            var den: f32 = 0;
            for (0..TAPS) |j| {
                scores[j] = @exp(scores[j] - peak);
                den += scores[j];
            }
            var alpha: [TAPS]f32 = undefined;
            for (0..TAPS) |j| alpha[j] = @floatCast(bf(scores[j] / den));
            for (0..rows) |row| {
                var square: f32 = 0;
                for (0..H) |d| {
                    var mixed: f32 = 0;
                    for (0..TAPS) |j| mixed += alpha[j] * self.f[0][(row * TAPS + j) * H + d];
                    square += mixed * mixed;
                }
                const scale: f32 = 1.0 / @sqrt(square / H + 1e-6);
                for (0..H) |d| {
                    var mixed: f32 = 0;
                    for (0..TAPS) |j| mixed += alpha[j] * self.f[0][(row * TAPS + j) * H + d];
                    const out = bf(mixed * scale * self.f[2][fusion * H + d]);
                    want[(fusion * rows + row) * H + d] = out;
                    if (fusion + 1 == FUSIONS and row + 1 == rows) want[FUSIONS * rows * H + d] = out;
                }
            }
        }
    }
    fn qkvReference(self: *Probe, want: []f64) void {
        for (0..8) |row| for (0..48) |h| for (0..128) |d| {
            const at = row * Q + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const other_at = row * Q + h * 128 + partner;
            const scale = rms(self.f[0], row * Q + h * 128, 128);
            const x: f32 = @floatCast(bf(self.f[0][at] * scale * self.f[3][d]));
            const other: f32 = @floatCast(bf(self.f[0][other_at] * scale * self.f[3][partner]));
            want[at] = rope(d, self.shape.k + row, x, other);
        };
        const offset = 8 * Q;
        for (0..8) |row| for (0..8) |h| for (0..128) |d| {
            const at = row * KV + h * 128 + d;
            const partner = if (d < 64) d + 64 else d - 64;
            const other_at = row * KV + h * 128 + partner;
            const scale = rms(self.f[1], row * KV + h * 128, 128);
            const x: f32 = @floatCast(bf(self.f[1][at] * scale * self.f[4][d]));
            const other: f32 = @floatCast(bf(self.f[1][other_at] * scale * self.f[4][partner]));
            want[offset + at] = rope(d, self.shape.k + row, x, other);
            want[offset + 8 * KV + at] = self.f[2][at];
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
        for (0..8) |row| for (0..48) |h| {
            const kv = h / 6;
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

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .init_rows => for (0..8) |row| {
                for (0..H) |d| want[row * H + d] = self.f[if (row == 0) 0 else 1][d];
            },
            .fuse_taps => self.fusionReference(want),
            .block_qkv => self.qkvReference(want),
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
            .kappa => for (0..self.shape.rows) |row| for (0..RNN) |channel| {
                var sum: f32 = 0;
                for (0..H) |d| sum += self.f[0][row * H + d] * self.f[2][channel * RNN_COLS + 2 * RANK + d];
                for (0..H) |d| sum += self.f[1][d] * self.f[2][channel * RNN_COLS + 2 * RANK + H + d];
                sum += self.f[3][channel];
                want[row * RNN + channel] = sum;
            },
            .base => for (0..self.shape.rows) |row| for (0..self.shape.n) |id| {
                var dot: f32 = 0;
                for (0..H) |d| dot += self.f[0][id * H + d] * self.f[1][row * H + d];
                want[row * self.shape.n + id] = bf(dot);
            },
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
                        .fuse_taps, .block_qkv, .residual_norm, .base => 0.02,
                        .attention => 0.03,
                        else => 0,
                    };
                    if (@abs(@as(f64, c.fromBf16(bits)) - w) > @max(1e-5, relative * @abs(w))) return "BF16 endpoint differs from the explicit FP32/FP64 reference";
                }
            }
        }
        if (self.kind == .base and gate == .shapes and self.f[0][0] == 1 and want[0] != H) return "raw base projection was softened instead of rounded once";
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
            const text = try controls.source(s.a, sources.inkling2_small_dspark, @fromBackingInt(@intCast(@backingInt(kind))));
            defer s.a.free(text);
            probe.* = try Probe.initSource(s.a, s.device, s.queue, kind, text);
        } else probe.* = try Probe.init(s.a, s.device, s.queue, kind);
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var k = probe.kernel();
        if (control) {
            k.spec.name = name;
            k.control = .shapes;
        }
        try s.kernels.append(s.a, k);
    }
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(host.Kind)) |kind| {
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
