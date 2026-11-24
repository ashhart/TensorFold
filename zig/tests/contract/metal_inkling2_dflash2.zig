//! Mini DFlash 2's eleven real Metal launches with guarded buffers, FP64/FP32 references and red controls.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_dflash2_controls.zig");
const ref = @import("metal_inkling2_dflash2_ref.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_dflash2_host;
const sources = @import("inkling2_metal");

const H = ref.H;
const Q = ref.Q;
const KV = ref.KV;
const QH = ref.QH;
const KVH = ref.KVH;
const D = ref.D;
const G = ref.G;
const INTER = ref.INTER;
const RANK = ref.RANK;
const GROUPS = ref.GROUPS;
const BLOCK = ref.BLOCK;
const PART = ref.PART;
const EPS = ref.EPS;
const THETA = ref.THETA;

fn depth(s: c.Shape) usize {
    return s.k + 16;
}
fn rounded(values: []f32) void {
    for (values) |*v| v.* = ref.bf(v.*);
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: host.Kind,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    served: [6]c.Shape = undefined,
    served_len: usize = 0,
    f: [6][]f32 = @splat(&.{}),
    u: [3][]u32 = @splat(&.{}),
    refs: [6]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    attend_args: host.AttendArgs = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: host.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.inkling2_dflash2);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: host.Kind, text: []const u8) !Probe {
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .pipes = try host.Pipelines.loadSource(device, text) };
        switch (kind) {
            .embed => self.setServed(&.{ .{ .rows = BLOCK, .n = H, .k = 0 }, .{ .rows = BLOCK, .n = H, .k = 1 } }),
            .norm => self.setServed(&.{ .{ .rows = BLOCK, .n = H, .k = 0 }, .{ .rows = BLOCK, .n = H, .k = 1 } }),
            .norm16 => self.setServed(&.{ .{ .rows = 5, .n = H, .k = 0 }, .{ .rows = 1, .n = H, .k = 0 }, .{ .rows = BLOCK, .n = H, .k = 0 } }),
            .conv => self.setServed(&.{ .{ .rows = BLOCK, .n = H, .k = 0 }, .{ .rows = BLOCK, .n = H, .k = 1 } }),
            .ctx_kv => self.setServed(&.{ .{ .rows = 5, .n = KV, .k = 17 }, .{ .rows = 1, .n = KV, .k = 31 }, .{ .rows = BLOCK, .n = KV, .k = 67 } }),
            .block_qkv => self.setServed(&.{ .{ .rows = BLOCK, .n = Q, .k = 17 }, .{ .rows = BLOCK, .n = Q, .k = 67 } }),
            .attend => self.setServed(&.{ .{ .rows = BLOCK, .n = 0, .k = 0 }, .{ .rows = BLOCK, .n = 17, .k = 0 }, .{ .rows = BLOCK, .n = 140, .k = 1 } }),
            .attend_merge => self.setServed(&.{ .{ .rows = BLOCK, .n = 1, .k = 0 }, .{ .rows = BLOCK, .n = 2, .k = 0 }, .{ .rows = BLOCK, .n = 3, .k = 0 } }),
            .silu_mul => self.setServed(&.{.{ .rows = BLOCK, .n = INTER, .k = 0 }}),
            .topk16 => self.setServed(&.{ .{ .rows = 7, .n = 64, .k = 0 }, .{ .rows = 7, .n = 4096, .k = 0 }, .{ .rows = 7, .n = 256, .k = 1 } }),
            .walk => self.setServed(&.{ .{ .rows = 7, .n = 64, .k = 1 }, .{ .rows = 7, .n = 64, .k = 0 } }),
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
            .out = switch (self.kind) {
                .embed, .attend, .topk16, .walk => .f32,
                else => .bf16,
            },
            .served = self.served[0..self.served_len],
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "explicit serial FP32 reference and required BF16/guard invariants checked directly" },
        } };
    }
    pub fn count(self: *Probe) usize {
        return switch (self.kind) {
            .embed => self.shape.rows * H,
            .norm => 2 * self.shape.rows * H,
            .norm16, .conv => self.shape.rows * H,
            .ctx_kv => 2 * self.shape.rows * KV,
            .block_qkv => self.shape.rows * (Q + 2 * KV),
            .attend => self.shape.rows * QH * self.attend_args.chunks * PART,
            .attend_merge => self.shape.rows * Q,
            .silu_mul => self.shape.rows * INTER,
            .topk16 => 2 * self.shape.rows * host.top_k,
            .walk => 1 + self.shape.rows,
        };
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
        return values;
    }
    fn inputF(self: *Probe, values: []const f32, is_bf: bool) !rig_.Ref {
        if (!is_bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }
    fn output(self: *Probe, slot: usize, elements: usize, elem: c.Elem) !void {
        self.outs[slot] = try self.rig.output(elements * c.size(elem));
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
        self.fill = fill;
        self.f = @splat(&.{});
        self.u = @splat(&.{});
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        const rows = shape.rows;

        switch (self.kind) {
            .embed => {
                const vocab: usize = 32;
                self.u[0] = try self.arena.allocator().alloc(u32, rows);
                for (self.u[0], 0..) |*id, i| id.* = @intCast((i * 5 + 3) % vocab);
                self.f[0] = try self.floats(vocab * H, rng);
                self.f[1] = try self.floats(H, rng);
                for (self.f[1]) |*w| w.* = 1.5 + w.*;
                if (nf) self.f[0][self.u[0][0] * H] = std.math.nan(f32);
                rounded(self.f[0]);
                rounded(self.f[1]);
                self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.u[0]));
                self.refs[1] = try self.inputF(self.f[0], true);
                self.refs[2] = try self.inputF(self.f[1], true);
                try self.output(0, rows * H, .f32);
            },
            .norm => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(rows * H, rng);
                self.f[2] = try self.floats(H, rng);
                for (self.f[2]) |*w| w.* = 1.25 + w.*;
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[1]);
                rounded(self.f[2]);
                try self.output(0, rows * H, .f32);
                @memcpy(self.rig.bytes(self.outs[0], rows * H * 4), std.mem.sliceAsBytes(self.f[0]));
                self.refs[0] = try self.inputF(self.f[1], true);
                self.refs[1] = try self.inputF(self.f[2], true);
                try self.output(1, rows * H, .bf16);
            },
            .norm16 => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(H, rng);
                for (self.f[1]) |*w| w.* = 1.5 + w.*;
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[0]);
                rounded(self.f[1]);
                self.refs[0] = try self.inputF(self.f[0], true);
                self.refs[1] = try self.inputF(self.f[1], true);
                try self.output(0, rows * H, .bf16);
            },
            .conv => {
                self.f[0] = try self.floats(rows * H, rng);
                self.f[1] = try self.floats(rows * 4 * GROUPS, rng);
                self.f[2] = try self.floats(4 * H, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |v| rounded(v);
                for (0..3) |j| self.refs[j] = try self.inputF(self.f[j], true);
                try self.output(0, rows * H, .bf16);
            },
            .ctx_kv => {
                self.f[0] = try self.floats(rows * KV, rng);
                self.f[1] = try self.floats(rows * KV, rng);
                self.f[2] = try self.floats(D, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |v| rounded(v);
                for (0..3) |j| self.refs[j] = try self.inputF(self.f[j], true);
                try self.output(0, rows * KV, .bf16);
                try self.output(1, rows * KV, .bf16);
            },
            .block_qkv => {
                const sizes = [_]usize{ rows * Q, rows * KV, rows * KV, D, D };
                for (sizes, 0..) |n, j| {
                    self.f[j] = try self.floats(n, rng);
                    rounded(self.f[j]);
                }
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (0..5) |j| self.refs[j] = try self.inputF(self.f[j], true);
                try self.output(0, rows * Q, .bf16);
                try self.output(1, rows * KV, .bf16);
                try self.output(2, rows * KV, .bf16);
            },
            .attend => {
                const retained = shape.n;
                const ring = @max(retained, 1);
                const keys = retained + rows;
                const chunks = (keys + 127) / 128;
                self.attend_args = .{
                    .rows = @intCast(rows),
                    .anchor = @intCast(retained + 20),
                    .from = 20,
                    .lo = 20,
                    .retained = @intCast(retained),
                    .keys = @intCast(keys),
                    .window = if (shape.k != 0) 64 else 2048,
                    .ring = @intCast(ring),
                    .chunks = @intCast(chunks),
                    .scale = 1.0 / @sqrt(@as(f32, D)),
                };
                const sizes = [_]usize{ rows * Q, rows * KV, rows * KV, ring * KV, ring * KV };
                for (sizes, 0..) |n, j| {
                    self.f[j] = try self.floats(n, rng);
                    rounded(self.f[j]);
                }
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (0..5) |j| self.refs[j] = try self.inputF(self.f[j], true);
                try self.output(0, rows * QH * chunks * PART, .f32);
            },
            .attend_merge => {
                const chunks = shape.n;
                self.attend_args = .{ .rows = @intCast(rows), .anchor = 0, .from = 0, .lo = 0, .retained = 0, .keys = @intCast(chunks * 128), .window = 2048, .ring = 1, .chunks = @intCast(chunks), .scale = 1.0 };
                self.f[0] = try self.floats(rows * QH * chunks * PART, rng);
                for (0..rows) |r| for (0..KVH) |kh| for (0..chunks) |ch| for (0..G) |gg| {
                    const base = (((r * KVH + kh) * chunks + ch) * G + gg) * PART;
                    self.f[0][base + D] = if (chunks >= 3 and ch == 1 and r == 0 and kh == 0 and gg == 0)
                        -std.math.inf(f32)
                    else
                        @as(f32, @floatFromInt(ch)) * 1.5 - 0.5 + rng.float(f32) * 0.25;
                    self.f[0][base + D + 1] = if (self.f[0][base + D] == -std.math.inf(f32)) 0.0 else 0.5 + rng.float(f32) * 2.0;
                };
                if (nf) self.f[0][D] = std.math.nan(f32);
                self.refs[0] = try self.inputF(self.f[0], false);
                try self.output(0, rows * Q, .bf16);
            },
            .silu_mul => {
                self.f[0] = try self.floats(rows * INTER, rng);
                self.f[1] = try self.floats(rows * INTER, rng);
                for (self.f[0]) |*v| v.* *= 8.0;
                if (nf) self.f[0][0] = std.math.nan(f32);
                rounded(self.f[0]);
                rounded(self.f[1]);
                self.refs[0] = try self.inputF(self.f[0], true);
                self.refs[1] = try self.inputF(self.f[1], true);
                try self.output(0, rows * INTER, .bf16);
            },
            .topk16 => {
                const vocab = shape.n;
                const stride = vocab + 16;
                self.f[0] = try self.floats(rows * stride, rng);
                for (self.f[0]) |*v| v.* = rng.float(f32) * 40.0 - 20.0;
                if (shape.k != 0) {
                    // Plant > 64 ties at the cut boundary to exercise the overflow tie-break path.
                    for (0..rows) |r| {
                        for (0..vocab) |tok| self.f[0][r * stride + tok] = 1.0;
                        self.f[0][r * stride + 5] = 5.0;
                        self.f[0][r * stride + 9] = 4.0;
                    }
                }
                if (nf) {
                    // Plant NaN, +inf, and -inf in separate rows while leaving other rows finite.
                    self.f[0][0 * stride + 3] = std.math.nan(f32);
                    self.f[0][1 * stride + 7] = std.math.inf(f32);
                    self.f[0][2 * stride + 11] = -std.math.inf(f32);
                }
                rounded(self.f[0]);
                self.refs[0] = try self.inputF(self.f[0], true);
                try self.output(0, rows * host.top_k, .f32);
                try self.output(1, rows * host.top_k, .f32);
            },
            .walk => {
                const slots = rows;
                const vocab = shape.n;
                const top: usize = host.top_k;
                self.f[0] = try self.floats(slots * top, rng);
                self.u[0] = try self.arena.allocator().alloc(u32, slots * top);
                self.f[1] = try self.floats(slots * RANK, rng);
                self.f[2] = try self.floats(vocab * RANK, rng);
                self.f[3] = try self.floats(vocab * RANK, rng);
                for (self.f[0]) |*v| v.* = 0.05 * v.*;
                for (self.f[1]) |*v| v.* = 1.0 + v.*;
                for (self.f[2]) |*v| v.* = 0.5 * v.*;
                for (self.f[3]) |*v| v.* = 0.5 * v.*;
                for (0..slots) |e| {
                    for (0..top) |ci| self.u[0][e * top + ci] = @intCast((e * 7 + ci * 3 + 1) % vocab);
                }
                // Bias transition scores toward candidate 5 so dropping the codebook term flips the choice.
                const seed_tok: u32 = 2;
                const target_cand = self.u[0][5];
                for (0..RANK) |d| {
                    self.f[1][d] = 1.0;
                    self.f[2][seed_tok * RANK + d] = 0.5;
                    self.f[3][target_cand * RANK + d] = 1.0;
                }
                self.f[0][0] = 2.0;
                self.f[0][5] = 0.0;
                for (self.f[1..4]) |v| rounded(v);
                if (nf) {
                    // Slot 0 mimics a rejected non-finite topk16 row so slot 1 must guard prev before reading A.
                    for (0..top) |ci| {
                        self.u[0][ci] = 0xffffffff;
                        self.f[0][ci] = -std.math.inf(f32);
                    }
                }
                self.refs[0] = try self.inputF(self.f[0], false);
                self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.u[0]));
                self.refs[2] = try self.inputF(self.f[1], true);
                self.refs[3] = try self.inputF(self.f[2], true);
                self.refs[4] = try self.inputF(self.f[3], true);
                try self.output(0, 1 + slots, .f32);
                const seq_init = try self.arena.allocator().alloc(u32, 1 + slots);
                seq_init[0] = seed_tok;
                for (seq_init[1..]) |*v| v.* = 0x12345678;
                @memcpy(self.rig.bytes(self.outs[0], (1 + slots) * 4), std.mem.sliceAsBytes(seq_init));
            },
        }
        return true;
    }

    fn hostRef(r: rig_.Ref) host.Ref {
        return .{ .buf = r.buf, .off = r.off };
    }
    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const e = pass.e;
        const R = self.refs;
        const O = self.outs;
        const p = &self.pipes;
        switch (self.kind) {
            .embed => host.embed(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), .{ .rows = @intCast(self.shape.rows), .width = H, .use_norm = @intCast(self.shape.k), .eps = EPS }),
            .norm => host.norm(p, e, hostRef(O[0]), hostRef(R[0]), hostRef(R[1]), hostRef(O[1]), .{ .rows = @intCast(self.shape.rows), .width = H, .eps = EPS, .add = @intCast(self.shape.k) }),
            .norm16 => host.norm16(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(O[0]), .{ .rows = @intCast(self.shape.rows), .width = H, .eps = EPS, .add = 0 }),
            .conv => host.conv(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), .{ .rows = @intCast(self.shape.rows), .width = H, .block = BLOCK, .groups = GROUPS, .side = @intCast(self.shape.k) }),
            .ctx_kv => host.ctxKv(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(O[0]), hostRef(O[1]), .{ .rows = @intCast(self.shape.rows), .pos0 = @intCast(self.shape.k), .ring = @intCast(self.shape.rows), .eps = EPS, .theta = THETA }),
            .block_qkv => host.blockQkv(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(R[4]), hostRef(O[0]), hostRef(O[1]), hostRef(O[2]), .{ .rows = @intCast(self.shape.rows), .block = BLOCK, .pos0 = @intCast(self.shape.k), .eps = EPS, .theta = THETA }),
            .attend => host.attendPart(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(R[4]), hostRef(O[0]), self.attend_args),
            .attend_merge => host.attendMerge(p, e, hostRef(R[0]), hostRef(O[0]), self.attend_args),
            .silu_mul => host.siluMul(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(O[0]), @intCast(self.shape.rows * INTER)),
            .topk16 => host.topk16(p, e, hostRef(R[0]), hostRef(O[0]), hostRef(O[1]), .{ .rows = @intCast(self.shape.rows), .vocab = @intCast(self.shape.n), .k = host.top_k, .stride = @intCast(self.shape.n + 16) }),
            .walk => host.walk(p, e, hostRef(R[0]), hostRef(R[1]), hostRef(R[2]), hostRef(R[3]), hostRef(R[4]), hostRef(O[0]), .{ .slots = @intCast(self.shape.rows), .rank = RANK, .top = host.top_k, .vocab = @intCast(self.shape.n), .softcap = if (self.shape.k != 0) 30.0 else 0.0 }),
        }
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        ref.computeReference(self.kind, self.shape, self.attend_args, self.f, self.u, want, mag);
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return "numeric hooks must be used";
        const bytes = try self.rig.outputBytes();
        if (self.kind == .topk16) return ref.checkTopk16(self.a, self.shape, self.f[0], bytes);
        if (self.kind == .walk) return ref.checkWalk(self.shape, self.f, self.u[0], bytes);
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        return ref.checkNumeric(self.kind, self.shape, want, bytes);
    }
};

fn controlKind(ctl: controls.Control) host.Kind {
    return switch (ctl) {
        .embed => .embed,
        .norm => .norm,
        .norm16 => .norm16,
        .conv => .conv,
        .ctx_kv => .ctx_kv,
        .block_qkv => .block_qkv,
        .attend => .attend,
        .attend_merge => .attend_merge,
        .silu_mul => .silu_mul,
        .topk16, .topk16_nonfinite => .topk16,
        .walk, .walk_nonfinite => .walk,
    };
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn addKernel(s: *Suite, kind: host.Kind, only: ?[]const u8) !void {
        const name = kind.label();
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        probe.* = try Probe.init(s.a, s.device, s.queue, kind);
        errdefer probe.deinit();
        try controls.publishProbe(Probe, s.a, &s.probes, s.kernels, probe, probe.kernel());
    }

    fn addControl(s: *Suite, ctl: controls.Control, only: ?[]const u8) !void {
        const name = controls.name(ctl);
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        const text = try controls.source(s.a, sources.inkling2_dflash2, ctl);
        defer s.a.free(text);
        probe.* = try Probe.initSource(s.a, s.device, s.queue, controlKind(ctl), text);
        errdefer probe.deinit();
        var k = probe.kernel();
        k.spec.name = name;
        k.control = ctl.gate();
        try controls.publishProbe(Probe, s.a, &s.probes, s.kernels, probe, k);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(host.Kind)) |kind| try s.addKernel(kind, only);
        for (std.enums.values(controls.Control)) |ctl| try s.addControl(ctl, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};

test "DFlash2 Suite addKernel and addControl keep probe ownership transactional on OOM" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const base = std.testing.allocator;
    for ([_]bool{ false, true }) |is_control| {
        var fail_at: usize = 0;
        var saw_kernel_oom = false;
        while (fail_at < 16) : (fail_at += 1) {
            var kernels: std.ArrayList(g.Kernel) = .empty;
            defer kernels.deinit(base);
            var failing = std.testing.FailingAllocator.init(base, .{ .fail_index = fail_at });
            var suite: Suite = .{ .a = failing.allocator(), .device = device, .queue = queue, .kernels = &kernels };
            defer suite.deinit();
            const res = if (is_control) suite.addControl(.walk_nonfinite, null) else suite.addKernel(.embed, null);
            if (res) |_| {
                try std.testing.expectEqual(@as(usize, 1), suite.probes.items.len);
                try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
                break;
            } else |err| {
                try std.testing.expectEqual(error.OutOfMemory, err);
                try std.testing.expectEqual(@as(usize, 0), suite.probes.items.len);
                try std.testing.expectEqual(@as(usize, 0), kernels.items.len);
                if (suite.probes.capacity > 0) saw_kernel_oom = true;
            }
        }
        try std.testing.expect(saw_kernel_oom);
    }
}
