//! Mini DFlash 2 batched kernel probes (conv, ctx_kv, block_qkv) and planted controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const controls = @import("metal_inkling2_dflash2_batch_controls.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const host = tf.inkling2_dflash2_host;
const sources = @import("inkling2_metal");

const KVH = 8;
const QH = 32;
const D = 128;
const KW = KVH * D;
const QW = QH * D;
const block_size = 8;
const ctx_ring = 64;

fn bf(v: anytype) f64 {
    const x: f32 = @floatCast(v);
    return c.fromBf16(c.toBf16(x));
}

fn rounded(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

fn rms(values: []const f32, start: usize, width: usize) f32 {
    var total: f32 = 0;
    for (0..32) |lane| {
        var sum: f32 = 0;
        var j: usize = 0;
        while (j < width) : (j += 32) {
            const x = values[start + lane + j];
            sum = @mulAdd(f32, x, x, sum);
        }
        total += sum;
    }
    return 1.0 / @sqrt(total / @as(f32, @floatFromInt(width)) + 1e-6);
}

fn rope(d: usize, position: usize, x: f32, other: f32) f64 {
    const freq = 1.0 / std.math.pow(f32, 1e6, @as(f32, @floatFromInt(2 * (d % 64))) / 128.0);
    const angle = @as(f32, @floatFromInt(position)) * freq;
    const cs = c.fromBf16(c.toBf16(@cos(angle)));
    const sn = c.fromBf16(c.toBf16(@sin(angle)));
    const rot: f32 = if (d < 64) -other else other;
    return bf(bf(x * cs) + bf(rot * sn));
}

const Case = enum {
    conv,
    conv_ctrl,
    ctx_kv,
    ctx_kv_ctrl,
    block_qkv,
    block_qkv_ctrl,
};

const case_list = [_]Case{ .conv, .conv_ctrl, .ctx_kv, .ctx_kv_ctrl, .block_qkv, .block_qkv_ctrl };

fn caseName(case: Case) []const u8 {
    return switch (case) {
        .conv => "metal inkling2 dflash2 conv (block isolation)",
        .conv_ctrl => controls.name(.conv),
        .ctx_kv => "metal inkling2 dflash2 ctx_kv (3 streams, ring wrap)",
        .ctx_kv_ctrl => controls.name(.ctx_kv),
        .block_qkv => "metal inkling2 dflash2 block_qkv (3 streams, per-stream RoPE)",
        .block_qkv_ctrl => controls.name(.block_qkv),
    };
}

fn caseControl(case: Case) ?controls.Kind {
    return switch (case) {
        .conv, .ctx_kv, .block_qkv => null,
        .conv_ctrl => .conv,
        .ctx_kv_ctrl => .ctx_kv,
        .block_qkv_ctrl => .block_qkv,
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    case: Case,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: [4]c.Shape = undefined,
    served_len: usize = 0,
    f: [5][]f32 = @splat(&.{}),
    refs: [5]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    rings: [3]host.Ring = undefined,
    segs: [3]host.CtxSeg = undefined,
    seg_len: usize = 0,
    pos0: [host.max_batch]u32 = @splat(0),
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, case: Case) !Probe {
        const pipes = if (caseControl(case)) |ck| blk: {
            const txt = try controls.source(a, sources.inkling2_dflash2, ck);
            defer a.free(txt);
            break :blk try host.Pipelines.loadSource(device, txt);
        } else try host.Pipelines.loadSource(device, sources.inkling2_dflash2);
        var self: Probe = .{ .a = a, .arena = .init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .case = case, .pipes = pipes };
        switch (case) {
            .conv, .conv_ctrl => self.setServed(&.{ .{ .rows = 8, .n = 2304, .k = 144 }, .{ .rows = 16, .n = 2304, .k = 144 }, .{ .rows = 24, .n = 576, .k = 144 } }),
            .ctx_kv, .ctx_kv_ctrl => self.setServed(&.{ .{ .rows = 8, .n = KW, .k = ctx_ring }, .{ .rows = 16, .n = KW, .k = ctx_ring }, .{ .rows = 9, .n = KW, .k = ctx_ring }, .{ .rows = 24, .n = KW, .k = ctx_ring } }),
            .block_qkv, .block_qkv_ctrl => self.setServed(&.{ .{ .rows = 8, .n = QW, .k = 17 }, .{ .rows = 16, .n = QW, .k = 17 }, .{ .rows = 24, .n = QW, .k = 131 } }),
        }
        return self;
    }

    fn setServed(self: *Probe, list: []const c.Shape) void {
        @memcpy(self.served[0..list.len], list);
        self.served_len = list.len;
    }

    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{
            .ctx = self,
            .vt = &vt,
            .control = if (caseControl(self.case) != null) .shapes else null,
            .spec = .{
                .name = caseName(self.case),
                .accum = .none,
                .depth = depthFn,
                .out = .bf16,
                .served = self.served[0..self.served_len],
                .sums = .{ .f16 = 0, .bf16 = 0, .why = "explicit serial FP32 reference and required BF16 nodes" },
            },
        };
    }

    fn depthFn(s: c.Shape) usize {
        return s.k + 16;
    }

    pub fn count(self: *Probe) usize {
        return switch (self.case) {
            .conv, .conv_ctrl => self.shape.rows * self.shape.n,
            .ctx_kv, .ctx_kv_ctrl => 2 * self.seg_len * ctx_ring * KW,
            .block_qkv, .block_qkv_ctrl => self.shape.rows * (QW + 2 * KW),
        };
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const out = try self.arena.allocator().alloc(f32, n);
        for (out) |*v| v.* = rng.float(f32) * 0.5 - 0.25;
        return out;
    }

    fn inputBf(self: *Probe, values: []const f32) !rig_.Ref {
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        var admitted = false;
        for (self.served[0..self.served_len]) |sv| if (std.meta.eql(shape, sv)) {
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
        switch (self.case) {
            .conv, .conv_ctrl => {
                const width = shape.n;
                const groups = shape.k;
                self.f[0] = try self.floats(shape.rows * width, rng);
                self.f[1] = try self.floats(shape.rows * 4 * groups, rng);
                self.f[2] = try self.floats(4 * width, rng);
                if (nf) self.f[0][0] = std.math.nan(f32);
                for (self.f[0..3]) |v| rounded(v);
                for (0..3) |j| self.refs[j] = try self.inputBf(self.f[j]);
                self.outs[0] = try self.rig.output(shape.rows * width * 2);
            },
            .ctx_kv, .ctx_kv_ctrl => {
                const rows = shape.rows;
                self.f[0] = try self.floats(rows * KW, rng);
                self.f[1] = try self.floats(rows * KW, rng);
                self.f[2] = try self.floats(D, rng);
                if (nf) {
                    self.f[0][0] = std.math.nan(f32);
                    self.f[1][KW + 3] = std.math.inf(f32);
                }
                for (self.f[0..3]) |v| rounded(v);
                for (0..3) |j| self.refs[j] = try self.inputBf(self.f[j]);
                const r0: u32 = @intCast(rows / 3);
                const r1: u32 = @intCast((rows - r0) / 2);
                const r2: u32 = @intCast(rows - r0 - r1);
                self.seg_len = 3;
                self.segs = .{
                    .{ .row0 = 0, .rows = r0, .pos0 = 60 },
                    .{ .row0 = r0, .rows = r1, .pos0 = 19 },
                    .{ .row0 = r0 + r1, .rows = r2, .pos0 = 62 },
                };
                for (0..self.seg_len) |s| {
                    const kr = try self.rig.output(ctx_ring * KW * 2);
                    const vr = try self.rig.output(ctx_ring * KW * 2);
                    self.rings[s] = .{ .k = .{ .buf = kr.buf, .off = kr.off }, .v = .{ .buf = vr.buf, .off = vr.off } };
                }
            },
            .block_qkv, .block_qkv_ctrl => {
                const rows = shape.rows;
                const sizes = [_]usize{ rows * QW, rows * KW, rows * KW, D, D };
                for (sizes, 0..) |sz, j| {
                    self.f[j] = try self.floats(sz, rng);
                    if (nf and j == 0) self.f[0][0] = std.math.nan(f32);
                    rounded(self.f[j]);
                    self.refs[j] = try self.inputBf(self.f[j]);
                }
                self.pos0 = .{ @intCast(shape.k), @intCast(shape.k + 7), @intCast(shape.k + 41), 0, 0, 0, 0, 0 };
                self.outs[0] = try self.rig.output(rows * QW * 2);
                self.outs[1] = try self.rig.output(rows * KW * 2);
                self.outs[2] = try self.rig.output(rows * KW * 2);
            },
        }
        return true;
    }

    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const r = struct {
            fn of(x: rig_.Ref) host.Ref {
                return .{ .buf = x.buf, .off = x.off };
            }
        }.of;
        switch (self.case) {
            .conv, .conv_ctrl => {
                const side: u32 = @intCast((self.shape.rows / block_size) % 2);
                host.conv(&self.pipes, pass.e, r(self.refs[0]), r(self.refs[1]), r(self.refs[2]), r(self.outs[0]), .{
                    .rows = @intCast(self.shape.rows),
                    .width = @intCast(self.shape.n),
                    .block = block_size,
                    .groups = @intCast(self.shape.k),
                    .side = side,
                });
            },
            .ctx_kv, .ctx_kv_ctrl => host.ctxKv(&self.pipes, pass.e, r(self.refs[0]), r(self.refs[1]), r(self.refs[2]), self.rings[0..self.seg_len], .{
                .rows = @intCast(self.shape.rows),
                .segs = @intCast(self.seg_len),
                .ring = ctx_ring,
                .eps = 1e-6,
                .theta = 1e6,
            }, self.segs[0..self.seg_len]),
            .block_qkv, .block_qkv_ctrl => host.blockQkv(&self.pipes, pass.e, r(self.refs[0]), r(self.refs[1]), r(self.refs[2]), r(self.refs[3]), r(self.refs[4]), r(self.outs[0]), r(self.outs[1]), r(self.outs[2]), .{
                .rows = @intCast(self.shape.rows),
                .block = block_size,
                .eps = 1e-6,
                .theta = 1e6,
                .pos0 = self.pos0,
            }),
        }
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        switch (self.case) {
            .conv, .conv_ctrl => {
                const width = self.shape.n;
                const groups = self.shape.k;
                const side: usize = (self.shape.rows / block_size) % 2;
                const gw = width / groups;
                for (0..self.shape.rows) |row| for (0..width) |ch| {
                    const gr = ch / gw;
                    const d = self.f[1][row * 4 * groups + side * 2 * groups ..];
                    const c0: f32 = @floatCast(bf(self.f[2][side * 2 * width + ch] + d[gr]));
                    const c1: f32 = @floatCast(bf(self.f[2][(side * 2 + 1) * width + ch] + d[groups + gr]));
                    var y: f64 = bf(c0 * self.f[0][row * width + ch]);
                    if (row % block_size >= 1) y = bf(y + bf(c1 * self.f[0][(row - 1) * width + ch]));
                    want[row * width + ch] = y;
                };
            },
            .ctx_kv, .ctx_kv_ctrl => {
                const slot_elems = ctx_ring * KW;
                for (want) |*w| w.* = std.math.nan(f64);
                for (self.segs[0..self.seg_len], 0..) |seg, s| {
                    const k_base = (2 * s) * slot_elems;
                    const v_base = (2 * s + 1) * slot_elems;
                    for (0..seg.rows) |off_r| {
                        const row = seg.row0 + off_r;
                        const pos = seg.pos0 + off_r;
                        const slot = pos % ctx_ring;
                        for (0..KVH) |h| {
                            const scale = rms(self.f[0], row * KW + h * D, D);
                            for (0..D) |d| {
                                const src = row * KW + h * D + d;
                                const partner = if (d < 64) d + 64 else d - 64;
                                const x: f32 = @floatCast(bf(self.f[2][d] * bf(self.f[0][src] * scale)));
                                const other: f32 = @floatCast(bf(self.f[2][partner] * bf(self.f[0][row * KW + h * D + partner] * scale)));
                                want[k_base + (slot * KVH + h) * D + d] = rope(d, pos, x, other);
                                want[v_base + (slot * KVH + h) * D + d] = self.f[1][src];
                            }
                        }
                    }
                }
            },
            .block_qkv, .block_qkv_ctrl => {
                const rows = self.shape.rows;
                for (0..rows) |row| {
                    const stream = row / block_size;
                    const pos = self.pos0[stream] + (row % block_size);
                    for (0..QH) |h| {
                        const scale = rms(self.f[0], row * QW + h * D, D);
                        for (0..D) |d| {
                            const src = row * QW + h * D + d;
                            const partner = if (d < 64) d + 64 else d - 64;
                            const x: f32 = @floatCast(bf(self.f[3][d] * bf(self.f[0][src] * scale)));
                            const other: f32 = @floatCast(bf(self.f[3][partner] * bf(self.f[0][row * QW + h * D + partner] * scale)));
                            want[src] = rope(d, pos, x, other);
                        }
                    }
                    for (0..KVH) |h| {
                        const scale = rms(self.f[1], row * KW + h * D, D);
                        for (0..D) |d| {
                            const src = row * KW + h * D + d;
                            const partner = if (d < 64) d + 64 else d - 64;
                            const x: f32 = @floatCast(bf(self.f[4][d] * bf(self.f[1][src] * scale)));
                            const other: f32 = @floatCast(bf(self.f[4][partner] * bf(self.f[1][row * KW + h * D + partner] * scale)));
                            want[rows * QW + src] = rope(d, pos, x, other);
                            want[rows * (QW + KW) + src] = self.f[2][src];
                        }
                    }
                }
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
        const is_ctx = self.case == .ctx_kv or self.case == .ctx_kv_ctrl;
        for (want, 0..) |w, i| {
            const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
            if (is_ctx and std.math.isNan(w) and bits == 0xffff) continue;
            if (std.math.isNan(w)) {
                if (!std.math.isNan(c.fromBf16(bits))) return "BF16 nonfinite value was not preserved";
            } else {
                const got = c.fromBf16(bits);
                const is_conv = self.case == .conv or self.case == .conv_ctrl;
                const rel: f64 = if (is_conv) 0 else 0.02;
                const abs_tol: f64 = if (is_conv) 0 else 5e-4;
                if (@abs(@as(f64, got) - w) > @max(abs_tol, rel * @abs(w))) {
                    return try std.fmt.bufPrint(&self.msg, "mismatch at {d}: got {d} want {d}", .{ i, got, w });
                }
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

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (case_list) |case| {
            const label = caseName(case);
            if (only) |o| if (std.mem.indexOf(u8, label, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, case);
            errdefer p.deinit();
            try contract.register(s.a, &s.probes, s.kernels, p, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
