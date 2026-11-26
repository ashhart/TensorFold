//! Mini cross-pass messages and rank-64 gates under FP64 oracles and mutation controls.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_loop_controls.zig");
const split_controls = @import("metal_inkling2_split_controls.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.loop;
const p = host.plan;
const sources = @import("inkling2_metal");

fn depth(s: c.Shape) usize {
    return if (s.n == p.message_entries * p.hidden) 2 * s.k + 16 else 2 * p.hidden + s.k + 16;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
const roles_label = "metal inkling2 loop roles tf_inkling2_loop_roles_norm, tf_inkling2_loop_roles_zeta and tf_inkling2_loop_roles_blend";
const split_label = "metal inkling2 loop gates split tf_inkling2_loop_norm, tf_inkling2_loop_down and tf_inkling2_loop_apply";
/// Production's m rows are one token's 48 message entries apart; the cell keeps that stride, with NaN between rows.
const m_stride = p.message_entries * p.hidden;

/// Each down dot opens with +B x0 - B x0 = 0 (B = 2^18, h1 = h0): any other order loses the low terms' bits.
fn orderProbe(h: []f32, down: []f32) void {
    for (0..h.len / p.hidden) |row| h[row * p.hidden + 1] = h[row * p.hidden];
    for (0..p.rank) |r| {
        down[r * 2 * p.hidden] = 0x1p18;
        down[r * 2 * p.hidden + 1] = -0x1p18;
    }
}

fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: [3]c.Shape = undefined,
    f: [6][]f32 = @splat(&.{}),
    refs: [6]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    /// The split gates against main's per-row tf_inkling2_loop_gates: 1, 33 and 64 rows, two down blocks a row.
    split: bool = false,
    role_batch: bool = false,
    scratch: [2]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.loop);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .pipes = try host.Pipelines.loadSource(device, text),
        };
        self.served = switch (kind) {
            .messages => .{
                .{ .rows = 1, .n = p.message_entries * p.hidden, .k = p.hidden },
                .{ .rows = 5, .n = p.message_entries * p.hidden, .k = p.hidden },
                .{ .rows = 7, .n = p.message_entries * p.hidden, .k = p.hidden },
            },
            .gates => .{
                .{ .rows = 1, .n = p.hidden, .k = p.rank },
                .{ .rows = 5, .n = p.hidden, .k = p.rank },
                .{ .rows = 7, .n = p.hidden, .k = p.rank },
            },
        };
        return self;
    }
    pub fn initSplit(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, text: []const u8) !Probe {
        var self = try initSource(a, device, queue, .gates, text);
        self.split = true;
        self.served = .{ .{ .rows = 1, .n = p.hidden, .k = p.rank }, .{ .rows = 33, .n = p.hidden, .k = p.rank }, .{ .rows = 64, .n = p.hidden, .k = p.rank } };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .spec = .{
            .name = if (self.role_batch) roles_label else if (self.split) split_label else self.kind.label(),
            .replaces = if (self.split) "tf_inkling2_loop_gates" else null,
            .accum = .f32,
            .depth = depth,
            .ops = if (self.kind == .messages) 32 else 64,
            .mids = if (self.kind == .gates) 2 else 0,
            .out = .f32,
            .served = &self.served,
            .sums = .{},
        } };
    }
    pub fn count(self: *Probe) usize {
        return p.count(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = value(rng);
        return values;
    }
    fn input(self: *Probe, values: []const f32, bf: bool) !rig_.Ref {
        if (!bf) return self.rig.input(std.mem.sliceAsBytes(values));
        const bits = try self.arena.allocator().alloc(u16, values.len);
        for (bits, values) |*b, v| b.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(bits));
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const call = p.Call{ .rows = shape.rows, .n = shape.n, .k = shape.k };
        if (!p.valid(self.kind, call)) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.f = @splat(&.{});
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        switch (self.kind) {
            .messages => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(2 * p.core_layers * p.hidden, rng);
                self.f[3] = try self.floats(2 * p.core_layers * p.hidden, rng);
                if (fill == .stress) {
                    // rho(+256) = +1 and rho(-256) = -1 in FP32: their square sum is 9 * 2^24.
                    @memset(self.f[0], 256);
                    @memset(self.f[1], -256);
                    // Mirrored integer queries leave 8, while every partial sum stays below 2^24.
                    for ([_][]f32{ self.f[2], self.f[3] }) |queries| for (0..2 * p.core_layers) |query_row| {
                        const row = queries[query_row * p.hidden ..][0..p.hidden];
                        for (0..p.hidden / 2) |d| {
                            const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                            const nudge: f32 = if (d < 8) 1 else 0;
                            row[d] = v;
                            row[p.hidden / 2 + d] = nudge - v;
                        }
                    };
                }
                if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
                round(self.f[0]);
                round(self.f[1]);
                self.refs[0] = try self.input(self.f[0], true);
                self.refs[1] = try self.input(self.f[1], true);
                self.refs[2] = try self.input(self.f[2], false);
                self.refs[3] = try self.input(self.f[3], false);
            },
            .gates => {
                self.f[0] = try self.floats(r * p.hidden, rng);
                self.f[1] = try self.floats(r * p.hidden, rng);
                self.f[2] = try self.floats(p.rank * 2 * p.hidden, rng);
                self.f[3] = try self.floats(p.hidden * p.rank, rng);
                self.f[4] = try self.floats(p.hidden * p.rank, rng);
                if (fill == .stress) {
                    @memset(self.f[0], 256);
                    @memset(self.f[1], -256);
                    // D sees [+1; -1]: mirrored integer rows leave zeta = 1, with exact FP32 partials.
                    for (0..p.rank) |rank| for (0..p.hidden) |d| {
                        const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
                        self.f[2][rank * 2 * p.hidden + d] = w;
                        self.f[2][rank * 2 * p.hidden + p.hidden + d] = if (d == 0) w - 1 else w;
                    };
                    // With zeta = 1, mirrored F/Gamma rows leave one, also exactly in every FP32 order.
                    for ([_][]f32{ self.f[3], self.f[4] }) |weights| for (0..p.hidden) |d| for (0..p.rank / 2) |rank| {
                        const w: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                        weights[d * p.rank + rank] = w;
                        weights[d * p.rank + rank + p.rank / 2] = if (rank == 0) 1 - w else -w;
                    };
                }
                if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
                if (self.split and fill == .random) orderProbe(self.f[0], self.f[2]);
                round(self.f[2]);
                round(self.f[3]);
                round(self.f[4]);
                self.refs[0] = try self.input(self.f[0], false);
                self.refs[1] = if (self.split) try self.spread(self.f[1]) else try self.input(self.f[1], false);
                self.refs[2] = try self.input(self.f[2], true);
                self.refs[3] = try self.input(self.f[3], true);
                self.refs[4] = try self.input(self.f[4], true);
                if (self.split) self.scratch = .{ try self.rig.scratch(r * 2 * p.hidden * 2), try self.rig.scratch(r * p.rank * 4) };
            },
        }
        self.out = try self.rig.output(self.count() * @sizeOf(f32));
        return true;
    }

    fn at(ref: rig_.Ref) host.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }
    pub fn run(self: *Probe) !void {
        const pass = self.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(self.shape.rows);
        switch (self.kind) {
            .messages => host.crossPassMessageAttention(&self.pipes, e, at(self.refs[0]), at(self.refs[1]), at(self.refs[2]), at(self.refs[3]), at(self.out), rows),
            .gates => if (self.role_batch)
                host.gatesEach(&self.pipes, e, &.{.{ .h = at(self.refs[0]), .m = at(self.refs[1]), .down = at(self.refs[2]), .forget = at(self.refs[3]), .gain = at(self.refs[4]), .out = at(self.out) }}, rows, m_stride)
            else if (self.split)
                host.gatesSplit(&self.pipes, e, at(self.refs[0]), at(self.refs[1]), m_stride, at(self.refs[2]), at(self.refs[3]), at(self.refs[4]), at(self.out), at(self.scratch[0]), at(self.scratch[1]), rows)
            else
                host.gates(&self.pipes, e, at(self.refs[0]), at(self.refs[1]), at(self.refs[2]), at(self.refs[3]), at(self.refs[4]), at(self.out), rows),
        }
        try self.rig.finish(pass);
    }

    /// m's rows at production's stride, NaN between them, so a kernel reading at the wrong stride cannot pass.
    fn spread(self: *Probe, m: []const f32) !rig_.Ref {
        const rows = m.len / p.hidden;
        const values = try self.arena.allocator().alloc(f32, (rows - 1) * m_stride + p.hidden);
        @memset(values, std.math.nan(f32));
        for (0..rows) |row| @memcpy(values[row * m_stride ..][0..p.hidden], m[row * p.hidden ..][0..p.hidden]);
        return self.rig.input(std.mem.sliceAsBytes(values));
    }

    /// Main's single-launch gates a row at a time, as the forward called them before the split.
    pub fn replaced(self: *Probe) ![]const u8 {
        const rows = self.shape.rows;
        const out = try self.rig.scratch(rows * p.hidden * 4);
        const pass = self.rig.begin();
        for (0..rows) |i| {
            const h: host.Ref = .{ .buf = self.refs[0].buf, .off = self.refs[0].off + i * p.hidden * 4 };
            const m: host.Ref = .{ .buf = self.refs[1].buf, .off = self.refs[1].off + i * m_stride * 4 };
            const y: host.Ref = .{ .buf = out.buf, .off = out.off + i * p.hidden * 4 };
            host.gates(&self.pipes, pass.e, h, m, at(self.refs[2]), at(self.refs[3]), at(self.refs[4]), y, 1);
        }
        try self.rig.finish(pass);
        return self.rig.bytes(out, rows * p.hidden * 4);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        switch (self.kind) {
            .messages => for (0..self.shape.rows) |row| {
                const b0 = self.f[0][row * p.hidden ..][0..p.hidden];
                const b1 = self.f[1][row * p.hidden ..][0..p.hidden];
                for (0..p.message_entries) |entry| {
                    const role: p.Role = @fromBackingInt(@intCast(entry % p.roles));
                    const dst = want[(row * p.message_entries + entry) * p.hidden ..][0..p.hidden];
                    p.message(b0, b1, self.f[2], self.f[3], entry / p.roles, role, dst);
                }
            },
            .gates => for (0..self.shape.rows) |row| {
                const h = self.f[0][row * p.hidden ..][0..p.hidden];
                const m = self.f[1][row * p.hidden ..][0..p.hidden];
                p.gate(h, m, self.f[2], self.f[3], self.f[4], want[row * p.hidden ..][0..p.hidden]);
            },
        }
        for (mag, want) |*m, w| m.* = if (self.kind == .messages)
            10 * @abs(w) + 1e-3
        else
            32 * @abs(w) + 8;
    }
    fn simRms(x: []const f32, acc: c.Accum, order: g.Order, terms: []f64) f64 {
        for (x, 0..) |v, d| terms[d] = @as(f64, v) * v;
        const ss = contract.rows.sumIn(acc, order, terms[0..p.hidden]);
        return 1 / @sqrt(ss / @as(f64, @floatFromInt(p.hidden)) + p.eps);
    }

    fn simMessage(bank0: []const f32, bank1: []const f32, q_mlp: []const f32, kv: []const f32, layer: usize, role: p.Role, inverse: [2]f64, acc: c.Accum, order: g.Order, terms: []f64, out: []f64) void {
        const queries = if (p.usesQueryMessageTensor(role)) q_mlp else kv;
        const query = queries[p.queryRow(role, layer) * p.hidden ..][0..p.hidden];
        var logits: [2]f64 = undefined;
        var peak = -std.math.inf(f64);
        inline for (0..2) |s| {
            const bank = if (s == 0) bank0 else bank1;
            for (0..p.hidden) |d| terms[d] = @as(f64, query[d]) * (@as(f64, bank[d]) * inverse[s]);
            logits[s] = contract.rows.sumIn(acc, order, terms[0..p.hidden]) * p.message_scale;
            peak = @max(peak, logits[s]);
        }
        var weights: [2]f64 = undefined;
        var den: f64 = 0;
        for (0..2) |s| {
            weights[s] = @exp(logits[s] - peak);
            den += weights[s];
        }
        for (0..p.hidden) |d| out[d] = weights[0] / den * (@as(f64, bank0[d]) * inverse[0]) + weights[1] / den * (@as(f64, bank1[d]) * inverse[1]);
    }

    fn simGate(h: []const f32, m: []const f32, down: []const f32, forget: []const f32, gain: []const f32, inverse: [2]f64, acc: c.Accum, order: g.Order, terms: []f64, normalized: *[2 * p.hidden]f64, zeta: *[p.rank]f64, out: []f64) void {
        for (0..p.hidden) |d| {
            normalized[d] = p.roundBf16(@as(f64, h[d]) * inverse[0]);
            normalized[p.hidden + d] = p.roundBf16(@as(f64, m[d]) * inverse[1]);
        }
        for (0..p.rank) |r| {
            for (0..2 * p.hidden) |d| terms[d] = @as(f64, down[r * 2 * p.hidden + d]) * normalized[d];
            zeta[r] = p.roundBf16(contract.rows.sumIn(acc, order, terms));
        }
        for (0..p.hidden) |d| {
            for (0..p.rank) |r| terms[r] = @as(f64, forget[d * p.rank + r]) * zeta[r];
            const fdot = contract.rows.sumIn(acc, order, terms[0..p.rank]);
            for (0..p.rank) |r| terms[r] = @as(f64, gain[d * p.rank + r]) * zeta[r];
            const gdot = contract.rows.sumIn(acc, order, terms[0..p.rank]);
            out[d] = p.blend(std.math.tanh(fdot), std.math.tanh(gdot), h[d], m[d]);
        }
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const terms = try self.a.alloc(f64, 2 * p.hidden);
        defer self.a.free(terms);
        switch (self.kind) {
            .messages => for (0..self.shape.rows) |row| {
                const banks = [2][]const f32{
                    self.f[0][row * p.hidden ..][0..p.hidden],
                    self.f[1][row * p.hidden ..][0..p.hidden],
                };
                var inverse: [2]f64 = undefined;
                inline for (0..2) |s| inverse[s] = simRms(banks[s], acc, order, terms);
                for (0..p.message_entries) |entry| {
                    const role: p.Role = @fromBackingInt(@intCast(entry % p.roles));
                    const dst = out[(row * p.message_entries + entry) * p.hidden ..][0..p.hidden];
                    simMessage(banks[0], banks[1], self.f[2], self.f[3], entry / p.roles, role, inverse, acc, order, terms, dst);
                }
            },
            .gates => {
                var normalized: [2 * p.hidden]f64 = undefined;
                var zeta: [p.rank]f64 = undefined;
                for (0..self.shape.rows) |row| {
                    const h = self.f[0][row * p.hidden ..][0..p.hidden];
                    const m = self.f[1][row * p.hidden ..][0..p.hidden];
                    var inverse: [2]f64 = undefined;
                    inverse[0] = simRms(h, acc, order, terms);
                    inverse[1] = simRms(m, acc, order, terms);
                    simGate(h, m, self.f[2], self.f[3], self.f[4], inverse, acc, order, terms, &normalized, &zeta, out[row * p.hidden ..][0..p.hidden]);
                }
            },
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, kind: p.Kind, control: bool, only: ?[]const u8) !void {
        const name = if (control) controls.name(kind) else kind.label();
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        if (control) {
            const text = try controls.source(s.a, sources.loop, kind);
            defer s.a.free(text);
            probe.* = try Probe.initSource(s.a, s.device, s.queue, kind, text);
        } else {
            probe.* = try Probe.init(s.a, s.device, s.queue, kind);
        }
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var k = probe.kernel();
        if (control) {
            k.spec.name = name;
            k.control = .shapes;
        }
        try s.kernels.append(s.a, k);
    }

    fn addSplit(s: *Suite, control: ?split_controls.Control, only: ?[]const u8) !void {
        const name = if (control) |x| split_controls.name(x) else split_label;
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        const text = if (control) |x| try split_controls.source(s.a, sources.loop, x) else try s.a.dupe(u8, sources.loop);
        defer s.a.free(text);
        probe.* = try Probe.initSplit(s.a, s.device, s.queue, text);
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var k = probe.kernel();
        if (control != null) {
            k.spec.name = name;
            k.control = .shapes;
        }
        try s.kernels.append(s.a, k);
    }

    fn addRoles(s: *Suite, control: bool, only: ?[]const u8) !void {
        const name = if (control) "control: " ++ roles_label else roles_label;
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const text = if (control) try @import("metal_inkling2_roles_controls.zig").source(s.a, sources.loop) else try s.a.dupe(u8, sources.loop);
        defer s.a.free(text);
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        probe.* = try Probe.initSplit(s.a, s.device, s.queue, text);
        probe.role_batch = true;
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
        for (std.enums.values(p.Kind)) |kind| {
            try s.add(kind, false, only);
            try s.add(kind, true, only);
        }
        try s.addSplit(null, only);
        try s.addSplit(.down_strided, only);
        try s.addSplit(.down_index, only);
        try s.addRoles(false, only);
        try s.addRoles(true, only);
    }

    /// Verify fused grid.z=3 Q/K/V dispatch matches three separate 1-role dispatches byte-for-byte.
    pub fn multiRole(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        for ([_]usize{ 1, 2, 3, 7, 8, 16, 32, 64 }) |rows| bad += try s.multiRoleRows(seed, rows);
        return bad;
    }

    fn multiRoleRows(s: *Suite, seed: u64, rows: usize) !usize {
        var prod: ?*Probe = null;
        for (s.probes.items) |pr| if (pr.kind == .gates and std.mem.eql(u8, pr.kernel().spec.name, p.Kind.gates.label())) {
            prod = pr;
            break;
        };
        const pr = prod orelse return 0;
        var rig = rig_.Rig{ .a = s.a, .device = s.device, .queue = s.queue };
        defer rig.deinit();
        var prng = std.Random.DefaultPrng.init(seed ^ 0x9e37);
        const rng = prng.random();
        var roles: [3]host.Gate = undefined;
        var sep_outs: [3]rig_.Ref = undefined;
        var arena: std.heap.ArenaAllocator = .init(s.a);
        defer arena.deinit();
        const aa = arena.allocator();
        const m_all = try aa.alloc(f32, rows * m_stride);
        for (m_all) |*v| v.* = value(rng);
        const m_base = try rig.input(std.mem.sliceAsBytes(m_all));
        for (0..3) |r_i| {
            const h = try aa.alloc(f32, rows * p.hidden);
            for (h) |*v| v.* = value(rng);
            const down = try aa.alloc(f32, p.rank * 2 * p.hidden);
            for (down) |*v| v.* = value(rng);
            round(down);
            const forget = try aa.alloc(f32, p.hidden * p.rank);
            for (forget) |*v| v.* = value(rng);
            round(forget);
            const gain = try aa.alloc(f32, p.hidden * p.rank);
            for (gain) |*v| v.* = value(rng);
            round(gain);
            const d_bf = try aa.alloc(u16, down.len);
            for (d_bf, down) |*b, v| b.* = c.toBf16(v);
            const f_bf = try aa.alloc(u16, forget.len);
            for (f_bf, forget) |*b, v| b.* = c.toBf16(v);
            const g_bf = try aa.alloc(u16, gain.len);
            for (g_bf, gain) |*b, v| b.* = c.toBf16(v);
            const out_fused = try rig.output(rows * p.hidden * @sizeOf(f32));
            sep_outs[r_i] = try rig.output(rows * p.hidden * @sizeOf(f32));
            roles[r_i] = .{
                .h = Probe.at(try rig.input(std.mem.sliceAsBytes(h))),
                .m = .{ .buf = m_base.buf, .off = m_base.off + r_i * p.hidden * @sizeOf(f32) },
                .down = Probe.at(try rig.input(std.mem.sliceAsBytes(d_bf))),
                .forget = Probe.at(try rig.input(std.mem.sliceAsBytes(f_bf))),
                .gain = Probe.at(try rig.input(std.mem.sliceAsBytes(g_bf))),
                .out = Probe.at(out_fused),
            };
        }
        const normalized = Probe.at(try rig.output(rows * 2 * p.hidden * 2));
        const zeta = Probe.at(try rig.output(rows * p.rank * 4));
        const pass = rig.begin();
        host.gatesEach(&pr.pipes, pass.e, &roles, @intCast(rows), @intCast(m_stride));
        pass.e.barrier();
        for (roles, 0..) |role, r_i| {
            var solo = role;
            solo.out = Probe.at(sep_outs[r_i]);
            host.gatesSplit(&pr.pipes, pass.e, solo.h, solo.m, m_stride, solo.down, solo.forget, solo.gain, solo.out, normalized, zeta, @intCast(rows));
            pass.e.barrier();
        }
        try rig.finish(pass);
        var bad: usize = 0;
        for (roles, sep_outs, 0..) |role, sep, r_i| {
            const nb = rows * p.hidden * @sizeOf(f32);
            const got = rig.bytes(.{ .buf = role.out.buf, .off = role.out.off }, nb);
            const want = rig.bytes(sep, nb);
            if (std.mem.indexOfDiff(u8, got, want)) |diff| {
                bad += 1;
                std.debug.print("FAIL metal inkling2 loop multiRole role={d}: first diff at byte {d}\n", .{ r_i, diff });
            }
        }
        const word = if (bad == 0) "PASS" else "FAIL";
        std.debug.print("{s} multi-role metal inkling2 loop: grid.z=3 byte-equal to main split gates, {d} differ\n", .{ word, bad });
        return bad;
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
