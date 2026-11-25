//! Inkling 2 Mini Metal tree conv/compaction gates 1-4 and shared tree contract Suite.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_inkling2_tree_controls.zig");
const attn_ = @import("metal_inkling2_tree_attn.zig");
const host = @import("tensorfold").inkling2_metal_host.tree;
const p = host.plan;
const c = contract.spec;
const g = contract.gates;
const Which = controls.Which;
const Mutation = controls.Mutation;

const bf16 = c.fromBf16;
fn toBf(x: f32) u16 {
    return c.toBf16(x);
}
fn bits(x: f64) u16 {
    return c.toBf16(@floatCast(x));
}
fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

const static_shapes = [_]c.Shape{ .{ .rows = 5, .n = 1024, .k = 4 }, .{ .rows = 6, .n = 2304, .k = 4 } };
const dynamic_shapes = [_]c.Shape{ .{ .rows = 5, .n = 1024, .k = 4 }, .{ .rows = 6, .n = 512, .k = 4 } };
const kv_shapes = [_]c.Shape{ .{ .rows = 6, .n = 128, .k = 1088 }, .{ .rows = 5, .n = 128, .k = 2048 } };
const conv_shapes = [_]c.Shape{ .{ .rows = 5, .n = p.conv_row_values, .k = 8 }, .{ .rows = 6, .n = p.conv_row_values, .k = 16 } };

fn depthStatic(_: c.Shape) usize {
    return 8;
}
fn depthDynamic(_: c.Shape) usize {
    return p.hidden + 8;
}
fn depthCopy(_: c.Shape) usize {
    return 1;
}

pub const OpProbe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    which: Which,
    mutation: Mutation,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: []const c.Shape = &.{},
    src_rows: []u32 = &.{},
    f: [5][]f32 = @splat(&.{}),
    kv_in: []u16 = &.{},
    conv_in: []u16 = &.{},
    taps_u16: []u16 = &.{},
    refs: [6]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    ring: u32 = 1088,
    start: u32 = 1084,
    keep_path: [p.window_rows]u32 = @splat(0),
    keep_path_len: usize = 0,
    conv_src3: [3]u32 = .{ 0, 1, 2 },

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, w: Which, m: Mutation) !OpProbe {
        const text = try controls.apply(a, w.baseSource(), m);
        defer a.free(text);
        return .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .which = w,
            .mutation = m,
            .pipes = try host.Pipelines.loadSource(device, text),
            .served = switch (w) {
                .conv_static_tree => &static_shapes,
                .conv_dynamic_tree => &dynamic_shapes,
                .keep_kv => &kv_shapes,
                .keep_conv => &conv_shapes,
                else => unreachable,
            },
        };
    }

    pub fn deinit(self: *OpProbe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }

    const vt = g.hooks(OpProbe);
    pub fn kernel(self: *OpProbe) g.Kernel {
        const label = switch (self.which) {
            .conv_static_tree => p.Kind.conv_static_tree.label(),
            .conv_dynamic_tree => p.Kind.conv_dynamic_tree.label(),
            .keep_kv => p.Kind.keep_kv.label(),
            .keep_conv => p.Kind.keep_conv.label(),
            else => unreachable,
        };
        return .{
            .ctx = self,
            .vt = &vt,
            .control = self.mutation.gate(),
            .spec = .{
                .name = if (self.mutation == .none) label else self.mutation.name(),
                .accum = if (self.which == .conv_dynamic_tree) .f32 else .none,
                .depth = switch (self.which) {
                    .conv_static_tree => depthStatic,
                    .conv_dynamic_tree => depthDynamic,
                    .keep_kv, .keep_conv => depthCopy,
                    else => unreachable,
                },
                .ops = if (self.which == .conv_dynamic_tree) 96 else 5,
                .mids = if (self.which == .conv_dynamic_tree) 1 else 0,
                .out = .bf16,
                .served = self.served,
                .sums = .{ .f16 = 0, .bf16 = 0, .why = "bf16 projection rounding before weighted sum" },
            },
        };
    }

    pub fn count(self: *OpProbe) usize {
        return switch (self.which) {
            .conv_static_tree, .conv_dynamic_tree => self.shape.rows * self.shape.n,
            .keep_kv => p.visit_slots * p.kv_heads * self.ring * p.head_dim * 2,
            .keep_conv => p.visit_slots * 3 * p.conv_row_values,
            else => unreachable,
        };
    }

    fn floats(self: *OpProbe, n: usize, rng: std.Random, scale: f32) ![]f32 {
        const out = try self.arena.allocator().alloc(f32, n);
        for (out) |*x| x.* = (rng.float(f32) * 0.5 + 0.05) * (if (rng.boolean()) @as(f32, 1) else -scale);
        round(out);
        return out;
    }

    fn uploadBf(self: *OpProbe, values: []const f32) !rig_.Ref {
        const raw = try self.arena.allocator().alloc(u16, values.len);
        for (raw, values) |*b, x| b.* = toBf(x);
        return self.rig.input(std.mem.sliceAsBytes(raw));
    }

    pub fn prepare(self: *OpProbe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > p.window_rows) return false;
        switch (self.which) {
            .conv_static_tree => if (shape.n == 0 or shape.k != 4 or shape.n > p.hidden) return false,
            .conv_dynamic_tree => if (shape.n == 0 or shape.n > 1024 or shape.k != 4) return false,
            .keep_kv => if (shape.rows < 2 or shape.n != p.head_dim or shape.k < shape.rows or shape.k > 4096) return false,
            .keep_conv => if (shape.n != p.conv_row_values or shape.k < shape.rows) return false,
            else => return false,
        }
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        self.f = @splat(&.{});
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const aa = self.arena.allocator();
        const rows = shape.rows;
        const n = shape.n;
        var parents: [p.window_rows]i32 = undefined;
        parents[0] = -1;
        for (1..rows) |r| parents[r] = switch (r) {
            1, 2 => 0,
            else => @intCast(r - 2),
        };
        if (self.which == .conv_static_tree or self.which == .conv_dynamic_tree) {
            self.src_rows = try aa.alloc(u32, rows * 4);
            p.buildConvSrc(parents[0..rows], self.src_rows);
            self.f[0] = try self.floats(3 * n, rng, 1);
            self.f[1] = try self.floats(rows * n, rng, 1);
            if (fill == .nonfinite) {
                self.f[0][1 % n] = std.math.nan(f32);
                self.f[1][(rows - 1) * n + 2 % n] = std.math.inf(f32);
            }
            if (self.which == .conv_static_tree) {
                self.f[2] = try self.floats(n * 4, rng, 1);
                self.refs[0] = try self.uploadBf(self.f[0]);
                self.refs[1] = try self.uploadBf(self.f[1]);
                self.refs[2] = try self.uploadBf(self.f[2]);
                self.refs[3] = try self.rig.input(std.mem.sliceAsBytes(self.src_rows));
            } else {
                self.f[2] = try self.floats(rows * p.hidden, rng, 1);
                self.f[3] = try self.floats(4 * n, rng, 1);
                self.f[4] = try self.floats(32 * p.hidden, rng, 1);
                for (0..5) |j| self.refs[j] = try self.uploadBf(self.f[j]);
                self.refs[5] = try self.rig.input(std.mem.sliceAsBytes(self.src_rows));
            }
            self.out = try self.rig.output(self.count() * 2);
            return true;
        }
        if (self.which == .keep_kv) {
            self.ring = @intCast(shape.k);
            self.start = self.ring - 3;
            const win = @min(rows, 5);
            self.keep_path_len = if (win >= 5) 3 else 2;
            self.keep_path[0] = 0;
            self.keep_path[1] = 2 % @as(u32, @intCast(rows));
            if (self.keep_path_len == 3) self.keep_path[2] = 4 % @as(u32, @intCast(rows));
            if (self.keep_path[1] == 1) self.keep_path[1] = @intCast(rows - 1);
            self.kv_in = try aa.alloc(u16, self.count());
            for (self.kv_in) |*v| v.* = bits(rng.floatNorm(f64));
            if (fill == .nonfinite) {
                const stride = self.ring * p.head_dim;
                self.kv_in[((self.start + self.keep_path[1]) % self.ring) * p.head_dim + 3] = bits(std.math.nan(f64));
                self.kv_in[stride + ((self.start + self.keep_path[0]) % self.ring) * p.head_dim + 7] = bits(std.math.inf(f64));
            }
            self.out = try self.rig.output(self.count() * 2);
            return true;
        }
        const tap_rows = shape.k;
        self.conv_src3 = .{ 1, @intCast(3 + tap_rows / 2), @intCast(3 + tap_rows - 1) };
        self.conv_in = try aa.alloc(u16, p.visit_slots * 3 * p.conv_row_values);
        for (self.conv_in) |*v| v.* = bits(rng.floatNorm(f64));
        self.taps_u16 = try aa.alloc(u16, p.visit_slots * tap_rows * p.conv_row_values);
        for (self.taps_u16) |*v| v.* = bits(rng.floatNorm(f64));
        if (fill == .nonfinite) {
            self.conv_in[1 * p.conv_row_values + 5] = bits(std.math.nan(f64));
            self.taps_u16[p.visit_slots * tap_rows * p.conv_row_values - 1 - 17] = bits(std.math.inf(f64));
        }
        self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.taps_u16));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    pub fn run(self: *OpProbe) !void {
        const pass = self.rig.begin();
        const e = pass.e;
        const r: u32 = @intCast(self.shape.rows);
        const n: u32 = @intCast(self.shape.n);
        const y: host.Ref = .{ .buf = self.out.buf, .off = self.out.off };
        switch (self.which) {
            .conv_static_tree => host.convStatic(&self.pipes, e, .{ .buf = self.refs[0].buf, .off = self.refs[0].off }, .{ .buf = self.refs[1].buf, .off = self.refs[1].off }, .{ .buf = self.refs[2].buf, .off = self.refs[2].off }, .{ .buf = self.refs[3].buf, .off = self.refs[3].off }, y, r, n),
            .conv_dynamic_tree => host.convDynamic(&self.pipes, e, .{ .buf = self.refs[0].buf, .off = self.refs[0].off }, .{ .buf = self.refs[1].buf, .off = self.refs[1].off }, .{ .buf = self.refs[2].buf, .off = self.refs[2].off }, .{ .buf = self.refs[3].buf, .off = self.refs[3].off }, .{ .buf = self.refs[4].buf, .off = self.refs[4].off }, .{ .buf = self.refs[5].buf, .off = self.refs[5].off }, y, r, n),
            .keep_kv => {
                @memcpy(self.rig.bytes(self.out, self.count() * 2), std.mem.sliceAsBytes(self.kv_in));
                var args: p.KeepKvArgs = .{ .key_at = undefined, .value_at = undefined, .rings = @splat(self.ring), .start = self.start, .path_len = @intCast(self.keep_path_len), .path = self.keep_path };
                const half = p.kv_heads * self.ring * p.head_dim * 2;
                for (0..p.visit_slots) |slot| {
                    args.key_at[slot] = slot * 2 * half;
                    args.value_at[slot] = slot * 2 * half + half;
                }
                host.keepKv(&self.pipes, e, y, args);
            },
            .keep_conv => {
                @memcpy(self.rig.bytes(self.out, self.count() * 2), std.mem.sliceAsBytes(self.conv_in));
                host.keepConv(&self.pipes, e, y, .{ .buf = self.refs[0].buf, .off = self.refs[0].off }, y, .{ .src = self.conv_src3, .tap_rows = @intCast(self.shape.k) });
            },
            else => unreachable,
        }
        try self.rig.finish(pass);
    }

    pub fn reference(self: *OpProbe, want: []f64, mag: []f64) !void {
        const rows = self.shape.rows;
        const n = self.shape.n;
        switch (self.which) {
            .conv_static_tree => for (0..rows) |r| for (0..n) |d| {
                want[r * n + d] = p.convStaticTree(self.f[0], self.f[1], self.f[2], self.src_rows, n, r, d);
            },
            .conv_dynamic_tree => for (0..rows) |r| for (0..n) |d| {
                want[r * n + d] = p.convDynamicTree(self.f[0], self.f[1], self.f[2], self.f[3], self.f[4], self.src_rows, n, r, d);
            },
            .keep_kv => {
                for (want, self.kv_in) |*w, x| w.* = bf16(x);
                const head_vals = self.ring * p.head_dim;
                for (0..p.visit_slots * 2 * p.kv_heads) |pair| {
                    const base = pair * head_vals;
                    for (0..self.keep_path_len) |j| {
                        const src_r = (self.start + self.keep_path[j]) % self.ring;
                        const dst_r = (self.start + j) % self.ring;
                        for (0..p.head_dim) |d| want[base + dst_r * p.head_dim + d] = bf16(self.kv_in[base + src_r * p.head_dim + d]);
                    }
                }
            },
            .keep_conv => {
                const tap_rows = self.shape.k;
                const widths = p.conv_widths;
                for (0..p.visit_slots) |slot| {
                    var conv_base: usize = 0;
                    var tap_base: usize = 0;
                    for (widths) |w| {
                        for (0..3) |out_r| {
                            const src_r = self.conv_src3[out_r];
                            for (0..w) |d| {
                                const dst = slot * 3 * p.conv_row_values + conv_base + out_r * w + d;
                                want[dst] = if (src_r < 3)
                                    bf16(self.conv_in[slot * 3 * p.conv_row_values + conv_base + src_r * w + d])
                                else
                                    bf16(self.taps_u16[slot * tap_rows * p.conv_row_values + tap_base + (src_r - 3) * w + d]);
                            }
                        }
                        conv_base += 3 * w;
                        tap_base += tap_rows * w;
                    }
                }
            },
            else => unreachable,
        }
        for (mag, want) |*m, w| m.* = 10 * @abs(w) + 1e-5;
    }

    pub fn simulate(self: *OpProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.which != .conv_dynamic_tree) {
            const mag = try self.a.alloc(f64, out.len);
            defer self.a.free(mag);
            return self.reference(out, mag);
        }
        var dot_terms: [p.hidden]f64 = undefined;
        const rows = self.shape.rows;
        const n = self.shape.n;
        for (0..rows) |r| {
            var projected: [32]f64 = undefined;
            for (0..@min(32, (n + 127) / 128 * 4)) |t| {
                for (&dot_terms, 0..) |*dt, d| dt.* = @as(f64, self.f[4][t * p.hidden + d]) * self.f[2][r * p.hidden + d];
                projected[t] = p.roundBf16(contract.rows.sumIn(acc, order, &dot_terms));
            }
            for (0..n) |ch| {
                const idx3 = self.src_rows[r * 4 + 3];
                var sum: f64 = if (idx3 < 3) self.f[0][idx3 * n + ch] else self.f[1][(idx3 - 3) * n + ch];
                for (0..4) |lag| {
                    const w = @as(f64, self.f[3][lag * n + ch]) + projected[4 * (ch / 128) + lag];
                    const idx = self.src_rows[r * 4 + 3 - lag];
                    const x = if (idx < 3) self.f[0][idx * n + ch] else self.f[1][(idx - 3) * n + ch];
                    sum += w * x;
                }
                out[r * n + ch] = sum / 3;
            }
        }
    }
};

fn supported(device: mtl.Device, w: Which) bool {
    return switch (w) {
        .prompt_m5_tree_prefix, .prompt_m5_tree_tail => device.tensorUnits(),
        .prompt_pre_m5_tree_prefix, .prompt_pre_m5_tree_tail => !device.tensorUnits(),
        else => true,
    };
}

fn isOpWhich(w: Which) bool {
    return w == .conv_static_tree or w == .conv_dynamic_tree or w == .keep_kv or w == .keep_conv;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    attn_probes: std.ArrayList(*attn_.Probe) = .empty,
    op_probes: std.ArrayList(*OpProbe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| {
            if (!supported(s.device, w)) continue;
            const name = if (isOpWhich(w)) switch (w) {
                .conv_static_tree => p.Kind.conv_static_tree.label(),
                .conv_dynamic_tree => p.Kind.conv_dynamic_tree.label(),
                .keep_kv => p.Kind.keep_kv.label(),
                .keep_conv => p.Kind.keep_conv.label(),
                else => unreachable,
            } else attn_.spec(w).name;
            if (only) |text| if (std.mem.indexOf(u8, name, text) == null) continue;
            try s.add(w, .none);
        }
        for (std.enums.values(Mutation)) |m| {
            if (m == .none) continue;
            const w = m.which();
            if (!supported(s.device, w)) continue;
            if (only) |text| if (std.mem.indexOf(u8, m.name(), text) == null) continue;
            try s.add(w, m);
        }
    }

    fn add(s: *Suite, w: Which, m: Mutation) !void {
        if (isOpWhich(w)) {
            const probe = try s.a.create(OpProbe);
            errdefer s.a.destroy(probe);
            probe.* = OpProbe.init(s.a, s.device, s.queue, w, m) catch |err| {
                var empty: OpProbe = .{ .a = s.a, .arena = .init(s.a), .rig = undefined, .which = w, .mutation = m, .pipes = undefined };
                var k = empty.kernel();
                k.init_error = err;
                try s.kernels.append(s.a, k);
                s.a.destroy(probe);
                return;
            };
            errdefer probe.deinit();
            try contract.register(s.a, &s.op_probes, &s.kernels, probe, probe.kernel());
            return;
        }
        const probe = try s.a.create(attn_.Probe);
        errdefer s.a.destroy(probe);
        probe.* = attn_.Probe.init(s.a, s.device, s.queue, w, m) catch |err| {
            var k: g.Kernel = .{ .spec = attn_.spec(w), .ctx = undefined, .vt = undefined, .control = m.gate(), .init_error = err };
            if (m != .none) k.spec.name = m.name();
            try s.kernels.append(s.a, k);
            s.a.destroy(probe);
            return;
        };
        errdefer probe.deinit();
        try contract.register(s.a, &s.attn_probes, &s.kernels, probe, probe.kernel());
    }

    pub fn deinit(s: *Suite) void {
        for (s.attn_probes.items) |pr| {
            pr.deinit();
            s.a.destroy(pr);
        }
        s.attn_probes.deinit(s.a);
        for (s.op_probes.items) |pr| {
            pr.deinit();
            s.a.destroy(pr);
        }
        s.op_probes.deinit(s.a);
    }
};
