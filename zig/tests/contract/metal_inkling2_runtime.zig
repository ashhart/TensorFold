//! Inkling 2 Mini runtime helpers under FP64 references, guarded buffers and exact ABI launches.
const std = @import("std");
const mtl = @import("metal");
const controls = @import("metal_inkling2_runtime_controls.zig");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const p = @import("inkling2_runtime_plan");
const sources = @import("inkling2_metal");

fn depth(_: c.Shape) usize {
    return 1;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    kind: p.Kind,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    served: [5]c.Shape = undefined,
    f: [2][]f32 = @splat(&.{}),
    refs: [2]rig_.Ref = undefined,
    blocks: rig_.Ref = undefined,
    blocks_before: []u16 = &.{},
    stage_hist_before: [2][]u16 = @splat(&.{}),
    stage_raw: [2][]u16 = @splat(&.{}),
    stage_hist: [2]rig_.Ref = undefined,
    stage_out: [2]rig_.Ref = undefined,
    taps_u16: []u16 = &.{},
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind) !Probe {
        return initSource(a, device, queue, kind, sources.runtime);
    }
    pub fn initSource(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: p.Kind, text: []const u8) !Probe {
        var self: Probe = .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .kind = kind,
            .pipe = undefined,
        };
        errdefer self.arena.deinit();
        self.pipe = try rig_.pipeline(device, text, kind.symbol());
        self.served = switch (kind) {
            .attention_partial => .{
                .{ .rows = 1, .n = p.hidden, .k = 0 },
                .{ .rows = 5, .n = p.hidden, .k = 1 },
                .{ .rows = 3, .n = p.hidden, .k = 2 },
                .{ .rows = 2, .n = p.hidden, .k = 3 },
                .{ .rows = 7, .n = p.hidden, .k = 4 },
            },
            .copy_bf16, .copy_f32 => .{
                .{ .rows = 1, .n = p.hidden, .k = 1 },
                .{ .rows = 5, .n = 1024, .k = 1 },
                .{ .rows = 7, .n = p.hidden, .k = 1 },
                .{ .rows = 16, .n = 1024, .k = 1 },
                .{ .rows = 32, .n = p.hidden, .k = 1 },
            },
            .capture_tap => .{
                .{ .rows = 1, .n = p.hidden, .k = 0 },
                .{ .rows = 5, .n = p.hidden, .k = 1 },
                .{ .rows = 7, .n = p.hidden, .k = 2 },
                .{ .rows = 16, .n = p.hidden, .k = 3 },
                .{ .rows = 32, .n = p.hidden, .k = 3 },
            },
            .stage_conv => .{
                .{ .rows = 1, .n = p.hidden, .k = 1 },
                .{ .rows = 2, .n = p.hidden, .k = 1 },
                .{ .rows = 1, .n = 1024, .k = 3 },
                .{ .rows = 7, .n = 1024, .k = 2 },
                .{ .rows = 512, .n = p.hidden, .k = 1 },
            },
            .commit_histories => .{
                .{ .rows = 48, .n = 19, .k = 1 },
                .{ .rows = 48, .n = 19, .k = 7 },
                .{ .rows = 48, .n = 19, .k = 16 },
                .{ .rows = 2, .n = 19, .k = 0 },
                .{ .rows = 1, .n = 7, .k = 4 },
            },
            else => .{
                .{ .rows = 1, .n = p.hidden, .k = 1 },
                .{ .rows = 5, .n = p.hidden, .k = 1 },
                .{ .rows = 7, .n = p.hidden, .k = 1 },
                .{ .rows = 16, .n = p.hidden, .k = 1 },
                .{ .rows = 32, .n = p.hidden, .k = 1 },
            },
        };
        return self;
    }
    pub fn deinit(self: *Probe) void {
        self.pipe.deinit();
        self.rig.deinit();
        self.arena.deinit();
    }

    const vt_plain = blk: {
        var vt = g.hooks(Probe);
        vt.check = null;
        break :blk vt;
    };
    const vt_checked = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = if (self.kind == .attention_partial) &vt_checked else &vt_plain, .spec = .{
            .name = self.kind.label(),
            .accum = .none,
            .depth = depth,
            .ops = if (self.kind == .attention_partial or self.kind == .mlp_partial) 1 else 0,
            .out = switch (self.kind) {
                .copy_bf16, .mlp_partial, .capture_tap, .stage_conv, .commit_histories => .bf16,
                .copy_f32, .widen, .attention_partial => .f32,
            },
            .in_place = self.kind == .stage_conv,
            .served = &self.served,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "pointwise moves and single additions have no reduction accumulator to discriminate" },
        } };
    }
    pub fn count(self: *Probe) usize {
        return p.outputCount(self.kind, .{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k });
    }

    fn floats(self: *Probe, n: usize, rng: std.Random) ![]f32 {
        const values = try self.arena.allocator().alloc(f32, n);
        for (values) |*v| v.* = value(rng);
        return values;
    }
    fn u16s(self: *Probe, n: usize, rng: std.Random) ![]u16 {
        const values = try self.arena.allocator().alloc(u16, n);
        for (values) |*v| v.* = c.toBf16(value(rng));
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
        self.blocks_before = &.{};
        self.stage_hist_before = @splat(&.{});
        self.stage_raw = @splat(&.{});
        self.taps_u16 = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.kind == .stage_conv) {
            const pairs = p.stagePairs(shape.k);
            for (0..pairs) |pair| {
                self.stage_hist_before[pair] = try self.u16s(3 * shape.n, rng);
                self.stage_raw[pair] = try self.u16s(shape.rows * shape.n, rng);
                if (fill == .nonfinite and pair == 0) self.stage_raw[0][0] = c.toBf16(std.math.nan(f32));
                self.refs[pair] = try self.rig.input(std.mem.sliceAsBytes(self.stage_raw[pair]));
                self.stage_hist[pair] = try self.rig.output(3 * shape.n * 2);
                self.stage_out[pair] = try self.rig.output((3 + shape.rows) * shape.n * 2);
            }
            return true;
        }
        if (self.kind == .commit_histories) {
            self.taps_u16 = try self.u16s(shape.rows * shape.n * p.conv_row_values, rng);
            if (fill == .nonfinite) self.taps_u16[shape.k * p.conv_widths[0]] = c.toBf16(std.math.nan(f32));
            self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.taps_u16));
            self.out = try self.rig.output(self.count() * 2);
            return true;
        }
        const n = shape.rows * shape.n;
        self.f[0] = try self.floats(n, rng);
        switch (self.kind) {
            .copy_bf16, .widen, .capture_tap => round(self.f[0]),
            .attention_partial => {
                self.f[1] = try self.floats(n, rng);
                round(self.f[0]);
                round(self.f[1]);
                const blocks = try self.arena.allocator().alloc(u16, shape.rows * p.block_slots * p.hidden);
                for (blocks) |*b| b.* = c.toBf16(value(rng));
                self.blocks_before = blocks;
                self.blocks = try self.rig.scratch(std.mem.sliceAsBytes(blocks).len);
                @memcpy(self.rig.bytes(self.blocks, std.mem.sliceAsBytes(blocks).len), std.mem.sliceAsBytes(blocks));
                self.refs[1] = try self.input(self.f[1], true);
            },
            .mlp_partial => {
                self.f[1] = try self.floats(n, rng);
                round(self.f[1]);
                self.refs[1] = try self.input(self.f[1], true);
            },
            .copy_f32 => {},
            .stage_conv, .commit_histories => unreachable,
        }
        if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
        self.refs[0] = try self.input(self.f[0], self.kind == .copy_bf16 or self.kind == .widen or self.kind == .attention_partial or self.kind == .capture_tap);
        const elem: c.Elem = switch (self.kind) {
            .copy_bf16, .mlp_partial, .capture_tap => .bf16,
            .copy_f32, .widen, .attention_partial => .f32,
            .stage_conv, .commit_histories => unreachable,
        };
        self.out = try self.rig.output(self.count() * c.size(elem));
        if (self.kind == .capture_tap) {
            const untouched = std.mem.bytesAsSlice(u16, self.rig.bytes(self.out, self.count() * 2));
            @memset(untouched, c.toBf16(1));
        }
        return true;
    }

    pub fn run(self: *Probe) !void {
        const call = p.Call{ .rows = self.shape.rows, .n = self.shape.n, .k = self.shape.k };
        if (self.kind == .stage_conv) {
            const pairs = p.stagePairs(self.shape.k);
            for (0..pairs) |pair| {
                @memcpy(self.rig.bytes(self.stage_hist[pair], 3 * self.shape.n * 2), std.mem.sliceAsBytes(self.stage_hist_before[pair]));
            }
        }
        const pass = self.rig.begin();
        const e = pass.e;
        e.setPipeline(self.pipe);
        switch (self.kind) {
            .stage_conv => {
                const second: usize = if (p.stagePairs(self.shape.k) == 2) 1 else 0;
                e.setBuffer(self.stage_hist[0].buf, self.stage_hist[0].off, 0);
                e.setBuffer(self.refs[0].buf, self.refs[0].off, 1);
                e.setBuffer(self.stage_out[0].buf, self.stage_out[0].off, 2);
                e.setBuffer(self.stage_hist[second].buf, self.stage_hist[second].off, 3);
                e.setBuffer(self.refs[second].buf, self.refs[second].off, 4);
                e.setBuffer(self.stage_out[second].buf, self.stage_out[second].off, 5);
                e.setValue(p.stageConvArgs(call), 6);
            },
            .commit_histories => {
                e.setBuffer(self.refs[0].buf, self.refs[0].off, 0);
                e.setBuffer(self.out.buf, self.out.off, 1);
                e.setValue(p.commitHistoriesArgs(call), 2);
            },
            .copy_bf16, .copy_f32, .widen, .capture_tap => {
                e.setBuffer(self.refs[0].buf, self.refs[0].off, 0);
                e.setBuffer(self.out.buf, self.out.off, 1);
                e.setValue(p.args(self.kind, call), 2);
            },
            .attention_partial => {
                e.setBuffer(self.refs[0].buf, self.refs[0].off, 0);
                e.setBuffer(self.refs[1].buf, self.refs[1].off, 1);
                e.setBuffer(self.blocks.buf, self.blocks.off, 2);
                e.setBuffer(self.out.buf, self.out.off, 3);
                e.setValue(p.args(self.kind, call), 4);
            },
            .mlp_partial => {
                e.setBuffer(self.refs[0].buf, self.refs[0].off, 0);
                e.setBuffer(self.refs[1].buf, self.refs[1].off, 1);
                e.setBuffer(self.out.buf, self.out.off, 2);
                e.setValue(p.args(self.kind, call), 3);
            },
        }
        e.dispatchThreads(mtl.Size.of(p.dispatchCount(self.kind, call), 1, 1), mtl.Size.of(p.threads, 1, 1));
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        if (self.kind == .stage_conv) {
            const pairs = p.stagePairs(self.shape.k);
            const commit = p.stageCommit(self.shape.k);
            const w = self.shape.n;
            const r = self.shape.rows;
            const h_buf = try self.a.alloc(u16, 3 * w);
            defer self.a.free(h_buf);
            const s_buf = try self.a.alloc(u16, (3 + r) * w);
            defer self.a.free(s_buf);
            var at: usize = 0;
            for (0..pairs) |pair| {
                p.stageConvRef(self.stage_hist_before[pair], self.stage_raw[pair], w, r, commit, h_buf, s_buf);
                for (h_buf) |v| {
                    want[at] = c.fromBf16(v);
                    mag[at] = @abs(want[at]);
                    at += 1;
                }
                for (s_buf) |v| {
                    want[at] = c.fromBf16(v);
                    mag[at] = @abs(want[at]);
                    at += 1;
                }
            }
            return;
        }
        if (self.kind == .commit_histories) {
            const out_u16 = try self.a.alloc(u16, self.count());
            defer self.a.free(out_u16);
            p.commitHistoriesRef(self.taps_u16, out_u16, self.shape.rows, self.shape.n, self.shape.k);
            for (out_u16, want, mag) |v, *w, *m| {
                w.* = c.fromBf16(v);
                m.* = @abs(w.*);
            }
            return;
        }
        if (self.kind == .capture_tap) {
            for (want, mag, 0..) |*w, *m, i| {
                const row = i / (p.tap_channels * p.hidden);
                const channel = (i / p.hidden) % p.tap_channels;
                const x = self.f[0][row * p.hidden + i % p.hidden];
                w.* = if (channel == self.shape.k) x else 1;
                m.* = @abs(@as(f64, x));
            }
            return;
        }
        for (want, mag, 0..) |*w, *m, i| {
            const x = self.f[0][i];
            const y = if (self.f[1].len == 0) 0 else self.f[1][i];
            w.* = switch (self.kind) {
                .copy_bf16, .copy_f32, .widen, .capture_tap => x,
                .attention_partial => p.attentionPartial(x, y, self.shape.k),
                .mlp_partial => p.mlpPartial(x, y),
                .stage_conv, .commit_histories => unreachable,
            };
            m.* = @abs(@as(f64, x)) + @abs(@as(f64, y));
        }
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (self.kind != .attention_partial or (gate != .shapes and gate != .nonfinite)) return "numeric hooks must be used";
        const want = try self.a.alloc(f64, self.count());
        defer self.a.free(want);
        const mag = try self.a.alloc(f64, self.count());
        defer self.a.free(mag);
        try self.reference(want, mag);
        const bytes = try self.rig.outputBytes();
        var planted: usize = 0;
        for (want, 0..) |w, i| {
            const got = c.read(.f32, bytes, i);
            if (c.classOf(got) != c.classOf(w)) return "partial's floating-point class differs";
            if (c.classOf(w) != .finite) planted += 1;
            if (gate == .shapes and std.math.isFinite(w) and got != w) return "FP32 partial differs from the FP64 reference's explicit rounding";
        }
        if (gate == .nonfinite and (planted == 0 or planted == want.len)) return "the planted NaN does not discriminate finite and non-finite partials";
        const block_bytes = self.rig.bytes(self.blocks, std.mem.sliceAsBytes(self.blocks_before).len);
        const got_blocks = std.mem.bytesAsSlice(u16, block_bytes);
        for (0..self.shape.rows) |row| for (0..p.block_slots) |slot| for (0..p.hidden) |d| {
            const at = (row * p.block_slots + slot) * p.hidden + d;
            const expected = if (self.shape.k != 0 and slot == self.shape.k - 1)
                c.toBf16(self.f[0][row * p.hidden + d])
            else
                self.blocks_before[at];
            if (got_blocks[at] != expected) return "completed-block row/slot layout differs";
        };
        return null;
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
            const text = try controls.source(s.a, sources.runtime, kind);
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

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(p.Kind)) |kind| {
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
