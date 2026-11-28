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

fn depth(s: c.Shape) usize {
    return if (s.rows > 1 and s.n >= 1024 and s.k <= 3) 12 else 1;
}
fn value(rng: std.Random) f32 {
    return rng.float(f32) * 0.5 - 0.25;
}
fn round(values: []f32) void {
    for (values) |*v| v.* = c.fromBf16(c.toBf16(v.*));
}
fn segSplit(total: usize, n_segs: usize, idx: usize) usize {
    if (total == 16 and n_segs == 4) return ([_]usize{ 3, 5, 4, 4 })[idx];
    if (total == 7 and n_segs == 3) return ([_]usize{ 1, 2, 4 })[idx];
    if (total == 5 and n_segs == 2) return ([_]usize{ 2, 3 })[idx];
    const q = total / n_segs;
    return if (idx + 1 == n_segs) total - q * idx else q;
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
    seg_hist_before: [8][]u16 = @splat(&.{}),
    seg_raw: [8][]u16 = @splat(&.{}),
    seg_weight: [2][]u16 = @splat(&.{}),
    seg_raw_refs: [8]rig_.Ref = undefined,
    seg_hist_refs: [8]rig_.Ref = undefined,
    seg_staged_refs: [8]rig_.Ref = undefined,
    seg_out_refs: [8]rig_.Ref = undefined,
    segs_ref: rig_.Ref = undefined,
    res_ids: [40]mtl.objc.Id = undefined,
    res_len: usize = 0,
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
                .{ .rows = 256, .n = p.hidden, .k = 1 },
            },
            .stage_conv_segs, .stage_conv_static => .{
                .{ .rows = 2, .n = 1024, .k = 1 },
                .{ .rows = 5, .n = 1024, .k = 3 },
                .{ .rows = 7, .n = p.hidden, .k = 0 },
                .{ .rows = 16, .n = p.hidden, .k = 2 },
                .{ .rows = 16, .n = 1024, .k = 3 },
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
        const staged = self.kind == .stage_conv or self.kind == .stage_conv_segs or self.kind == .stage_conv_static;
        return .{ .ctx = self, .vt = if (self.kind == .attention_partial) &vt_checked else &vt_plain, .spec = .{
            .name = self.kind.label(),
            .accum = if (self.kind == .stage_conv_static) .f32 else .none,
            .depth = depth,
            .ops = if (self.kind == .stage_conv_static) 96 else if (self.kind == .attention_partial or self.kind == .mlp_partial) 1 else 0,
            .out = switch (self.kind) {
                .copy_bf16, .mlp_partial, .capture_tap, .stage_conv, .stage_conv_segs, .stage_conv_static, .commit_histories => .bf16,
                .copy_f32, .widen, .attention_partial => .f32,
            },
            .in_place = staged,
            .served = &self.served,
            .sums = if (self.kind == .stage_conv_static)
                .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "short convolution rounds to BF16 after a four-tap dot" }
            else
                .{ .f16 = 0, .bf16 = 0, .why = "pointwise moves and single additions have no reduction accumulator to discriminate" },
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
        self.seg_hist_before = @splat(&.{});
        self.seg_raw = @splat(&.{});
        self.seg_weight = @splat(&.{});
        self.res_len = 0;
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
        if (self.kind == .stage_conv_segs or self.kind == .stage_conv_static) {
            const n_segs = p.segCount(shape.rows);
            const pairs = p.stagePairs(shape.k);
            const is_static = self.kind == .stage_conv_static;
            if (is_static) for (0..pairs) |pair| {
                self.seg_weight[pair] = try self.u16s(4 * shape.n, rng);
                self.refs[pair] = try self.rig.input(std.mem.sliceAsBytes(self.seg_weight[pair]));
            };
            var descs: [4]p.InkConvSeg = undefined;
            for (0..n_segs) |si| {
                const sr = segSplit(shape.rows, n_segs, si);
                for (0..pairs) |pair| {
                    const idx = si * 2 + pair;
                    self.seg_hist_before[idx] = try self.u16s(3 * shape.n, rng);
                    self.seg_raw[idx] = try self.u16s(sr * shape.n, rng);
                    if (fill == .nonfinite and si == 0 and pair == 0) self.seg_raw[0][0] = c.toBf16(std.math.nan(f32));
                    self.seg_raw_refs[idx] = try self.rig.input(std.mem.sliceAsBytes(self.seg_raw[idx]));
                    self.seg_hist_refs[idx] = try self.rig.output(3 * shape.n * 2);
                    self.seg_staged_refs[idx] = try self.rig.output((3 + sr) * shape.n * 2);
                    self.res_ids[self.res_len] = self.seg_raw_refs[idx].buf.id;
                    self.res_ids[self.res_len + 1] = self.seg_hist_refs[idx].buf.id;
                    self.res_ids[self.res_len + 2] = self.seg_staged_refs[idx].buf.id;
                    self.res_len += 3;
                    if (is_static) {
                        self.seg_out_refs[idx] = try self.rig.output(sr * shape.n * 2);
                        self.res_ids[self.res_len] = self.seg_out_refs[idx].buf.id;
                        self.res_len += 1;
                    }
                }
                const s1 = si * 2 + @as(usize, if (pairs == 2) 1 else 0);
                descs[si] = .{
                    .hist0 = self.seg_hist_refs[si * 2].buf.gpuAddress() + self.seg_hist_refs[si * 2].off,
                    .raw0 = self.seg_raw_refs[si * 2].buf.gpuAddress() + self.seg_raw_refs[si * 2].off,
                    .staged0 = self.seg_staged_refs[si * 2].buf.gpuAddress() + self.seg_staged_refs[si * 2].off,
                    .out0 = if (is_static) self.seg_out_refs[si * 2].buf.gpuAddress() + self.seg_out_refs[si * 2].off else 0,
                    .hist1 = self.seg_hist_refs[s1].buf.gpuAddress() + self.seg_hist_refs[s1].off,
                    .raw1 = self.seg_raw_refs[s1].buf.gpuAddress() + self.seg_raw_refs[s1].off,
                    .staged1 = self.seg_staged_refs[s1].buf.gpuAddress() + self.seg_staged_refs[s1].off,
                    .out1 = if (is_static) self.seg_out_refs[s1].buf.gpuAddress() + self.seg_out_refs[s1].off else 0,
                    .rows = @intCast(sr),
                    .commit = @intFromBool(p.stageCommit(shape.k)),
                };
            }
            self.segs_ref = try self.rig.input(std.mem.sliceAsBytes(descs[0..n_segs]));
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
            .stage_conv, .stage_conv_segs, .stage_conv_static, .commit_histories => unreachable,
        }
        if (fill == .nonfinite) self.f[0][0] = std.math.nan(f32);
        self.refs[0] = try self.input(self.f[0], self.kind == .copy_bf16 or self.kind == .widen or self.kind == .attention_partial or self.kind == .capture_tap);
        const elem: c.Elem = switch (self.kind) {
            .copy_bf16, .mlp_partial, .capture_tap => .bf16,
            .copy_f32, .widen, .attention_partial => .f32,
            .stage_conv, .stage_conv_segs, .stage_conv_static, .commit_histories => unreachable,
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
        if (self.kind == .stage_conv_segs or self.kind == .stage_conv_static) {
            const n_segs = p.segCount(self.shape.rows);
            const pairs = p.stagePairs(self.shape.k);
            for (0..n_segs) |si| for (0..pairs) |pair| {
                const idx = si * 2 + pair;
                @memcpy(self.rig.bytes(self.seg_hist_refs[idx], 3 * self.shape.n * 2), std.mem.sliceAsBytes(self.seg_hist_before[idx]));
            };
        }
        const pass = self.rig.begin();
        const e = pass.e;
        e.setPipeline(self.pipe);
        const n_segs: usize = if (self.kind == .stage_conv_segs or self.kind == .stage_conv_static) p.segCount(self.shape.rows) else 1;
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
            .stage_conv_segs => {
                e.useResources(self.res_ids[0..self.res_len], mtl.ResourceUsage.read | mtl.ResourceUsage.write);
                e.setBuffer(self.segs_ref.buf, self.segs_ref.off, 0);
                e.setValue(p.StageConvSegsArgs{ .width = @intCast(self.shape.n), .n_segs = @intCast(n_segs), .pairs = @intCast(p.stagePairs(self.shape.k)) }, 1);
            },
            .stage_conv_static => {
                const second: usize = if (p.stagePairs(self.shape.k) == 2) 1 else 0;
                e.useResources(self.res_ids[0..self.res_len], mtl.ResourceUsage.read | mtl.ResourceUsage.write);
                e.setBuffer(self.segs_ref.buf, self.segs_ref.off, 0);
                e.setBuffer(self.refs[0].buf, self.refs[0].off, 1);
                e.setBuffer(self.refs[second].buf, self.refs[second].off, 2);
                e.setValue(p.StageConvSegsArgs{ .width = @intCast(self.shape.n), .n_segs = @intCast(n_segs), .pairs = @intCast(p.stagePairs(self.shape.k)) }, 3);
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
        e.dispatchThreads(mtl.Size.of(p.dispatchCount(self.kind, call), n_segs, 1), mtl.Size.of(p.threads, 1, 1));
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        if (self.kind == .stage_conv_segs or self.kind == .stage_conv_static) {
            const n_segs = p.segCount(self.shape.rows);
            const pairs = p.stagePairs(self.shape.k);
            const commit = p.stageCommit(self.shape.k);
            const is_static = self.kind == .stage_conv_static;
            const w = self.shape.n;
            const h_buf = try self.a.alloc(u16, 3 * w);
            defer self.a.free(h_buf);
            var at: usize = 0;
            for (0..n_segs) |si| {
                const sr = segSplit(self.shape.rows, n_segs, si);
                const s_buf = try self.a.alloc(u16, (3 + sr) * w);
                defer self.a.free(s_buf);
                const c_buf: []f64 = if (is_static) try self.a.alloc(f64, sr * w) else &.{};
                defer if (is_static) self.a.free(c_buf);
                for (0..pairs) |pair| {
                    const idx = si * 2 + pair;
                    if (is_static) {
                        p.stageConvStaticRef(self.seg_hist_before[idx], self.seg_raw[idx], self.seg_weight[pair], w, sr, commit, h_buf, s_buf, c_buf);
                    } else {
                        p.stageConvRef(self.seg_hist_before[idx], self.seg_raw[idx], w, sr, commit, h_buf, s_buf);
                    }
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
                    if (is_static) for (c_buf) |v| {
                        want[at] = v;
                        mag[at] = 10 * @abs(v) + 1e-5;
                        at += 1;
                    };
                }
            }
            return;
        }
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
                .stage_conv, .stage_conv_segs, .stage_conv_static, .commit_histories => unreachable,
            };
            m.* = @abs(@as(f64, x)) + @abs(@as(f64, y));
        }
    }

    pub fn simulate(self: *Probe, _: c.Accum, _: g.Order, out: []f64) !void {
        const mag = try self.a.alloc(f64, out.len);
        defer self.a.free(mag);
        return self.reference(out, mag);
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
