//! The fused BF16 read-out's tiles under the contract: each bit-equal to the launches it replaces, with two controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const tc = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const tiles = flash.tiles;
const tk = flash.tile_kernels;

const dims = 2560;
const streams = 4;
const wide = dims * streams;
const low = 320;
const n_down = low + streams;
const nc = dims / 256;
const eps: f32 = 1e-6;
const most = 3072;

pub const Kind = enum { down, up };
pub const Fault = enum { none, rz, raw };
pub const Entry = struct { name: []const u8, kind: Kind, di: usize = 1, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda flashnext fn_hcn_down2 (normalize and fp32 down, 32-row tiles)", .kind = .down, .di = 0 },
    .{ .name = "cuda flashnext fn_hcn_down3 (normalize and fp32 down, 48-row tiles)", .kind = .down, .di = 1 },
    .{ .name = "cuda flashnext fn_hcn_down4 (normalize and fp32 down, 64-row tiles)", .kind = .down, .di = 2 },
    .{ .name = "cuda flashnext fn_hcn_up3 (bf16 up, gates and the four-stream mix)", .kind = .up },
    .{ .name = "control: fn_hcn_down3 rounding its normed rows toward zero", .kind = .down, .fault = .rz },
    .{ .name = "control: fn_hcn_up3 gating its up rows before their bf16 rounding", .kind = .up, .fault = .raw },
};

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(e: Entry) c.Spec {
    return .{
        .name = e.name,
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = "every output byte equal to the launches it replaces, planted non-finite rows included" },
        .replaces = if (e.kind == .down) "fn_hc_normalize, then the qmmf lane's fp32 down" else "the qmmf lane's bf16 up, then fn_hc_dense_mix over fn_hc_normalize's rows",
        .served = &.{ .{ .rows = 768, .n = dims, .k = 1 }, .{ .rows = 769, .n = dims, .k = 1 }, .{ .rows = 1000, .n = dims, .k = 1 }, .{ .rows = 1536, .n = dims, .k = 1 }, .{ .rows = 2053, .n = dims, .k = 1 }, .{ .rows = 3072, .n = dims, .k = 1 } },
    };
}

pub const Hcn = struct {
    rig: rig_.Rig,
    entry: Entry,
    glue: *const flash.Glue,
    lane: *const cuda.qmmf.Lane,
    t: *const tk.TileKernels,
    ctl: ?cuda.Function = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    w: cuda.qmmf.Weight = undefined,
    h: u64 = 0,
    pss: u64 = 0,
    scale: u64 = 0,
    act: u64 = 0,
    norm: u64 = 0,
    up: u64 = 0,
    part: u64 = 0,
    lens: [2]usize = @splat(0),
    outs: [2]u64 = @splat(0),
    nout: usize = 0,
    base: u64 = 0,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Hcn) void {
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    const vt = g.hooks(Hcn);

    pub fn kernel(self: *Hcn) g.Kernel {
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = if (self.entry.fault == .none) null else .shapes };
    }

    pub fn prepare(self: *Hcn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.nout = 0;
        self.shape = shape;
        const m = shape.rows;
        if (m == 0 or m > most or shape.n != dims) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x68636e);
        const r = prng.random();
        const h = try a.alloc(u16, m * wide);
        defer a.free(h);
        for (h) |*x| x.* = c.toBf16(r.float(f32) * 6 - 3);
        if (fill == .nonfinite) for (0..m) |row| {
            h[row * wide + (row % streams) * dims + 5] = switch (row % 4) {
                0 => 0x7fc1,
                1 => 0x7f80,
                2 => 0xff80,
                else => 0x8000,
            };
        };
        const pss = try a.alloc(f32, m * nc * streams);
        defer a.free(pss);
        for (pss) |*x| x.* = 0.5 + r.float(f32) * 2;
        const scale = try a.alloc(f32, wide);
        defer a.free(scale);
        for (scale) |*x| x.* = 0.5 + r.float(f32);
        const rows: usize = if (self.entry.kind == .down) n_down else wide;
        const k: usize = if (self.entry.kind == .down) wide else low;
        const wt = try a.alloc(u16, rows * k);
        defer a.free(wt);
        const spread: f32 = if (self.entry.kind == .down) 0.04 else 0.4;
        for (wt) |*x| x.* = c.toBf16((r.float(f32) * 2 - 1) * spread);
        const packed_w = try a.alloc(u16, cuda.nvfp4.padded(rows) * k);
        defer a.free(packed_w);
        cuda.qmmf.packBf16(packed_w, wt, rows, k);
        self.h = try self.rig.input(std.mem.sliceAsBytes(h));
        self.pss = try self.rig.input(std.mem.sliceAsBytes(pss));
        self.scale = try self.rig.input(std.mem.sliceAsBytes(scale));
        self.w = cuda.modelopt.bf16Weight(try self.rig.input(std.mem.sliceAsBytes(packed_w)), rows, k);
        self.norm = try self.rig.scratch(m * wide * 2);
        if (self.entry.kind == .down) {
            try self.out(m * n_down * 4); // the down's fp32 rows
            self.part = try self.rig.scratch(8 * m * n_down * 4);
        } else {
            const act = try a.alloc(u16, m * low);
            defer a.free(act);
            for (act) |*x| x.* = c.toBf16(r.float(f32) * 2 - 1);
            self.act = try self.rig.input(std.mem.sliceAsBytes(act));
            self.up = try self.rig.scratch(m * wide * 2);
            try self.out(m * dims * 2); // the mixed rows
            try self.out(m * (dims / 32) * 4); // their 32-column sums
        }
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        self.base = try self.rig.scratch(total);
        return true;
    }

    fn out(self: *Hcn, len: usize) !void {
        self.outs[self.nout] = try self.rig.output(len);
        self.lens[self.nout] = len;
        self.nout += 1;
    }

    fn ours(self: *Hcn, o: []const u64) !void {
        const m = self.shape.rows;
        const s = self.rig.s;
        if (self.entry.kind == .down) {
            const f = self.ctl orelse self.t.hcn_down[self.entry.di];
            return tiles.fusedDown(f, tk.hcn_down_mt[self.entry.di], s, self.h, self.pss, self.scale, self.w, o[0], m, eps);
        }
        try tiles.fusedUpMix(self.ctl orelse self.t.hcn_up, s, self.act, self.w, self.h, self.pss, self.scale, o[0], o[1], m, eps);
    }

    fn theirs(self: *Hcn, o: []const u64) !void {
        const m = self.shape.rows;
        const s = self.rig.s;
        try self.glue.hcNormalize(s, self.h, self.pss, self.scale, self.norm, eps, m, dims, streams);
        if (self.entry.kind == .down) return self.lane.matmulF32(s, self.norm, wide, m, self.w, o[0], self.part);
        try self.lane.matmul(s, self.act, low, m, self.w, self.up, 0);
        try self.glue.hcDenseMix(s, self.up, self.norm, o[0], o[1], m, dims, streams);
    }

    pub fn run(self: *Hcn) !void {
        try self.ours(self.outs[0..self.nout]);
        try self.rig.s.synchronize();
    }

    /// The scratch the replaced launches write: the outputs' layout back to back, poisoned like them (0xff).
    fn baseline(self: *Hcn) ![]const u8 {
        var at = self.base;
        var o: [2]u64 = undefined;
        var total: usize = 0;
        for (self.lens[0..self.nout], 0..) |l, i| {
            o[i] = at;
            at += l;
            total += l;
        }
        const ff = try self.rig.gpu.gpa.alloc(u8, total);
        defer self.rig.gpu.gpa.free(ff);
        @memset(ff, 0xff);
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.base, .len = total }).upload(0, ff);
        try self.theirs(o[0..self.nout]);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, total);
    }

    pub fn replaced(self: *Hcn) ![]const u8 {
        return self.baseline();
    }

    /// Gate 1's stated rule: the outputs on the planted inputs equal the replaced launches' bytes.
    pub fn check(self: *Hcn, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.baseline();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from the replaced launches' from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *Hcn) usize {
        var total: usize = 0;
        for (self.lens[0..self.nout]) |l| total += l;
        return total;
    }

    pub fn reference(_: *Hcn, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};

/// The kernels, the chain they replace and the controls, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    glue: flash.Glue,
    lane: cuda.qmmf.Lane,
    t: tk.TileKernels,
    ctl: cuda.Module,
    items: [entries.len]Hcn = undefined,

    const controls = struct {
        pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
    };

    /// The set, its kernels appended to `list`, each control after the kernels it breaks.
    pub fn init(gpu: tc.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var glue = try flash.Glue.load(gpu.d);
        errdefer glue.deinit();
        var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
        errdefer lane.unload();
        var t = try tk.TileKernels.load(gpu.d);
        errdefer t.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .glue = glue, .lane = lane, .t = t, .ctl = ctl };
        try set.lane.loadPrompt(gpu.d); // the engine's lane: qmmf's BF16 prompt tiles (TF_FLASHNEXT_PROMPT_BF16)
        const rz = try set.ctl.function("tf_ctl_hcn_down3_rz");
        try rz.allowDynamicShared(tk.hcnDownSmem(3));
        const raw = try set.ctl.function("tf_ctl_hcn_up3_raw");
        try raw.allowDynamicShared(tk.hcnUpSmem(tk.hcn_up_mt));
        for (&set.items, entries) |*x, e| x.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .entry = e,
            .glue = &set.glue,
            .lane = &set.lane,
            .t = &set.t,
            .ctl = switch (e.fault) {
                .none => null,
                .rz => rz,
                .raw => raw,
            },
        };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.ctl.unload();
        set.t.unload();
        set.lane.unload();
        set.glue.deinit();
        set.gpa.destroy(set);
    }
};
