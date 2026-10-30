//! Flash Next EXL3 prefill experts under the contract: tf_exl3pf_gu and tf_exl3pf_down replace the decode form.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const pk_ = flash.exl3.prefill_k;
const exl3e = cuda.exl3_experts;

/// The gate-up pass (rotated inputs, SwiGLU outputs), the down pass (y), and the gate-up pass against a wrong baseline.
pub const Kind = enum { gu, down, control };

const dims = 2560;
const width = 640;
const table_n = 16; // 15 routed experts and the shared one as id 15
const slots = 11; // ten routed picks and the shared expert a row
const max_rows = 1408;
const tiles = (dims / 16) * (width / 16);
const k2s = [3]u32{ 6, 6, 8 }; // half-bits a weight of gate, up and down: 3, 3 and 4 bits
const rule = "as the decode form it replaces gives them: a NaN or infinite activation reaches every value its row's picks make";

fn depth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

// Zero trellis words decode to one dyadic weight; the stress dots are exact FP32 integers.
const stress_svh: f16 = -0x1.094p-14;
const stress_suh: f16 = 0.1104736328125;
const had: f64 = 1.0 / @sqrt(128.0);

/// Rows of 11 picks over 640-wide experts: the gate-up pass reads 2,560 inputs, the down pass 640.
pub fn spec(kind: Kind) c.Spec {
    const up = kind != .down;
    const common = .{ .accum = c.Accum.f32, .depth = depth, .nan = c.Nan{ .stated = rule }, .tiles = [3]c.Tile{ .{ .size = 6 }, .{}, .{} } };
    const served: []const c.Shape = if (up) &.{ .{ .rows = 130, .n = width, .k = dims }, .{ .rows = 1, .n = width, .k = dims }, .{ .rows = max_rows, .n = width, .k = dims } } else &.{ .{ .rows = 130, .n = dims, .k = width }, .{ .rows = 1, .n = dims, .k = width }, .{ .rows = max_rows, .n = dims, .k = width } };
    return .{
        .name = switch (kind) {
            .gu => "cuda Flash Next EXL3 prefill rotate and gate-up tf_exl3pf_gu (decode sums)",
            .down => "cuda Flash Next EXL3 prefill down tf_exl3pf_down (decode sums)",
            .control => "control: tf_exl3pf_gu against a decode form with the other SwiGLU roundings",
        },
        .accum = common.accum,
        .depth = common.depth,
        .out = if (up) .raw else .f32,
        .replaces = if (up) "grouped gate/up and SwiGLU of exl3_experts" else "grouped down of exl3_experts",
        .nan = common.nan,
        // Down has one rounded Hadamard normalization constant and one FP32 multiply.
        .ops = if (up) 0 else 2,
        .tiles = common.tiles,
        .served = served,
    };
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// One layer of random trellis words (any bits decode), random signs and scales: 15 routed experts and the shared one.
const Weights = struct {
    mem: cuda.DeviceBuffer,
    tab: cuda.DeviceBuffer,
    scales: [6]cuda.DeviceBuffer,
    layer: exl3e.Layer,

    fn init(gpu: check.Gpu, seed: u64, stress: bool) !Weights {
        const a = gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var made: usize = 0;
        var sc: [6]cuda.DeviceBuffer = undefined;
        errdefer for (sc[0..made]) |*b| b.free();
        for (&sc, [6]usize{ dims, dims, width, width, width, dims }, 0..) |*b, n, si| {
            const v = try a.alloc(u16, table_n * n);
            defer a.free(v);
            for (v) |*h| {
                const m: f16 = @floatCast(0.5 + rng.float(f32));
                h.* = @bitCast(if (stress) switch (si) {
                    2, 3 => stress_svh,
                    4 => stress_suh,
                    else => @as(f16, 1),
                } else if (rng.boolean()) m else -m);
            }
            b.* = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(v));
            made += 1;
        }
        var total: usize = 0;
        for (k2s) |k2| total += align256(tiles * 16 * k2) * table_n;
        const host = try a.alloc(u8, total);
        defer a.free(host);
        if (stress) @memset(host, 0) else rng.bytes(host);
        var mem = try cuda.DeviceBuffer.fromHost(gpu.d, host);
        errdefer mem.free();
        var ptrs: [3][table_n]u64 = undefined;
        var k2v: [3][table_n]i32 = undefined;
        var at: usize = 0;
        for (0..table_n) |e| for (0..3) |p| {
            ptrs[p][e] = mem.ptr + at;
            k2v[p][e] = @intCast(k2s[p]);
            at += align256(tiles * 16 * k2s[p]);
        };
        const tl = cuda.exl3_pool.Tables.init(table_n);
        const tb = try a.alloc(u8, tl.bytes);
        defer a.free(tb);
        @memset(tb, 0);
        for (0..3) |p| {
            @memcpy(tb[tl.ptr[p]..][0 .. table_n * 8], std.mem.sliceAsBytes(&ptrs[p]));
            @memcpy(tb[tl.k2[p]..][0 .. table_n * 4], std.mem.sliceAsBytes(&k2v[p]));
        }
        const tab = try cuda.DeviceBuffer.fromHost(gpu.d, tb);
        return .{ .mem = mem, .tab = tab, .scales = sc, .layer = .{
            .gate = tab.ptr + tl.ptr[0],
            .up = tab.ptr + tl.ptr[1],
            .down = tab.ptr + tl.ptr[2],
            .gate_k2 = tab.ptr + tl.k2[0],
            .up_k2 = tab.ptr + tl.k2[1],
            .down_k2 = tab.ptr + tl.k2[2],
            .suh_g = sc[0].ptr,
            .suh_u = sc[1].ptr,
            .svh_g = sc[2].ptr,
            .svh_u = sc[3].ptr,
            .suh_d = sc[4].ptr,
            .svh_d = sc[5].ptr,
            .experts = table_n,
            .dims = dims,
            .width = width,
            .codebook = .mul1,
            .act_mode = 1,
        } };
    }

    fn deinit(w: *Weights) void {
        w.tab.free();
        w.mem.free();
        for (&w.scales) |*b| b.free();
    }
};

pub const Prefill = struct {
    rig: rig_.Rig,
    kind: Kind,
    w: *const Weights,
    sw: *const Weights,
    stress: bool = false,
    k: *const pk_.Kernels,
    ex: *const exl3e.Experts,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    picks: []i32 = &.{},
    plan: pk_.Plan = undefined,
    planned: bool = false,
    want: []u8 = &.{},
    xd: u64 = 0,
    pd: u64 = 0,
    dv: pk_.Device = undefined,
    scr: u64 = 0,
    y: u64 = 0,
    base: u64 = 0,
    out_len: usize = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Prefill) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Prefill) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.picks);
        a.free(self.want);
        if (self.planned) self.plan.deinit(a);
        self.x = &.{};
        self.picks = &.{};
        self.want = &.{};
        self.planned = false;
    }

    pub fn kernel(self: *Prefill) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null };
    }

    const vt = g.hooks(Prefill);

    fn up(self: *const Prefill) bool {
        return self.kind != .down;
    }

    /// Random bf16 rows, ten distinct routed picks a row and the shared expert; non-finite: a NaN and an infinity.
    pub fn prepare(self: *Prefill, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n: usize = if (self.up()) width else dims;
        const k: usize = if (self.up()) dims else width;
        if (shape.rows == 0 or shape.rows > max_rows or shape.n != n or shape.k != k) return false;
        self.free();
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        self.stress = fill == .stress;
        const rows = shape.rows;
        const pairs = rows * slots;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x70667830);
        const rng = prng.random();
        self.x = try a.alloc(u16, rows * dims);
        for (self.x, 0..) |*v, i| {
            const col = i % dims;
            v.* = if (self.stress) c.toBf16(if (col % 128 != 0 or col >= dims - 128) 0 else if (col < dims / 2) 11584 else -11584) else c.toBf16((rng.float(f32) - 0.5) * 2);
        }
        if (fill == .nonfinite) {
            self.x[(rows - 1) * dims] = c.toBf16(std.math.nan(f32));
            if (rows > 1) self.x[(rows - 2) * dims + 1] = c.toBf16(std.math.inf(f32));
        }
        self.picks = try a.alloc(i32, pairs);
        for (0..rows) |r| {
            var ids: [table_n - 1]i32 = undefined;
            for (&ids, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(i32, &ids);
            @memcpy(self.picks[r * slots ..][0 .. slots - 1], ids[0 .. slots - 1]);
            self.picks[r * slots + slots - 1] = table_n - 1;
        }
        self.plan = try pk_.Plan.init(a, pairs, table_n);
        self.planned = true;
        var counts: [table_n]u32 = undefined;
        pk_.build(self.picks, null, &counts, &self.plan);
        const nm = self.plan.n_members;
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.pd = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const sb = pk_.scratchBytes(nm, dims, width);
        self.dv = .{ .members = try self.rig.input(self.plan.memberBytes()), .items = try self.rig.input(self.plan.itemBytes()), .scratch = 0 };
        if (self.up()) {
            self.scr = try self.rig.output(sb);
            self.y = try self.rig.scratch(pairs * dims * 4);
            self.out_len = sb;
        } else {
            self.scr = try self.rig.scratch(sb);
            self.y = try self.rig.output(pairs * dims * 4);
            self.out_len = pairs * dims * 4;
        }
        self.dv.scratch = self.scr;
        self.base = try self.rig.scratch(try exl3e.scratchBytes(self.layer(), rows, slots));
        return true;
    }

    fn layer(self: *const Prefill) exl3e.Layer {
        return if (self.stress) self.sw.layer else self.w.layer;
    }

    pub fn run(self: *Prefill) !void {
        if (self.stress and !self.up()) {
            // Isolate the down sum from the nonlinear GU epilogue with exact half inputs.
            const a = self.rig.gpu.gpa;
            const x = try a.alloc(f16, self.plan.n_members * width);
            defer a.free(x);
            @memset(x, 1024);
            const xd = try self.rig.input(std.mem.sliceAsBytes(x));
            const l = self.layer();
            var dn: cuda.Args = .{};
            dn.add(xd);
            dn.add(l.down);
            dn.add(l.down_k2);
            dn.add(self.dv.items);
            dn.add(self.dv.members);
            dn.add(l.svh_d);
            dn.add(self.y);
            dn.add(@as(i32, width));
            dn.add(@as(i32, dims));
            try cuda.launch.launch(self.k.down, .{ .grid = .{ .x = dims / 256, .y = @intCast(self.plan.n_items) }, .block = .{ .x = 256 } }, self.rig.s, &dn);
        } else try pk_.run(self.k, self.rig.s, self.layer(), self.xd, dims, self.pd, slots, self.plan.n_members, self.plan.n_items, self.dv, self.y);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Prefill) usize {
        return self.out_len / c.size(spec(self.kind).out);
    }

    pub fn reference(self: *Prefill, out: []f64, mag: []f64) !void {
        try self.stressOutput(.none, .sequential, out);
        for (mag, out) |*m, v| m.* = @abs(v);
    }

    pub fn simulate(self: *Prefill, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.stressOutput(acc, order, out);
    }

    // This reference decodes the zero-state weight in FP64, not through either GPU kernel.
    fn stressOutput(self: *Prefill, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!self.stress) return error.ExpectedStressInputs;
        const w: f64 = core.exl3_format.codebookValue(.mul1, 0);
        var zeros: [128]i16 = @splat(0);
        var decoded: [256]f16 = undefined;
        for (k2s) |k2| {
            const bits: core.exl3_format.Width = .{ .halves = @intCast(k2) };
            core.exl3_format.unpack(bits, .mul1, zeros[0..bits.tileWords()], 1, 1, &decoded);
            for (decoded) |v| if (v != w) return error.StressWeightIsNotConstant;
        }
        const rotated: f16 = @floatCast(11584 * had);
        if (rotated != 1024 or @abs(w * 1024) * dims >= 0x1p24) return error.StressIsNotExactFP32;
        var terms: [dims]f64 = undefined;
        const n: usize = if (self.up()) dims else width;
        for (terms[0..n], 0..) |*v, i| v.* = w * (if (!self.up()) @as(f64, 1024) else @as(f64, if (i >= dims - 128) 0 else if (i < dims / 2) 1024 else -1024));
        const dot = core.contract.rows.sumIn(acc, order, terms[0..n]);
        if (!self.up()) {
            // Constant dot rows transform to a single nonzero value per 128-column block.
            for (out, 0..) |*v, i| v.* = if (i % 128 == 0) dot / had else 0;
            return;
        }
        const gg = dot / had * @as(f64, stress_svh);
        const act = gg / (1 + @exp(-gg)) * gg;
        const raw = act * @as(f64, stress_suh) * had;
        // A wide margin to both half midpoints absorbs the FP32 nonlinear epilogue's roundings.
        if (acc == .none and !(raw > 1023.9 and raw < 1024.1)) return error.StressHalfMargin;
        const xd: f16 = @floatCast(raw);
        const a = self.rig.gpu.gpa;
        const bytes = try a.alloc(u8, self.out_len);
        defer a.free(bytes);
        @memset(bytes, 0xff);
        const nm = self.plan.n_members;
        const xu_at = align256(nm * dims * 2);
        for (0..2) |p| for (0..nm * dims) |i| {
            const col = i % dims;
            const xh: f16 = if (col >= dims - 128) 0 else if (col < dims / 2) 1024 else -1024;
            std.mem.writeInt(u16, bytes[p * xu_at + i * 2 ..][0..2], @bitCast(xh), .little);
        };
        for (0..nm * width) |i| std.mem.writeInt(u16, bytes[2 * xu_at + i * 2 ..][0..2], @bitCast(xd), .little);
        for (out, bytes) |*v, b| v.* = @floatFromInt(b);
    }

    /// The bytes of `n` rows of `row` bytes at device `ptr`, member m taking the decode form's row of pair members[m].
    fn gather(self: *Prefill, dst: []u8, ptr: u64, row: usize) !void {
        const src = try self.rig.bytes(ptr, self.shape.rows * slots * row);
        for (self.plan.members[0..self.plan.n_members], 0..) |pair, m| @memcpy(dst[m * row ..][0..row], src[@as(usize, @intCast(pair)) * row ..][0..row]);
    }

    /// The decode form (runSlots) on the same picks, in this pass's bytes: xg, xu, xd by member, or y.
    pub fn replaced(self: *Prefill) ![]const u8 {
        const a = self.rig.gpu.gpa;
        var l = self.layer();
        if (self.kind == .control) l.act_mode = 0;
        const sc = try exl3e.scratch(self.base, l, self.shape.rows, slots);
        try self.ex.runSlots(self.rig.s, l, self.xd, dims, .bf16, self.pd, self.shape.rows, sc);
        try self.rig.s.synchronize();
        a.free(self.want);
        self.want = try a.alloc(u8, self.out_len);
        @memset(self.want, 0xff);
        if (!self.up()) {
            @memcpy(self.want, try self.rig.bytes(sc.y, self.out_len));
            return self.want;
        }
        const nm = self.plan.n_members;
        const xu_at = align256(nm * dims * 2);
        try self.gather(self.want[0 .. nm * dims * 2], sc.xg, dims * 2);
        try self.gather(self.want[xu_at..][0 .. nm * dims * 2], sc.xu, dims * 2);
        try self.gather(self.want[2 * xu_at ..][0 .. nm * width * 2], sc.xd, width * 2);
        return self.want;
    }

    /// Gate 1: the decode form's bytes, and some output non-finite (the planted values reach it).
    pub fn check(self: *Prefill, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        if (self.up()) {
            for (std.mem.bytesAsSlice(u16, got[0 .. self.plan.n_members * dims * 2])) |h| if (h & 0x7c00 == 0x7c00) return null;
        } else for (std.mem.bytesAsSlice(f32, got)) |v| if (!std.math.isFinite(v)) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }
};

/// The three tests over one layer of random weights and the decode form's experts.
pub const Group = struct {
    a: std.mem.Allocator,
    w: Weights,
    sw: Weights,
    k: pk_.Kernels,
    ex: exl3e.Experts,
    entries: [3]Prefill,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.w = try Weights.init(gpu, 0x65786c33, false);
        errdefer self.w.deinit();
        self.sw = try Weights.init(gpu, 0, true);
        errdefer self.sw.deinit();
        self.k = try pk_.Kernels.load(gpu.d);
        errdefer self.k.unload();
        self.ex = try exl3e.Experts.load(gpu.d);
        errdefer self.ex.unload();
        for (&self.entries, std.enums.values(Kind)) |*x, kind| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kind = kind, .w = &self.w, .sw = &self.sw, .k = &self.k, .ex = &self.ex };
        errdefer for (&self.entries) |*x| x.deinit();
        for (&self.entries) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.entries) |*x| x.deinit();
        self.ex.unload();
        self.k.unload();
        self.w.deinit();
        self.sw.deinit();
        self.a.destroy(self);
    }
};

/// tf-cuda-test exl3-prefill-contract: just these kernels through the gates and the control (the full runner is hours).
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(gpu.gpa);
    const group = try Group.create(gpu.gpa, gpu, s, &list);
    defer group.destroy();
    var t: g.Tally = .{};
    try g.runAll(gpu.gpa, list.items, 0x6b63, &t);
    std.debug.print("SUMMARY {d} kernels: {d} pass, {d} fail, {d} known; {d} controls: {d} red, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "exl3-prefill-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
