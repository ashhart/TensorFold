//! Qwen3-ASR's tower kernels under the contract (qwen3-asr-contract): vit.cu's conv taps and flatten, attention at 64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const full = @import("contract_qwen3_asr_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const controls = struct {
    pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
};

fn depth(_: c.Shape) usize {
    return 0;
}

/// A conv's geometry in a shape: rows the chunks, n the channels, k = fi 256 + ti.
fn geometry(s: c.Shape) struct { n: usize, c: usize, fi: usize, ti: usize } {
    return .{ .n = s.rows, .c = s.n, .fi = s.k / 256, .ti = s.k % 256 };
}

/// The served convs (two chunks of each of the tower's three) and odd small grids for the padding's edges.
const conv_shapes = [_]c.Shape{ .{ .rows = 2, .n = 1, .k = 128 * 256 + 100 }, .{ .rows = 2, .n = 480, .k = 64 * 256 + 50 }, .{ .rows = 2, .n = 480, .k = 32 * 256 + 25 }, .{ .rows = 3, .n = 3, .k = 7 * 256 + 5 }, .{ .rows = 1, .n = 1, .k = 1 * 256 + 1 }, .{ .rows = 1, .n = 33, .k = 2 * 256 + 9 } };
const flat_shapes = [_]c.Shape{ .{ .rows = 2, .n = 480, .k = 16 * 256 + 13 }, .{ .rows = 3, .n = 8, .k = 5 * 256 + 3 }, .{ .rows = 1, .n = 1, .k = 1 * 256 + 1 } };

pub fn spec(name: []const u8, served: []const c.Shape) c.Spec {
    return .{
        .name = name,
        .accum = .none, // copies: every output checked bit for bit against the host's replay
        .depth = depth,
        .out = .bf16,
        .served = served,
    };
}

/// tf_vit_conv_cols (`flatten` false) or tf_vit_conv_flatten on random bf16 rows, against the host's copy.
pub const Move = struct {
    rig: rig_.Rig,
    vk: *const cuda.vit.Kernels,
    flatten: bool,
    name: []const u8,
    ctl: ?cuda.Function = null,
    x: []u16 = &.{},
    want: []u16 = &.{},
    dev: [2]u64 = undefined,
    shape: c.Shape = undefined,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Move) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.want);
        self.rig.deinit();
    }

    const vt = g.hooks(Move);

    pub fn kernel(self: *Move) g.Kernel {
        return .{ .spec = spec(self.name, if (self.flatten) &flat_shapes else &conv_shapes), .ctx = self, .vt = &vt, .control = if (self.ctl != null) .shapes else null };
    }

    fn ld(gm: anytype) usize {
        return std.mem.alignForward(usize, 9 * gm.c, 8);
    }

    /// Output rows and their width.
    fn outShape(self: *const Move) [2]usize {
        const gm = geometry(self.shape);
        if (self.flatten) return .{ gm.n * gm.ti, gm.c * gm.fi };
        return .{ gm.n * ((gm.fi - 1) / 2 + 1) * ((gm.ti - 1) / 2 + 1), ld(gm) };
    }

    pub fn prepare(self: *Move, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const gm = geometry(shape);
        if (gm.n == 0 or gm.c == 0 or gm.fi == 0 or gm.ti == 0) return false;
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        self.x = &.{};
        self.x = try a.alloc(u16, gm.n * gm.fi * gm.ti * gm.c);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.x) |*v| v.* = c.toBf16(rng.floatNorm(f32));
        if (fill == .nonfinite) {
            self.x[0] = c.toBf16(std.math.nan(f32));
            self.x[self.x.len / 2] = c.toBf16(std.math.inf(f32));
            self.x[self.x.len - 1] = c.toBf16(-std.math.inf(f32));
        }
        const o = self.outShape();
        a.free(self.want);
        self.want = &.{};
        self.want = try a.alloc(u16, o[0] * o[1]);
        self.replay(gm);
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.dev[1] = try self.rig.output(self.want.len * 2);
        return true;
    }

    /// The host's copy: zero where the conv's padding or the column tail is.
    fn replay(self: *Move, gm: anytype) void {
        if (self.flatten) {
            for (0..gm.n) |n| for (0..gm.ti) |t| for (0..gm.c) |ci| for (0..gm.fi) |f| {
                self.want[(n * gm.ti + t) * gm.c * gm.fi + ci * gm.fi + f] = self.x[((n * gm.fi + f) * gm.ti + t) * gm.c + ci];
            };
            return;
        }
        const fo = (gm.fi - 1) / 2 + 1;
        const to = (gm.ti - 1) / 2 + 1;
        const w = ld(gm);
        for (0..gm.n) |n| for (0..fo) |f| for (0..to) |t| for (0..w) |j| {
            var v: u16 = 0;
            if (j < 9 * gm.c) {
                const sf = @as(isize, @intCast(2 * f + j % 9 / 3)) - 1;
                const st = @as(isize, @intCast(2 * t + j % 3)) - 1;
                if (sf >= 0 and st >= 0 and sf < gm.fi and st < gm.ti) v = self.x[((n * gm.fi + @as(usize, @intCast(sf))) * gm.ti + @as(usize, @intCast(st))) * gm.c + j / 9];
            }
            self.want[((n * fo + f) * to + t) * w + j] = v;
        };
    }

    pub fn run(self: *Move) !void {
        const gm = geometry(self.shape);
        const s = self.rig.s;
        if (self.ctl) |f| {
            var a: cuda.Args = .{};
            a.add(self.dev[0]);
            a.add(self.dev[1]);
            const o = self.outShape();
            if (self.flatten) {
                for ([_]usize{ gm.fi, gm.ti, gm.c }) |v| a.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(o[0]) }, .block = .{ .x = 256 } }, s, &a);
            } else {
                for ([_]usize{ gm.fi, gm.ti, gm.c, o[1] }) |v| a.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(o[0]), .y = @intCast((o[1] + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
            }
        } else if (self.flatten) {
            try self.vk.convFlatten(s, self.dev[0], self.dev[1], gm.n, gm.fi, gm.ti, gm.c);
        } else try self.vk.convCols(s, self.dev[0], self.dev[1], gm.n, gm.fi, gm.ti, gm.c, ld(gm));
        try s.synchronize();
    }

    pub fn count(self: *Move) usize {
        return self.want.len;
    }

    pub fn reference(self: *Move, out: []f64, mag: []f64) !void {
        for (out, self.want) |*o, w| o.* = c.fromBf16(w);
        @memset(mag, 0);
    }

    /// Every output's bits equal to the host's copy (NaN payloads too).
    pub fn check(self: *Move, _: g.Gate) !?[]const u8 {
        const got = try self.rig.bytes(self.dev[1], self.want.len * 2);
        for (self.want, 0..) |w, i| {
            const v = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
            if (v != w) return try std.fmt.bufPrint(&self.msg, "output {d} has bits {x:0>4} where the copy has {x:0>4}", .{ i, v, w });
        }
        return null;
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    vk: cuda.vit.Kernels,
    ctl: cuda.Module,
    att: cuda.vit.Attention,
    moves: [4]Move = undefined,
    full: full.Full = undefined,

    /// The set, its kernels appended to `list` (each control after the kernel it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var vk = try cuda.vit.Kernels.load(gpu.d);
        errdefer vk.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .vk = vk, .ctl = ctl, .att = try cuda.vit.Attention.load(gpu.d) };
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        set.moves[0] = .{ .rig = rig, .vk = &set.vk, .flatten = false, .name = "cuda vit tf_vit_conv_cols (Qwen3-ASR's 3x3 stride-2 conv taps)" };
        set.moves[1] = .{ .rig = rig, .vk = &set.vk, .flatten = false, .name = "control: tf_vit_conv_cols reads the clamped edge, not the zero padding", .ctl = try ctl.function("tf_ctl_vit_conv_cols_edge") };
        set.moves[2] = .{ .rig = rig, .vk = &set.vk, .flatten = true, .name = "cuda vit tf_vit_conv_flatten (Qwen3-ASR's conv_out rows)" };
        set.moves[3] = .{ .rig = rig, .vk = &set.vk, .flatten = true, .name = "control: tf_vit_conv_flatten writes f-major rows", .ctl = try ctl.function("tf_ctl_vit_conv_flatten_fc") };
        set.full = .{ .rig = rig, .att = &set.att };
        for (&set.moves) |*x| try list.append(a, x.kernel());
        try list.append(a, set.full.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.moves) |*x| x.deinit();
        set.full.deinit();
        set.att.unload();
        set.ctl.unload();
        set.vk.unload();
        set.gpa.destroy(set);
    }
};

/// The Qwen3-ASR tower kernels' gates 1-4 and their controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x71776e33, &t);
    std.debug.print("SUMMARY qwen3-asr: {d} kernels, {d} pass gates 1-4, {d} fail; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "qwen3-asr-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}
