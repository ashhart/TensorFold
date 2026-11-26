//! Raw identity A/B coverage for FlashNext's direct attention programs and their PO/PM/PL state.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const G = 12;
const HD = 256;
const DI = 128;
const CH = 512;
const IDW = 2052;
const DENSE_KEYS = 2051;
const SCALE: f32 = 0.0625;
const NCH_SPARSE = 5;
const NCH_DENSE = 3;

const Mode = enum { chunks, one, ahead, fused, fused_paged, one_paged, ahead_paged };
const modes = [_]Mode{ .chunks, .ahead, .fused, .fused_paged, .one_paged, .ahead_paged };

fn sparse(shape: c.Shape) bool {
    return shape.n == 2 or shape.n == 4;
}

fn qsa(shape: c.Shape) bool {
    return shape.n != 1;
}

fn ptok(shape: c.Shape) usize {
    return if (shape.n >= 3) 512 else 48;
}

fn paged(mode: Mode) bool {
    return switch (mode) {
        .fused_paged, .one_paged, .ahead_paged => true,
        else => false,
    };
}

fn fused(mode: Mode) bool {
    return mode == .fused or mode == .fused_paged;
}

fn ahead(mode: Mode) bool {
    return mode == .ahead or mode == .ahead_paged;
}

fn baseline(mode: Mode) ?Mode {
    return switch (mode) {
        .chunks, .one => null,
        .ahead => .chunks,
        .fused => .one,
        .fused_paged => .fused,
        .one_paged => .one,
        .ahead_paged => .ahead,
    };
}

fn replacement(mode: Mode) ?[]const u8 {
    return switch (mode) {
        .chunks, .one => null,
        .ahead => "fn_attn_chunks",
        .fused => "fn_attn_chunks_one then fn_attn_merge",
        .fused_paged => "fn_attn_fused",
        .one_paged => "fn_attn_chunks_one",
        .ahead_paged => "fn_attn_chunks_ahead",
    };
}

fn depth(_: c.Shape) usize {
    return 0;
}

const short = [_]c.Shape{
    .{ .rows = 4, .n = 1, .k = 33 },
    .{ .rows = 1, .n = 1, .k = 1 },
    .{ .rows = 16, .n = 1, .k = 512 },
    .{ .rows = 16, .n = 1, .k = 511 },
    .{ .rows = 4, .n = 2, .k = 0 },
    .{ .rows = 1, .n = 2, .k = 33 },
    .{ .rows = 16, .n = 2, .k = 511 },
    .{ .rows = 16, .n = 2, .k = 1025 },
};
const wide = short ++ [_]c.Shape{
    .{ .rows = 17, .n = 1, .k = 31 },
    .{ .rows = 256, .n = 1, .k = 32 },
    .{ .rows = 17, .n = 2, .k = 513 },
    .{ .rows = 256, .n = 2, .k = 2049 },
};
const fused_shapes = wide ++ [_]c.Shape{
    .{ .rows = 257, .n = 1, .k = 33 },
    .{ .rows = 1024, .n = 1, .k = 32 },
    .{ .rows = 257, .n = 2, .k = 1025 },
    .{ .rows = 1024, .n = 2, .k = 65 },
};
const paged_short = short ++ [_]c.Shape{
    .{ .rows = 4, .n = 3, .k = 513 },
    .{ .rows = 4, .n = 4, .k = 0 },
    .{ .rows = 16, .n = 4, .k = 1025 },
};
const paged_wide = paged_short ++ wide[short.len..] ++ [_]c.Shape{
    .{ .rows = 256, .n = 3, .k = 32 },
    .{ .rows = 256, .n = 4, .k = 513 },
};
const paged_fused = fused_shapes ++ [_]c.Shape{
    .{ .rows = 4, .n = 3, .k = 513 },
    .{ .rows = 4, .n = 4, .k = 0 },
    .{ .rows = 16, .n = 4, .k = 1025 },
    .{ .rows = 1024, .n = 4, .k = 65 },
};
const control_shapes = [_]c.Shape{
    .{ .rows = 4, .n = 1, .k = 97 },
    .{ .rows = 4, .n = 2, .k = 1025 },
};

fn served(mode: Mode) []const c.Shape {
    return switch (mode) {
        .chunks, .one => &wide,
        .ahead => &short,
        .fused => &fused_shapes,
        .fused_paged => &paged_fused,
        .one_paged => &paged_wide,
        .ahead_paged => &paged_short,
    };
}

const Bind = struct {
    q: u64 = 0,
    kc: u64 = 0,
    vc: u64 = 0,
    pos: u64 = 0,
    po: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    out: u64 = 0,
    ids: u64 = 0,
    nk: u64 = 0,
    spr: u64 = 0,
    pg: flash.Pg = .{},
};

const Host = struct {
    q: []u16,
    key: []u16,
    value: []u16,
    ids: []i32,
    nk: []i32,
    spr: []i32,
    pos: [4]i32,
    pages: []u16,
    table: []i32,
    bad_table: []i32,
};

const Fns = struct {
    chunks: cuda.Function,
    one: cuda.Function,
    ahead: cuda.Function,
    fused: cuda.Function,
    fused_paged: cuda.Function,
    one_paged: cuda.Function,
    ahead_paged: cuda.Function,
    merge: cuda.Function,
};

const Runner = struct {
    rig: rig_.Rig,
    base: rig_.Rig,
    fns: *const Fns,
    mode: Mode,
    half: bool = false,
    control: bool = false,
    shape: c.Shape = undefined,
    label: [160]u8 = undefined,
    arena: ?std.heap.ArenaAllocator = null,
    dev: Bind = .{},
    base_dev: Bind = .{},
    host: Host = undefined,
    output_len: usize = 0,

    fn deinit(self: *Runner) void {
        self.rig.deinit();
        self.base.deinit();
        if (self.arena) |*a| a.deinit();
    }

    fn qh(self: *const Runner) usize {
        return if (self.half) 12 else 24;
    }

    fn kh(self: *const Runner) usize {
        return if (self.half) 1 else 2;
    }

    fn kernel(self: *Runner) g.Kernel {
        const base = if (self.control)
            "control: cuda fn_attn_chunks_ahead_paged with a rotated page table"
        else switch (self.mode) {
            .chunks => "cuda FlashNext fn_attn_chunks raw PO PM PL",
            .one => unreachable,
            .ahead => "cuda FlashNext fn_attn_chunks_ahead raw PO PM PL",
            .fused => "cuda FlashNext fn_attn_fused raw output",
            .fused_paged => "cuda FlashNext fn_attn_fused_paged raw output",
            .one_paged => "cuda FlashNext fn_attn_chunks_one_paged raw PO PM PL",
            .ahead_paged => "cuda FlashNext fn_attn_chunks_ahead_paged raw PO PM PL",
        };
        const name = if (self.control) base else std.fmt.bufPrint(&self.label, "{s}, {d}/{d} heads", .{ base, self.qh(), self.kh() }) catch unreachable;
        return .{
            .ctx = self,
            .vt = &vt,
            .control = if (self.control) .shapes else null,
            .spec = .{
                .name = name,
                .accum = .none,
                .depth = depth,
                .out = .raw,
                .replaces = replacement(self.mode),
                .nan = .{ .stated = "raw outputs and every unwritten partial-state byte, including NaN and infinities" },
                .served = if (self.control) &control_shapes else served(self.mode),
            },
        };
    }

    const vt = g.hooks(Runner);

    fn valid(self: *const Runner, shape: c.Shape) bool {
        if (shape.rows == 0 or shape.k > IDW or shape.n == 0 or shape.n > (if (paged(self.mode)) 4 else 2)) return false;
        if (!sparse(shape) and shape.k == 0) return false;
        const rows = if (fused(self.mode)) 1024 else if (ahead(self.mode)) 16 else 256;
        return shape.rows <= rows;
    }

    fn live(self: *const Runner, shape: c.Shape) usize {
        _ = self;
        return if (sparse(shape)) 2048 + shape.rows else shape.k - 1 + shape.rows;
    }

    fn prepare(self: *Runner, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!self.valid(shape)) return false;
        self.rig.reset();
        self.base.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = .init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        const cap = self.live(shape) + 4;
        self.host = try self.makeHost(a, cap, fill, seed);
        self.dev = try self.bind(&self.rig, self.mode, self.host, false);
        const need_base = baseline(self.mode) != null;
        self.base_dev = if (need_base) try self.bind(&self.base, baseline(self.mode).?, self.host, self.mode == .fused) else .{};
        self.output_len = if (fused(self.mode))
            shape.rows * self.qh() * HD * 2
        else
            shape.rows * self.nch() * self.qh() * (HD + 2) * 4;
        return true;
    }

    fn makeHost(self: *Runner, a: std.mem.Allocator, cap: usize, fill: g.Fill, seed: u64) !Host {
        const rows = self.shape.rows;
        var random = std.Random.DefaultPrng.init(seed);
        const rng = random.random();
        const q = try a.alloc(u16, rows * self.qh() * HD);
        const key = try a.alloc(u16, cap * self.kh() * HD);
        const value = try a.alloc(u16, key.len);
        for (q) |*v| v.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        for (key) |*v| v.* = c.toBf16(if (fill == .stress) 0 else rng.float(f32) * 2 - 1);
        for (value, 0..) |*v, i| {
            const token = i / (self.kh() * HD);
            const d = i % HD;
            const half = cap / 2;
            const positive: f32 = if (token % 2 == 0) 127 else 16256;
            const residual: f32 = if (token < 8 and token % 2 == 0) 1 else 0;
            const stress_value = if (token < half) positive else residual - positive;
            v.* = c.toBf16(if (fill == .stress) stress_value else rng.float(f32) * 2 - 1);
            if (fill == .nonfinite and token == 0 and d < 3) v.* = ([_]u16{ 0x7fc0, 0x7f80, 0xff80 })[d];
        }
        const ids = try a.alloc(i32, rows * IDW);
        @memset(ids, -1);
        const nk = try a.alloc(i32, rows);
        const spr = try a.alloc(i32, rows);
        for (0..rows) |r| {
            nk[r] = @intCast(self.shape.k);
            const dense_row = !sparse(self.shape) or (self.shape.k != 0 and rows > 1 and rows <= 256 and r % 3 == 1);
            spr[r] = @intFromBool(qsa(self.shape) and sparse(self.shape) and !dense_row);
            if (spr[r] != 0) {
                for (0..self.shape.k) |j| ids[r * IDW + j] = @intCast(2048 + rows - 1 - j);
            }
        }
        if (fill == .nonfinite) q[0] = 0x7fc1;
        if (fill == .random) plant(self.shape, self.kh(), q, key, value, ids, nk, spr, cap);
        const pos: [4]i32 = .{ @intCast(if (sparse(self.shape)) 2048 else self.shape.k - 1), 0, 0, 0 };
        const per = ptok(self.shape);
        const np = @max(1, (cap + per - 1) / per);
        const page_elems = per * (2 * self.kh() * HD + DI) + per / 4 * DI;
        const pages = try a.alloc(u16, (np + 2) * page_elems);
        @memset(pages, 0);
        const table = try a.alloc(i32, np);
        const bad = try a.alloc(i32, np);
        for (0..np) |p| {
            table[p] = @intCast(np - 1 - p);
            bad[p] = @intCast((np - p) % np);
        }
        for (0..cap) |token| for (0..self.kh() * HD) |d| {
            const physical: usize = @intCast(table[token / per]);
            const at = physical * page_elems + token % per * self.kh() * HD + d;
            pages[at] = key[token * self.kh() * HD + d];
            pages[at + per * self.kh() * HD] = value[token * self.kh() * HD + d];
        };
        return .{ .q = q, .key = key, .value = value, .ids = ids, .nk = nk, .spr = spr, .pos = pos, .pages = pages, .table = table, .bad_table = bad };
    }

    fn bind(self: *Runner, rig: *rig_.Rig, variant: Mode, host: Host, for_merge: bool) !Bind {
        var b: Bind = .{};
        b.q = try rig.input(std.mem.sliceAsBytes(host.q));
        if (paged(variant)) {
            b.kc = try rig.input(std.mem.sliceAsBytes(host.pages));
            b.vc = b.kc + ptok(self.shape) * self.kh() * HD * 2;
            const table = if (self.control and variant == self.mode) host.bad_table else host.table;
            b.pg = .{ .pt = try rig.input(std.mem.sliceAsBytes(table)), .ps = @intCast(host.pages.len / (host.table.len + 2)), .ptok = @intCast(ptok(self.shape)) };
        } else {
            b.kc = try rig.input(std.mem.sliceAsBytes(host.key));
            b.vc = try rig.input(std.mem.sliceAsBytes(host.value));
        }
        b.pos = try rig.input(std.mem.asBytes(&host.pos));
        b.ids = try rig.input(std.mem.sliceAsBytes(host.ids));
        b.nk = try rig.input(std.mem.sliceAsBytes(host.nk));
        b.spr = try rig.input(std.mem.sliceAsBytes(host.spr));
        const chunks = self.nch();
        if (fused(variant) or for_merge) b.out = try rig.output(self.shape.rows * self.qh() * HD * 2);
        if (for_merge) {
            b.po = try rig.scratch(self.shape.rows * chunks * self.qh() * HD * 4);
            b.pm = try rig.scratch(self.shape.rows * chunks * self.qh() * 4);
            b.pl = try rig.scratch(self.shape.rows * chunks * self.qh() * 4);
        } else if (!fused(variant)) {
            b.po = try rig.output(self.shape.rows * chunks * self.qh() * HD * 4);
            b.pm = try rig.output(self.shape.rows * chunks * self.qh() * 4);
            b.pl = try rig.output(self.shape.rows * chunks * self.qh() * 4);
        }
        return b;
    }

    fn nch(self: *const Runner) usize {
        return if (qsa(self.shape) or fused(self.mode)) NCH_SPARSE else NCH_DENSE;
    }

    fn function(self: *const Runner, mode: Mode) cuda.Function {
        return switch (mode) {
            .chunks => self.fns.chunks,
            .one => self.fns.one,
            .ahead => self.fns.ahead,
            .fused => self.fns.fused,
            .fused_paged => self.fns.fused_paged,
            .one_paged => self.fns.one_paged,
            .ahead_paged => self.fns.ahead_paged,
        };
    }

    fn launch(self: *Runner, rig: *rig_.Rig, b: Bind, variant: Mode) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ b.q, b.kc, b.vc, b.pos }) |p| args.add(p);
        if (fused(variant)) {
            args.add(b.out);
        } else for ([_]u64{ b.po, b.pm, b.pl }) |p| args.add(p);
        for ([_]u64{ b.ids, b.nk, b.spr }) |p| args.add(p);
        args.add(@as(c_int, @intCast(self.qh())));
        args.add(@as(c_int, @intCast(self.kh())));
        args.add(@as(c_int, G));
        args.add(@as(c_int, CH));
        if (!fused(variant)) args.add(@as(c_int, @intCast(self.nch())));
        args.add(SCALE);
        args.add(@as(c_int, IDW));
        args.add(@as(c_int, @intFromBool(qsa(self.shape) or fused(self.mode))));
        if (paged(variant)) args.add(b.pg);
        const chunks = (@min(self.live(self.shape), DENSE_KEYS) + CH - 1) / CH;
        const shared: usize = switch (variant) {
            .chunks => 43520,
            .ahead, .ahead_paged => 77312,
            else => 26624,
        };
        try cuda.launch.launch(self.function(variant), .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.kh()), .z = @intCast(if (fused(variant)) 1 else chunks) }, .block = .{ .x = 128 }, .shared = @intCast(shared) }, rig.s, &args);
        if (variant == .one and fused(self.mode)) {
            args = .{};
            for ([_]u64{ b.po, b.pm, b.pl, b.pos, b.out, b.nk, b.spr }) |p| args.add(p);
            for ([_]c_int{ @intCast(self.qh()), @intCast(self.kh()), G, CH, @intCast(self.nch()), 1 }) |v| args.add(v);
            try cuda.launch.launch(self.fns.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.qh()) }, .block = .{ .x = HD } }, rig.s, &args);
        }
        try rig.s.synchronize();
    }

    fn run(self: *Runner) !void {
        try self.launch(&self.rig, self.dev, self.mode);
    }

    fn replaced(self: *Runner) ![]const u8 {
        const mode = baseline(self.mode) orelse return error.NoReplacement;
        try self.launch(&self.base, self.base_dev, mode);
        if (!try self.base.intact() or self.base.overlap()) return error.ReplacementGuard;
        return self.base.outputBytes();
    }

    fn count(self: *const Runner) usize {
        return self.output_len;
    }

    fn reference(_: *Runner, _: []f64, _: []f64) !void {
        return error.RawIdentityUsesReplacementOrBytes;
    }

    fn tally(value: f64, finite: *usize, exceptional: *usize) void {
        if (c.classOf(value) == .finite) finite.* += 1 else exceptional.* += 1;
    }

    fn f32At(bytes: []const u8, bias: usize, index: usize) f32 {
        return @bitCast(std.mem.readInt(u32, bytes[bias + 4 * index ..][0..4], .little));
    }

    fn check(self: *Runner, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const bytes = try self.rig.outputBytes();
        var finite: usize = 0;
        var exceptional: usize = 0;
        if (fused(self.mode)) {
            for (0..self.output_len / 2) |i| {
                const bits = std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
                tally(c.fromBf16(bits), &finite, &exceptional);
            }
        } else {
            const po_values = self.shape.rows * self.nch() * self.qh() * HD;
            const state_values = self.shape.rows * self.nch() * self.qh();
            for (0..self.shape.rows) |r| {
                const keys: usize = if (self.host.spr[r] != 0)
                    @intCast(self.host.nk[r])
                else
                    @intCast(self.host.pos[0] + @as(i32, @intCast(r)) + 1);
                for (0..(keys + CH - 1) / CH) |chunk| for (0..self.qh()) |head| {
                    const base = (r * self.nch() + chunk) * self.qh() + head;
                    for (0..HD) |d| tally(f32At(bytes, 0, base * HD + d), &finite, &exceptional);
                    tally(f32At(bytes, po_values * 4, base), &finite, &exceptional);
                    tally(f32At(bytes, (po_values + state_values) * 4, base), &finite, &exceptional);
                };
            }
        }
        if (finite == 0 or exceptional == 0) return "nonfinite plants must leave both finite and exceptional written outputs";
        return null;
    }
};

fn plant(shape: c.Shape, kv: usize, q: []u16, key: []u16, value: []u16, ids: []i32, nk: []const i32, spr: []const i32, cap: usize) void {
    @memset(q[0..HD], 0);
    q[0] = c.toBf16(1);
    q[1] = 0x8000;
    q[2] = 0x0001;
    q[3] = 0x8001;
    q[4] = 0x007f;
    q[5] = 0x807f;
    @memset(key, 0);
    const points = [_]usize{ 31, 32, 63, 64, 65, 511, 512 };
    for (points, 0..) |j, i| {
        if (j >= nk[0] and spr[0] != 0) continue;
        if (j >= shape.k and spr[0] == 0) continue;
        const source: usize = if (spr[0] != 0) @intCast(ids[j]) else j;
        if (source >= cap) continue;
        const k = key[source * kv * HD ..][0..HD];
        @memset(k, 0);
        k[0] = c.toBf16(if (j == 65) 32 else 16);
        k[1] = if (i % 2 == 0) 0x8000 else 0;
        k[2] = if (i % 2 == 0) 0x0001 else 0x8001;
        k[3] = 0x807f;
        const v = value[source * kv * HD ..][0..HD];
        v[0] = c.toBf16(@floatFromInt(1 + 2 * i));
        v[1] = 0x8000;
        v[2] = 0x0001;
        v[3] = 0x8001;
        v[4] = 0x007f;
        v[5] = 0x807f;
    }
}

/// Both served head geometries for six primary symbols, plus a real rotated-page control.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: ?cuda.Module = null,
    fns: Fns = undefined,
    items: [2 * modes.len + 1]Runner = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_attn);
        const m = self.mod.?;
        self.fns = .{
            .chunks = try m.function("fn_attn_chunks"),
            .one = try m.function("fn_attn_chunks_one"),
            .ahead = try m.function("fn_attn_chunks_ahead"),
            .fused = try m.function("fn_attn_fused"),
            .fused_paged = try m.function("fn_attn_fused_paged"),
            .one_paged = try m.function("fn_attn_chunks_one_paged"),
            .ahead_paged = try m.function("fn_attn_chunks_ahead_paged"),
            .merge = try m.function("fn_attn_merge"),
        };
        try self.fns.chunks.allowDynamicShared(43520);
        try self.fns.one.allowDynamicShared(26624);
        try self.fns.ahead.allowDynamicShared(77312);
        try self.fns.fused.allowDynamicShared(26624);
        try self.fns.fused_paged.allowDynamicShared(26624);
        try self.fns.one_paged.allowDynamicShared(26624);
        try self.fns.ahead_paged.allowDynamicShared(77312);
        for (modes) |mode| for ([_]bool{ false, true }) |half| try self.add(gpu, s, mode, half, false, kernels);
        try self.add(gpu, s, .ahead_paged, false, true, kernels);
        return self;
    }

    fn add(self: *Group, gpu: check.Gpu, s: cuda.Stream, mode: Mode, half: bool, control: bool, kernels: *std.ArrayList(g.Kernel)) !void {
        self.items[self.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .base = .{ .gpu = gpu, .s = s }, .fns = &self.fns, .mode = mode, .half = half, .control = control };
        self.made += 1;
        try kernels.append(self.a, self.items[self.made - 1].kernel());
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*item| item.deinit();
        if (self.mod) |m| m.unload();
        self.a.destroy(self);
    }
};
