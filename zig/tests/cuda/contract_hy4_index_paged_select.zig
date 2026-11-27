//! Paged indexer top-511 + own-group selection kernels (select_1cta, select_scan + select_merge) under the contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_index_paged");
const rig_ = @import("contract_rig.zig");
const set = @import("contract_hy4_index_paged.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const rule = "top-511 complete groups < own in ascending order + own at [511], NaN above +inf, -0 as +0, lower group on ties, expanded to 2048 tokens";

fn zeroDepth(_: c.Shape) usize {
    return 0;
}

fn orderKey(s: f32) u32 {
    const b: i32 = if (std.math.isNan(s)) std.math.maxInt(i32) else @bitCast(s + 0.0);
    return @bitCast(b ^ ((b >> 31) | std.math.minInt(i32)));
}

fn before(keys: []const u32, x: u32, y: u32) bool {
    return keys[x] > keys[y] or (keys[x] == keys[y] and x < y);
}

fn fillScores(host: []f32, rows: usize, n: usize, gcap: usize, fill: g.Fill, seed: u64, multi: bool) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (0..rows) |r| {
        const dst = host[r * gcap ..][0..gcap];
        if (multi and fill == .random and n > hy4.select_cap and r == 1) {
            @memset(dst[0..n], 0.0);
            var placed: usize = 0;
            while (placed < 400) : (placed += 1) dst[(placed * 17) % n] = 1.0;
        } else if (multi and fill == .random and n > hy4.select_cap and r == 2) {
            @memset(dst[0..n], 0.0);
            for (0..@min(n, 5000)) |p| dst[p] = 1.0;
        } else {
            for (dst[0..n]) |*v| {
                const lvl: f32 = @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 128))) - 64);
                v.* = if (lvl == 0 and rng.boolean()) -0.0 else lvl * 0.25;
            }
        }
        for (dst[n..]) |*v| v.* = std.math.nan(f32);
    }
    if (fill == .nonfinite) {
        const neg_nan: f32 = @bitCast(@as(u32, 0xffc00000));
        const plants = [_]f32{ std.math.nan(f32), neg_nan, std.math.inf(f32), -std.math.inf(f32) };
        for (plants, 0..) |v, r| if (r < rows) {
            const step = n / 8;
            for (0..8) |e| host[r * gcap + e * step + r] = v;
        };
    }
}

fn verifyOutputs(a: std.mem.Allocator, host: []const f32, pos: []const i32, rows: usize, n: usize, gcap: usize, out_bytes: []const u8, msg: *[200]u8) !?[]const u8 {
    const grp_bytes = rows * hy4.groups * 4;
    const grp = out_bytes[0..grp_bytes];
    const tok = out_bytes[grp_bytes..];
    const keys = try a.alloc(u32, n);
    defer a.free(keys);
    const idx = try a.alloc(u32, n);
    defer a.free(idx);
    const k = @min(hy4.groups - 1, n);
    for (0..rows) |r| {
        for (keys, host[r * gcap ..][0..n]) |*kv, sv| kv.* = orderKey(sv);
        for (idx, 0..) |*iv, p| iv.* = @intCast(p);
        std.mem.sort(u32, idx, keys, before);
        const picks = idx[0..k];
        std.mem.sort(u32, picks, {}, std.sort.asc(u32));
        for (0..hy4.groups) |j| {
            const want: i32 = if (j < k) @intCast(picks[j]) else if (j < hy4.groups - 1) -1 else @intCast(n);
            const have = std.mem.readInt(i32, grp[(r * hy4.groups + j) * 4 ..][0..4], .little);
            if (have != want) return try std.fmt.bufPrint(msg, "row {d} group slot {d}: got {d}, host gives {d}", .{ r, j, have, want });
            for (0..hy4.ratio) |o| {
                const tok_id: i32 = want * 4 + @as(i32, @intCast(o));
                const want_tok: i32 = if (want >= 0 and tok_id <= pos[r]) tok_id else -1;
                const have_tok = std.mem.readInt(i32, tok[(r * hy4.tokens_per_row + j * hy4.ratio + o) * 4 ..][0..4], .little);
                if (have_tok != want_tok) return try std.fmt.bufPrint(msg, "row {d} token {d}: got {d}, expected {d}", .{ r, j * 4 + o, have_tok, want_tok });
            }
        }
    }
    return null;
}

const sel1_served = [_]c.Shape{ .{ .rows = 8, .n = 4096, .k = 511 }, .{ .rows = 1, .n = 512, .k = 511 }, .{ .rows = 4, .n = 4097, .k = 511 }, .{ .rows = 2, .n = 8192, .k = 511 } };

pub const Select1Cta = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fault: set.Fault = .none,
    shape: c.Shape = undefined,
    gcap: usize = 0,
    host: []f32 = &.{},
    pos: []i32 = &.{},
    in: [2]u64 = @splat(0),
    out_g: u64 = 0,
    out_t: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Select1Cta) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Select1Cta) void {
        const a = self.rig.gpu.gpa;
        a.free(self.host);
        a.free(self.pos);
        self.host = &.{};
        self.pos = &.{};
    }

    pub fn kernel(self: *Select1Cta) g.Kernel {
        var s: c.Spec = .{
            .name = "cuda hy4_index_paged tf_hy4_idx_select_1cta (top-511 + own + 2,048-token expand)",
            .accum = .none,
            .depth = zeroDepth,
            .out = .raw,
            .nan = .{ .stated = rule },
            .tiles = .{ .{}, .{ .size = 1024, .step = 1, .min = 512 }, .{} },
            .served = &sel1_served,
        };
        if (self.fault != .none) s.name = set.controlName("select_1cta", self.fault);
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(Select1Cta);

    pub fn count(self: *const Select1Cta) usize {
        return self.shape.rows * (hy4.groups + hy4.tokens_per_row) * 4;
    }

    pub fn prepare(self: *Select1Cta, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 511 or shape.n < 512 or shape.n > 8192 or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.gcap = std.mem.alignForward(usize, shape.n, 4);
        const a = self.rig.gpu.gpa;
        self.host = try a.alloc(f32, shape.rows * self.gcap);
        self.pos = try a.alloc(i32, shape.rows);
        fillScores(self.host, shape.rows, shape.n, self.gcap, fill, seed, false);
        for (self.pos, 0..) |*p, r| p.* = @intCast(4 * shape.n + (r % 4));
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.host));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.out_g = try self.rig.output(shape.rows * hy4.groups * 4);
        self.out_t = try self.rig.output(shape.rows * hy4.tokens_per_row * 4);
        return true;
    }

    pub fn run(self: *Select1Cta) !void {
        var a: cuda.Args = .{};
        inline for (.{ @as(u64, 0), self.in[1], self.in[0], @as(c_int, @intCast(self.gcap)), self.out_g, self.out_t }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 1024 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn reference(_: *Select1Cta, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Select1Cta, _: g.Gate) !?[]const u8 {
        const bytes = try self.rig.outputBytes();
        return verifyOutputs(self.rig.gpu.gpa, self.host, self.pos, self.shape.rows, self.shape.n, self.gcap, bytes, &self.msg);
    }
};

const selm_served = [_]c.Shape{ .{ .rows = 4, .n = 8192, .k = 511 }, .{ .rows = 1, .n = 4096, .k = 511 }, .{ .rows = 2, .n = 4097, .k = 511 }, .{ .rows = 2, .n = 16384, .k = 511 }, .{ .rows = 2, .n = 24576, .k = 511 } };

pub const SelectMulti = struct {
    rig: rig_.Rig,
    scan: cuda.Function,
    merge: cuda.Function,
    base_1cta: cuda.Function,
    fault: set.Fault = .none,
    fault_in_scan: bool = false,
    shape: c.Shape = undefined,
    gcap: usize = 0,
    host: []f32 = &.{},
    pos: []i32 = &.{},
    in: [2]u64 = @splat(0),
    bits: u64 = 0,
    out_g: u64 = 0,
    out_t: u64 = 0,
    base_g: u64 = 0,
    base_t: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *SelectMulti) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *SelectMulti) void {
        const a = self.rig.gpu.gpa;
        a.free(self.host);
        a.free(self.pos);
        self.host = &.{};
        self.pos = &.{};
    }

    pub fn kernel(self: *SelectMulti) g.Kernel {
        var s: c.Spec = .{
            .name = "cuda hy4_index_paged tf_hy4_idx_select_scan + tf_hy4_idx_select_merge (long context top-511 + expand)",
            .accum = .none,
            .depth = zeroDepth,
            .out = .raw,
            .nan = .{ .stated = rule },
            .tiles = .{ .{}, .{ .size = 2048, .step = 1, .min = 512 }, .{} },
            .served = &selm_served,
        };
        if (self.fault != .none) s.name = set.controlName(if (self.fault_in_scan) "select_scan" else "select_merge", self.fault);
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(SelectMulti);

    pub fn count(self: *const SelectMulti) usize {
        return self.shape.rows * (hy4.groups + hy4.tokens_per_row) * 4;
    }

    pub fn prepare(self: *SelectMulti, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 511 or shape.n < 512 or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.gcap = std.mem.alignForward(usize, shape.n, 4);
        const a = self.rig.gpu.gpa;
        self.host = try a.alloc(f32, shape.rows * self.gcap);
        self.pos = try a.alloc(i32, shape.rows);
        fillScores(self.host, shape.rows, shape.n, self.gcap, fill, seed, true);
        for (self.pos, 0..) |*p, r| p.* = @intCast(4 * shape.n + (r % 4));
        const words = (self.gcap + 31) / 32;
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.host));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.bits = try self.rig.scratch(@max(16, shape.rows * 2 * words * 4));
        self.out_g = try self.rig.output(shape.rows * hy4.groups * 4);
        self.out_t = try self.rig.output(shape.rows * hy4.tokens_per_row * 4);
        self.base_g = try self.rig.scratch(shape.rows * hy4.groups * 4);
        self.base_t = try self.rig.scratch(shape.rows * hy4.tokens_per_row * 4);
        return true;
    }

    pub fn run(self: *SelectMulti) !void {
        const words = (self.gcap + 31) / 32;
        const scan_blocks = (self.shape.n + hy4.select_scan_span - 1) / hy4.select_scan_span;
        if (scan_blocks > 0) {
            var sa: cuda.Args = .{};
            inline for (.{ @as(u64, 0), self.in[1], self.in[0], @as(c_int, @intCast(self.gcap)), self.bits, @as(c_int, @intCast(words)) }) |v| sa.add(v);
            try cuda.launch.launch(self.scan, .{ .grid = .{ .x = @intCast(scan_blocks), .y = @intCast(self.shape.rows) }, .block = .{ .x = 256 } }, self.rig.s, &sa);
        }
        var ma: cuda.Args = .{};
        inline for (.{ @as(u64, 0), self.in[1], self.in[0], @as(c_int, @intCast(self.gcap)), self.bits, @as(c_int, @intCast(words)), self.out_g, self.out_t, @as(u64, 0) }) |v| ma.add(v);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 1024 } }, self.rig.s, &ma);
        try self.rig.s.synchronize();
    }

    pub fn reference(_: *SelectMulti, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *SelectMulti, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (self.shape.n <= 8192 and self.fault == .none) {
            var ba: cuda.Args = .{};
            inline for (.{ @as(u64, 0), self.in[1], self.in[0], @as(c_int, @intCast(self.gcap)), self.base_g, self.base_t }) |v| ba.add(v);
            try cuda.launch.launch(self.base_1cta, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 1024 } }, self.rig.s, &ba);
            const gbytes = self.shape.rows * hy4.groups * 4;
            const tbytes = self.shape.rows * hy4.tokens_per_row * 4;
            if (std.mem.indexOfDiff(u8, got[0..gbytes], try self.rig.bytes(self.base_g, gbytes))) |at|
                return try std.fmt.bufPrint(&self.msg, "groups byte {d} differs from select_1cta", .{at});
            if (std.mem.indexOfDiff(u8, got[gbytes..][0..tbytes], try self.rig.bytes(self.base_t, tbytes))) |at|
                return try std.fmt.bufPrint(&self.msg, "tokens byte {d} differs from select_1cta", .{at});
        }
        return verifyOutputs(a, self.host, self.pos, self.shape.rows, self.shape.n, self.gcap, got, &self.msg);
    }
};
