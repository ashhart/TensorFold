//! Actual native LightOnOCR HIP launches under independent FP64 contracts; profiles state their geometry coverage.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const lighton = @import("lighton_ocr");
const model_ref = @import("contract_lighton_ref.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const bf = core.bf16_round;
pub const Profile = model_ref.Profile;
pub const entries = model_ref.entries;
const Ops = lighton.ops.Kernels;

pub const Bundle = struct {
    a: std.mem.Allocator,
    ops: Ops,
    cases: [entries.len]Case = undefined,
    made: usize = 0,

    pub fn deinit(b: *Bundle) void {
        for (b.cases[0..b.made]) |*item| item.deinit();
        b.ops.unload();
        b.a.destroy(b);
    }
};

/// The `lighton_ocr` contract entry (contract_families.zig): every case at `profile`'s shapes through gates 1-4.
pub fn run(a: std.mem.Allocator, _: std.Io, profile: @import("contract_entry.zig").Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    const bundle = try createProfile(a, &r, s, &l, &kernels, if (profile == .smoke) .smoke else .full);
    defer bundle.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6b63, &t);
    return t;
}

pub fn create(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, launcher: *const hip.Launcher, list: *std.ArrayList(g.Kernel)) !*Bundle {
    return createProfile(a, r, s, launcher, list, .full);
}

pub fn createProfile(a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, launcher: *const hip.Launcher, list: *std.ArrayList(g.Kernel), profile: Profile) !*Bundle {
    const bundle = try a.create(Bundle);
    errdefer a.destroy(bundle);
    const image = hip.kernels.images[@backingInt(hip.kernels.Group.lighton_ocr)];
    bundle.* = .{ .a = a, .ops = try Ops.load(r, image, launcher) };
    errdefer {
        for (bundle.cases[0..bundle.made]) |*item| item.deinit();
        bundle.ops.unload();
    }
    const before = list.items.len;
    errdefer list.items.len = before;
    for (entries) |entry| {
        const f = if (entry.control) try bundle.ops.mod.function(model_ref.symbol(entry)) else null;
        bundle.cases[bundle.made] = .{ .rig = .{ .a = a, .r = r, .s = s }, .ops = &bundle.ops, .entry = entry, .profile = profile, .control = f };
        try list.append(a, bundle.cases[bundle.made].kernel());
        bundle.made += 1;
    }
    return bundle;
}

pub const Case = struct {
    rig: Rig,
    ops: *const Ops,
    entry: model_ref.Entry,
    profile: Profile,
    control: ?hip.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    data: model_ref.Data = undefined,
    db: [5]u64 = @splat(0),
    df: [3]u64 = @splat(0),
    pos: u64 = 0,
    out: u64 = 0,
    whole: u64 = 0,
    poison: []u8 = &.{},
    msg: [180]u8 = undefined,

    pub fn deinit(self: *Case) void {
        if (self.arena) |*a| a.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Case);
    const numerical = blk: {
        var v = vt;
        v.check = null;
        break :blk v;
    };

    pub fn kernel(self: *Case) g.Kernel {
        const raw = self.entry.kind == .merge or self.entry.kind == .rows_to_heads or self.entry.kind == .heads_to_rows;
        return .{ .spec = model_ref.spec(self.entry, self.profile), .ctx = self, .vt = if (raw) &vt else &numerical, .control = if (self.entry.control) .shapes else null };
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (self.arena) |*arena| arena.deinit();
        self.arena = .init(self.rig.a);
        self.data = model_ref.Data.init(self.arena.?.allocator(), self.entry, shape, fill, seed) catch |err| return if (err == error.Shape) false else err;
        self.db = @splat(0);
        self.df = @splat(0);
        self.pos = 0;
        self.whole = 0;
        self.poison = &.{};
        for (self.data.b, &self.db) |b, *p| if (b.len > 0) {
            p.* = try self.rig.input(std.mem.sliceAsBytes(b));
        };
        for (self.data.f, &self.df) |f, *p| if (f.len > 0) {
            p.* = try self.rig.input(std.mem.sliceAsBytes(f));
        };
        if (self.data.pos.len > 0) self.pos = try self.rig.input(std.mem.sliceAsBytes(self.data.pos));
        self.out = try self.rig.output(self.data.want.len * c.size(model_ref.spec(self.entry, self.profile).out));
        if (self.entry.kind == .attention) {
            const bytes = self.data.total * shape.rows * 64 * 2;
            self.whole = try self.rig.scratch(bytes);
            self.poison = try self.arena.?.allocator().alloc(u8, bytes);
            @memset(self.poison, 0xa5);
        }
        return true;
    }

    pub fn run(self: *Case) !void {
        const sh = self.data.shape;
        var ops = self.ops.*;
        if (self.control) |f| switch (self.entry.kind) {
            .tables => ops.tables_fn = f,
            .rotary => ops.rotary_fn = f,
            .merge => ops.merge_fn = f,
            .gelu => ops.gelu_fn = f,
            .rms => ops.rms_fn = f,
            .text_qk => ops.text_qk_fn = f,
            .attention => ops.attention_fn = f,
            else => unreachable,
        };
        switch (self.entry.kind) {
            .tables => try ops.tables(self.rig.s, self.df[0], self.out, self.out + self.data.want.len, sh.rows, sh.n),
            .rotary => {
                const bytes = self.data.b[0].len * 2;
                try ops.rotary(self.rig.s, self.db[0], self.db[1], self.db[2], self.db[3], self.db[4], self.out, self.out + bytes, self.out + 2 * bytes, sh.rows * sh.n, 16);
            },
            .merge => try ops.merge(self.rig.s, self.db[0], self.out, sh.rows, sh.n, sh.k),
            .gelu => try ops.gelu(self.rig.s, self.db[0], self.out, sh.rows, sh.n),
            .rms => try ops.rms(self.rig.s, self.db[0], self.df[0], self.out, sh.rows, sh.n, if (self.entry.projector) 1e-6 else 1e-5),
            .text_qk => {
                const qbytes = self.data.b[0].len * 2;
                const kbytes = self.data.b[1].len * 2;
                try ops.textQk(self.rig.s, self.db[0], self.db[1], self.db[2], self.df[0], self.df[1], self.df[2], self.pos, self.out, self.out + qbytes, self.out + qbytes + kbytes, sh.rows, .{});
            },
            .rows_to_heads => try ops.rowsToHeads(self.rig.s, self.db[0], self.out, sh.rows, sh.n, sh.k),
            .heads_to_rows => try ops.headsToRows(self.rig.s, self.df[0], self.out, sh.rows, sh.n, sh.k),
            .attention => {
                try self.rig.put(self.whole, self.poison);
                try ops.attention(self.rig.s, self.db[0], self.db[1], self.db[2], self.whole, self.data.total, sh.rows, self.data.start, sh.k, 0.125);
                try self.rig.s.synchronize();
                const offset = self.data.start * sh.rows * 64 * 2;
                const length = self.data.want.len * 2;
                const selected = try self.rig.bytes(self.whole + offset, length);
                try self.rig.put(self.out, selected);
                const all = try self.rig.bytes(self.whole, self.poison.len);
                if (!std.mem.eql(u8, all[0..offset], self.poison[0..offset]) or !std.mem.eql(u8, all[offset + length ..], self.poison[offset + length ..])) return error.OutputSpanModified;
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.data.want.len;
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        @memcpy(want, self.data.want);
        @memcpy(mag, self.data.mag);
    }
    pub fn intervals(self: *Case, low: []f64, high: []f64) !void {
        @memcpy(low, self.data.low);
        @memcpy(high, self.data.high);
    }
    pub fn bounds(self: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        if (self.entry.kind != .gelu and self.entry.kind != .rms and self.entry.kind != .text_qk) return;
        for (want, self.data.low, self.data.high, allow) |v, lo, hi, *a| a.* = @max(@abs(lo - v), @abs(hi - v));
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try self.rig.a.alloc(f64, 3 * out.len);
        defer self.rig.a.free(buf);
        try self.data.calculate(acc, order, out, buf[0..out.len], buf[out.len .. 2 * out.len], buf[2 * out.len ..]);
    }

    pub fn check(self: *Case, gate: g.Gate) !?[]const u8 {
        const data = self.data;
        const sh = data.shape;
        const out = try self.rig.outputBytes();
        var nonfinite: usize = 0;
        for (0..self.count()) |i| {
            const raw: u32 = switch (self.entry.kind) {
                .merge => blk: {
                    const token = i / (sh.k * 4);
                    const lane = i % 4;
                    const ch = (i / 4) % sh.k;
                    const row = (token / (sh.n / 2)) * 2 + lane / 2;
                    const col = (token % (sh.n / 2)) * 2 + lane % 2;
                    break :blk data.b[0][(row * sh.n + col) * sh.k + ch];
                },
                .rows_to_heads => blk: {
                    const row = (i / sh.k) % sh.rows;
                    const head = i / (sh.rows * sh.k);
                    const col = i % sh.k;
                    break :blk @as(u32, data.b[0][(row * sh.n + head) * sh.k + col]) << 16;
                },
                .heads_to_rows => blk: {
                    const row = i / (sh.n * sh.k);
                    const head = (i / sh.k) % sh.n;
                    const col = i % sh.k;
                    break :blk bf.bits(data.f[0][(head * sh.rows + row) * sh.k + col]);
                },
                else => unreachable,
            };
            const value = c.read(model_ref.spec(self.entry, self.profile).out, out, i);
            if (!std.math.isFinite(value)) nonfinite += 1;
            const got: u32 = if (self.entry.kind == .rows_to_heads) std.mem.readInt(u32, out[4 * i ..][0..4], .little) else std.mem.readInt(u16, out[2 * i ..][0..2], .little);
            if (got != raw and !(self.entry.kind == .heads_to_rows and std.math.isNan(value) and std.math.isNan(data.want[i]))) return try std.fmt.bufPrint(&self.msg, "layout word {d} differs: {x} versus {x}", .{ i, got, raw });
        }
        if (gate == .nonfinite and (nonfinite == 0 or nonfinite == self.count())) return "layout nonfinite fixture lacks mixed classes";
        return null;
    }
};
