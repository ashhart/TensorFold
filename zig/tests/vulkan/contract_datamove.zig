// The datamove group's Vulkan contract: stage 1's exact data-movement and format kernels — cast_f32_bf16, cast_bf16_f32, copy_u32, copy_u16, gather_u32, gather_u16 and dequant_bf16 — checked bit for bit against a host model (the argmax rule: the output equals the reference bytes).
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const fx = @import("stage1_fixture.zig");

const c = contract.spec;
const g = contract.gates;

/// Guard bytes on both sides of every rig payload (the largest minStorageBufferOffsetAlignment).
pub const guard = 256;
const pattern = 0xa5;

// One call's Vulkan buffers: inputs uploaded, outputs poisoned 0xff, every payload between guard bytes. The gates read a kernel test's outputs, inputs and guards through its `rig` field.
pub const Rig = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    timeout_ns: u64,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,

    const Kind = enum { input, output };
    const Slot = struct { buf: vk.Buffer, len: usize, kind: Kind };

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.a);
        r.seen.deinit(r.a);
        r.outs.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        for (r.slots.items) |*s| s.buf.deinit();
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, data: ?[]const u8, len: usize, kind: Kind) !usize {
        const all = try r.a.alloc(u8, len + 2 * guard);
        defer r.a.free(all);
        @memset(all, pattern);
        if (data) |d| {
            @memcpy(all[guard..][0..len], d);
        } else {
            @memset(all[guard..][0..len], 0xff); // an output element no launch writes reads as NaN
        }
        var b = try vk.Buffer.init(r.dev, .device, all.len);
        errdefer b.deinit();
        try vk.upload(r.cmds, b, 0, r.staging, all, r.timeout_ns);
        try r.slots.append(r.a, .{ .buf = b, .len = len, .kind = kind });
        return r.slots.items.len - 1;
    }

    pub fn input(r: *Rig, data: []const u8) !usize {
        return r.slot(data, data.len, .input);
    }

    /// An output of `len` bytes, all 0xff until the launch writes it.
    pub fn output(r: *Rig, len: usize) !usize {
        return r.slot(null, len, .output);
    }

    /// The payload of slot `i` as a typed view for a binding; call between prepare and the next prepare.
    pub fn view(r: *Rig, i: usize, kind: vk.tensor.Kind, elements: usize) !vk.tensor.View {
        const s = &r.slots.items[i];
        return vk.tensor.View.init(&s.buf, kind, guard, elements);
    }

    fn gatherBytes(r: *Rig, kind: Kind, out: *std.ArrayList(u8), whole: bool) ![]const u8 {
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind or whole) {
            const at = out.items.len;
            const n = if (whole) s.len + 2 * guard else s.len;
            try out.resize(r.a, at + n);
            try vk.download(r.cmds, s.buf, if (whole) 0 else guard, r.staging, out.items[at..][0..n], r.timeout_ns);
        };
        return out.items;
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.gatherBytes(.output, &r.outs, false);
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.gatherBytes(.input, &r.seen, false);
    }

    /// Every guard of every buffer still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        const all = try r.gatherBytes(.input, &r.seen, true);
        var at: usize = 0;
        for (r.slots.items) |s| {
            for (all[at..][0..guard]) |v| if (v != pattern) return false;
            for (all[at + guard + s.len ..][0..guard]) |v| if (v != pattern) return false;
            at += s.len + 2 * guard;
        }
        return true;
    }

    /// Every payload is its own buffer, so no output range overlaps an input range.
    pub fn overlap(r: *Rig) bool {
        _ = r;
        return false;
    }
};

fn none(_: c.Shape) usize {
    return 0;
}

fn read16(bytes: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, bytes[2 * i ..][0..2], .little);
}

fn write16(bytes: []u8, i: usize, bits: u16) void {
    std.mem.writeInt(u16, bytes[2 * i ..][0..2], bits, .little);
}

fn read32(bytes: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
}

fn write32(bytes: []u8, i: usize, bits: u32) void {
    std.mem.writeInt(u32, bytes[4 * i ..][0..4], bits, .little);
}

/// The contract's gather id scheme: [0] out of range, [1] == [2] (a repeated row), 0, then cycling rows.
fn gatherIds(ids: []u32, table_rows: usize) void {
    for (ids, 0..) |*id, i| {
        id.* = switch (i) {
            0 => @intCast(table_rows), // out of range: the shader's documented zero row
            1, 2 => @intCast(table_rows - 1),
            3 => 0,
            else => @intCast(i % table_rows),
        };
    }
}

// The dequant's exact model: bf16 widen, one fp32 multiply-add (the q4 product is exact, so one rounding either way), then the contract's RNE.
fn dequantBits(scale_bits: u16, q: u4, bias_bits: u16) u16 {
    const s: f32 = c.fromBf16(scale_bits);
    const b: f32 = c.fromBf16(bias_bits);
    const v = s * @as(f32, @floatFromInt(q)) + b;
    return c.toBf16(v);
}

// The dequant's random scale or bias: normal bf16 magnitudes from about 2^-113 up, so no fp32 subnormal result an FTZ device could flush where the host does not.
fn randomBf16(rng: std.Random) u16 {
    const sign: u16 = rng.int(u1);
    const exp: u16 = 14 + rng.uintLessThan(u8, 113);
    const mant: u16 = rng.int(u7);
    return sign << 15 | exp << 7 | mant;
}

fn buf(msg: *[160]u8, comptime fmt: []const u8, args: anytype) ?[]const u8 {
    return std.fmt.bufPrint(msg, fmt, args) catch "message too long";
}

// One cast direction: f32 -> bf16 is the contract's RNE (a NaN keeps its class), bf16 -> f32 is exact widening; both outputs must equal the reference bytes.
pub fn Cast(comptime to_bf16: bool) type {
    return struct {
        const Self = @This();
        const esz: usize = if (to_bf16) 2 else 4;
        const in_kind: vk.tensor.Kind = if (to_bf16) .f32 else .bf16;
        const out_kind: vk.tensor.Kind = if (to_bf16) .bf16 else .f32;
        const out_elem: c.Elem = if (to_bf16) .bf16 else .f32;
        const rule = if (to_bf16)
            "outputs equal the contract's RNE bytes: ties to even, bf16 subnormals included, a NaN keeps its class"
        else
            "outputs equal the reference bits: bf16 widening is exact, every value and NaN byte copies";

        pub const spec: c.Spec = .{
            .name = if (to_bf16) "vulkan cast_f32_bf16 (RNE)" else "vulkan cast_bf16_f32 (exact widening)",
            .accum = .none,
            .depth = none,
            .out = out_elem,
            .nan = .{ .stated = rule },
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 146, .k = 0 }, .{ .rows = 3, .n = 700, .k = 0 } },
        };

        rig: Rig,
        o: *vk.ops.Ops,
        arena: std.heap.ArenaAllocator,
        in_at: usize = 0,
        out_at: usize = 0,
        elements: usize = 0,
        want: []u8 = &.{},
        msg: [160]u8 = undefined,

        pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout_ns: u64) Self {
            return .{ .rig = .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .timeout_ns = timeout_ns }, .o = o, .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = spec, .ctx = self, .vt = &vt };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.n == 0) return false;
            const elements = std.math.mul(usize, shape.rows, shape.n) catch return false;
            // A 16-bit range with an odd element count is not bindable (dispatch.bind needs whole 32-bit words; View.init refuses the same tail): the kernel refuses that edge shape, it is not a served shape.
            if (elements % 2 != 0 and (in_kind.size() == 2 or esz == 2)) return false;
            self.rig.reset();
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            const in = try a.alloc(u8, elements * in_kind.size());
            const want = try a.alloc(u8, elements * esz);
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            for (0..elements) |i| {
                if (to_bf16) {
                    const x: f32 = if (i < fx.bf16_vectors.len) fx.bf16_vectors[i].in else (rng.float(f32) * 2 - 1) * std.math.pow(f32, 2, @floatFromInt(@as(i32, @intCast(i % 41)) - 20));
                    write32(in, i, @bitCast(x));
                    write16(want, i, c.toBf16(x));
                } else {
                    const bits: u16 = if (i < fx.bf16_vectors.len) fx.bf16_vectors[i].want else rng.int(u16);
                    write16(in, i, bits);
                    write32(want, i, @as(u32, bits) << 16);
                }
            }
            if (fill == .nonfinite) {
                const planted = [_]u32{ 0x7fc00000, 0x7f800000, 0xff800000 };
                for (planted, 0..) |bits, k| {
                    const i = (k * 37 + 11) % elements;
                    if (to_bf16) {
                        write32(in, i, bits);
                        write16(want, i, c.toBf16(@bitCast(bits)));
                    } else {
                        write16(in, i, @truncate(bits >> 16));
                        write32(want, i, bits & 0xffff0000);
                    }
                }
            }
            self.in_at = try self.rig.input(in);
            self.out_at = try self.rig.output(want.len);
            self.elements = elements;
            self.want = want;
            return true;
        }

        pub fn run(self: *Self) !void {
            const src = try self.rig.view(self.in_at, in_kind, self.elements);
            const out = try self.rig.view(self.out_at, out_kind, self.elements);
            if (to_bf16) {
                try self.o.castF32ToBf16(self.rig.cmds, out, src, self.rig.timeout_ns);
            } else {
                try self.o.castBf16ToF32(self.rig.cmds, out, src, self.rig.timeout_ns);
            }
        }

        pub fn count(self: *Self) usize {
            return self.elements;
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            _ = self;
            @memset(want, 0);
            @memset(mag, 0); // never called: the check hook owns every comparison
        }

        pub fn check(self: *Self, at: g.Gate) !?[]const u8 {
            _ = at;
            const got = try self.rig.outputBytes();
            for (0..self.elements) |i| {
                const want_bits: u32 = if (to_bf16) read16(self.want, i) else read32(self.want, i);
                const got_bits: u32 = if (to_bf16) read16(got, i) else read32(got, i);
                if (got_bits != want_bits) return buf(&self.msg, "element {d} read 0x{x:0>8}, want 0x{x:0>8}", .{ i, got_bits, want_bits });
            }
            return null;
        }
    };
}

/// A contiguous same-kind raw copy: exact bytes out, guards and source untouched.
pub fn Copy(comptime kind: vk.tensor.Kind) type {
    return struct {
        const Self = @This();
        const esz = kind.size();

        pub const spec: c.Spec = .{
            .name = "vulkan copy_" ++ (if (kind == .u32) "u32" else "u16") ++ " (bit copy)",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .nan = .{ .stated = "outputs equal the source bytes: a raw bit copy, every value and NaN pattern included" },
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 300, .k = 0 }, .{ .rows = 2, .n = 520, .k = 0 } },
        };

        rig: Rig,
        o: *vk.ops.Ops,
        arena: std.heap.ArenaAllocator,
        in_at: usize = 0,
        out_at: usize = 0,
        elements: usize = 0,
        want: []u8 = &.{},
        msg: [160]u8 = undefined,

        pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout_ns: u64) Self {
            return .{ .rig = .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .timeout_ns = timeout_ns }, .o = o, .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = spec, .ctx = self, .vt = &vt };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows == 0 or shape.n == 0) return false;
            const elements = std.math.mul(usize, shape.rows, shape.n) catch return false;
            // See the casts: an odd 16-bit element count cannot be bound as whole 32-bit words.
            if (elements % 2 != 0 and esz == 2) return false;
            self.rig.reset();
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            const data = try a.alloc(u8, elements * esz);
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            rng.bytes(data);
            if (fill == .nonfinite) {
                const planted32 = [_]u32{ 0x7fc00000, 0x7f800000, 0xff800000 };
                const planted16 = [_]u16{ 0x7fc0, 0x7f80, 0xff80 };
                for (0..3) |k| {
                    const i = (k * 53 + 7) % elements;
                    if (esz == 4) write32(data, i, planted32[k]) else write16(data, i, planted16[k]);
                }
            }
            self.in_at = try self.rig.input(data);
            self.out_at = try self.rig.output(data.len);
            self.elements = elements;
            self.want = data;
            return true;
        }

        pub fn run(self: *Self) !void {
            const src = try self.rig.view(self.in_at, kind, self.elements);
            const dst = try self.rig.view(self.out_at, kind, self.elements);
            try self.o.copy(self.rig.cmds, dst, src, self.rig.timeout_ns);
        }

        pub fn count(self: *Self) usize {
            return self.elements;
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            _ = self;
            @memset(want, 0);
            @memset(mag, 0); // never called: the check hook owns every comparison
        }

        pub fn check(self: *Self, at: g.Gate) !?[]const u8 {
            _ = at;
            const got = try self.rig.outputBytes();
            if (std.mem.indexOfDiff(u8, got, self.want)) |byte| {
                return buf(&self.msg, "element {d} differs from the source bytes", .{byte / esz});
            }
            return null;
        }
    };
}

// gather_take on u32 or u16 rows: valid ids copy their row bit for bit, an out-of-range id gathers zeros (the rule documented at ops_q4.zig), never another row.
pub fn Gather(comptime kind: vk.tensor.Kind) type {
    return struct {
        const Self = @This();
        const esz = kind.size();

        pub const spec: c.Spec = .{
            .name = "vulkan gather_" ++ (if (kind == .u32) "u32" else "u16") ++ " (bit rows)",
            .accum = .none,
            .depth = none,
            .out = .raw,
            .nan = .{ .stated = "outputs equal the addressed row's bytes; an out-of-range id's row is zeros, never another row (ops_q4.zig)" },
            .tiles = .{ .{}, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 4, .n = 10, .k = 0 }, .{ .rows = 8, .n = 300, .k = 0 } },
        };

        rig: Rig,
        o: *vk.ops.Ops,
        arena: std.heap.ArenaAllocator,
        tokens: usize = 0,
        cols: usize = 0,
        table_rows: usize = 0,
        table_at: usize = 0,
        ids_at: usize = 0,
        out_at: usize = 0,
        ids: []u32 = &.{},
        want: []u8 = &.{},
        msg: [160]u8 = undefined,

        pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout_ns: u64) Self {
            return .{ .rig = .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .timeout_ns = timeout_ns }, .o = o, .arena = .init(a) };
        }

        pub fn deinit(self: *Self) void {
            self.rig.deinit();
            self.arena.deinit();
        }

        pub fn kernel(self: *Self) g.Kernel {
            return .{ .spec = spec, .ctx = self, .vt = &vt };
        }

        const vt = g.hooks(Self);

        pub fn prepare(self: *Self, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            if (shape.rows < 4 or shape.n == 0) return false;
            const tokens = shape.rows;
            const cols = shape.n;
            const cells = std.math.mul(usize, tokens, cols) catch return false;
            const table_rows = tokens + 2;
            self.rig.reset();
            _ = self.arena.reset(.retain_capacity);
            const a = self.arena.allocator();
            const table = try a.alloc(u8, table_rows * cells * esz);
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            rng.bytes(table);
            if (fill == .nonfinite) {
                if (esz == 4) {
                    const planted = [_]u32{ 0x7fc00000, 0x7f800000, 0xff800000 };
                    for (planted, [_]usize{ 0, 1, cols }) |bits, cell| write32(table, cell, bits);
                } else {
                    const planted = [_]u16{ 0x7fc0, 0x7f80, 0xff80 };
                    for (planted, [_]usize{ 0, 1, cols }) |bits, cell| write16(table, cell, bits);
                }
            }
            const ids = try a.alloc(u32, tokens);
            gatherIds(ids, table_rows);
            const want = try a.alloc(u8, cells * esz);
            for (ids, 0..) |id, tok| {
                const row = want[tok * cols * esz ..][0 .. cols * esz];
                if (id < table_rows) {
                    @memcpy(row, table[id * cols * esz ..][0 .. cols * esz]);
                } else {
                    @memset(row, 0); // the documented out-of-range read
                }
            }
            self.table_at = try self.rig.input(table);
            self.ids_at = try self.rig.input(std.mem.sliceAsBytes(ids));
            self.out_at = try self.rig.output(want.len);
            self.tokens = tokens;
            self.cols = cols;
            self.table_rows = table_rows;
            self.ids = ids;
            self.want = want;
            return true;
        }

        pub fn run(self: *Self) !void {
            const out = try self.rig.view(self.out_at, kind, self.tokens * self.cols);
            const table = try self.rig.view(self.table_at, kind, self.table_rows * self.cols);
            const ids = try self.rig.view(self.ids_at, .u32, self.tokens);
            try self.o.gather(self.rig.cmds, out, table, ids, self.table_rows, self.rig.timeout_ns);
        }

        pub fn count(self: *Self) usize {
            return self.tokens * self.cols;
        }

        pub fn reference(self: *Self, want: []f64, mag: []f64) !void {
            _ = self;
            @memset(want, 0);
            @memset(mag, 0); // never called: the check hook owns every comparison
        }

        pub fn check(self: *Self, at: g.Gate) !?[]const u8 {
            _ = at;
            const got = try self.rig.outputBytes();
            if (std.mem.indexOfDiff(u8, got, self.want)) |byte| {
                const tok = byte / (self.cols * esz);
                const id = self.ids[tok];
                if (id >= self.table_rows) return buf(&self.msg, "out-of-range id {d} gathered nonzero bytes at {d}: it must gather zeros (ops_q4.zig)", .{ id, byte });
                return buf(&self.msg, "token {d} (id {d}) differs at byte {d}", .{ tok, id, byte });
            }
            return null;
        }
    };
}

// The Q4 g64 dequant: one fp32 expression scale*q + bias per code, then the contract's RNE; the output equals that model bit for bit, and a non-finite scale or bias gives its class.
pub const Dequant = struct {
    const group: u64 = 64;

    pub const spec: c.Spec = .{
        .name = "vulkan dequant_bf16 (q4 g64 RNE)",
        .accum = .none,
        .depth = none,
        .out = .bf16,
        .nan = .{ .stated = "one fp32 expression, then the contract's RNE: a non-finite scale or bias makes the output the fp32 expression's class" },
        .tiles = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} },
        .served = &.{ .{ .rows = 6, .n = 128, .k = 0 }, .{ .rows = 2, .n = 2048, .k = 0 } },
    };

    rig: Rig,
    o: *vk.ops.Ops,
    arena: std.heap.ArenaAllocator,
    rows: usize = 0,
    elements: usize = 0,
    words: usize = 0,
    groups: usize = 0,
    w_at: usize = 0,
    s_at: usize = 0,
    b_at: usize = 0,
    out_at: usize = 0,
    want: []u8 = &.{},
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, o: *vk.ops.Ops, timeout_ns: u64) Dequant {
        return .{ .rig = .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .timeout_ns = timeout_ns }, .o = o, .arena = .init(a) };
    }

    pub fn deinit(self: *Dequant) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Dequant) g.Kernel {
        return .{ .spec = spec, .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Dequant);

    pub fn prepare(self: *Dequant, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        // two rows minimum: the nonfinite plants need a second row's first group
        if (shape.rows < 2 or shape.n < group or shape.n % group != 0) return false;
        const rows = shape.rows;
        const per_row = shape.n;
        const elements = std.math.mul(usize, rows, per_row) catch return false;
        const words = per_row / 8;
        const groups = per_row / @as(usize, @intCast(group));
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const w = try a.alloc(u8, rows * words * 4);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        rng.bytes(w);
        const scales = try a.alloc(u16, rows * groups);
        const biases = try a.alloc(u16, rows * groups);
        for (scales, biases) |*s, *b| {
            s.* = randomBf16(rng);
            b.* = randomBf16(rng);
        }
        if (fill == .nonfinite) {
            scales[0] = 0x7fc0; // a NaN scale: its group's outputs are NaN
            scales[groups] = 0x7f80; // +inf scale with -inf bias on the next row: inf + -inf is NaN
            biases[0] = 0x0000;
            biases[groups] = 0xff80;
            biases[rows * groups - 1] = 0xff80; // -inf bias on a finite scale: -inf out
        }
        const want = try a.alloc(u8, elements * 2);
        for (0..rows * words) |unit| {
            const row = unit / words;
            const col = unit % words;
            const grp = col / (@as(usize, @intCast(group)) / 8);
            const sbits = scales[row * groups + grp];
            const bbits = biases[row * groups + grp];
            const word = read32(w, unit);
            for (0..8) |lane| {
                const q: u4 = @truncate(word >> @intCast(4 * lane));
                write16(want, unit * 8 + lane, dequantBits(sbits, q, bbits));
            }
        }
        self.w_at = try self.rig.input(w);
        self.s_at = try self.rig.input(std.mem.sliceAsBytes(scales));
        self.b_at = try self.rig.input(std.mem.sliceAsBytes(biases));
        self.out_at = try self.rig.output(want.len);
        self.rows = rows;
        self.elements = elements;
        self.words = words;
        self.groups = groups;
        self.want = want;
        return true;
    }

    pub fn run(self: *Dequant) !void {
        const out = try self.rig.view(self.out_at, .bf16, self.elements);
        const w = try self.rig.view(self.w_at, .u32, self.rows * self.words);
        const scales = try self.rig.view(self.s_at, .bf16, self.rows * self.groups);
        const biases = try self.rig.view(self.b_at, .bf16, self.rows * self.groups);
        try self.o.dequant(self.rig.cmds, out, w, scales, biases, self.words * 8, group, self.rig.timeout_ns);
    }

    pub fn count(self: *Dequant) usize {
        return self.elements;
    }

    pub fn reference(self: *Dequant, want: []f64, mag: []f64) !void {
        _ = self;
        @memset(want, 0);
        @memset(mag, 0); // never called: the check hook owns every comparison
    }

    pub fn check(self: *Dequant, at: g.Gate) !?[]const u8 {
        _ = at;
        const got = try self.rig.outputBytes();
        for (0..self.elements) |i| {
            const wbits = read16(self.want, i);
            const gbits = read16(got, i);
            const wv = c.fromBf16(wbits);
            if (c.classOf(wv) == .finite) {
                if (gbits != wbits) return buf(&self.msg, "element {d} read 0x{x:0>4}, want the model's 0x{x:0>4}", .{ i, gbits, wbits });
            } else if (c.classOf(c.fromBf16(gbits)) != c.classOf(wv)) {
                return buf(&self.msg, "element {d} is {t} where the fp32 expression is {t}", .{ i, c.classOf(c.fromBf16(gbits)), c.classOf(wv) });
            }
        }
        return null;
    }
};

const testing = std.testing;

test "the dequant model is one fp32 multiply-add then the contract's RNE" {
    try testing.expectEqual(@as(u16, 0x3f80), dequantBits(0x3f80, 1, 0x0000));
    try testing.expectEqual(@as(u16, 0x4170), dequantBits(0x3f80, 15, 0x0000)); // 15.0 is a grid point
    try testing.expectEqual(@as(u16, 0x3f80), dequantBits(0x3f80, 1, 0x3b80)); // 1 + 2^-8: tie to even, down
    try testing.expectEqual(@as(u16, 0x3f82), dequantBits(0x3f80, 1, 0x3dc0)); // 1 + 3*2^-8: tie with lsb 1, up
    try testing.expectEqual(@as(u16, 0x3f81), dequantBits(0x3f80, 1, 0x3b81)); // a hair above half rounds up
    try testing.expectEqual(@as(u16, 0x7f80), dequantBits(0x7f80, 1, 0x0000)); // +inf scale stays +inf
}

test "gather ids include an out-of-range id and a repeat, the rest are valid rows" {
    const ids = try testing.allocator.alloc(u32, 6);
    defer testing.allocator.free(ids);
    gatherIds(ids, 5);
    try testing.expectEqual(@as(u32, 5), ids[0]); // out of range: gathers zeros, never a table row
    try testing.expectEqual(ids[1], ids[2]); // a repeated row
    for (ids[1..]) |id| try testing.expect(id < 5);
}
