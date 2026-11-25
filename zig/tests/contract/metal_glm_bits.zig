//! GLM Metal launches that keep a predecessor's bits: each case's served launches against it on the same inputs.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;

/// A buffer of one call: read only, written whole, read then written (the predecessor gets its own copy), or working.
pub const Kind = enum { input, output, inout, scratch };
/// How gate 1 counts a written buffer's exceptional values: fp32 or bf16 elements, or none (indices, counts).
pub const Elem = enum { f32, bf16, int };
pub const Slot = struct { kind: Kind, len: usize, elem: Elem = .int };
pub const max_slots = 24;

/// A case's own layout for a shape: its slots, or null where the shape is not one it serves.
pub const Layout = struct {
    slots: [max_slots]Slot = undefined,
    n: usize = 0,
    pub fn add(l: *Layout, kind: Kind, len: usize, elem: Elem) usize {
        l.slots[l.n] = .{ .kind = kind, .len = len, .elem = elem };
        l.n += 1;
        return l.n - 1;
    }
};

/// Exceptional elements (NaN, either infinity) of a written buffer.
fn exceptional(elem: Elem, bytes: []const u8) usize {
    var n: usize = 0;
    switch (elem) {
        .int => {},
        .f32 => for (0..bytes.len / 4) |i| {
            n += @intFromBool(!std.math.isFinite(c.read(.f32, bytes, i)));
        },
        .bf16 => for (0..bytes.len / 2) |i| {
            n += @intFromBool(!std.math.isFinite(c.read(.bf16, bytes, i)));
        },
    }
    return n;
}

/// The probe of case type C: C has `spec`, `layout`, `fill`, `encode` (new or predecessor) and `deinit`.
pub fn Probe(comptime C: type) type {
    return struct {
        const P = @This();
        a: std.mem.Allocator,
        rig: rig_.Rig,
        case: C,
        shape: c.Shape = undefined,
        layout: Layout = .{},
        data: [max_slots][]u8 = @splat(&.{}),
        refs: [max_slots]rig_.Ref = undefined,
        base: [max_slots]rig_.Ref = undefined,
        got: std.ArrayList(u8) = .empty,
        want: std.ArrayList(u8) = .empty,
        msg: [200]u8 = undefined,

        fn free(p: *P) void {
            for (&p.data) |*d| {
                p.a.free(d.*);
                d.* = &.{};
            }
            p.rig.reset();
        }
        pub fn deinit(p: *P) void {
            p.free();
            p.rig.deinit();
            p.got.deinit(p.a);
            p.want.deinit(p.a);
            p.case.deinit();
        }
        const vt = blk: {
            var v = g.hooks(P);
            v.release = release;
            break :blk v;
        };
        fn release(ctx: *anyopaque) void {
            const p: *P = @ptrCast(@alignCast(ctx));
            p.free();
        }
        pub fn kernel(p: *P) g.Kernel {
            const s = p.case.spec();
            return .{ .spec = s, .ctx = p, .vt = &vt, .control = if (p.case.control) .shapes else null };
        }
        pub fn prepare(p: *P, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
            p.free();
            p.layout = p.case.layout(shape) orelse return false;
            p.shape = shape;
            const slots = p.layout.slots[0..p.layout.n];
            for (slots, 0..) |s, i| if (s.kind == .input or s.kind == .inout) {
                p.data[i] = try p.a.alloc(u8, s.len);
            };
            try p.case.fill(shape, fill, seed, p.data[0..p.layout.n]);
            for (slots, 0..) |s, i| p.refs[i] = switch (s.kind) {
                .input => try p.rig.input(p.data[i]),
                .output, .inout => try p.rig.output(s.len),
                .scratch => try p.rig.scratch(s.len),
            };
            for (slots, 0..) |s, i| p.base[i] = if (s.kind == .output or s.kind == .inout) try p.rig.scratch(s.len) else p.refs[i];
            return true;
        }
        /// Outputs poisoned (an element no launch writes stays 0xff), in-place buffers as prepared, scratch zeroed.
        fn reset(p: *P, refs: []const rig_.Ref) void {
            for (p.layout.slots[0..p.layout.n], refs, 0..) |s, r, i| {
                const bytes = p.rig.bytes(r, s.len);
                switch (s.kind) {
                    .input => {},
                    .output => @memset(bytes, 0xff),
                    .inout => @memcpy(bytes, p.data[i]),
                    .scratch => @memset(bytes, 0),
                }
            }
        }
        fn launch(p: *P, refs: []const rig_.Ref, new: bool) !void {
            p.reset(refs);
            const pass = p.rig.begin();
            p.case.encode(pass.e, refs, p.shape, new);
            try p.rig.finish(pass);
        }
        pub fn run(p: *P) !void {
            try p.launch(p.refs[0..p.layout.n], true);
        }
        /// The predecessor's bytes on the same inputs, its written buffers in the order the rig gathers the probe's.
        pub fn replaced(p: *P) ![]const u8 {
            try p.launch(p.base[0..p.layout.n], false);
            p.want.clearRetainingCapacity();
            for (p.layout.slots[0..p.layout.n], 0..) |s, i| if (s.kind == .output or s.kind == .inout) {
                try p.want.appendSlice(p.a, p.rig.bytes(p.base[i], s.len));
            };
            return p.want.items;
        }
        /// Gate 1: the predecessor's bytes, NaN payloads included, where the planted values reach a written buffer.
        pub fn check(p: *P, _: g.Gate) !?[]const u8 {
            p.got.clearRetainingCapacity();
            try p.got.appendSlice(p.a, try p.rig.outputBytes());
            const want = try p.replaced();
            if (!std.mem.eql(u8, p.got.items, want)) {
                const at = std.mem.indexOfDiff(u8, p.got.items, want).?;
                return try std.fmt.bufPrint(&p.msg, "its bytes differ from the predecessor's from byte {d} of {d}", .{ at, want.len });
            }
            var floats = false;
            var planted: usize = 0;
            var off: usize = 0;
            for (p.layout.slots[0..p.layout.n]) |s| if (s.kind == .output or s.kind == .inout) {
                floats = floats or s.elem != .int;
                planted += exceptional(s.elem, want[off..][0..s.len]);
                off += s.len;
            };
            if (floats and planted == 0) return "the planted values reach no written buffer: no proof";
            return null;
        }
        pub fn count(p: *P) usize {
            var n: usize = 0;
            for (p.layout.slots[0..p.layout.n]) |s| n += if (s.kind == .output or s.kind == .inout) s.len else 0;
            return n;
        }
        pub fn reference(_: *P, want: []f64, mag: []f64) !void {
            @memset(want, 0);
            @memset(mag, 0);
        }
    };
}

/// A registered probe, freed through its own type.
const Item = struct { ctx: *anyopaque, destroy: *const fn (*anyopaque, std.mem.Allocator) void };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    items: std.ArrayList(Item) = .empty,
    loader: ?Loader = null,

    pub fn wanted(_: *const Suite, only: ?[]const u8, name: []const u8) bool {
        const o = only orelse return true;
        return std.mem.indexOf(u8, name, o) != null;
    }
    pub fn add(s: *Suite, comptime C: type, case: C) !void {
        var x = case;
        errdefer x.deinit();
        try s.items.ensureUnusedCapacity(s.a, 1);
        try s.kernels.ensureUnusedCapacity(s.a, 1);
        const p = try s.a.create(Probe(C));
        p.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .case = x };
        const destroy = struct {
            fn f(ctx: *anyopaque, a: std.mem.Allocator) void {
                const q: *Probe(C) = @ptrCast(@alignCast(ctx));
                q.deinit();
                a.destroy(q);
            }
        }.f;
        s.items.appendAssumeCapacity(.{ .ctx = p, .destroy = destroy });
        s.kernels.appendAssumeCapacity(p.kernel());
    }
    /// A test type of its own (created by the caller), freed with the suite.
    pub fn own(s: *Suite, comptime T: type, p: *T) !void {
        errdefer {
            p.deinit();
            s.a.destroy(p);
        }
        try s.items.ensureUnusedCapacity(s.a, 1);
        try s.kernels.ensureUnusedCapacity(s.a, 1);
        const destroy = struct {
            fn f(ctx: *anyopaque, a: std.mem.Allocator) void {
                const q: *T = @ptrCast(@alignCast(ctx));
                q.deinit();
                a.destroy(q);
            }
        }.f;
        s.items.appendAssumeCapacity(.{ .ctx = p, .destroy = destroy });
        s.kernels.appendAssumeCapacity(p.kernel());
    }
    /// The GLM loader's libraries, assembled once: every case compiles text the loader compiles.
    pub fn production(s: *Suite) !*Loader {
        if (s.loader == null) {
            s.loader = .{ .a = s.a };
            try tf.glm.kernels.visitSources(s.a, s.device, &s.loader.?);
        }
        return &s.loader.?;
    }
    pub fn deinit(s: *Suite) void {
        for (s.items.items) |x| x.destroy(x.ctx, s.a);
        s.items.deinit(s.a);
        if (s.loader) |*l| l.deinit();
    }
};

/// The texts kernels.zig's loader compiles, as its source visitor reports them.
pub const Loader = struct {
    a: std.mem.Allocator,
    texts: std.ArrayList([]u8) = .empty,
    pub fn raw(l: *Loader, _: []const u8, _: []const u8, text: []const u8) !void {
        try l.texts.append(l.a, try l.a.dupe(u8, text));
    }
    /// `text` when the loader compiles exactly it; a contract text that drifted from the loader's is refused.
    pub fn require(l: *const Loader, text: []const u8) ![]const u8 {
        for (l.texts.items) |t| if (std.mem.eql(u8, t, text)) return text;
        std.debug.print("GLM contract text ({d} bytes) is not one the loader compiles\n", .{text.len});
        return error.NotTheLoadersLibrary;
    }
    fn deinit(l: *Loader) void {
        for (l.texts.items) |t| l.a.free(t);
        l.texts.deinit(l.a);
    }
};

/// A gate-5 launch's two texts: this GPU's and the pre-M5 one (equal unless the source includes nax.h).
pub const Texts = struct { here: []const u8, pre_m5: []const u8 };

/// One function of `text`, compiled as the GLM loader compiles every library (MLX's options).
pub fn pipeline(device: mtl.Device, text: []const u8, name: []const u8) !mtl.Pipeline {
    return rig_.pipeline(device, text, name);
}

/// `text` with its first `from` after `kernel`'s name replaced: a control's planted fault (owned either way).
pub fn edited(a: std.mem.Allocator, text: []const u8, kernel: []const u8, from: []const u8, to: []const u8) ![]u8 {
    const k = std.mem.indexOf(u8, text, kernel) orelse return error.ControlEditMissing;
    const at = k + (std.mem.indexOf(u8, text[k..], from) orelse return error.ControlEditMissing);
    return std.mem.concat(a, u8, &.{ text[0..at], to, text[at + from.len ..] });
}

/// fp32 to bf16 bits (round to nearest even, NaN kept).
pub fn bf(v: f32) u16 {
    return c.toBf16(v);
}

/// Every exceptional bf16 value a planted row can hold: NaN of both signs, both infinities, both zeros.
pub const special_bf16 = [_]u16{ 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x0000, 0x8000 };

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A replacement's spec: every launch's written bytes against the predecessor's, which the gates run beside it.
pub fn spec(name: []const u8, replaces: []const u8, nan: []const u8, served: []const c.Shape, why: []const u8) c.Spec {
    return .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .replaces = replaces, .nan = .{ .stated = nan }, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
}
