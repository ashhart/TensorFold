//! DSpine's eight Metal kernels (qwen27/dspine.metal) at the stock 27B drafter's shapes, and three planted controls.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const ref = @import("metal_qwen27_dspine_ref.zig");
const c = contract.spec;
const g = contract.gates;
const Kind = ref.Kind;

/// Rows of one, two and eight drafting streams' 16-row blocks (forward.zig runs every block in one pass).
fn bridge(n: usize, k: usize) [3]c.Shape {
    return .{ .{ .rows = 16, .n = n, .k = k }, .{ .rows = 32, .n = n, .k = k }, .{ .rows = 128, .n = n, .k = k } };
}
const served = struct {
    const add = bridge(ref.hidden, 1);
    const read = bridge(ref.hidden, ref.hidden);
    const code = bridge(ref.code, ref.code);
    const message = bridge(ref.rank, 1);
    const senders = [_]c.Shape{.{ .rows = ref.heads, .n = ref.code, .k = ref.topk }};
    const base = [_]c.Shape{.{ .rows = ref.heads, .n = ref.topk, .k = ref.hidden }};
    const table = [_]c.Shape{.{ .rows = ref.heads, .n = ref.topk * ref.topk, .k = ref.rank }};
};

fn reduction(s: c.Shape) usize {
    return s.k;
}
/// The table's rank sums plus 24 terms' worth for the fast-math sigmoid each term's message carries.
fn tableDepth(s: c.Shape) usize {
    return s.k + 24;
}

const fp32_sums = c.Sums{ .bf16 = 0, .exact = false, .why = "the stress rows overflow fp16 sums; bf16 ones are not separated" };
/// The write's two bf16 nodes (RES, then RES * inv) widen its bound 4 half-ulps, so overflowed fp16 sums miss ~50x.
const write_sums = c.Sums{ .f16 = 40, .bf16 = 0, .exact = false, .why = "fp16 sums overflow; the norm's bf16 nodes bound the miss" };

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .add => .{ .name = "metal qwen27 dspine add", .accum = .none, .depth = reduction, .out = .bf16, .served = &served.add, .sums = .{ .f16 = 0, .bf16 = 0, .why = "elementwise: no sums" } },
        .read_xn => .{ .name = "metal qwen27 dspine read", .accum = .f32, .depth = reduction, .ops = 6, .out = .bf16, .served = &served.read, .sums = fp32_sums },
        .read_rms => .{ .name = "metal qwen27 dspine read rms", .accum = .f32, .depth = reduction, .ops = 6, .out = .f32, .served = &served.read, .sums = fp32_sums },
        .code => .{ .name = "metal qwen27 dspine code", .accum = .f32, .depth = reduction, .ops = 6, .out = .bf16, .served = &served.code, .sums = fp32_sums },
        .message => .{ .name = "metal qwen27 dspine message", .accum = .none, .depth = reduction, .ops = 24, .out = .bf16, .served = &served.message, .sums = .{ .f16 = 0, .bf16 = 0, .why = "elementwise: no sums" } },
        .write => .{ .name = "metal qwen27 dspine write", .accum = .f32, .depth = reduction, .ops = 6, .mids = 4, .out = .bf16, .served = &served.read, .sums = write_sums },
        .senders => .{ .name = "metal qwen27 dspine senders", .accum = .none, .depth = reduction, .out = .raw, .served = &served.senders, .nan = .{ .stated = "NaN and infinities in a codebook row pass through its scaled copy" }, .sums = .{ .f16 = 0, .bf16 = 0, .why = "copies: no sums" } },
        .base => .{ .name = "metal qwen27 dspine base", .accum = .f32, .depth = reduction, .ops = 2, .out = .f32, .served = &served.base, .sums = fp32_sums },
        .table => .{ .name = "metal qwen27 dspine table", .accum = .f32, .depth = tableDepth, .ops = 4, .out = .f32, .served = &served.table, .sums = fp32_sums },
    };
}

fn function(kind: Kind) []const u8 {
    return switch (kind) {
        .read_xn, .read_rms => "tf_dspine_read",
        inline else => |k| "tf_dspine_" ++ @tagName(k),
    };
}

/// One fault planted in dspine.metal's text, and the gate it must turn red.
pub const Control = enum {
    none,
    message_own_row,
    read_half_sums,
    table_first_sender,

    pub fn kind(m: Control) Kind {
        return switch (m) {
            .none => unreachable,
            .message_own_row => .message,
            .read_half_sums => .read_rms,
            .table_first_sender => .table,
        };
    }
    fn name(m: Control) []const u8 {
        return switch (m) {
            .none => "",
            .message_own_row => "control: metal qwen27 dspine message, a row's gate reads its own S, not the row before's",
            .read_half_sums => "control: metal qwen27 dspine read rms, the squares summed in half",
            .table_first_sender => "control: metal qwen27 dspine table, every sender's message from the first sender's G and V",
        };
    }
    fn gate(m: Control) ?g.Gate {
        return switch (m) {
            .none => null,
            .read_half_sums => .sums,
            else => .shapes,
        };
    }
    fn edit(m: Control) [2][]const u8 {
        return switch (m) {
            .none => unreachable,
            .message_own_row => .{ "float(PACKED[prev + r + i.x])", "float(PACKED[q + r + i.x])" },
            .read_half_sums => .{ "const float x = float(RAW[base + i]); ss = fma(x, x, ss); }\n  const float rms", "const half x = half(RAW[base + i]); hs = fma(x, x, hs); }\n  ss = hs;\n  const float rms" },
            .table_first_sender => .{ "sv = size_t(i * p.k + s) * 2 * r", "sv = size_t(i * p.k) * 2 * r" },
        };
    }
};

/// dspine.metal as kernels.zig compiles it, or with a control's one edit (which must apply exactly once).
fn source(a: std.mem.Allocator, m: Control) ![]const u8 {
    if (m == .none) return sources.qwen27_dspine;
    const e = m.edit();
    if (std.mem.count(u8, sources.qwen27_dspine, e[0]) != 1) return error.ControlEditMissing;
    const text = try std.mem.replaceOwned(u8, a, sources.qwen27_dspine, e[0], e[1]);
    if (m != .read_half_sums) return text;
    defer a.free(text);
    const decl = "float ss = 0.0f;\n  for (uint i = t; i < p.width; i += threads) { const half x";
    if (std.mem.count(u8, text, decl) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, decl, "float ss = 0.0f;\n  half hs = 0.0h;\n  for (uint i = t; i < p.width; i += threads) { const half x");
}

const Rows = extern struct { rows: u32, width: u32, block: u32, eps: f32 = 0, scale: f32 = 0 };
const Senders = extern struct { rows: u32, k: u32, code: u32, row_words: u32, anchor: u32, scale: f32 };
const Base = extern struct { rows: u32, k: u32, hidden: u32, row_words: u32, raw_first: u32 };
const Table = extern struct { rows: u32, k: u32, rank: u32, row_words: u32, rms_first: u32 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    control: Control,
    arena: ?std.heap.ArenaAllocator = null,
    f: ref.Fixture = undefined,
    refs: [9]rig_.Ref = undefined,
    bound: usize = 0,
    copies: Copies = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: Control) !Probe {
        const text = try source(a, control);
        defer if (control != .none) a.free(text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, text, function(kind)), .kind = kind, .control = control };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        if (p.arena) |*x| x.deinit();
        p.pipe.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(p: *Probe) g.Kernel {
        var s = spec(p.kind);
        if (p.control != .none) s.name = p.control.name();
        if (p.kind == .senders) {
            p.copies = .{ .rig = &p.rig, .p = p };
            return .{ .ctx = &p.copies, .vt = &Copies.vt, .spec = s, .control = p.control.gate() };
        }
        return .{ .ctx = p, .vt = &vt, .spec = s, .control = p.control.gate() };
    }

    fn in(p: *Probe, data: anytype) !void {
        p.refs[p.bound] = try p.rig.input(std.mem.sliceAsBytes(data));
        p.bound += 1;
    }
    fn value(p: *Probe, v: anytype) !void {
        try p.in(std.mem.asBytes(&v));
    }
    fn out(p: *Probe, bytes: usize, checked: bool) !void {
        p.refs[p.bound] = if (checked) try p.rig.output(bytes) else try p.rig.scratch(bytes);
        p.bound += 1;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        const want = spec(p.kind).served[0];
        const head = p.kind == .senders or p.kind == .base or p.kind == .table;
        if (s.n != want.n or s.k != want.k or s.rows == 0 or s.rows > @as(usize, if (head) ref.heads else 128)) return false;
        p.rig.reset();
        if (p.arena) |*x| x.deinit();
        p.arena = .init(p.a);
        const a = p.arena.?.allocator();
        p.f = try ref.make(a, p.kind, s.rows, fill, seed);
        const f = &p.f;
        const r: u32 = @intCast(s.rows);
        const k: u32 = ref.topk;
        p.bound = 0;
        switch (p.kind) {
            .add => {
                try p.in(f.x);
                try p.in(f.y);
                try p.value(Rows{ .rows = r, .width = ref.hidden, .block = 1 });
                try p.out(s.rows * ref.hidden * 2, true);
            },
            .read_xn, .read_rms => {
                try p.in(f.x);
                try p.value(Rows{ .rows = r, .width = ref.hidden, .block = 1, .eps = ref.eps });
                try p.out(s.rows * ref.hidden * 2, p.kind == .read_xn);
                try p.out(s.rows * 4, p.kind == .read_rms);
            },
            .code => {
                try p.in(f.x);
                try p.in(f.cb);
                try p.in(f.ids);
                try p.value(Rows{ .rows = r, .width = ref.code, .block = ref.block, .eps = ref.eps, .scale = 32 });
                try p.out(s.rows * ref.code * 2, true);
            },
            .message => {
                try p.in(f.packed_rows);
                try p.in(f.gb);
                try p.value(Rows{ .rows = r, .width = ref.rank, .block = ref.block });
                try p.out(s.rows * ref.rank * 2, true);
            },
            .write => {
                try p.in(f.x);
                try p.in(f.y);
                try p.in(f.rms);
                try p.in(f.w);
                try p.value(Rows{ .rows = r, .width = ref.hidden, .block = ref.block, .eps = ref.eps });
                try p.out(s.rows * ref.hidden * 2, true);
                try p.out(s.rows * ref.hidden * 2, true);
            },
            .senders => {
                try p.in(f.top);
                try p.in(f.cb);
                try p.value(Senders{ .rows = r, .k = k, .code = ref.code, .row_words = ref.row_words, .anchor = f.anchor, .scale = 32 });
                try p.out(s.rows * k * ref.code * 2, true);
                try p.out(s.rows * k * 4, true);
            },
            .base => {
                for ([_][]const u8{ std.mem.sliceAsBytes(f.x), std.mem.sliceAsBytes(f.w), std.mem.sliceAsBytes(f.hw), std.mem.sliceAsBytes(f.hs), std.mem.sliceAsBytes(f.hb), std.mem.sliceAsBytes(f.top) }) |b| try p.in(b);
                try p.value(Base{ .rows = r, .k = k, .hidden = ref.hidden, .row_words = ref.row_words, .raw_first = 1 });
                try p.out(s.rows * k * 4, true);
            },
            .table => {
                for ([_][]const u8{ std.mem.sliceAsBytes(f.packed_rows), std.mem.sliceAsBytes(f.tsv), std.mem.sliceAsBytes(f.gb), std.mem.sliceAsBytes(f.tg), std.mem.sliceAsBytes(f.top), std.mem.sliceAsBytes(f.base), std.mem.sliceAsBytes(f.rms) }) |b| try p.in(b);
                try p.value(Table{ .rows = r, .k = k, .rank = ref.rank, .row_words = ref.row_words, .rms_first = 1 });
                try p.out(s.rows * k * k * 4, true);
            },
        }
        return true;
    }

    pub fn count(p: *Probe) usize {
        return if (p.kind == .senders) p.f.rows * ref.topk * (ref.code * 2 + 4) else p.f.count();
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.refs[0..p.bound], 0..) |x, i| e.setBuffer(x.buf, x.off, i);
        const rows = p.f.rows;
        switch (p.kind) {
            .add => e.dispatchThreads(mtl.Size.of(ref.hidden, rows, 1), mtl.Size.of(256, 1, 1)),
            .message => e.dispatchThreads(mtl.Size.of(ref.rank, rows, 1), mtl.Size.of(256, 1, 1)),
            .read_xn, .read_rms, .code, .write => e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1)),
            .senders, .base => e.dispatchGroups(mtl.Size.of(ref.topk, rows, 1), mtl.Size.of(128, 1, 1)),
            .table => e.dispatchGroups(mtl.Size.of(ref.topk, rows, 1), mtl.Size.of(256, 1, 1)),
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.f.compute(p.a, null, .sequential, want, mag);
    }

    /// The base's allowance grows by what a fused dequant (scale * q + bias in one fma) moves its bf16 head values.
    pub fn bounds(p: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.kind == .base) for (allow, p.f.extra) |*l, x| {
            l.* += x;
        };
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: contract.rows.Order, got: []f64) !void {
        const mag = try p.a.alloc(f64, got.len);
        defer p.a.free(mag);
        try p.f.compute(p.a, acc, order, got, mag);
    }
};

/// The senders' copies: a Probe whose own check replaces the float64 bound (bf16 copies bit-equal, ids equal).
pub const Copies = struct {
    rig: *rig_.Rig,
    p: *Probe,

    pub fn prepare(x: *Copies, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        return x.p.prepare(s, fill, seed);
    }
    pub fn run(x: *Copies) !void {
        return x.p.run();
    }
    pub fn count(x: *Copies) usize {
        return x.p.count();
    }
    pub fn reference(_: *Copies, want: []f64, mag: []f64) !void {
        @memset(want, 0); // the copies check themselves (check)
        @memset(mag, 0);
    }
    const vt = g.hooks(Copies);

    /// Every scaled codebook value bit for bit (a NaN by class), and every sender id.
    pub fn check(x: *Copies, _: g.Gate) !?[]const u8 {
        const p = x.p;
        const f = &p.f;
        const n = f.rows * ref.topk;
        const want = try p.a.alloc(f64, n * ref.code);
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, n * ref.code);
        defer p.a.free(mag);
        try f.compute(p.a, null, .sequential, want, mag);
        const codes = p.rig.bytes(p.refs[3], n * ref.code * 2);
        const sids = p.rig.bytes(p.refs[4], n * 4);
        for (want, 0..) |w, i| {
            const got = c.read(.bf16, codes, i);
            const bits = c.toBf16(@floatCast(w));
            if (std.math.isNan(w)) {
                if (!std.math.isNan(got)) return "a NaN codebook value is not NaN in its scaled copy";
            } else if (std.mem.readInt(u16, codes[2 * i ..][0..2], .little) != bits) return "a scaled codebook value differs from bf16(scale * value)";
        }
        for (0..n) |e| {
            const row = e / ref.topk;
            const sid = if (row == 0) f.anchor else f.top[(row - 1) * ref.row_words + 2 + 2 * (e % ref.topk)];
            if (std.mem.readInt(u32, sids[4 * e ..][0..4], .little) != sid) return "a sender id is not the anchor or the row before's top-k";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (std.enums.values(Control)) |m| for (std.enums.values(Kind)) |kind| {
            if (m != .none and kind != m.kind()) continue;
            const name = if (m == .none) spec(kind).name else m.name();
            if (only) |needle| if (std.mem.indexOf(u8, name, needle) == null) continue;
            const p = try a.create(Probe);
            errdefer a.destroy(p);
            p.* = try Probe.init(a, device, queue, kind, m);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        };
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};

test {
    _ = ref;
}
