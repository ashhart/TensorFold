//! Unlimited-OCR-2's moves (tower patches, SAM windows, rel_pos, CLIP embeds, MoE rows): exact, controls planted.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const uocr = @import("metal_uocr.zig");
const kit = @import("metal_uocr_moves_kit.zig");

const c = contract.spec;
const g = contract.gates;
const bf = uocr.bf;
const planted = uocr.planted;
const clip_width = uocr.clip_width;
const hidden = uocr.hidden;
const chunk = kit.chunk;
const slots = kit.slots;
const Edit = kit.Edit;
const Control = kit.Control;
const build = kit.build;
const finiteBits = kit.finiteBits;
const value = kit.value;
const bits = kit.bits;
const val = kit.val;
const distinct = kit.distinct;
const geos = kit.geos;
pub const Im2col = kit.Im2col;
pub const Window = kit.Window;
pub const RelPos = kit.RelPos;
// --- uv_concat: y [rows, left + right] = [a row | b row] ------------------------------------------------------------

pub const Concat = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?Control,
    shape: c.Shape = undefined,
    l: []u16 = &.{},
    r: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: the view's tokens (16^2 or 10^2); n: CLIP's width (left); k: SAM's net3 width (right).
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr moves uv_concat (CLIP | SAM rows)",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = .{ .{}, .{ .size = 256 }, .{ .size = 256 } },
            .served = &.{ .{ .rows = 256, .n = clip_width, .k = 1024 }, .{ .rows = 100, .n = clip_width, .k = 1024 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?Control) !Concat {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, "uv_concat", if (ctl) |x| x.edit else null), .ctl = ctl };
    }
    pub fn deinit(t: *Concat) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Concat) void {
        inline for (.{ "l", "r" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Concat) g.Kernel {
        var s = spec();
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Concat);

    pub fn prepare(t: *Concat, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.l = try t.a.alloc(u16, shape.rows * shape.n);
        t.r = try t.a.alloc(u16, shape.rows * shape.k);
        for (t.l) |*v| v.* = finiteBits(rng);
        for (t.r) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            t.l[0 * shape.n + 1] = bits(planted[0]);
            t.r[1 * shape.k + 2 % shape.k] = bits(planted[1]);
            t.l[2 * shape.n + 3 % shape.n] = bits(planted[2]);
            t.l[3 * shape.n + shape.n - 1] = bits(planted[1]);
            t.r[3 * shape.k] = bits(planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.l));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.r));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Concat) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue([3]u32{ @intCast(t.shape.rows), @intCast(t.shape.n), @intCast(t.shape.k) }, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n + t.shape.k, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Concat) usize {
        return t.shape.rows * (t.shape.n + t.shape.k);
    }

    pub fn reference(t: *Concat, want: []f64, mag: []f64) !void {
        const w = t.shape.n + t.shape.k;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / w;
            const col = i % w;
            o.* = if (col < t.shape.n) val(t.l[row * t.shape.n + col]) else val(t.r[row * t.shape.k + col - t.shape.n]);
            m.* = 0;
        }
    }
};

// --- uv_clip_embed: y[0] = bf16(cls + pos[0]), y[r] = bf16(sam[r - 1] + pos[r]) -----------------------------------

pub const ClipEmbed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    ctl: ?Control,
    shape: c.Shape = undefined,
    cls: []u16 = &.{},
    sam: []u16 = &.{},
    pos: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: CLIP's tokens (the class row and 16^2 or 10^2 of SAM's); n: CLIP's width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal uocr moves uv_clip_embed (class row, positions, one bf16 add)",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 8 }, .{ .size = 256 }, .{} },
            .served = &.{ .{ .rows = 257, .n = clip_width, .k = 1 }, .{ .rows = 101, .n = clip_width, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: ?Control) !ClipEmbed {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .vision, "uv_clip_embed", if (ctl) |x| x.edit else null), .ctl = ctl };
    }
    pub fn deinit(t: *ClipEmbed) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *ClipEmbed) void {
        inline for (.{ "cls", "sam", "pos" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *ClipEmbed) g.Kernel {
        var s = spec();
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(ClipEmbed);

    pub fn prepare(t: *ClipEmbed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.cls = try t.a.alloc(u16, n);
        t.sam = try t.a.alloc(u16, (shape.rows - 1) * n);
        t.pos = try t.a.alloc(u16, shape.rows * n);
        for (t.cls) |*v| v.* = value(rng, 4);
        for (t.sam) |*v| v.* = value(rng, 4);
        for (t.pos) |*v| v.* = value(rng, 1);
        if (fill == .nonfinite and shape.rows > 3 and n > 9) {
            t.cls[2] = bits(planted[0]); // row 0
            t.sam[0 * n + 5] = bits(planted[1]); // row 1
            t.pos[2 * n + 7] = bits(planted[2]); // row 2
            t.sam[2 * n + 9] = bits(planted[1]); // row 3, both infinities
            t.pos[3 * n + 9] = bits(planted[2]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.cls));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.sam));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.pos));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *ClipEmbed) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.setValue(@as(u32, @intCast(t.shape.n)), 4);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *ClipEmbed) usize {
        return t.shape.rows * t.shape.n;
    }

    pub fn reference(t: *ClipEmbed, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, mag, 0..) |*o, *m, i| {
            const row = i / n;
            const v = if (row == 0) val(t.cls[i]) else val(t.sam[i - n]);
            o.* = bf(v + val(t.pos[i]));
            m.* = 0;
        }
    }
};

// --- ud_gather and ud_scatter: an expert's rows by its index list, y[i] = x[rows[i]] and slots[dest[i]] = y[i] -------

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    scatter: bool,
    ctl: ?Control,
    shape: c.Shape = undefined,
    src: []u16 = &.{},
    ids: []u32 = &.{},
    /// ud_scatter's slot rows before the launch: the rows no index names keep them.
    bg: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: expert rows (1..2048); n: hidden (1280); k: indexed table (tokens or slots); indices distinct, shuffled.
    const gather_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = chunk }, .{ .rows = chunk, .n = hidden, .k = chunk }, .{ .rows = 1, .n = hidden, .k = 1 } };
    const scatter_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = slots }, .{ .rows = chunk, .n = hidden, .k = slots }, .{ .rows = 1, .n = hidden, .k = uocr.topk } };

    pub fn spec(scatter: bool) c.Spec {
        return .{
            .name = if (scatter) "metal uocr moves ud_scatter 1280 (slots[dest[i]] = y[i])" else "metal uocr moves ud_gather 1280 (y[i] = x[rows[i]])",
            .accum = .none,
            .depth = uocr.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 64 }, .{}, .{} },
            .served = if (scatter) &scatter_served else &gather_served,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, scatter: bool, ctl: ?Control) !Rows {
        const name = if (scatter) "ud_scatter" else "ud_gather";
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try build(a, device, .decoder, name, if (ctl) |x| x.edit else null), .scatter = scatter, .ctl = ctl };
    }
    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Rows) void {
        inline for (.{ "src", "ids", "bg" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Rows) g.Kernel {
        var s = spec(t.scatter);
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Rows);

    pub fn prepare(t: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.rows > shape.k) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = hidden;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // gather reads table rows by index; scatter reads its own rows in order
        const src_rows = if (t.scatter) shape.rows else shape.k;
        t.src = try t.a.alloc(u16, src_rows * n);
        for (t.src) |*v| v.* = finiteBits(rng);
        t.ids = try distinct(t.a, rng, shape.rows, shape.k);
        t.bg = try t.a.alloc(u16, if (t.scatter) shape.k * n else 0);
        for (t.bg) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            // the rows the first four indices move: NaN, +Inf, -Inf, both infinities
            const moved = [_]usize{ 0, 1, 2, 3 };
            for (moved, 0..) |j, x| {
                const row = if (t.scatter) j else t.ids[j];
                if (x < 3) t.src[row * n + 11 * x] = bits(planted[x]) else {
                    t.src[row * n + 40] = bits(planted[1]);
                    t.src[row * n + n - 1] = bits(planted[2]);
                }
            }
            // gather: a NaN in a row no index names must stay out
            if (!t.scatter and shape.k > shape.rows) {
                const seen = try t.a.alloc(bool, shape.k);
                defer t.a.free(seen);
                @memset(seen, false);
                for (t.ids) |i| seen[i] = true;
                const free_row = std.mem.indexOfScalar(bool, seen, false).?;
                t.src[free_row * n] = bits(planted[0]);
            }
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.src));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Rows) !void {
        if (t.scatter) @memcpy(t.rig.bytes(t.out, t.bg.len * 2), std.mem.sliceAsBytes(t.bg));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.dispatchThreads(mtl.Size.of(hidden, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Rows) usize {
        return (if (t.scatter) t.shape.k else t.shape.rows) * hidden;
    }

    pub fn reference(t: *Rows, want: []f64, mag: []f64) !void {
        const n = hidden;
        @memset(mag, 0);
        if (t.scatter) {
            for (want, t.bg) |*o, v| o.* = val(v);
            for (t.ids, 0..) |d, i| for (0..n) |j| {
                want[d * n + j] = val(t.src[i * n + j]);
            };
            return;
        }
        for (want, 0..) |*o, i| o.* = val(t.src[t.ids[i / n] * n + i % n]);
    }
};

// --- the group -------------------------------------------------------------------------------------------------------

/// The planted controls: each edits one index or padding rule (or the sums' type) and must go red at its gate.
const im2col_controls = [_]Control{
    .{ .name = "metal uocr moves control uv_im2col padding 1 not 0", .edit = .{ .from = "bfloat v = bfloat(0.0f);", .to = "bfloat v = bfloat(1.0f);" } },
    .{ .name = "metal uocr moves control uv_im2col ky and kx swapped", .edit = .{ .from = "ky = (at.x / p.k) % p.k, kx = at.x % p.k;", .to = "kx = (at.x / p.k) % p.k, ky = at.x % p.k;" } },
};
const window_control = Control{ .name = "metal uocr moves control uv_window rows and columns swapped in a window", .edit = .{
    .from = "gy = (win / per) * p.window + in / p.window, gx = (win % per) * p.window + in % p.window;",
    .to = "gy = (win / per) * p.window + in % p.window, gx = (win % per) * p.window + in / p.window;",
} };
const window_pad_control = Control{ .name = "metal uocr moves control uv_window padding 1 not 0", .edit = .{
    .from = "p.channels + at.x] : bfloat(0.0f);",
    .to = "p.channels + at.x] : bfloat(1.0f);",
} };
const unwindow_control = Control{ .name = "metal uocr moves control uv_unwindow_add rows and columns swapped in a window", .edit = .{
    .from = "in = (gy % p.window) * p.window + gx % p.window;",
    .to = "in = (gx % p.window) * p.window + gy % p.window;",
} };
const rel_controls = [_]Control{
    .{ .name = "metal uocr moves control uv_rel_pos rel_h and rel_w positions swapped", .edit = .{
        .from = "((rows ? qh : qw) - kk + p.side - 1)",
        .to = "((rows ? qw : qh) - kk + p.side - 1)",
    } },
    .{ .name = "metal uocr moves control uv_rel_pos summed in half", .gate = .sums, .edit = .{
        .from = "float s = 0.0f;\n  for (uint d = 0; d < 64; d++) s += float(q[d]) * float(r[d]);",
        .to = "half s = 0.0h;\n  for (uint d = 0; d < 64; d++) s += half(float(q[d]) * float(r[d]));",
    } },
};
const concat_control = Control{ .name = "metal uocr moves control uv_concat left row one column wider", .edit = .{ .from = "at.x < p.left ?", .to = "at.x <= p.left ?" } };
const clip_control = Control{ .name = "metal uocr moves control uv_clip_embed class row takes position 1", .edit = .{
    .from = "float(pos[ulong(at.y) * width + at.x])",
    .to = "float(pos[ulong(max(at.y, 1u)) * width + at.x])",
} };
const gather_control = Control{ .name = "metal uocr moves control ud_gather odd rows read the even row's index", .edit = .{ .from = "x[ulong(rows[at.y]) * 1280", .to = "x[ulong(rows[at.y & ~1u]) * 1280" } };
const scatter_control = Control{ .name = "metal uocr moves control ud_scatter odd rows written to the even row's slot", .edit = .{ .from = "slots[ulong(dest[at.y]) * 1280", .to = "slots[ulong(dest[at.y & ~1u]) * 1280" } };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    im2col: std.ArrayList(*Im2col) = .empty,
    windows: std.ArrayList(*Window) = .empty,
    rel: std.ArrayList(*RelPos) = .empty,
    concat: std.ArrayList(*Concat) = .empty,
    clip: std.ArrayList(*ClipEmbed) = .empty,
    rows: std.ArrayList(*Rows) = .empty,

    fn wanted(only: ?[]const u8, name: []const u8) bool {
        const o = only orelse return true;
        return std.mem.indexOf(u8, name, o) != null;
    }

    /// One test of T (T.init with these args) when its name is wanted.
    fn add(s: *Suite, comptime T: type, list: *std.ArrayList(*T), only: ?[]const u8, name: []const u8, args: anytype) !void {
        if (!wanted(only, name)) return;
        const x = try s.a.create(T);
        errdefer s.a.destroy(x);
        x.* = try @call(.auto, T.init, .{ s.a, s.device, s.queue } ++ args);
        try list.append(s.a, x);
        try s.kernels.append(s.a, x.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (geos) |geo| try s.add(Im2col, &s.im2col, only, geo.name, .{ geo, @as(?Control, null) });
        // the controls run at the neck's geometry (padding on every side) and the patches' (no padding: the swap)
        for (im2col_controls, 0..) |ctl, i| try s.add(Im2col, &s.im2col, only, ctl.name, .{ geos[if (i == 0) 1 else 0], @as(?Control, ctl) });
        for ([_]bool{ false, true }) |back| try s.add(Window, &s.windows, only, Window.spec(back).name, .{ back, @as(?Control, null) });
        try s.add(Window, &s.windows, only, window_control.name, .{ false, @as(?Control, window_control) });
        try s.add(Window, &s.windows, only, window_pad_control.name, .{ false, @as(?Control, window_pad_control) });
        try s.add(Window, &s.windows, only, unwindow_control.name, .{ true, @as(?Control, unwindow_control) });
        try s.add(RelPos, &s.rel, only, RelPos.spec().name, .{@as(?Control, null)});
        for (rel_controls) |ctl| try s.add(RelPos, &s.rel, only, ctl.name, .{@as(?Control, ctl)});
        try s.add(Concat, &s.concat, only, Concat.spec().name, .{@as(?Control, null)});
        try s.add(Concat, &s.concat, only, concat_control.name, .{@as(?Control, concat_control)});
        try s.add(ClipEmbed, &s.clip, only, ClipEmbed.spec().name, .{@as(?Control, null)});
        try s.add(ClipEmbed, &s.clip, only, clip_control.name, .{@as(?Control, clip_control)});
        for ([_]bool{ false, true }) |scatter| try s.add(Rows, &s.rows, only, Rows.spec(scatter).name, .{ scatter, @as(?Control, null) });
        try s.add(Rows, &s.rows, only, gather_control.name, .{ false, @as(?Control, gather_control) });
        try s.add(Rows, &s.rows, only, scatter_control.name, .{ true, @as(?Control, scatter_control) });
    }

    pub fn deinit(s: *Suite) void {
        inline for (.{ "im2col", "windows", "rel", "concat", "clip", "rows" }) |f| {
            for (@field(s, f).items) |x| {
                x.deinit();
                s.a.destroy(x);
            }
            @field(s, f).deinit(s.a);
        }
    }
};

/// Gate 5's texts (`uocr moves <case> <function>`): the tower's library for uv_*, the decoder's for ud_*.
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, name: []const u8) !?uocr.Texts {
    if (!std.mem.startsWith(u8, name, "uocr moves ")) return null;
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    const which: uocr.Library = if (std.mem.startsWith(u8, fun, "ud_")) .decoder else if (std.mem.startsWith(u8, fun, "uv_")) .vision else return error.UnknownLaunch;
    const text = uocr.source(which);
    return .{ .here = text, .pre_m5 = text };
}
