//! Unlimited-OCR-2's CUDA moves under the contract: im2col, windows, concat, embeddings, the MoE's gather and scatter.
const std = @import("std");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const uocr = @import("contract_uocr.zig");
const kit = @import("contract_uocr_moves_kit.zig");
const tower = @import("contract_uocr_moves_tower.zig");

const c = kit.c;
const g = kit.g;
const Kernels = kit.Kernels;
const Control = kit.Control;
const bits = kit.bits;
const val = kit.val;
const finiteBits = kit.finiteBits;
const planted = kit.planted;
const chunk = kit.chunk;
const slots = kit.slots;
const hidden = uocr.hidden;
/// The tokenizer's image token and <PAGE> marker (config.zig's).
const image_token = 128815;
const page_token = 128827;

// --- tf_uocr_gather and tf_uocr_scatter: an expert's rows by index, y[i] = x[rows[i]], slots[dest[i]] = y[i] ------

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: Kernels,
    scatter: bool,
    ctl: ?Control,
    shape: c.Shape = undefined,
    src: []u16 = &.{},
    ids: []u32 = &.{},
    /// tf_uocr_scatter's slot rows before the launch: the rows no index names keep them.
    bg: []u16 = &.{},
    ins: [2]u64 = @splat(0),
    bg_in: u64 = 0,
    out: u64 = 0,

    /// Rows: expert rows (1..2048); n: hidden (1280); k: indexed table (tokens or slots); indices distinct, shuffled.
    const gather_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = chunk }, .{ .rows = chunk, .n = hidden, .k = chunk }, .{ .rows = 1, .n = hidden, .k = 1 } };
    const scatter_served = [_]c.Shape{ .{ .rows = 192, .n = hidden, .k = slots }, .{ .rows = chunk, .n = hidden, .k = slots }, .{ .rows = 1, .n = hidden, .k = uocr.topk } };

    pub fn spec(scatter: bool) c.Spec {
        return .{
            .name = if (scatter) "cuda uocr moves tf_uocr_scatter 1280 (slots[dest[i]] = y[i])" else "cuda uocr moves tf_uocr_gather 1280 (y[i] = x[rows[i]])",
            .accum = .none,
            .depth = kit.noSums,
            .out = .bf16,
            .tiles = .{ .{ .size = 64 }, .{}, .{} },
            .served = if (scatter) &scatter_served else &gather_served,
        };
    }
    pub fn init(ctx: uocr.Ctx, scatter: bool, ctl: ?Control) !Rows {
        const k = if (scatter) try kit.kernelsFor(ctx, "scatter", ctl) else try kit.kernelsFor(ctx, "gather", ctl);
        return .{ .a = ctx.gpu.gpa, .rig = ctx.rig(), .k = k, .scatter = scatter, .ctl = ctl };
    }
    pub fn deinit(t: *Rows) void {
        t.free();
        t.rig.deinit();
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
        t.ids = try kit.distinct(t.a, rng, shape.rows, shape.k);
        t.bg = try t.a.alloc(u16, if (t.scatter) shape.k * n else 0);
        for (t.bg) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite and shape.rows > 3) {
            // the rows the first four indices move: NaN, +Inf, -Inf, both infinities
            for (0..4) |x| {
                const row = if (t.scatter) x else t.ids[x];
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
        if (t.scatter) t.bg_in = try t.rig.input(std.mem.sliceAsBytes(t.bg));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Rows) !void {
        if (t.scatter) try kit.preset(&t.rig, t.out, t.bg_in, t.bg.len * 2);
        const grid = [3]usize{ t.shape.rows, (hidden + 255) / 256, 1 };
        if (t.scatter) try t.k.run("scatter", t.rig.s, grid, 256, .{ t.ins[0], t.ins[1], t.out }) else try t.k.run("gather", t.rig.s, grid, 256, .{ t.ins[0], t.ins[1], t.out });
        try t.rig.s.synchronize();
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

// --- tf_uocr_embed: x[i] = embed[ids[i]] for text tokens; an image token's row is left for its picture ----------

pub const Embed = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    k: Kernels,
    ctl: ?Control,
    shape: c.Shape = undefined,
    table: []u16 = &.{},
    ids: []u32 = &.{},
    /// The rows before the launch: an image token's keeps its own.
    bg: []u16 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: prompt tokens (a 2048-row chunk; a page: BOS, <PAGE>, 903 image tokens, 4 more); n: hidden; k: vocab.
    pub fn spec() c.Spec {
        return .{
            .name = "cuda uocr moves tf_uocr_embed 1280 (token rows, image tokens skipped)",
            .accum = .none,
            .depth = kit.noSums,
            .out = .bf16,
            .served = &.{ .{ .rows = chunk, .n = hidden, .k = uocr.vocab }, .{ .rows = 909, .n = hidden, .k = uocr.vocab } },
        };
    }
    pub fn init(ctx: uocr.Ctx, ctl: ?Control) !Embed {
        return .{ .a = ctx.gpu.gpa, .rig = ctx.rig(), .k = try kit.kernelsFor(ctx, "embed", ctl), .ctl = ctl };
    }
    pub fn deinit(t: *Embed) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Embed) void {
        inline for (.{ "table", "ids", "bg" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Embed) g.Kernel {
        var s = spec();
        if (t.ctl) |x| s.name = x.name;
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const vt = g.hooks(Embed);

    /// A text id: any token of the vocabulary but the image token.
    fn text(rng: std.Random, k: usize) u32 {
        while (true) {
            const id = rng.uintLessThan(u32, @intCast(k));
            if (id != image_token) return id;
        }
    }

    pub fn prepare(t: *Embed, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != hidden or shape.k <= page_token) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        const n = hidden;
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.table = try t.a.alloc(u16, shape.k * n);
        for (t.table) |*v| v.* = finiteBits(rng);
        t.ids = try t.a.alloc(u32, rows);
        for (t.ids) |*id| id.* = text(rng, shape.k);
        // a page's prompt: BOS, <PAGE>, its image tokens, the instruction's 4 tokens
        if (rows >= 8) {
            t.ids[0] = 0;
            t.ids[1] = page_token;
            for (t.ids[2 .. rows - 4]) |*id| id.* = image_token;
        }
        t.bg = try t.a.alloc(u16, rows * n);
        for (t.bg) |*v| v.* = finiteBits(rng);
        if (fill == .nonfinite) {
            // the last four text rows' table rows: NaN, +Inf, -Inf, both infinities
            for (0..@min(4, rows)) |x| {
                const row = t.ids[rows - 1 - x];
                if (x < 3) t.table[row * n + 11 * x] = bits(planted[x]) else {
                    t.table[row * n + 40] = bits(planted[1]);
                    t.table[row * n + n - 1] = bits(planted[2]);
                }
            }
            // the image token's own table row and a background image row: the first stays out, the second stays
            t.table[image_token * n + 7] = bits(planted[0]);
            if (rows >= 8) t.bg[2 * n + 5] = bits(planted[0]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.table));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.bg));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Embed) !void {
        try kit.preset(&t.rig, t.out, t.ins[2], t.bg.len * 2);
        try t.k.run("embed", t.rig.s, .{ t.shape.rows, (hidden + 255) / 256, 1 }, 256, .{ t.ins[0], t.ins[1], t.out, @as(u32, image_token) });
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Embed) usize {
        return t.shape.rows * hidden;
    }

    pub fn reference(t: *Embed, want: []f64, mag: []f64) !void {
        const n = hidden;
        @memset(mag, 0);
        for (want, 0..) |*o, i| {
            const id = t.ids[i / n];
            o.* = if (id == image_token) val(t.bg[i]) else val(t.table[id * n + i % n]);
        }
    }
};

// --- the group -------------------------------------------------------------------------------------------------------

/// The planted controls: each breaks one index or padding rule (or the sums' type) and must go red at its gate.
const im2col_controls = [_]Control{
    .{ .name = "cuda uocr moves control tf_uocr_im2col padding 1 not 0", .symbol = "tf_ctl_uocr_im2col_pad_one" },
    .{ .name = "cuda uocr moves control tf_uocr_im2col ky and kx swapped", .symbol = "tf_ctl_uocr_im2col_kykx" },
};
const window_control = Control{ .name = "cuda uocr moves control tf_uocr_window rows and columns swapped in a window", .symbol = "tf_ctl_uocr_window_swap" };
const window_pad_control = Control{ .name = "cuda uocr moves control tf_uocr_window padding 1 not 0", .symbol = "tf_ctl_uocr_window_pad_one" };
const unwindow_control = Control{ .name = "cuda uocr moves control tf_uocr_unwindow_add rows and columns swapped in a window", .symbol = "tf_ctl_uocr_unwindow_add_swap" };
const rel_controls = [_]Control{
    .{ .name = "cuda uocr moves control tf_uocr_rel_pos rel_h and rel_w positions swapped", .symbol = "tf_ctl_uocr_rel_pos_swap" },
    .{ .name = "cuda uocr moves control tf_uocr_rel_pos summed in half", .symbol = "tf_ctl_uocr_rel_pos_half", .gate = .sums },
};
const concat_control = Control{ .name = "cuda uocr moves control tf_uocr_concat left row one column wider", .symbol = "tf_ctl_uocr_concat_wide" };
const clip_control = Control{ .name = "cuda uocr moves control tf_uocr_clip_embed class row takes position 1", .symbol = "tf_ctl_uocr_clip_embed_pos1" };
const gather_control = Control{ .name = "cuda uocr moves control tf_uocr_gather odd rows read the even row's index", .symbol = "tf_ctl_uocr_gather_even" };
const scatter_control = Control{ .name = "cuda uocr moves control tf_uocr_scatter odd rows written to the even row's slot", .symbol = "tf_ctl_uocr_scatter_even" };
const embed_control = Control{ .name = "cuda uocr moves control tf_uocr_embed image tokens' rows written from the table", .symbol = "tf_ctl_uocr_embed_images" };

/// The group's registrations, on the heap: the gates hold pointers to them.
pub const Group = struct {
    gpa: std.mem.Allocator,
    im2col: std.ArrayList(*kit.Im2col) = .empty,
    windows: std.ArrayList(*kit.Window) = .empty,
    rel: std.ArrayList(*tower.RelPos) = .empty,
    concat: std.ArrayList(*tower.Concat) = .empty,
    clip: std.ArrayList(*tower.ClipEmbed) = .empty,
    rows: std.ArrayList(*Rows) = .empty,
    embed: std.ArrayList(*Embed) = .empty,

    /// One test of T (T.init with these args), appended to the gates' list.
    fn add(grp: *Group, comptime T: type, into: *std.ArrayList(*T), list: *std.ArrayList(g.Kernel), ctx: uocr.Ctx, args: anytype) !void {
        const x = try grp.gpa.create(T);
        errdefer grp.gpa.destroy(x);
        x.* = try @call(.auto, T.init, .{ctx} ++ args);
        try into.append(grp.gpa, x);
        try list.append(grp.gpa, x.kernel());
    }

    pub fn init(ctx: uocr.Ctx, list: *std.ArrayList(g.Kernel)) !*Group {
        const grp = try ctx.gpu.gpa.create(Group);
        grp.* = .{ .gpa = ctx.gpu.gpa };
        errdefer grp.deinit();
        const none: ?Control = null;
        for (kit.geos) |geo| try grp.add(kit.Im2col, &grp.im2col, list, ctx, .{ geo, none });
        // the controls run at the neck's geometry (padding on every side) and the patches' (no padding: the swap)
        for (im2col_controls, 0..) |ctl, i| try grp.add(kit.Im2col, &grp.im2col, list, ctx, .{ kit.geos[if (i == 0) 1 else 0], @as(?Control, ctl) });
        for ([_]bool{ false, true }) |back| try grp.add(kit.Window, &grp.windows, list, ctx, .{ back, none });
        try grp.add(kit.Window, &grp.windows, list, ctx, .{ false, @as(?Control, window_control) });
        try grp.add(kit.Window, &grp.windows, list, ctx, .{ false, @as(?Control, window_pad_control) });
        try grp.add(kit.Window, &grp.windows, list, ctx, .{ true, @as(?Control, unwindow_control) });
        try grp.add(tower.RelPos, &grp.rel, list, ctx, .{none});
        for (rel_controls) |ctl| try grp.add(tower.RelPos, &grp.rel, list, ctx, .{@as(?Control, ctl)});
        try grp.add(tower.Concat, &grp.concat, list, ctx, .{none});
        try grp.add(tower.Concat, &grp.concat, list, ctx, .{@as(?Control, concat_control)});
        try grp.add(tower.ClipEmbed, &grp.clip, list, ctx, .{none});
        try grp.add(tower.ClipEmbed, &grp.clip, list, ctx, .{@as(?Control, clip_control)});
        for ([_]bool{ false, true }) |scatter| try grp.add(Rows, &grp.rows, list, ctx, .{ scatter, none });
        try grp.add(Rows, &grp.rows, list, ctx, .{ false, @as(?Control, gather_control) });
        try grp.add(Rows, &grp.rows, list, ctx, .{ true, @as(?Control, scatter_control) });
        try grp.add(Embed, &grp.embed, list, ctx, .{none});
        try grp.add(Embed, &grp.embed, list, ctx, .{@as(?Control, embed_control)});
        return grp;
    }

    pub fn deinit(grp: *Group) void {
        inline for (.{ "im2col", "windows", "rel", "concat", "clip", "rows", "embed" }) |f| {
            for (@field(grp, f).items) |x| {
                x.deinit();
                grp.gpa.destroy(x);
            }
            @field(grp, f).deinit(grp.gpa);
        }
        grp.gpa.destroy(grp);
    }
};
