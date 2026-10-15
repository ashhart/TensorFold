//! GLM-5.3-Flash's weights in the Python family's layout (stacked projections and experts, repacked mixes), read by parallel preads.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const st = @import("../../core/safetensors.zig");
const cfg = @import("config.zig");
const quant = @import("quant.zig");
const affine = @import("affine.zig");
const layout = @import("names.zig");
const load_plan = @import("load_plan.zig");

pub const Width = quant.Width;

pub const Ref = affine.Ref;
pub const Affine = affine.Affine;

/// The members of the fused matrices, in their rows' order.
pub const kda_in = [_][]const u8{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" };
pub const mla_x = [_][]const u8{ "q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj" };
pub const mla_qr = [_][]const u8{ "q_b_proj", "indexer.wq_b" };
pub const gate_up = [_][]const u8{ "gate_proj", "up_proj" };

pub const Hc = struct { fnp: Ref, base: Ref, scale: Ref };

pub const Kda = struct { in_proj: Affine, f_b: Affine, g_b: Affine, o_proj: Affine, conv_w: Ref, a_log: Ref, a: Ref, dt_bias: Ref, o_norm: Ref };

pub const Mla = struct {
    x_proj: Affine,
    qr_proj: Affine,
    kv_b: Affine, // none when the checkpoint stores mlx-lm's absorbed pair
    o_proj: Affine,
    q_norm: Ref,
    kv_norm: Ref,
    k_norm_w: Ref,
    k_norm_b: Ref,
    ape: Ref,
    igate: Ref, // packed as a router
    absorbed: ?[2]Affine = null, // mlx-lm's embed_q [H kv_lora, nope] and unembed_out [H v, kv_lora]
};

pub const Dense = struct { gate_up: Affine, down: Affine };

pub const Moe = struct { router: Ref, bias: Ref, gate: Affine, up: Affine, down: Affine, sh_gate_up: Affine, sh_down: Affine };

pub const Layer = struct {
    hc: ?[2]Hc, // attention's and the MLP's hyper-connections (none on the MTP layer)
    in_norm: Ref,
    post_norm: Ref,
    attn: union(cfg.Kind) { kda: Kda, mla: Mla },
    mlp: union(enum) { dense: Dense, moe: Moe },
};

pub const Mtp = struct { eh_proj: Affine, enorm: Ref, hnorm: Ref, norm: Ref, eh_bf16: ?Ref = null }; // stored bf16

pub const Weights = struct {
    layers: [cfg.max_layers]Layer = undefined, // the backbone's, then the MTP layer at index c.layers
    mtp: ?Mtp = null,
    embed: Affine = undefined,
    head: Affine = undefined,
    norm: Ref = undefined,
    buffers: std.ArrayList(mtl.Buffer) = .empty,
    bytes: usize = 0,
    gpa: std.mem.Allocator,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |b| b.deinit();
        w.buffers.deinit(w.gpa);
    }
};

pub const prefix = "model.language_model.";

const load_fix = @import("weights_fix.zig");
const Src = load_fix.Src;
const Fix = load_fix.Fix;
const Transform = load_fix.Transform;
const Copy = load_fix.Copy;

const Loader = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    device: mtl.Device,
    c: *const cfg.Config,
    w: *Weights,
    dir: []const u8,
    shards: std.ArrayList([:0]const u8) = .empty,
    names: std.StringHashMapUnmanaged(Src) = .empty,
    copies: std.ArrayList(Copy) = .empty,
    transforms: std.ArrayList(Transform) = .empty,
    dry: bool = false, // plan and check only: no buffers, no reads
    widths: quant.Widths = undefined,
    // the buffer being filled and its next free byte
    cur: ?mtl.Buffer = null,
    cur_len: usize = 0,
    cur_at: usize = 0,

    fn src(l: *Loader, comptime fmt: []const u8, args: anytype) !Src {
        var name: [200]u8 = undefined;
        const full = try std.fmt.bufPrint(&name, fmt, args);
        return l.names.get(full) orelse {
            std.log.err("glm: the checkpoint has no tensor {s}", .{full});
            return error.MissingTensor;
        };
    }

    fn expect(s: Src, name: []const u8, dtype: st.DType, shape: []const usize) !void {
        if (s.dtype == dtype and s.rank == shape.len and std.mem.eql(usize, s.shape[0..s.rank], shape)) return;
        quant.say("glm: {s} is {t} {any}; the kernels read {t} {any}", .{ name, s.dtype, s.shape[0..s.rank], dtype, shape });
        return error.UnexpectedTensor;
    }

    /// Refuses two members of one fused matrix stored at different widths.
    fn mixed(a: []const u8, aw: Width, b: []const u8, bw: Width) error{MixedWidths} {
        quant.say("glm: {s} is {d}-bit in groups of {d} and {s} {d}-bit in groups of {d}; one matrix holds both", .{ a, aw.bits, aw.group, b, bw.bits, bw.group });
        return error.MixedWidths;
    }

    /// Input columns [lo, lo + k) of `module` at `w`, refused unless whole groups and words.
    fn cutOf(module: []const u8, lo: usize, k: usize, w: Width) error{UnexpectedTensor}![2][2]usize {
        return affine.cut(lo, k, w) orelse {
            quant.say("glm: {s}: columns [{d}, {d}) split a group or word at {d} bits in groups of {d}", .{ module, lo, lo + k, w.bits, w.group });
            return error.UnexpectedTensor;
        };
    }

    /// A new buffer of `len` bytes that the next `take`s fill.
    fn begin(l: *Loader, len: usize) !void {
        const b: ?mtl.Buffer = if (l.dry) null else try l.device.buffer(@max(len, 16), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        if (b) |x| l.w.buffers.append(l.gpa, x) catch |err| {
            x.deinit();
            return err;
        };
        l.cur = b;
        l.cur_len = len;
        l.cur_at = 0;
        l.w.bytes += len;
    }

    fn take(l: *Loader, len: usize) Ref {
        const at = l.cur_at;
        l.cur_at = std.mem.alignForward(usize, at + len, 256);
        if (l.cur_at > l.cur_len) std.debug.panic("glm: a weights buffer of {d} bytes is too small for {d}", .{ l.cur_len, l.cur_at });
        return .{ .buf = l.cur orelse .{ .id = undefined }, .off = at };
    }

    fn copyTo(l: *Loader, s: Src, dst: Ref) !void {
        if (l.dry) return;
        try l.copies.append(l.gpa, .{ .shard = s.shard, .off = s.off, .len = s.len, .dst = dst.addr() });
    }

    /// A tensor as stored (bf16 norms, fp32 biases), checked; fp32 stored as bf16 (mlx-lm's) is widened, exactly.
    fn plain(l: *Loader, comptime fmt: []const u8, args: anytype, dtype: st.DType, shape: []const usize) !Ref {
        const s = try l.src(fmt, args);
        var name: [200]u8 = undefined;
        const widen = dtype == .f32 and s.dtype == .bf16;
        try expect(s, try std.fmt.bufPrint(&name, fmt, args), if (widen) .bf16 else dtype, shape);
        if (widen) return l.fixed(s, 2 * s.len, .to_f32);
        const r = l.take(s.len);
        try l.copyTo(s, r);
        return r;
    }

    fn fixed(l: *Loader, s: Src, bytes: usize, fix: Fix) !Ref {
        const r = l.take(bytes);
        if (!l.dry) try l.transforms.append(l.gpa, .{ .src = s, .dst = r.addr(), .fix = fix });
        return r;
    }

    pub fn has(l: *Loader, comptime fmt: []const u8, args: anytype) bool {
        var name: [200]u8 = undefined;
        return l.names.contains(std.fmt.bufPrint(&name, fmt, args) catch return false);
    }

    /// No matrix: the place of one the checkpoint stores another way (refused on the device before it is read).
    fn none(l: *Loader, k: usize) Affine {
        const r = l.take(0);
        return .{ .w = r, .s = r, .b = r, .n = 0, .k = @intCast(k), .bits = 0, .group = 0 };
    }

    /// Expert `e`'s three tensors cut from layer i's mlx-lm stack `switch_mlp.proj` [experts, n, ...], checked.
    fn slab(l: *Loader, i: usize, e: usize, proj: []const u8, n: usize, k: usize, w: Width) ![3]Src {
        var buf: [200]u8 = undefined;
        const module = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.switch_mlp.{s}", .{ i, proj });
        const sh = affine.shapes(n, k, w) orelse return error.UnexpectedTensor;
        var out: [3]Src = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, &out, 0..) |comp, *o, j| {
            var full: [240]u8 = undefined;
            const nm = try std.fmt.bufPrint(&full, "{s}.{s}", .{ module, comp });
            const s = l.names.get(nm) orelse return error.MissingTensor;
            try expect(s, nm, if (j == 0) .u32 else .bf16, &.{ l.c.experts, sh[@min(j, 1)][0], sh[@min(j, 1)][1] });
            const per = s.len / l.c.experts;
            o.* = .{ .shard = s.shard, .off = s.off + e * per, .len = per, .dtype = s.dtype, .shape = .{ n, sh[@min(j, 1)][1], 1, 1, 1 }, .rank = 2 };
        }
        return out;
    }

    /// This Mac's heads of mlx-lm's per-head `module` stored [heads, r, k]: rows [h0 r, (h0 + H) r) as one matrix.
    fn perHead(l: *Loader, comptime fmt: []const u8, args: anytype, heads: usize, h0: usize, H: usize, r: usize, k: usize) !Affine {
        var buf: [200]u8 = undefined;
        const module = try std.fmt.bufPrint(&buf, fmt, args);
        const w = try l.widths.of(module);
        const sh = affine.shapes(r, k, w) orelse return error.UnexpectedTensor;
        const out: Affine = .{ .w = l.take(affine.wBytes(H * r, k, w)), .s = l.take(affine.sBytes(H * r, k, w)), .b = l.take(affine.sBytes(H * r, k, w)), .n = @intCast(H * r), .k = @intCast(k), .bits = w.bits, .group = w.group };
        for ([_][]const u8{ "weight", "scales", "biases" }, [_]Ref{ out.w, out.s, out.b }, 0..) |comp, dst, j| {
            var full: [240]u8 = undefined;
            const nm = try std.fmt.bufPrint(&full, "{s}.{s}", .{ module, comp });
            const s = l.names.get(nm) orelse return error.MissingTensor;
            try expect(s, nm, if (j == 0) .u32 else .bf16, &.{ heads, r, sh[@min(j, 1)][1] });
            const row = if (j == 0) affine.wBytes(1, k, w) else affine.sBytes(1, k, w);
            try l.copyTo(.{ .shard = s.shard, .off = s.off + h0 * r * row, .len = H * r * row, .dtype = s.dtype, .shape = s.shape, .rank = s.rank }, dst);
        }
        return out;
    }

    /// The width config.json gives the module `fmt` names, for a buffer's budget.
    pub fn widthAt(l: *Loader, comptime fmt: []const u8, args: anytype) !Width {
        var name: [200]u8 = undefined;
        return l.widths.of(try std.fmt.bufPrint(&name, fmt, args));
    }

    /// The weight, scales and biases of `module` stored [all, k] at `w`, checked.
    fn stored(l: *Loader, module: []const u8, all: usize, k: usize, w: Width) ![3]Src {
        const sh = affine.shapes(all, k, w) orelse {
            quant.say("glm: {s} has {d} input columns, which do not pack at {d} bits in groups of {d}", .{ module, k, w.bits, w.group });
            return error.UnexpectedTensor;
        };
        var out: [3]Src = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, &out, 0..) |comp, *o, j| {
            var full: [240]u8 = undefined;
            const nm = try std.fmt.bufPrint(&full, "{s}.{s}", .{ module, comp });
            o.* = l.names.get(nm) orelse {
                quant.say("glm: the checkpoint has no tensor {s}", .{nm});
                return error.MissingTensor;
            };
            try expect(o.*, nm, if (j == 0) .u32 else .bf16, &sh[@min(j, 1)]);
        }
        return out;
    }

    /// Projections that read one input, stacked by rows in order; each part checked [n_i, k].
    fn mat(l: *Loader, path: affine.Path, k: usize, comptime fmt: []const u8, parts: []const []const u8, ns: []const usize, args: anytype) !Affine {
        const zeros: [8]usize = @splat(0);
        return l.stack(path, k, fmt, parts, zeros[0..parts.len], ns, ns, 1, args);
    }

    /// Rows [lo, lo + n) of each part stored [all, k], stacked, padded to `pad`-row tiles; mixed widths: a run a part.
    fn stack(l: *Loader, path: affine.Path, k: usize, comptime fmt: []const u8, parts: []const []const u8, los: []const usize, ns: []const usize, alls: []const usize, pad: usize, args: anytype) !Affine {
        var name: [200]u8 = undefined;
        const base = try std.fmt.bufPrint(&name, fmt, args);
        var srcs: [affine.max_runs][3]Src = undefined;
        var ws: [affine.max_runs]Width = undefined;
        var mixed_widths = false;
        for (parts, alls, 0..) |p, all, i| {
            var full: [240]u8 = undefined;
            const module = try std.fmt.bufPrint(&full, "{s}{s}", .{ base, p });
            ws[i] = try l.widths.of(module);
            srcs[i] = try l.stored(module, all, k, ws[i]);
            mixed_widths = mixed_widths or !std.meta.eql(ws[0], ws[i]);
        }
        // 0.6.6's QSplit: once widths differ, every part is its own product, even beside one of its width
        const runs = if (mixed_widths) parts.len else 1;
        var starts: [affine.max_runs + 1]usize = undefined; // run r: parts [starts[r], starts[r + 1])
        for (0..runs) |r| starts[r] = r;
        starts[runs] = parts.len;
        for (parts, ws[0..parts.len], ns) |p, pw, rows| {
            var full: [240]u8 = undefined;
            try affine.admit(try std.fmt.bufPrint(&full, "{s}{s}", .{ base, p }), path, pw, rows);
        }
        var out: Affine = undefined;
        var row: usize = 0;
        for (0..runs) |r| {
            const w = ws[starts[r]];
            var n: usize = 0;
            for (ns[starts[r]..starts[r + 1]]) |x| n += x;
            const room = std.mem.alignForward(usize, n, pad);
            const run: affine.Run = .{ .w = l.take(affine.wBytes(room, k, w)), .s = l.take(affine.sBytes(room, k, w)), .b = l.take(affine.sBytes(room, k, w)), .row = @intCast(row), .n = @intCast(n), .bits = w.bits, .group = w.group };
            if (r == 0) out = .{ .w = run.w, .s = run.s, .b = run.b, .n = 0, .k = @intCast(k), .bits = w.bits, .group = w.group } else out.runs[r - 1] = run;
            var at: usize = 0; // the run's own rows
            for (srcs[starts[r]..starts[r + 1]], los[starts[r]..starts[r + 1]], ns[starts[r]..starts[r + 1]]) |three, lo, x| {
                const dsts = [_]Ref{ run.w.at(affine.wBytes(at, k, w)), run.s.at(affine.sBytes(at, k, w)), run.b.at(affine.sBytes(at, k, w)) };
                for (three, dsts, 0..) |s, dst, j| {
                    const row_bytes = if (j == 0) affine.wBytes(1, k, w) else affine.sBytes(1, k, w);
                    try l.copyTo(.{ .shard = s.shard, .off = s.off + lo * row_bytes, .len = x * row_bytes, .dtype = s.dtype, .shape = s.shape, .rank = s.rank }, dst);
                }
                at += x;
            }
            row += n;
        }
        out.n = @intCast(row);
        out.more = @intCast(runs - 1);
        return out;
    }

    /// One matrix stored [n, k_all]: its input columns [lo, lo + k) (TP2: o_proj over one Mac's heads).
    fn cols(l: *Loader, path: affine.Path, comptime fmt: []const u8, args: anytype, n: usize, k_all: usize, lo: usize, k: usize) !Affine {
        var name: [200]u8 = undefined;
        const module = try std.fmt.bufPrint(&name, fmt, args);
        const w = try l.widths.of(module);
        const three = try l.stored(module, n, k_all, w);
        try affine.admit(module, path, w, n);
        const span = try cutOf(module, lo, k, w);
        const out: Affine = .{ .w = l.take(affine.wBytes(n, k, w)), .s = l.take(affine.sBytes(n, k, w)), .b = l.take(affine.sBytes(n, k, w)), .n = @intCast(n), .k = @intCast(k), .bits = w.bits, .group = w.group };
        for (three, [_]Ref{ out.w, out.s, out.b }, 0..) |s, dst, j| {
            const row_all = if (j == 0) affine.wBytes(1, k_all, w) else affine.sBytes(1, k_all, w);
            const at = span[@min(j, 1)];
            if (!l.dry) try l.transforms.append(l.gpa, .{ .src = s, .dst = dst.addr(), .fix = .cols, .arg = .{ row_all, at[0], at[1] } });
        }
        return out;
    }

    /// Elements [lo, lo + n) of a vector stored with `all` elements of `size` bytes (TP2: one Mac's heads).
    fn plainRange(l: *Loader, comptime fmt: []const u8, args: anytype, dtype: st.DType, all: usize, lo: usize, n: usize, size: usize) !Ref {
        const s = try l.src(fmt, args);
        var name: [200]u8 = undefined;
        const widen = dtype == .f32 and s.dtype == .bf16;
        try expect(s, try std.fmt.bufPrint(&name, fmt, args), if (widen) .bf16 else dtype, &.{all});
        const at = if (widen) size / 2 else size; // a stored element's bytes
        const part: Src = .{ .shard = s.shard, .off = s.off + lo * at, .len = n * at, .dtype = s.dtype, .shape = s.shape, .rank = s.rank };
        if (widen) return l.fixed(part, n * size, .to_f32);
        const r = l.take(n * size);
        try l.copyTo(part, r);
        return r;
    }

    /// This Mac's routed experts' `proj` stacked [own, n, k] at one width; by rows, gate/up keep rows `inter`.
    fn experts(l: *Loader, i: usize, proj: []const u8, n_all: usize, k_all: usize, part: enum { whole, rows, cols }) !Affine {
        const every = i == l.c.layers; // the MTP layer: every expert whole on every Mac (its drafts need no exchange)
        const lo = if (every) 0 else l.c.own[0];
        const e = (if (every) l.c.experts else l.c.own[1]) - lo;
        const in_lo: usize = l.c.inter[0];
        const in_n: usize = l.c.inter[1] - l.c.inter[0];
        const n = if (part == .rows) in_n else n_all;
        const k = if (part == .cols) in_n else k_all;
        var first: [200]u8 = undefined;
        const name0 = try std.fmt.bufPrint(&first, prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, lo, proj });
        const w = try l.widths.expert(i, lo, proj);
        try affine.admit(name0, .routed_experts, w, n);
        const span = if (part == .cols) try cutOf(name0, in_lo, in_n, w) else undefined;
        const out: Affine = .{ .w = l.take(e * affine.wBytes(n, k, w)), .s = l.take(e * affine.sBytes(n, k, w)), .b = l.take(e * affine.sBytes(n, k, w)), .n = @intCast(n), .k = @intCast(k), .bits = w.bits, .group = w.group };
        for (0..e) |x| {
            var buf: [200]u8 = undefined;
            const module = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, lo + x, proj });
            const xw = try l.widths.expert(i, lo + x, proj);
            if (!std.meta.eql(w, xw)) return mixed(name0, w, module, xw);
            const own = l.has("{s}.weight", .{module});
            const three = if (own) try l.stored(module, n_all, k_all, w) else try l.slab(i, lo + x, proj, n_all, k_all, w);
            const dsts = [_]Ref{ out.w.at(x * affine.wBytes(n, k, w)), out.s.at(x * affine.sBytes(n, k, w)), out.b.at(x * affine.sBytes(n, k, w)) };
            for (three, dsts, 0..) |s, dst, j| {
                const row_all = if (j == 0) affine.wBytes(1, k_all, w) else affine.sBytes(1, k_all, w);
                switch (part) {
                    .whole => try l.copyTo(s, dst),
                    .rows => try l.copyTo(.{ .shard = s.shard, .off = s.off + in_lo * row_all, .len = in_n * row_all, .dtype = s.dtype, .shape = s.shape, .rank = s.rank }, dst),
                    .cols => if (!l.dry) try l.transforms.append(l.gpa, .{ .src = s, .dst = dst.addr(), .fix = .cols, .arg = .{ row_all, span[@min(j, 1)][0], span[@min(j, 1)][1] } }),
                }
            }
        }
        return out;
    }

    fn hc(l: *Loader, i: usize, site: []const u8) !Hc {
        const D: usize = l.c.hidden;
        const f = try l.src(prefix ++ "layers.{d}.hc_{s}_fn", .{ i, site });
        try expect(f, "hc fn", if (f.dtype == .f32) .f32 else .bf16, &.{ 24, 4 * D }); // fp32 (mlx-lm's): bf16 if exact
        return .{
            .fnp = try l.fixed(f, 24 * 4 * D * 2, .hc_pack),
            .base = try l.plain(prefix ++ "layers.{d}.hc_{s}_base", .{ i, site }, .f32, &.{24}),
            .scale = try l.plain(prefix ++ "layers.{d}.hc_{s}_scale", .{ i, site }, .f32, &.{3}),
        };
    }

    /// Layer i's q, k, v conv taps [channels, 1, taps]: three tensors, or mlx-lm's one [3 channels, taps, 1] cut up.
    fn convs(l: *Loader, i: usize, channels: usize) ![3]Src {
        const a = prefix ++ "layers.{d}.self_attn.";
        if (l.has(a ++ "conv1d.weight", .{i})) {
            const s = try l.src(a ++ "conv1d.weight", .{i});
            try expect(s, "conv1d", .bf16, &.{ 3 * channels, l.c.conv, 1 });
            var out: [3]Src = undefined;
            for (&out, 0..) |*o, p| o.* = .{ .shard = s.shard, .off = s.off + p * s.len / 3, .len = s.len / 3, .dtype = .bf16, .shape = .{ channels, 1, l.c.conv, 1, 1 }, .rank = 3 };
            return out;
        }
        var out: [3]Src = undefined;
        inline for (.{ "q", "k", "v" }, 0..) |x, p| {
            out[p] = try l.src(a ++ x ++ "_conv1d.weight", .{i});
            try expect(out[p], x ++ "_conv1d", .bf16, &.{ channels, 1, l.c.conv });
        }
        return out;
    }

    fn kda(l: *Loader, i: usize) !Kda {
        const c = l.c;
        const D: usize = c.hidden;
        const W: usize = c.kdaWidth(); // this Mac's heads' channels
        const H: usize = c.kda_heads;
        const dk: usize = c.kda_dim;
        const h0: usize = c.tp_rank * H; // TP2: this Mac's first head (0 on one Mac)
        const ch0 = h0 * dk;
        const Wa = W * c.tp; // every head's channels, as stored
        const Ha = H * c.tp;
        const a = prefix ++ "layers.{d}.self_attn.";
        const qkv = try l.convs(i, Wa);
        const conv = try l.fixed(qkv[0], 3 * W * c.conv * 4, .conv);
        const extra: [3]?Src = .{ qkv[1], qkv[2], null };
        if (!l.dry) {
            const t = &l.transforms.items[l.transforms.items.len - 1];
            t.extra = extra;
            t.arg = .{ ch0, W, 0 };
        }
        const o_norm = try l.src(a ++ "o_norm.weight", .{i});
        try expect(o_norm, "o_norm", .bf16, &.{dk});
        return .{
            .in_proj = try l.stack(.kda_in, D, a, &kda_in, &.{ ch0, ch0, ch0, 0, 0, h0 }, &.{ W, W, W, dk, dk, H }, &.{ Wa, Wa, Wa, dk, dk, Ha }, 64, .{i}),
            .f_b = try l.stack(.kda_gate, dk, a, &.{"f_b_proj"}, &.{ch0}, &.{W}, &.{Wa}, 1, .{i}),
            .g_b = try l.stack(.kda_gate, dk, a, &.{"g_b_proj"}, &.{ch0}, &.{W}, &.{Wa}, 1, .{i}),
            .o_proj = if (c.tp == 1) try l.mat(.kda_out, W, a, &.{"o_proj"}, &.{D}, .{i}) else try l.cols(.kda_out, a ++ "o_proj", .{i}, D, Wa, ch0, W),
            .conv_w = conv,
            .a_log = try l.plainRange(a ++ "A_log", .{i}, .f32, Ha, h0, H, 4),
            .a = l.take(H * 4),
            .dt_bias = try l.plainRange(a ++ "dt_bias", .{i}, .f32, Wa, ch0, W, 4),
            .o_norm = try l.fixed(o_norm, dk * 4, .to_f32),
        };
    }

    fn mla(l: *Loader, i: usize) !Mla {
        const c = l.c;
        const D: usize = c.hidden;
        const a = prefix ++ "layers.{d}.self_attn.";
        const H: usize = c.mla_heads; // this Mac's heads
        const h0: usize = c.tp_rank * H; // TP2: its first head (0 on one Mac)
        const Ha = H * c.tp; // every head, as stored
        const gate = try l.src(a ++ "indexer.index_kpool_compress_gate", .{i});
        try expect(gate, "index_kpool_compress_gate", .bf16, &.{ c.i_dim, D });
        const pair = l.has(a ++ "embed_q.weight", .{i});
        return .{
            .x_proj = try l.stack(.mla_in, D, a, &mla_x, &.{ 0, 0, 0, 0 }, &.{ c.q_lora, c.kv_lora, c.i_dim, c.i_heads }, &.{ c.q_lora, c.kv_lora, c.i_dim, c.i_heads }, 64, .{i}),
            .qr_proj = try l.stack(.mla_q, c.q_lora, a, &mla_qr, &.{ h0 * c.nope, 0 }, &.{ H * c.nope, c.i_heads * c.i_dim }, &.{ Ha * c.nope, c.i_heads * c.i_dim }, 1, .{i}),
            .kv_b = if (pair) l.none(c.kv_lora) else try l.stack(.kv_b, c.kv_lora, a, &.{"kv_b_proj"}, &.{h0 * (c.nope + c.v_dim)}, &.{H * (c.nope + c.v_dim)}, &.{Ha * (c.nope + c.v_dim)}, 1, .{i}),
            .absorbed = if (pair) .{ try l.perHead(a ++ "embed_q", .{i}, Ha, h0, H, c.kv_lora, c.nope), try l.perHead(a ++ "unembed_out", .{i}, Ha, h0, H, c.v_dim, c.kv_lora) } else null,
            .o_proj = if (c.tp == 1) try l.mat(.mla_out, H * c.v_dim, a, &.{"o_proj"}, &.{D}, .{i}) else try l.cols(.mla_out, a ++ "o_proj", .{i}, D, Ha * c.v_dim, h0 * c.v_dim, H * c.v_dim),
            .q_norm = try l.plain(a ++ "q_a_layernorm.weight", .{i}, .bf16, &.{c.q_lora}),
            .kv_norm = try l.plain(a ++ "kv_a_layernorm.weight", .{i}, .bf16, &.{c.kv_lora}),
            .k_norm_w = try l.plain(a ++ "indexer.k_norm.weight", .{i}, .bf16, &.{c.i_dim}),
            .k_norm_b = try l.plain(a ++ "indexer.k_norm.bias", .{i}, .bf16, &.{c.i_dim}),
            .ape = try l.plain(a ++ "indexer.index_kpool_compress_ape", .{i}, .bf16, &.{ c.kpool, c.i_dim }),
            .igate = try l.fixed(gate, gate.len, .router_pack), // core/moe_route.zig's layout: the gate is the same gemv_t
        };
    }

    fn layer(l: *Loader, i: usize) !Layer {
        const c = l.c;
        const D: usize = c.hidden;
        const plain_layer = i == c.layers; // the MTP block: no hyper-connections
        try l.begin(try load_plan.layerBytes(l, i));
        var out: Layer = .{
            .hc = if (plain_layer) null else .{ try l.hc(i, "attn"), try l.hc(i, "ffn") },
            .in_norm = try l.plain(prefix ++ "layers.{d}.input_layernorm.weight", .{i}, .bf16, &.{D}),
            .post_norm = try l.plain(prefix ++ "layers.{d}.post_attention_layernorm.weight", .{i}, .bf16, &.{D}),
            .attn = switch (c.kind(@intCast(i))) {
                .kda => .{ .kda = try l.kda(i) },
                .mla => .{ .mla = try l.mla(i) },
            },
            .mlp = undefined,
        };
        if (plain_layer) { // the MTP head's own tensors, in the layer's buffer (before the experts begin theirs)
            const p = prefix ++ "layers.{d}.";
            const bf16 = !l.has(p ++ "eh_proj.scales", .{i});
            l.w.mtp = .{
                .eh_proj = if (bf16) l.none(2 * D) else try l.mat(.eh_proj, 2 * D, p, &.{"eh_proj"}, &.{D}, .{i}),
                .eh_bf16 = if (bf16) try l.plain(p ++ "eh_proj.weight", .{i}, .bf16, &.{ D, 2 * D }) else null,
                .enorm = try l.plain(p ++ "enorm.weight", .{i}, .bf16, &.{D}),
                .hnorm = try l.plain(p ++ "hnorm.weight", .{i}, .bf16, &.{D}),
                .norm = try l.plain(p ++ "shared_head.norm.weight", .{i}, .bf16, &.{D}),
            };
        }
        const m = prefix ++ "layers.{d}.mlp.";
        if (c.isMoe(@intCast(i))) {
            const router = try l.src(m ++ "gate.weight", .{i});
            try expect(router, "mlp.gate.weight", if (router.dtype == .f32) .f32 else .bf16, &.{ c.experts, D });
            var moe: Moe = .{
                .router = try l.fixed(router, @as(usize, c.experts) * D * 2, .router_pack),
                .bias = try l.plain(m ++ "gate.e_score_correction_bias", .{i}, .f32, &.{c.experts}),
                .sh_gate_up = try l.mat(.shared_expert, D, m ++ "shared_experts.", &gate_up, &.{ c.moe_inter, c.moe_inter }, .{i}),
                .sh_down = try l.mat(.shared_expert, c.moe_inter, m ++ "shared_experts.", &.{"down_proj"}, &.{D}, .{i}),
                .gate = undefined,
                .up = undefined,
                .down = undefined,
            };
            try l.begin(try load_plan.expertBytes(l, i));
            const by_rows = c.byRows() and i != c.layers;
            moe.gate = try l.experts(i, "gate_proj", c.moe_inter, D, if (by_rows) .rows else .whole);
            moe.up = try l.experts(i, "up_proj", c.moe_inter, D, if (by_rows) .rows else .whole);
            if (!std.meta.eql(moe.gate.width(), moe.up.width())) { // one launch takes both, SwiGLU'd
                var buf: [96]u8 = undefined;
                return mixed(try std.fmt.bufPrint(&buf, "layer {d}'s routed gate_proj", .{i}), moe.gate.width(), "up_proj", moe.up.width());
            }
            moe.down = try l.experts(i, "down_proj", D, c.moe_inter, if (by_rows) .cols else .whole);
            out.mlp = .{ .moe = moe };
        } else {
            const n: usize = c.dense_inter; // this Mac's rows; TP2: rank r's half [r n, (r + 1) n)
            const lo = c.tp_rank * n;
            out.mlp = .{ .dense = .{
                .gate_up = try l.stack(.dense_mlp, D, m, &gate_up, &.{ lo, lo }, &.{ n, n }, &.{ n * c.tp, n * c.tp }, 1, .{i}),
                .down = if (c.tp == 1) try l.mat(.dense_mlp, n, m, &.{"down_proj"}, &.{D}, .{i}) else try l.cols(.dense_mlp, m ++ "down_proj", .{i}, D, n * c.tp, lo, n),
            } };
        }
        return out;
    }
};

/// The first `c.run` layers, the MTP layer (if stored), the head; `dry`: check names, dtypes and shapes, read nothing.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, c: *const cfg.Config, threads: usize, dry: bool) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const w = try gpa.create(Weights);
    w.* = .{ .gpa = gpa };
    errdefer {
        w.deinit();
        gpa.destroy(w);
    }
    var l: Loader = .{ .gpa = gpa, .arena = arena_state.allocator(), .device = device, .c = c, .w = w, .dir = dir, .dry = dry };
    defer l.shards.deinit(gpa);
    defer l.copies.deinit(gpa);
    defer l.transforms.deinit(gpa);
    try layout.read(gpa, l.arena, dir, c.layers, &l.shards, &l.names);
    l.widths = try readWidths(l.arena, dir, c);
    const D: usize = c.hidden;
    const V: usize = c.vocab;
    try l.begin(affine.bytes(V, D, try l.widths.of(prefix ++ "embed_tokens")) + affine.bytes(V, D, try l.widths.of("lm_head")) + D * 2 + 8 * 256);
    w.embed = try l.mat(.embed, D, prefix ++ "embed_tokens", &.{""}, &.{V}, .{});
    const vp = c.vocabPart(); // TP2: this Mac's half of the head's rows
    w.head = try l.stack(.head, D, "lm_head", &.{""}, &.{vp[0]}, &.{vp[1] - vp[0]}, &.{V}, 1, .{});
    w.norm = try l.plain(prefix ++ "norm.weight", .{}, .bf16, &.{D});
    for (0..c.run) |i| w.layers[i] = try l.layer(i);
    const has_mtp = c.mtp > 0 and l.names.contains(prefix ++ "layers.45.eh_proj.weight");
    if (has_mtp) w.layers[c.layers] = try l.layer(c.layers);
    if ((c.tp > 1 or c.byRows() or c.own[1] - c.own[0] != c.experts) and !native(w, c)) {
        quant.say("glm: this checkpoint has widths other than 4/64; the pair's split kernels read only 4/64 (load it on one Mac)", .{});
        return error.UnsupportedWidth;
    }
    if (dry) {
        var buf: [256]u8 = undefined;
        const widths = tally(w, c).summary(&buf);
        std.debug.print("glm plan: {d} of {d} layers{s}, {d:.2} GB in buffers, every tensor's name, dtype and shape checked; {s}\n", .{ c.run, c.layers, if (has_mtp) " and the MTP layer" else "", @as(f64, @floatFromInt(w.bytes)) / 1e9, widths });
        return w;
    }
    try load_fix.runCopies(gpa, l.shards.items, l.copies.items, l.transforms.items, threads);
    return w;
}

/// A plan's matrices by width: the head, the embedding, every loaded layer.
pub fn tally(w: *const Weights, c: *const cfg.Config) affine.Tally {
    var t: affine.Tally = .{};
    t.add(w.embed);
    t.add(w.head);
    t.add(w.mtp);
    for (w.layers[0..c.run]) |l| t.add(l);
    if (w.mtp != null) t.add(w.layers[c.layers]);
    return t;
}

/// Whether every loaded matrix takes the generated 4/64 kernels (no other width's pipelines needed).
pub fn native(w: *const Weights, c: *const cfg.Config) bool {
    if (!affine.allNative(.{ w.embed, w.head, w.mtp })) return false;
    if (w.mtp) |m| if (m.eh_bf16 != null) return false;
    for (w.layers[0..c.run]) |l| if (!layerNative(l)) return false;
    return w.mtp == null or layerNative(w.layers[c.layers]);
}

fn layerNative(l: Layer) bool {
    return switch (l.attn) {
        .mla => |m| m.absorbed == null and affine.allNative(l),
        .kda => affine.allNative(l),
    };
}

/// config.json's widths by module.
fn readWidths(arena: std.mem.Allocator, dir: []const u8, c: *const cfg.Config) !quant.Widths {
    const path = try std.fmt.allocPrintSentinel(arena, "{s}/config.json", .{dir}, 0);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    return quant.parse(arena, f.bytes[0..f.size], c.layers);
}
