//! FP64 reference per Qwen3.8 projection class (MLX 4-bit, real weights) against decode, window and prefill kernels.

const std = @import("std");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qb = @import("qwen_xpu").blocks;

const gpa = std.heap.page_allocator;
const Buf = rt.Buffer;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);

const modes = [_]struct { name: []const u8, rows: u32 }{ .{ .name = "decode", .rows = 1 }, .{ .name = "window", .rows = 8 }, .{ .name = "prefill", .rows = 64 } };
/// Output columns checked per class at most (evenly spread); smaller classes are checked in full.
const max_cols = 2048;
/// Relative error is taken against max(column maximum, col_floor * class maximum): near-zero columns carry fp32 noise.
const col_floor = 1e-4;

fn bf(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}

fn toBf(x: f64) u16 {
    const f: f32 = @floatCast(x);
    const b: u32 = @bitCast(f);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// A quantized projection read from the checkpoint file in its stored row order (independent of the device layout).
const Host = struct {
    w: []u32,
    s: []u16,
    b: []u16,
    in: usize,

    fn read(l: *ld.Loader, prefix: []const u8, rows: usize, in: usize) !Host {
        const h: Host = .{ .w = try gpa.alloc(u32, rows * in / 8), .s = try gpa.alloc(u16, rows * in / 64), .b = try gpa.alloc(u16, rows * in / 64), .in = in };
        inline for (.{ "weight", "scales", "biases" }, .{ std.mem.sliceAsBytes(h.w), std.mem.sliceAsBytes(h.s), std.mem.sliceAsBytes(h.b) }) |f, dst| {
            const name = try std.fmt.allocPrint(gpa, "{s}.{s}", .{ prefix, f });
            const inf = try l.info(name);
            try ld.readExact(inf.fd, dst, inf.off);
        }
        return h;
    }

    fn free(h: Host) void {
        gpa.free(h.w);
        gpa.free(h.s);
        gpa.free(h.b);
    }

    /// sum_i x[i] * (code * scale + bias) of one row in f64, and the sum of the absolute terms (the error scale).
    fn dot(h: Host, row: usize, x: []const f64) [2]f64 {
        const words = h.in / 8;
        const groups = h.in / 64;
        var y: f64 = 0;
        var mag: f64 = 0;
        for (0..groups) |g| {
            var d: f64 = 0;
            var sx: f64 = 0;
            const sc = bf(h.s[row * groups + g]);
            const bi = bf(h.b[row * groups + g]);
            for (0..8) |wj| {
                const pk = h.w[row * words + g * 8 + wj];
                for (0..8) |j| {
                    const v = x[g * 64 + wj * 8 + j];
                    const q: f64 = @floatFromInt((pk >> @as(u5, @intCast(4 * j))) & 15);
                    d += v * q;
                    sx += v;
                    mag += @abs(v * (q * sc + bi));
                }
            }
            y += sc * d + bi * sx;
        }
        return .{ y, mag };
    }
};

const Ctx = struct {
    r: *rt.Runtime,
    m: *model.Model,
    l: *ld.Loader,
    prng: std.Random.DefaultPrng,
    xb: Buf,
    yb: Buf,
    bad: u32 = 0,
    verbose: bool = false,
    filter: []const u8 = "",

    fn gauss(c: *Ctx) f64 {
        const rnd = c.prng.random();
        const u1_ = @max(rnd.float(f64), 1e-300);
        return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * rnd.float(f64));
    }

    /// One class in one mode: random bf16 rows through the kernel and through the f64 reference.
    fn check(c: *Ctx, label: []const u8, layer: usize, t: qb.Table, prefix: []const u8, mode: usize) !void {
        const n = modes[mode].rows;
        const in: usize = t.in;
        const rows: usize = t.rows;
        c.prng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(layer * 4 + mode, label)); // the same rows for the same case
        const x = try gpa.alloc(f64, n * in);
        defer gpa.free(x);
        const xh = try gpa.alloc(u16, n * in);
        defer gpa.free(xh);
        for (x, xh) |*v, *h| {
            h.* = toBf(c.gauss());
            v.* = bf(h.*);
        }
        try c.r.upload(c.xb, std.mem.sliceAsBytes(xh));
        try c.r.sync();
        if (n == 1) try c.m.ops.matvec(t, c.xb, c.yb, 0) else try c.m.win.proj(t, c.xb, n, c.yb, 0);
        const y = try gpa.alloc(u16, n * rows);
        defer gpa.free(y);
        try c.r.download(std.mem.sliceAsBytes(y), c.yb);
        try c.r.sync();
        const host = try Host.read(c.l, prefix, rows, in);
        defer host.free();
        const step = @max(1, rows / max_cols);
        const cols = (rows + step - 1) / step;
        const refs = try gpa.alloc(f64, cols * n);
        defer gpa.free(refs);
        const mags = try gpa.alloc(f64, cols * n);
        defer gpa.free(mags);
        var gmax: f64 = 0;
        for (0..cols) |ci| for (0..n) |ri| {
            const d = host.dot(ci * step, x[ri * in ..][0..in]);
            refs[ci * n + ri] = d[0];
            mags[ci * n + ri] = d[1];
            gmax = @max(gmax, @abs(d[0]));
        };
        // a column whose own maximum is below this floor (cancellation to ~0) is judged against the floor
        const floor = col_floor * gmax;
        var sq_err: f64 = 0;
        var sq_ref: f64 = 0;
        var worst_own: f64 = 0; // relative to the column's own maximum
        var worst: f64 = 0; // relative to max(column maximum, floor): the pass/fail figure
        var worst_abs: f64 = 0;
        var worst_scaled: f64 = 0; // relative to the sum of the absolute terms (the scale of fp32 accumulation error)
        var own_ref: f64 = 0;
        var own_mag: f64 = 0;
        var own_err: f64 = 0;
        var exact: usize = 0;
        for (0..cols) |ci| {
            var e_max: f64 = 0;
            var r_max: f64 = 0;
            var m_max: f64 = 0;
            var ref_at_e: f64 = 0;
            var mag_at_e: f64 = 0;
            for (0..n) |ri| {
                const ref = refs[ci * n + ri];
                const e = @abs(bf(y[ri * rows + ci * step]) - ref);
                sq_err += e * e;
                sq_ref += ref * ref;
                if (e >= e_max) {
                    ref_at_e = ref;
                    mag_at_e = mags[ci * n + ri];
                }
                e_max = @max(e_max, e);
                r_max = @max(r_max, @abs(ref));
                m_max = @max(m_max, mags[ci * n + ri]);
                exact += @intFromBool(y[ri * rows + ci * step] == toBf(ref));
            }
            worst_scaled = @max(worst_scaled, e_max / m_max);
            worst_abs = @max(worst_abs, e_max);
            worst = @max(worst, e_max / @max(r_max, floor));
            if (r_max > 0 and e_max / r_max > worst_own) {
                worst_own = e_max / r_max;
                own_ref = ref_at_e;
                own_mag = mag_at_e;
                own_err = e_max;
            }
        }
        const agg = @sqrt(sq_err / @max(sq_ref, 1e-300));
        const ok = agg < 4e-3 and worst < 2e-2;
        if (!ok) c.bad += 1;
        if (c.verbose or worst_own > 2e-2) std.debug.print("    diag {s} L{d} {s}: column with the largest error relative to its own maximum {e:.2}: |ref| {e:.2}, abs error {e:.2}, sum of |terms| {e:.2}, error/terms {e:.2}; class max {e:.2}, worst error/terms of any column {e:.2}\n", .{ label, layer, modes[mode].name, worst_own, @abs(own_ref), own_err, own_mag, own_err / own_mag, gmax, worst_scaled });
        std.debug.print("{s:<12} L{d:<2} {s:<8} n={d:<3} cols={d:<5} agg {e:.2} worst-col abs {e:.2} rel {e:.2} (own max {e:.2}) bf16-exact {d:.1}% {s}\n", .{ label, layer, modes[mode].name, n, cols, agg, worst_abs, worst, worst_own, 100.0 * @as(f64, @floatFromInt(exact)) / @as(f64, @floatFromInt(cols * n)), if (ok) "ok" else "OVER" });
    }

    fn classes(c: *Ctx, base: []const u8, layer: usize, attn: bool) !void {
        const ly = c.m.layers[layer];
        var nb: [160]u8 = undefined;
        const p = try std.fmt.bufPrint(&nb, "{s}.layers.{d}", .{ base, layer });
        const list: []const struct { name: []const u8, sub: []const u8, t: qb.Table } = if (attn) &.{
            .{ .name = "attn.q", .sub = ".self_attn.q_proj", .t = ly.mixer.attn.q },
            .{ .name = "attn.k", .sub = ".self_attn.k_proj", .t = ly.mixer.attn.k },
            .{ .name = "attn.v", .sub = ".self_attn.v_proj", .t = ly.mixer.attn.v },
            .{ .name = "attn.o", .sub = ".self_attn.o_proj", .t = ly.mixer.attn.o },
            .{ .name = "mlp.gate", .sub = ".mlp.gate_proj", .t = ly.mlp.gate },
            .{ .name = "mlp.up", .sub = ".mlp.up_proj", .t = ly.mlp.up },
            .{ .name = "mlp.down", .sub = ".mlp.down_proj", .t = ly.mlp.down },
        } else &.{
            .{ .name = "gdn.qkv", .sub = ".linear_attn.in_proj_qkv", .t = ly.mixer.gdn.qkv },
            .{ .name = "gdn.z", .sub = ".linear_attn.in_proj_z", .t = ly.mixer.gdn.z },
            .{ .name = "gdn.b", .sub = ".linear_attn.in_proj_b", .t = ly.mixer.gdn.b },
            .{ .name = "gdn.a", .sub = ".linear_attn.in_proj_a", .t = ly.mixer.gdn.a },
            .{ .name = "gdn.out", .sub = ".linear_attn.out_proj", .t = ly.mixer.gdn.out },
            .{ .name = "mlp.gate", .sub = ".mlp.gate_proj", .t = ly.mlp.gate },
            .{ .name = "mlp.up", .sub = ".mlp.up_proj", .t = ly.mlp.up },
            .{ .name = "mlp.down", .sub = ".mlp.down_proj", .t = ly.mlp.down },
        };
        for (list) |e| {
            if (std.mem.indexOf(u8, e.name, c.filter) == null) continue;
            var pb: [200]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&pb, "{s}{s}", .{ p, e.sub });
            for (0..modes.len) |mode| try c.check(e.name, layer, e.t, prefix, mode);
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const dir = if (args.len > 1) args[1] else return error.Usage;
    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    _ = l.info("language_model.model.embed_tokens.weight") catch {
        std.debug.print("qwen_fp64 reads the MLX 4-bit checkpoint layout only\n", .{});
        return;
    };
    var m = try model.Model.load(gpa, &r, &l, cfg.value, 256);
    m.bf16_logits = true;
    var c: Ctx = .{ .verbose = args.len > 2, .filter = if (args.len > 2) args[2] else "", .r = &r, .m = &m, .l = &l, .prng = std.Random.DefaultPrng.init(7), .xb = try r.alloc(64 * 17408 * 2), .yb = try r.alloc(64 * 17408 * 2) };
    const n_layers = cfg.value.text_config.num_hidden_layers;
    var first = [2]?usize{ null, null };
    var last = [2]?usize{ null, null };
    for (0..n_layers) |i| {
        const k: usize = @intFromBool(cfg.value.isAttention(i));
        if (first[k] == null) first[k] = i;
        last[k] = i;
    }
    for (0..2) |k| for ([_]?usize{ first[k], if (last[k] != first[k]) last[k] else null }) |li| if (li) |i| try c.classes("language_model.model", i, k == 1);
    std.debug.print("{s}\n", .{if (c.bad == 0) "qwen_fp64: every class within bounds" else "qwen_fp64: classes over the bounds"});
    if (c.bad != 0) return error.OverBounds;
}
