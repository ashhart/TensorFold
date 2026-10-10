//! Kernel choice: `select` takes the cheapest entry by tuning table, so ranks agree; one family writes equal bits.

const std = @import("std");
const quant = @import("quant/quant.zig");
const tuning = @import("tuning.zig");

/// What is launched: a product, up to four sharing x, a plan over stacked experts, or its gate and up with activation.
pub const Op = enum { project, group, routed, routed_act };

/// A prompt's span, or anything that decodes (a lane round, the logits head, a draft).
pub const Path = enum { decode, prefill };

/// Entries that write the same bytes for any row of a product.
pub const Family = enum { stream, row, lanes, wide, tiled, reference, split, gemm };

/// A product as the choice sees it.
pub const Shape = struct {
    /// Rows of x; a plan's most rows in an item.
    m: u32,
    n: u32,
    k: u32,
    bits: u8,
    group: u16,
    /// x is fp16 (bf16 otherwise).
    fp16: bool,
    tables: quant.Tables = .bf16,
    tables_alike: bool = true,
    /// x on 16 bytes.
    x_aligned: bool = true,
    /// The words on the load size of their width.
    words_aligned: bool = true,
    /// A plan's items; 0 is a plain product.
    items: u32 = 0,
    /// Parts the groups are split into.
    parts: u32 = 1,
    /// Inside a lane round: the decode tile keeps every row count.
    round: bool = false,
    /// The activation of a stacked gate and up is the epilogue.
    pairs: bool = false,
};

/// What the GPU has and the run switched on.
pub const Env = struct {
    /// bf16 activations (fp16 otherwise).
    bf16: bool,
    /// Matrix cores a tile can use exist.
    matrix: bool,
    /// ...and the run lets them take the rows above the decode tiles'.
    matrix_on: bool,
    stream_on: bool,
    /// The GEMM tile (else the reference one) takes the products.
    gemm_on: bool,
};

/// Whether an entry writes the activation type itself instead of fp32.
pub const Rounds = enum { never, always, fp16 };

pub fn Entry(comptime Launch: type) type {
    return struct {
        id: []const u8,
        format: quant.Format,
        op: Op,
        path: Path,
        family: Family,
        /// What the GPU must have.
        caps: *const fn (Env) bool,
        /// What the run must not have switched off.
        policy: *const fn (Env) bool,
        /// What the kernel can do: alignments, widths, row counts it has code for.
        fits: *const fn (Shape) bool,
        rounds: Rounds = .never,
        /// A lane round takes it at any row count its shape fits.
        round_any: bool = false,
        launch: Launch,

        /// Whether the entry writes the activation type at this shape.
        pub fn roundsAct(e: *const @This(), s: Shape) bool {
            return switch (e.rounds) {
                .never => false,
                .always => true,
                .fp16 => s.fp16,
            };
        }
    };
}

/// Room for the entries' tuned rows: a Registry sits in a backend's kernel table, so its size is fixed.
pub const max_entries = 64;

/// The most rows the family rule is checked at.
pub const rule_rows = 256;

pub fn Registry(comptime Launch: type) type {
    return struct {
        const Self = @This();
        pub const E = Entry(Launch);

        entries: []const E,
        table: *const tuning.Table,
        tuned: [max_entries]?tuning.Row,

        pub fn init(table: *const tuning.Table, entries: []const E) Self {
            std.debug.assert(entries.len <= max_entries);
            var r: Self = .{ .entries = entries, .table = table, .tuned = @splat(null) };
            for (entries, 0..) |e, i| r.tuned[i] = tuning.find(table, e.id);
            return r;
        }

        /// The cost of entry `i` at `shape`, or null where its row has no room for the shape.
        fn cost(r: *const Self, i: usize, shape: Shape, env: Env) ?f32 {
            const e = &r.entries[i];
            const t = r.tuned[i] orelse return null;
            const plan = shape.items > 0;
            if (plan and !t.routed) return null;
            if (!plan and !t.plain) return null;
            if (t.items != 0 and plan and shape.items > t.items) return null;
            // a routed prompt's kernel is chosen by its items, whatever rows they hold
            if (!plan or e.path == .decode) {
                const limit = if (t.rows_matrix != 0 and env.matrix_on and env.matrix) t.rows_matrix else t.rows;
                if (!(e.round_any and shape.round) and limit != 0 and shape.m > limit) return null;
                if (shape.m < t.min_rows) return null;
            }
            return t.cost;
        }

        /// The cheapest entry of (format, op, path) the GPU, run and shape allow; null when none takes the shape.
        pub fn select(r: *const Self, env: Env, format: quant.Format, op: Op, path: Path, shape: Shape) ?*const E {
            var best: ?*const E = null;
            var best_cost: f32 = std.math.inf(f32);
            for (r.entries, 0..) |*e, i| {
                if (e.format != format or e.op != op or e.path != path or !e.caps(env) or !e.policy(env) or !e.fits(shape)) continue;
                const c = r.cost(i, shape, env) orelse continue;
                if (c < best_cost) {
                    best = e;
                    best_cost = c;
                }
            }
            return best;
        }

        /// The family rule for one (op, path) of a format: at every row count it selects the one family.
        fn rule(r: *const Self, env: Env, format: quant.Format, op: Op, path: Path, base: Shape, quiet: bool) error{FamilyChanges}!void {
            var shape = base;
            var seen: ?Family = null;
            for (1..rule_rows + 1) |m| {
                shape.m = @intCast(m);
                const e = r.select(env, format, op, path, shape) orelse continue;
                if (seen) |f| {
                    if (f != e.family) {
                        if (!quiet) std.log.err("registry: {s} at {d} rows is {t}, not {t} ({d} bits, group {d})", .{ e.id, m, e.family, f, shape.bits, shape.group });
                        return error.FamilyChanges;
                    }
                } else seen = e.family;
            }
        }

        /// Checks the family rule for a round's decode, a prompt's products and routed plans, at every width and group.
        pub fn verify(r: *const Self, env: Env, format: quant.Format) error{FamilyChanges}!void {
            // the reference decode tiles and the reference GEMM tile are the Python engine's rules, by row count
            if (!env.stream_on or !env.gemm_on) return;
            for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u16{ 32, 64, 128 }) |group| {
                var base: Shape = .{ .m = 1, .n = 4096, .k = 4096, .bits = bits, .group = group, .fp16 = !env.bf16, .tables = if (env.bf16) .bf16 else .f16 };
                base.round = true;
                try r.rule(env, format, .project, .decode, base, false);
                base.round = false;
                try r.rule(env, format, .project, .prefill, base, false);
                base.items = 64;
                try r.rule(env, format, .routed, .prefill, base, false);
            };
        }

        /// What each (op, path) chose across row counts, one line a run of rows: for `--explain-kernels`.
        pub fn explain(r: *const Self, env: Env, format: quant.Format, w: *std.Io.Writer, bits: u8, group: u16) std.Io.Writer.Error!void {
            const base: Shape = .{ .m = 1, .n = 4096, .k = 4096, .bits = bits, .group = group, .fp16 = !env.bf16, .tables = if (env.bf16) .bf16 else .f16 };
            try w.print("kernel choice for {t} on {s}: {d}-bit, group {d}, {s} activations\n", .{ format, r.table.gfx, bits, group, if (env.bf16) "bf16" else "fp16" });
            const rows = [_]struct { Op, Path, bool, u32 }{
                .{ .project, .decode, true, 0 },
                .{ .project, .decode, false, 0 },
                .{ .group, .decode, true, 0 },
                .{ .routed, .decode, false, 64 },
                .{ .routed_act, .decode, false, 64 },
                .{ .project, .prefill, false, 0 },
                .{ .routed, .prefill, false, 64 },
            };
            for (rows) |row| {
                var shape = base;
                shape.round = row[2];
                shape.items = row[3];
                shape.pairs = row[0] == .routed_act;
                try w.print("  {t} {t}{s}{s}:", .{ row[0], row[1], if (row[2]) " in a lane round" else "", if (row[3] != 0) " over a plan" else "" });
                var from: u32 = 1;
                var last: ?*const E = null;
                var m: u32 = 1;
                while (m <= rule_rows) : (m += 1) {
                    shape.m = m;
                    const e = r.select(env, format, row[0], row[1], shape);
                    if (m > 1 and e != last) {
                        try printRun(w, from, m - 1, last);
                        from = m;
                    }
                    last = e;
                }
                try printRun(w, from, rule_rows, last);
                try w.writeAll("\n");
            }
        }

        fn printRun(w: *std.Io.Writer, from: u32, to: u32, e: ?*const E) std.Io.Writer.Error!void {
            const name = if (e) |x| x.id[std.mem.lastIndexOfScalar(u8, x.id, '.').? + 1 ..] else "none";
            if (from == to) try w.print(" {d} {s}", .{ from, name }) else try w.print(" {d}-{d} {s}", .{ from, to, name });
        }
    };
}

// ---- the rules, on entries of no backend

const Toy = Registry(*const fn () void);
const ToyEntry = Toy.E;

fn nothing() void {}

fn anyEnv(_: Env) bool {
    return true;
}

fn anyShape(_: Shape) bool {
    return true;
}

fn toy(id: []const u8, path: Path, family: Family, round_any: bool) ToyEntry {
    return .{ .id = id, .format = .mlx, .op = .project, .path = path, .family = family, .caps = &anyEnv, .policy = &anyEnv, .fits = &anyShape, .round_any = round_any, .launch = &nothing };
}

const toy_env: Env = .{ .bf16 = true, .matrix = true, .matrix_on = true, .stream_on = true, .gemm_on = true };
const toy_shape: Shape = .{ .m = 1, .n = 64, .k = 64, .bits = 4, .group = 32, .fp16 = false };

test "the cheapest entry that has room for the rows runs" {
    const entries = [_]ToyEntry{
        toy("a.decode.few", .decode, .stream, true),
        toy("a.decode.cores", .decode, .gemm, false),
        toy("a.decode.any", .decode, .lanes, false),
    };
    const rows = [_]tuning.Row{
        .{ .id = "a.decode.few", .cost = 1, .rows = 16, .rows_matrix = 15 },
        .{ .id = "a.decode.cores", .cost = 2, .min_rows = 16 },
        .{ .id = "a.decode.any", .cost = 3 },
    };
    const r = Toy.init(&.{ .gfx = "toy", .rows = &rows }, &entries);
    var shape = toy_shape;
    const want = [_]struct { u32, []const u8 }{ .{ 1, "few" }, .{ 15, "few" }, .{ 16, "cores" }, .{ 100, "cores" } };
    for (want) |w| {
        shape.m = w[0];
        try std.testing.expect(std.mem.endsWith(u8, r.select(toy_env, .mlx, .project, .decode, shape).?.id, w[1]));
    }
    // the matrix cores off: the 16th row is still the decode tile's
    var off = toy_env;
    off.matrix_on = false;
    shape.m = 16;
    try std.testing.expect(std.mem.endsWith(u8, r.select(off, .mlx, .project, .decode, shape).?.id, "few"));
    // a lane round keeps its tile at any rows; another format has no entry
    shape.round = true;
    shape.m = 200;
    try std.testing.expect(std.mem.endsWith(u8, r.select(toy_env, .mlx, .project, .decode, shape).?.id, "few"));
    try std.testing.expect(r.select(toy_env, .dense, .project, .decode, shape) == null);
    // an entry the table does not list never runs
    const partial = Toy.init(&.{ .gfx = "toy", .rows = rows[2..] }, &entries);
    try std.testing.expect(std.mem.endsWith(u8, partial.select(toy_env, .mlx, .project, .decode, shape).?.id, "any"));
}

test "a plan is chosen by its items, a plain product by its rows" {
    const entries = [_]ToyEntry{
        toy("a.prefill.short", .prefill, .gemm, false),
        toy("a.prefill.plan", .prefill, .gemm, false),
        toy("a.prefill.big", .prefill, .gemm, false),
    };
    const rows = [_]tuning.Row{
        .{ .id = "a.prefill.short", .cost = 1, .rows = 4, .routed = false },
        .{ .id = "a.prefill.plan", .cost = 2, .items = 300, .plain = false },
        .{ .id = "a.prefill.big", .cost = 3 },
    };
    const r = Toy.init(&.{ .gfx = "toy", .rows = &rows }, &entries);
    var shape = toy_shape;
    shape.m = 3;
    try std.testing.expect(std.mem.endsWith(u8, r.select(toy_env, .mlx, .project, .prefill, shape).?.id, "short"));
    shape.m = 5;
    try std.testing.expect(std.mem.endsWith(u8, r.select(toy_env, .mlx, .project, .prefill, shape).?.id, "big"));
    // the rows an item holds do not count for a plan
    shape.items = 100;
    shape.m = 40;
    try std.testing.expect(std.mem.endsWith(u8, r.select(toy_env, .mlx, .project, .prefill, shape).?.id, "plan"));
    shape.items = 301;
    try std.testing.expect(std.mem.endsWith(u8, r.select(toy_env, .mlx, .project, .prefill, shape).?.id, "big"));
}

test "the family rule refuses a path whose family changes with the rows" {
    const entries = [_]ToyEntry{
        toy("a.decode.small", .decode, .stream, false),
        toy("a.decode.large", .decode, .gemm, false),
        toy("a.prefill.all", .prefill, .gemm, false),
    };
    const rows = [_]tuning.Row{
        .{ .id = "a.decode.small", .cost = 1, .rows = 8 },
        .{ .id = "a.decode.large", .cost = 2, .min_rows = 9 },
        .{ .id = "a.prefill.all", .cost = 1 },
    };
    const r = Toy.init(&.{ .gfx = "toy", .rows = &rows }, &entries);
    try r.rule(toy_env, .mlx, .project, .prefill, toy_shape, true);
    try std.testing.expectError(error.FamilyChanges, r.rule(toy_env, .mlx, .project, .decode, toy_shape, true));
}
