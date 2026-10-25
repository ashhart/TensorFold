//! The family's EXL3 admission widened over the checkpoint's own trellis headers: per-module bit-width
//! discovery through the core reader (`core/exl3_load.zig`), which admits mul1, odd half-bit widths and
//! mixed per-tensor widths natively — its `admit` (:134-157) parameterizes nothing: widths, scales and
//! the codebook marker all come from the header. This file adds the family's own admission around it:
//! everything under `language_model.`, one codebook (mul1), the served shapes, per class a complete set.

const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");
const wnames = @import("weights.zig"); // the spellings module: the fixture writes the same names the plan reads

const exl3 = core.exl3_load;
pub const Codebook = exl3.Codebook;

const layers_root = "language_model.model.layers.";

/// The module class the group's tensor name puts it in: the load's projection families and the head.
pub const Class = enum {
    attn, // self_attn.{q,k,v,o}_proj (the pack: 6-bit mul1)
    dense_mlp, // layers 0..dense_layers' mlp.{gate,up,down}_proj (5-bit)
    shared, // block_sparse_moe.shared_experts.{gate,up,down}_proj (5-bit)
    routed, // block_sparse_moe.experts.<e>.{w1,w2,w3} (mixed, odd half-bit widths)
    head, // language_model.lm_head (head_bits 8 → 16 halves)

    /// The projection's [k, n]: the reader's rows and columns of the matrix this trellis stores.
    /// The attention projections are GQA: q fans out to every head (64×128 = 8192), k/v to the
    /// KV heads only (4×128 = 512), o gathers the head dim back — measured on the real pack's
    /// trellis headers (q [384,512,96·], k/v [384,32,96·], o [512,384,96·] in K/16, N/16, 16·bits).
    pub fn shape(klas: Class, c: *const cfg.Config, proj: u32) [2]u32 {
        const inter: u32 = if (klas == .dense_mlp) c.dense_inter else c.moe_inter;
        return switch (klas) {
            .attn => switch (proj) {
                0 => .{ c.hidden, c.attn_heads * c.head_dim }, // q
                1, 2 => .{ c.hidden, c.kv_heads * c.head_dim }, // k, v
                else => .{ c.attn_heads * c.head_dim, c.hidden }, // o
            },
            .head => .{ c.hidden, c.vocab }, // measured: the pack's head trellis is [384,12504,128·] — k=hidden, n=vocab, the EXL3 in→out convention
            else => switch (proj) {
                0, 1 => .{ c.hidden, inter }, // gate/up: hidden → intermediate
                else => .{ inter, c.hidden }, // down: intermediate → hidden
            },
        };
    }

    /// How many groups of this class the checkpoint must carry, verbatim from the Config's counts.
    pub fn count(klas: Class, c: *const cfg.Config) usize {
        const layers: usize = switch (klas) {
            .attn => c.layers,
            .head => 1,
            .dense_mlp => c.dense_layers,
            .shared, .routed => c.layers - c.dense_layers,
        };
        return layers * switch (klas) {
            .attn => @as(usize, 4),
            .dense_mlp, .shared => 3,
            .routed => 3 * @as(usize, c.experts),
            .head => 1,
        };
    }
};

/// One discovered group: the reader's admission plus its class, projection and layer, verbatim [k, n].
pub const Group = struct {
    prefix: []const u8, // owned; the tensor name without its EXL3 suffixes
    class: Class,
    proj: u32, // attn: 0..3 = q,k,v,o; else 0..2 = gate,up,down
    layer: u32,
    halves: u16, // 2 * bits; odd values are mul1's half steps (the pack mixes them per tensor)
    codebook: Codebook,
    k: u32,
    n: u32,

    /// The kernel's k2 for this group: exactly the header's half-bit count.
    pub fn k2(g: Group) u16 {
        return g.halves;
    }
};

/// Per-class width distribution: how many groups carry each half-bit width (the load's k2 tables).
pub const Widths = struct {
    counts: [5][17]usize = @splat(@splat(0)),

    pub fn bump(w: *Widths, klas: Class, halves: u16) void {
        w.counts[@intFromEnum(klas)][halves] += 1;
    }

    pub fn at(w: Widths, klas: Class, halves: u16) usize {
        return w.counts[@intFromEnum(klas)][halves];
    }
};

/// Every trellis group the checkpoint carries, classified and header-checked against the family's Config.
pub const Plan = struct {
    gpa: std.mem.Allocator,
    groups: []Group,
    widths: Widths = .{},
    summary: exl3.Summary = .{},
    bad: usize = 0, // the reader's own refusals (foreign-format tensors, broken groups): a load refuses

    pub fn deinit(p: *Plan) void {
        for (p.groups) |g| p.gpa.free(g.prefix);
        p.gpa.free(p.groups);
        p.* = undefined;
    }
};

/// The group's owner by its tensor name: class, projection and layer (norms, router and embed carry none).
fn classify(c: *const cfg.Config, prefix: []const u8) ?struct { class: Class, proj: u32, layer: u32 } {
    if (std.mem.eql(u8, prefix, "language_model.lm_head")) return .{ .class = .head, .proj = 0, .layer = 0 };
    if (!std.mem.startsWith(u8, prefix, layers_root)) return null; // another tower's tensors (vision stays unquantized)
    const rest = prefix[layers_root.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    const module = rest[dot + 1 ..];
    const attn = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" };
    const mlp = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
    for (attn, 0..) |t, i| {
        if (std.mem.startsWith(u8, module, "self_attn.")) {
            if (std.mem.startsWith(u8, module["self_attn.".len..], t)) return .{ .class = .attn, .proj = @intCast(i), .layer = layer };
        }
    }
    if (std.mem.startsWith(u8, module, "mlp.")) {
        for (mlp, 0..) |t, i| if (std.mem.startsWith(u8, module["mlp.".len..], t)) {
            return .{ .class = .dense_mlp, .proj = @intCast(i), .layer = layer };
        };
    }
    const shared = "block_sparse_moe.shared_experts.";
    if (std.mem.startsWith(u8, module, shared)) {
        for (mlp, 0..) |t, i| if (std.mem.startsWith(u8, module[shared.len..], t)) {
            return .{ .class = .shared, .proj = @intCast(i), .layer = layer };
        };
    }
    const experts = "block_sparse_moe.experts.";
    if (std.mem.startsWith(u8, module, experts)) {
        const tail = module[experts.len..];
        const edot = std.mem.indexOfScalar(u8, tail, '.') orelse return null;
        const e = std.fmt.parseInt(u32, tail[0..edot], 10) catch return null;
        const w = tail[edot + 1 ..];
        const proj: u32 = if (std.mem.eql(u8, w, "w1")) 0 else if (std.mem.eql(u8, w, "w3")) 1 else if (std.mem.eql(u8, w, "w2")) 2 else return null;
        if (e >= c.experts) return null;
        return .{ .class = .routed, .proj = proj, .layer = layer };
    }
    return null;
}

/// The checkpoint directory's every EXL3 group, geometry-checked against the family's Config. The reader
/// owns the format's admission (marker values, scale presence, whole-128 tiles); this adds the family's:
/// everything under `language_model.`, mul1, and the shapes the family serves, per class a complete set.
pub fn discover(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const cfg.Config) !Plan {
    var scan = try exl3.scan(gpa, io, dir);
    defer scan.deinit();
    var out: Plan = .{ .gpa = gpa, .groups = &.{}, .summary = scan.summary, .bad = scan.bad };
    var groups: std.ArrayList(Group) = .empty;
    errdefer {
        for (groups.items) |g| gpa.free(g.prefix);
        groups.deinit(gpa);
    }
    for (scan.groups) |g| {
        const who = classify(c, g.prefix) orelse return error.ForeignTrellis; // a quantized tensor the family never reads
        const want = who.class.shape(c, who.proj);
        if (g.k != want[0] or g.n != want[1]) return error.BadTrellisGeo;
        if (g.codebook != .mul1) return error.NotMul1; // the pack's one codebook
        if (who.class == .head and g.halves != 2 * c.qc_head_bits) return error.HeadBits; // lm_head's own width
        try groups.append(gpa, .{
            .prefix = try gpa.dupe(u8, g.prefix),
            .class = who.class,
            .proj = who.proj,
            .layer = who.layer,
            .halves = g.halves,
            .codebook = g.codebook,
            .k = @intCast(g.k),
            .n = @intCast(g.n),
        });
        out.widths.bump(who.class, g.halves);
    }
    out.groups = try groups.toOwnedSlice(gpa);
    var missing: bool = false;
    for ([_]Class{ .attn, .dense_mlp, .shared, .routed, .head }) |klas| if (countOf(&out, klas) != Class.count(klas, c)) {
        missing = true;
    };
    if (missing) {
        out.deinit(); // the plan holds on to nothing when the pack's set is incomplete
        return error.MissingTrellis;
    }
    if (scan.bad != 0) {
        out.deinit();
        return error.BrokenTrellis;
    }
    return out;
}

/// One class's discovered groups (the loader walks them per layer, checkpoint order not guaranteed).
pub fn countOf(p: *const Plan, klas: Class) usize {
    var n: usize = 0;
    for (p.groups) |g| if (g.class == klas) {
        n += 1;
    };
    return n;
}

/// The trellis-and-scales bytes a group reads from the checkpoint (the raw device footprint's read).
pub fn groupBytes(g: Group) u64 {
    return @as(u64, g.k / 16) * (g.n / 16) * g.halves * 8 * 2 // the trellis int16 words
        + @as(u64, g.k) * 2 + @as(u64, g.n) * 2 + 4; // suh + svh fp16, the codebook marker int32
}

test "classify maps the pack's spellings onto the family's classes by projection" {
    const c = cfg.Config{};
    const cases = [_]struct { prefix: []const u8, class: Class, proj: u32, layer: u32 }{
        .{ .prefix = "language_model.model.layers.7.self_attn.q_proj", .class = .attn, .proj = 0, .layer = 7 },
        .{ .prefix = "language_model.model.layers.7.self_attn.o_proj", .class = .attn, .proj = 3, .layer = 7 },
        .{ .prefix = "language_model.model.layers.0.mlp.down_proj", .class = .dense_mlp, .proj = 2, .layer = 0 },
        .{ .prefix = "language_model.model.layers.9.block_sparse_moe.shared_experts.up_proj", .class = .shared, .proj = 1, .layer = 9 },
        .{ .prefix = "language_model.model.layers.57.block_sparse_moe.experts.3.w3", .class = .routed, .proj = 1, .layer = 57 },
        .{ .prefix = "language_model.model.layers.57.block_sparse_moe.experts.3.w2", .class = .routed, .proj = 2, .layer = 57 },
        .{ .prefix = "language_model.lm_head", .class = .head, .proj = 0, .layer = 0 },
    };
    for (cases) |t| {
        const who = classify(&c, t.prefix) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(t.class, who.class);
        try std.testing.expectEqual(t.proj, who.proj);
        try std.testing.expectEqual(t.layer, who.layer);
    }
    try std.testing.expect(classify(&c, "language_model.model.layers.7.self_attn.index_q_proj") == null); // unquantized parts
    try std.testing.expect(classify(&c, "language_model.model.layers.7.block_sparse_moe.gate") == null); // router fp32
    try std.testing.expect(classify(&c, "vision_tower.encoder.layers.0.mlp.fc1") == null); // not this family's
}

test "geometry, counts and the head's own bits are the Config's, and k2 carries odd half steps verbatim" {
    try std.testing.expectEqual([2]u32{ 6144, 3072 }, Class.routed.shape(&cfg.Config{}, 0));
    try std.testing.expectEqual([2]u32{ 3072, 6144 }, Class.routed.shape(&cfg.Config{}, 2));
    try std.testing.expectEqual([2]u32{ 6144, 12288 }, Class.dense_mlp.shape(&cfg.Config{}, 1));
    try std.testing.expectEqual([2]u32{ 6144, 200064 }, Class.head.shape(&cfg.Config{}, 0));
    const c = cfg.Config{};
    try std.testing.expectEqual(@as(usize, 240), Class.count(.attn, &c)); // 60 layers * q,k,v,o
    try std.testing.expectEqual(@as(usize, 9), Class.count(.dense_mlp, &c)); // 3 dense layers * gate,up,down
    try std.testing.expectEqual(@as(usize, 171), Class.count(.shared, &c)); // 57 MoE layers * 3
    try std.testing.expectEqual(@as(usize, 21888), Class.count(.routed, &c)); // 57 * 128 experts * w1,w2,w3
    try std.testing.expectEqual(@as(usize, 22309), 240 + 9 + 171 + 21888 + 1); // the pack's whole group set
    const g = Group{ .prefix = "", .class = .routed, .proj = 0, .layer = 3, .halves = 9, .codebook = .mul1, .k = 6144, .n = 3072 };
    try std.testing.expectEqual(@as(u16, 9), g.k2()); // 4.5-bit: the odd mul1 half step, carried verbatim
    try std.testing.expectEqual(@as(u64, (6144 / 16) * (3072 / 16) * 9 * 8 * 2 + 6144 * 2 + 3072 * 2 + 4), groupBytes(g)); // 11.25 MB
}

// --- a synthetic checkpoint fixture, small enough to disk-test, shapes from a scaled Config ---

/// The fixture's shapes: the REAL pack's widths (config.parse's width gate requires exactly those, so
/// admission against a synthetic pack must carry them) with small counts: 2 layers, one dense wave, 4
/// whole-128 experts and a 512-wide vocabulary (whole 128s at every tile edge).
pub fn synthConfig() cfg.Config {
    var c = cfg.Config{}; // the pack's own widths: hidden 6144, head_dim 128, 4×128 index heads, etc.
    c.layers = 2;
    c.run = 2;
    c.vocab = 512;
    c.dense_layers = 1;
    c.experts = 4;
    c.topk = 4;
    c.qc_head_bits = 8; // the head's own width: admission reads it from the config, the fixture mirrors it
    return c;
}

pub const Fixture = struct {
    foreign: bool = false, // one quantized tensor outside language_model.
    drop_last_w2: bool = false, // expert 3's up projection missing: an incomplete set refuses
    head_halves: u16 = 16, // the 8-bit head's halves
    mcg: bool = false, // one routed group's marker names the other codebook
};

/// One EXL3 group's four tensors into the fixture's header and zero-filled data (headers only are read).
const Builder = struct {
    gpa: std.mem.Allocator,
    header: std.ArrayListUnmanaged(u8) = .empty,
    data: std.ArrayListUnmanaged(u8) = .empty,
    first: bool = true,

    fn group(b: *Builder, name: []const u8, k: usize, n: usize, halves: u16, cb: exl3.Codebook) !void {
        const trellis = (k / 16) * (n / 16) * @as(usize, halves) * 8 * 2;
        const suh: usize = k * 2;
        const svh: usize = n * 2;
        const base = b.data.items.len;
        try b.data.appendNTimes(b.gpa, 0, trellis + suh + svh + 4);
        const marker: [4]u8 = if (cb == .mul1) .{ 0x2D, 0xD1, 0xDC, 0x83 } else .{ 0xED, 0x1F, 0xAC, 0xCB }; // mul1 / mcg magic
        @memcpy(b.data.items[base + trellis + suh + svh ..][0..4], &marker);
        const parts = [_]struct { suffix: []const u8, dtype: []const u8, len: usize, shape: []const u8 }{
            .{ .suffix = ".trellis", .dtype = "I16", .len = trellis, .shape = "K,N,H" },
            .{ .suffix = ".suh", .dtype = "F16", .len = suh, .shape = "K" },
            .{ .suffix = ".svh", .dtype = "F16", .len = svh, .shape = "N" },
            .{ .suffix = if (cb == .mul1) ".mul1" else ".mcg", .dtype = "I32", .len = 4, .shape = "1" },
        };
        var off = base;
        for (parts) |e| {
            if (!b.first) try b.header.appendSlice(b.gpa, ",");
            b.first = false;
            try b.header.print(b.gpa, "\"{s}{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, e.suffix, e.dtype });
            if (std.mem.eql(u8, e.shape, "K,N,H")) {
                try b.header.print(b.gpa, "{d},{d},{d}", .{ k / 16, n / 16, halves * 8 });
            } else if (std.mem.eql(u8, e.shape, "K")) {
                try b.header.print(b.gpa, "{d}", .{k});
            } else if (std.mem.eql(u8, e.shape, "N")) {
                try b.header.print(b.gpa, "{d}", .{n});
            } else try b.header.appendSlice(b.gpa, "1");
            try b.header.print(b.gpa, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + e.len });
            off += e.len;
        }
    }

    fn done(b: *Builder) ![]u8 {
        try b.header.insertSlice(b.gpa, 0, "{"); // the safetensors header is one JSON object of the tensors
        try b.header.appendSlice(b.gpa, "}");
        var file: std.ArrayListUnmanaged(u8) = .empty;
        errdefer file.deinit(b.gpa);
        try file.appendNTimes(b.gpa, 0, 8);
        std.mem.writeInt(u64, file.items[0..8], b.header.items.len, .little);
        try file.appendSlice(b.gpa, b.header.items);
        try file.appendSlice(b.gpa, b.data.items);
        return file.toOwnedSlice(b.gpa);
    }
};

test "discover admits the whole synthetic pack and records its mixed widths" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(path);
    try writeFixture(gpa, io, tmp.dir, .{});
    const c = synthConfig();
    var p = try discover(gpa, io, path, &c);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 27), p.groups.len); // 8 attn + 3 dense + 3 shared + 12 routed + 1 head
    try std.testing.expectEqual(@as(usize, 8), countOf(&p, .attn));
    try std.testing.expectEqual(@as(usize, 3), countOf(&p, .dense_mlp));
    try std.testing.expectEqual(@as(usize, 3), countOf(&p, .shared));
    try std.testing.expectEqual(@as(usize, 12), countOf(&p, .routed));
    try std.testing.expectEqual(@as(usize, 1), countOf(&p, .head));
    try std.testing.expectEqual(@as(usize, 6), p.widths.at(.routed, 4)); // e0,e2: w1,w2,w3 at 2-bit
    try std.testing.expectEqual(@as(usize, 6), p.widths.at(.routed, 5)); // e1,e3: the odd 2.5-bit half step
    try std.testing.expectEqual(@as(usize, 1), p.widths.at(.head, 16)); // the 8-bit head's one group
    var head_halves: u16 = 0;
    for (p.groups) |g| if (g.class == .head) {
        head_halves = g.k2();
        try std.testing.expectEqual(@as(u32, 512), g.n); // cols of the lm_head matrix: the whole vocab at TP1 (k=hidden, measured pack convention)
    };
    try std.testing.expectEqual(@as(u16, 16), head_halves);
}

test "a head at the wrong bits, a foreign quantized tensor, another codebook and an incomplete set refuse" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    inline for ([_]struct { name: []const u8, o: Fixture, want: anyerror }{
        .{ .name = "head_halves", .o = .{ .head_halves = 12 }, .want = error.HeadBits }, // not this pack's 8-bit row
        .{ .name = "foreign", .o = .{ .foreign = true }, .want = error.ForeignTrellis }, // another tower's quantized mlp
        .{ .name = "mcg", .o = .{ .mcg = true }, .want = error.NotMul1 }, // one expert on the other codebook
        .{ .name = "incomplete", .o = .{ .drop_last_w2 = true }, .want = error.MissingTrellis }, // expert 3's w2 missing
    }) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeFixture(gpa, io, tmp.dir, case.o);
        const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(path);
        const c = synthConfig();
        try std.testing.expectError(case.want, discover(gpa, io, path, &c));
    }
}

/// The synthetic pack written into `dir` as model.safetensors (header + zero-filled groups).
pub fn writeFixture(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, o: Fixture) !void {
    const file = try buildFixture(gpa, o);
    defer gpa.free(file);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file });
}

/// A 2-layer synthetic checkpoint: layer 0 dense, layer 1 MoE (4 experts, one odd half step, mixed).
pub fn buildFixture(gpa: std.mem.Allocator, o: Fixture) ![]u8 {
    var b: Builder = .{ .gpa = gpa };
    defer {
        b.header.deinit(gpa);
        b.data.deinit(gpa);
    }
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    const c = synthConfig();
    const keep = struct {
        fn f(a: std.mem.Allocator, list: *std.ArrayList([]u8), comptime fmt: []const u8, args: anytype) ![]const u8 {
            const owned = try std.fmt.allocPrint(a, fmt, args);
            try list.append(a, owned);
            return owned;
        }
    }.f;
    // the head, whole vocab at TP1: its own width is the pack's 8 halves *2 qc_head_bits — unless the
    // fixture deliberately varies it. Attention's and the MLPs' widths are any legal width: admission
    // checks geometry and the codebook, not per-class bits; the head's is the only width rule.
    const hs = Class.head.shape(&c, 0);
    try b.group(try keep(gpa, &names, "language_model.lm_head", .{}), hs[0], hs[1], o.head_halves, .mul1);
    for ([_]u32{ 0, 1 }) |l| {
        inline for (.{ "q_proj", "k_proj", "v_proj", "o_proj" }, 0..) |p, pi| {
            const s = Class.attn.shape(&c, pi);
            try b.group(try keep(gpa, &names, "language_model.model.layers.{d}.self_attn.{s}", .{ l, p }), s[0], s[1], 2, .mul1);
        }
        if (l < c.dense_layers) {
            inline for ([_]wnames.ExpertProj{ .gate, .up, .down }) |proj| {
                const s = Class.dense_mlp.shape(&c, @intFromEnum(proj));
                try b.group(try keep(gpa, &names, "language_model.model.layers.{d}.mlp.{s}_proj", .{ l, @tagName(proj) }), s[0], s[1], 2, .mul1);
            }
            continue;
        }
        inline for ([_]wnames.ExpertProj{ .gate, .up, .down }) |proj| {
            const s = Class.shared.shape(&c, @intFromEnum(proj));
            try b.group(try keep(gpa, &names, "language_model.model.layers.{d}.block_sparse_moe.shared_experts.{s}_proj", .{ l, @tagName(proj) }), s[0], s[1], 2, .mul1);
        }
        for (0..@as(u32, c.experts)) |e| {
            const odd: u16 = if (e % 2 == 1) 5 else 4; // mixed per-expert widths, one odd half step
            for ([_]wnames.ExpertProj{ .gate, .up, .down }) |proj| {
                if (e == 3 and o.drop_last_w2 and proj == .up) continue; // expert 3's up projection missing
                const s = Class.routed.shape(&c, @intFromEnum(proj));
                try b.group(try keep(gpa, &names, "language_model.model.layers.{d}.block_sparse_moe.experts.{d}.{s}", .{ l, e, proj.ckpt() }), s[0], s[1], odd, if (e == 2 and o.mcg) .mcg else .mul1);
            }
        }
    }
    if (o.foreign) try b.group(try keep(gpa, &names, "vision_tower.blocks.0.mlp.fc1", .{}), c.hidden, c.hidden, 2, .mul1);
    return b.done(); // the owned names outlive only this build: the file bytes carry the text on
}
/// The synthetic checkpoint's config.json: this scaled Config's counts in the pack's spellings, so the
/// admission tests open `dir` as a whole model. The indexer's and EXL3's shapes come from Config's
/// defaults (the real pack's values — 4×128 index heads, 16-of-128 blocks, swigluoai, sigmoid router);
/// rms_norm_eps matches the Config default exactly, as parse compares it.
pub fn writeSynthConfigJson(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const c = synthConfig();
    const json = try std.fmt.allocPrint(gpa,
        "{{\"model_type\":\"minimax_m3_vl\",\"text_config\":{{" ++
            "\"hidden_size\":{d},\"num_hidden_layers\":{d},\"vocab_size\":{d}," ++
            "\"num_attention_heads\":64,\"num_key_value_heads\":4,\"head_dim\":128," ++
            "\"rms_norm_eps\":0.000001,\"num_local_experts\":{d},\"num_experts_per_tok\":{d}," ++
            "\"intermediate_size\":{d},\"dense_intermediate_size\":{d},\"shared_intermediate_size\":{d}," ++
            "\"routed_scaling_factor\":2.0,\"swiglu_alpha\":1.702,\"swiglu_limit\":7.0," ++
            "\"scoring_func\":\"sigmoid\",\"use_routing_bias\":true," ++
            "\"use_qk_norm\":true,\"qk_norm_type\":\"per_head\",\"moe_layer_freq\":[0,1]," ++
            "\"sparse_attention_config\":{{\"sparse_num_index_heads\":4,\"sparse_index_dim\":128," ++
            "\"sparse_topk_blocks\":16,\"sparse_block_size\":128,\"sparse_score_type\":\"max\"}}," ++
            "\"quantization_config\":{{\"quant_method\":\"exl3\",\"avg_bits\":3.06,\"head_bits\":8," ++
            "\"codebook\":\"mul1\"}}}}}}",
        .{ c.hidden, c.layers, c.vocab, c.experts, c.topk, c.moe_inter, c.dense_inter, c.shared_inter });
    defer gpa.free(json);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = json });
}
