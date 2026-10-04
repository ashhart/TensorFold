//! Kimi K3 layers on synthetic or checkpoint weights: row-exact rounds checked here, outputs for k3_check.py.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const synth = @import("synth.zig");
const rounds = @import("rounds.zig");
const gpu_mod = @import("gpu.zig");
const Gpu = gpu_mod.Gpu;
const resident = gpu_mod.resident;

fn writeFile(io: std.Io, dir: []const u8, name: []const u8, data: []const u8, gpa: std.mem.Allocator) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

/// Where a check's weights come from: hash-made at real shapes, or the checkpoint's shards.
const Src = union(enum) {
    synth: *synth.Synth,
    shards: *k3.shards.Shards,

    fn source(s: Src) k3.store.Source {
        return switch (s) {
            inline else => |x| x.source(),
        };
    }

    fn buffers(s: Src, out: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator) !void {
        switch (s) {
            .synth => |x| {
                x.sync();
                try out.appendSlice(gpa, x.bufs.items);
            },
            .shards => |x| try x.buffers(out, gpa),
        }
    }
};

fn scratchBuffers(sc: *const k3.round.Scratch, out: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator) !void {
    const info = @typeInfo(k3.round.Scratch).@"struct";
    inline for (info.field_names, info.field_types) |f, T| if (T == mtl.Buffer) try out.append(gpa, @field(sc, f));
}

/// Layer i through the schedule three ways, compared bit for bit; mixed rows and an expert decode go to `dir`.
fn checkLayer(g: *Gpu, gpa: std.mem.Allocator, io: std.Io, i: u32, dir: []const u8, model: ?*k3.shards.Shards) !void {
    const cfg = k3.config.Config{};
    var own = synth.Synth{ .gpa = gpa, .device = g.device, .queue = g.queue, .k = &g.k };
    defer own.deinit();
    const src: Src = if (model) |m| .{ .shards = m } else .{ .synth = &own };
    const t0 = mtl.clock.seconds();
    var w = try k3.weights.layer(&cfg, i, src.source(), g.device, gpa, 0, cfg.experts);
    var owned = k3.prepare.Owned{ .gpa = gpa };
    defer owned.deinit();
    try k3.prepare.layer(&g.k, g.device, g.queue, &w, &owned);
    var weight_bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer weight_bufs.deinit(gpa);
    try src.buffers(&weight_bufs, gpa);
    std.debug.print("layer {d} ({s}, {s}): {s} weights ready in {d:.1} s\n", .{ i, @tagName(cfg.kind(i)), if (cfg.isMoe(i)) "MoE" else "dense", @tagName(src), mtl.clock.seconds() - t0 });
    var state = try k3.state.State.init(gpa, g.device, cfg, .{ .slots = 9, .log_rows = 16, .max_ctx = 512 }, &.{i}, cfg.kda_heads);
    defer state.deinit();
    var sc = try k3.round.Scratch.init(g.device, &cfg, 256);
    defer sc.deinit();
    var bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer bufs.deinit(gpa);
    try state.buffers(&bufs, gpa);
    try scratchBuffers(&sc, &bufs, gpa);
    if (w.mlp == .moe) try bufs.append(gpa, w.mlp.moe.table);
    const set = try resident(g, &.{ weight_bufs.items, bufs.items });
    defer {
        g.queue.removeResidencySet(set);
        set.deinit();
    }
    var local = k3.experts.Local.all(&g.k, &cfg);
    const run = rounds.Run{ .gpa = gpa, .queue = g.queue, .ctx = .{ .k = &g.k, .c = &cfg, .sc = &sc, .st = &state, .experts = local.experts() }, .w = &w, .layer = i, .state = &state };
    var res: [3]rounds.Results = .{ .{ .gpa = gpa }, .{ .gpa = gpa }, .{ .gpa = gpa } };
    defer for (&res) |*r| r.deinit();
    for (std.enums.values(rounds.Comp), 0..) |comp, ci| {
        const t = mtl.clock.seconds();
        try rounds.all(run, comp, @intCast(3 * ci), &res[ci]);
        std.debug.print("  {s}: {d} rows in {d:.2} s\n", .{ @tagName(comp), res[ci].rows.count(), mtl.clock.seconds() - t });
    }
    const n_split = try rounds.same(&res[0], &res[1], "a segment a round");
    const n_serial = try rounds.same(&res[0], &res[2], "a kept row a round");
    std.debug.print("  PASS row-exact: {d} rows equal in mixed rounds (1-256 rows) and alone, {d} kept rows equal to serial\n", .{ n_split, n_serial });
    const sub = try std.fmt.allocPrint(gpa, "{s}/layer{d}", .{ dir, i });
    defer gpa.free(sub);
    try std.Io.Dir.cwd().createDirPath(io, sub);
    var rows_bin: std.ArrayList(u8) = .empty;
    defer rows_bin.deinit(gpa);
    var keys: std.ArrayList(u8) = .empty;
    defer keys.deinit(gpa);
    try keys.appendSlice(gpa, "[");
    for (res[0].rows.keys(), res[0].rows.values(), 0..) |key, v, n| {
        try rows_bin.appendSlice(gpa, std.mem.sliceAsBytes(v));
        try keys.print(gpa, "{s}[{d},{d},{d}]", .{ if (n == 0) "" else ",", key.stream, key.round, key.row });
    }
    try keys.appendSlice(gpa, "]");
    try writeFile(io, sub, "rows.bin", rows_bin.items, gpa);
    try writeFile(io, sub, "keys.json", keys.items, gpa);
    if (w.mlp == .moe) try dequant(g, gpa, io, src.source(), i, sub);
}

/// Expert 5's w1 decoded by the kernels' own MXFP4 path, for a bit-exact check against the reference decode.
fn dequant(g: *Gpu, gpa: std.mem.Allocator, io: std.Io, src: k3.store.Source, i: u32, sub: []const u8) !void {
    const cfg = k3.config.Config{};
    const base = try std.fmt.allocPrint(gpa, "layers.{d}.block_sparse_moe.experts.5.w1.", .{i});
    defer gpa.free(base);
    const pn = try std.mem.concat(gpa, u8, &.{ base, "weight_packed" });
    defer gpa.free(pn);
    const sn = try std.mem.concat(gpa, u8, &.{ base, "weight_scale" });
    defer gpa.free(sn);
    const p = try src.get(pn, .u8, &.{ cfg.moe_inter, cfg.latent / 2 });
    const s = try src.get(sn, .u8, &.{ cfg.moe_inter, cfg.latent / 32 });
    const n = @as(usize, cfg.moe_inter) * cfg.latent;
    const out = try g.device.buffer(n * 4, mtl.ResourceOptions.shared);
    defer out.deinit();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    g.k.dequant(e, p.ref, s.ref, .{ .buf = out }, cfg.moe_inter, cfg.latent);
    e.end();
    cb.commit();
    cb.wait();
    try writeFile(io, sub, "dequant_e5_w1.bin", std.mem.sliceAsBytes(out.slice(f32, n)), gpa);
}

/// Output residual, final norm and LM head on 8 rows, with greedy tokens; embedding rows checked here.
fn checkHead(g: *Gpu, gpa: std.mem.Allocator, io: std.Io, dir: []const u8, model: ?*k3.shards.Shards) !void {
    const cfg = k3.config.Config{};
    var own = synth.Synth{ .gpa = gpa, .device = g.device, .queue = g.queue, .k = &g.k };
    defer own.deinit();
    const src: Src = if (model) |m| .{ .shards = m } else .{ .synth = &own };
    var h = try k3.weights.head(&cfg, src.source(), gpa);
    var owned = k3.prepare.Owned{ .gpa = gpa };
    defer owned.deinit();
    try k3.prepare.head(&g.k, g.device, g.queue, &h, &owned);
    var weight_bufs: std.ArrayList(mtl.Buffer) = .empty;
    defer weight_bufs.deinit(gpa);
    try src.buffers(&weight_bufs, gpa);
    var state = try k3.state.State.init(gpa, g.device, cfg, .{ .slots = 1, .max_ctx = 16 }, &.{}, cfg.kda_heads);
    defer state.deinit();
    var sc = try k3.round.Scratch.init(g.device, &cfg, 40);
    defer sc.deinit();
    const R = 8;
    const ids = [_]u32{ 0, 1, 77, 4096, 65535, 100000, 163583, 163839 };
    const ctx = k3.layer.Ctx{ .k = &g.k, .c = &cfg, .sc = &sc, .st = &state, .experts = undefined };
    const V: usize = cfg.vocab;
    const wide = try gpa.alloc(u16, R * V);
    defer gpa.free(wide);
    for ([_]u32{ 40, R }) |rows| {
        headInputs(&sc, &cfg, rows);
        @memcpy(sc.ids.slice(u32, R), &ids);
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        k3.layer.head(ctx, e, &h, cfg.layers);
        g.k.greedy(e, sc.ref("logits"), sc.ref("tokens"), rows, cfg.vocab);
        g.k.embedRows(e, sc.ref("ids"), h.embed.ref, sc.ref("y"), R, cfg.hidden);
        e.end();
        cb.commit();
        cb.wait();
        if (rows != R) @memcpy(wide, sc.logits.slice(u16, R * V));
    }
    if (!std.mem.eql(u16, wide, sc.logits.slice(u16, R * V))) return error.HeadRowsDiffer;
    const H = cfg.hidden;
    const table = @as([*]const u16, @ptrCast(@alignCast(h.embed.ref.buf.contents() + h.embed.ref.off)))[0 .. @as(usize, cfg.vocab) * H];
    for (ids, 0..) |id, r| {
        if (!std.mem.eql(u16, sc.y.slice(u16, (r + 1) * H)[r * H ..], table[id * H .. (id + 1) * H])) return error.EmbedRow;
    }
    std.debug.print("head: logits of 8 rows equal alone and in a 40-row round; embedding rows exact; greedy tokens {any}\n", .{sc.tokens.slice(u32, R)});
    const sub = try std.fmt.allocPrint(gpa, "{s}/head", .{dir});
    defer gpa.free(sub);
    try std.Io.Dir.cwd().createDirPath(io, sub);
    try writeFile(io, sub, "logits.bin", std.mem.sliceAsBytes(sc.logits.slice(u16, R * @as(usize, cfg.vocab))), gpa);
    try writeFile(io, sub, "tokens.bin", std.mem.sliceAsBytes(sc.tokens.slice(u32, R)), gpa);
}

/// The head check's inputs for `rows` rows: rows 0-7 are k3_check.py's (stream 9), the rest other values.
fn headInputs(sc: *k3.round.Scratch, cfg: *const k3.config.Config, rows: u32) void {
    const H = cfg.hidden;
    const nb = (cfg.layers - 1) / cfg.block + 1;
    var buf: [96]u8 = undefined;
    for (0..rows) |r| {
        const key = rounds.Key{ .stream = if (r < 8) 9 else 8, .round = 0, .row = @intCast(r) };
        synth.rowValues(sc.prefix.slice(u16, (r + 1) * H)[r * H ..], rounds.inputName(&buf, key, "prefix", null));
        synth.rowValues(sc.delta.slice(u16, (r + 1) * H)[r * H ..], rounds.inputName(&buf, key, "delta", null));
        for (0..nb) |e| synth.rowValues(sc.blocks.slice(u16, (e * rows + r + 1) * H)[(e * rows + r) * H ..], rounds.inputName(&buf, key, "block", e));
    }
    sc.rows = rows;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var args = try init.minimal.args.toSlice(init.arena.allocator());
    var model: ?*k3.shards.Shards = null;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var g = try Gpu.init();
    defer g.deinit();
    var shards: k3.shards.Shards = undefined;
    if (args.len > 2 and std.mem.eql(u8, args[1], "--model")) {
        shards = try k3.shards.Shards.open(gpa, init.io, g.device, args[2]);
        model = &shards;
        args = args[2..];
    }
    defer if (model) |m| m.deinit();
    if (args.len < 3) {
        std.debug.print("usage: tf-k3-check [--model K3_DIR] OUT_DIR (LAYER | head)...\n", .{});
        std.process.exit(2);
    }
    std.debug.print("device: {s}; weights: {s}\n", .{ g.device.name(), if (model != null) "checkpoint" else "synthetic" });
    try std.Io.Dir.cwd().createDirPath(init.io, args[1]);
    for (args[2..]) |a| {
        if (std.mem.eql(u8, a, "head")) {
            try checkHead(&g, gpa, init.io, args[1], model);
        } else try checkLayer(&g, gpa, init.io, try std.fmt.parseInt(u32, a, 10), args[1], model);
    }
}
