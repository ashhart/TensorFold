//! Kimi K3's checkpoint against itself and our family: index vs headers, byte coverage, every tensor's dtype and shape.
const std = @import("std");
const k3 = @import("kimi_k3");
const cluster = @import("cluster");

const ckpt = cluster.checkpoint;
const Spec = k3.weights.Spec;

const Report = struct {
    problems: usize = 0,
    shown: usize = 0,

    fn bad(r: *Report, comptime fmt: []const u8, args: anytype) void {
        r.problems += 1;
        if (r.shown < 40) std.debug.print("  PROBLEM " ++ fmt ++ "\n", args);
        r.shown += 1;
    }
};

fn dtypeOf(d: k3.store.DType) ckpt.DType {
    return switch (d) {
        .bf16 => .bf16,
        .f32 => .f32,
        .u8 => .u8,
    };
}

fn fullName(a: std.mem.Allocator, short: []const u8) ![]const u8 {
    return if (std.mem.eql(u8, short, "lm_head.weight")) "language_model.lm_head.weight" else std.mem.concat(a, u8, &.{ "language_model.model.", short });
}

/// Each file's bytes: header, then tensors tiling [8 + n, size) with no gap or overlap.
fn coverage(io: std.Io, a: std.mem.Allocator, dir: []const u8, c: ckpt.Checkpoint, r: *Report) !void {
    for (c.files, 0..) |f, fi| {
        const path = try std.fs.path.join(a, &.{ dir, f.name });
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        var head: [8]u8 = undefined;
        _ = try file.readPositionalAll(io, &head, 0);
        const data = 8 + std.mem.readInt(u64, &head, .little);
        var spans: std.ArrayList([2]u64) = .empty;
        for (c.tensors) |t| if (t.file == fi) try spans.append(a, .{ t.start, t.start + t.bytes });
        std.mem.sort([2]u64, spans.items, {}, struct {
            fn less(_: void, x: [2]u64, y: [2]u64) bool {
                return x[0] < y[0];
            }
        }.less);
        var at = data;
        for (spans.items) |sp| {
            if (sp[0] != at) r.bad("{s}: bytes {d}..{d} {s}", .{ f.name, @min(at, sp[0]), @max(at, sp[0]), if (sp[0] > at) "unindexed" else "overlap" });
            at = @max(at, sp[1]);
        }
        if (at != f.size) r.bad("{s}: tensors end at {d}, file is {d} bytes", .{ f.name, at, f.size });
        if (data % 8 != 0) r.bad("{s}: data starts at {d}, not 8-byte aligned", .{ f.name, data });
    }
}

/// Every index entry in its file's header and every header entry in the index, file for file.
fn indexMatches(io: std.Io, a: std.mem.Allocator, dir: []const u8, c: ckpt.Checkpoint, r: *Report) !void {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), a, .limited(1 << 30));
    const index = try ckpt.parseIndex(a, text);
    if (index.count() != c.tensors.len) r.bad("index lists {d} tensors, headers hold {d}", .{ index.count(), c.tensors.len });
    for (c.tensors) |t| {
        const file = index.get(t.name) orelse {
            r.bad("{s} is in {s} but not in the index", .{ t.name, c.files[t.file].name });
            continue;
        };
        if (!std.mem.eql(u8, file, c.files[t.file].name)) r.bad("{s}: index says {s}, found in {s}", .{ t.name, file, c.files[t.file].name });
    }
    std.debug.print("index: {d} tensors in {d} files, every entry where its header puts it\n", .{ index.count(), c.files.len });
}

const Groups = struct { kda: u64 = 0, mla: u64 = 0, moe: u64 = 0, dense: u64 = 0, layer_norms: u64 = 0, head: u64 = 0, experts: u64 = 0 };

/// Every tensor our family reads, as the config predicts it; `used` marks them so extras can be listed.
fn specs(a: std.mem.Allocator, cfg: *const k3.config.Config, c: ckpt.Checkpoint, by_name: *const std.StringHashMapUnmanaged(u32), used: []bool, r: *Report) !Groups {
    var g = Groups{};
    var list: std.ArrayList(Spec) = .empty;
    var align16: usize = 0;
    var checked: usize = 0;
    for (0..cfg.layers + 1) |li| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const la = arena.allocator();
        list.clearRetainingCapacity();
        const i: u32 = @intCast(li);
        if (i == cfg.layers) try k3.weights.headSpecs(cfg, la, &list) else {
            try k3.weights.layerSpecs(cfg, i, la, &list);
            if (cfg.isMoe(i)) for (0..cfg.experts) |e| try k3.weights.expertSpecs(cfg, i, @intCast(e), la, &list);
        }
        for (list.items) |s| {
            const full = try fullName(la, s.name);
            const ti = by_name.get(full) orelse {
                r.bad("missing {s}", .{full});
                continue;
            };
            used[ti] = true;
            checked += 1;
            const t = c.tensors[ti];
            const want = dtypeOf(s.dtype);
            var shape_ok = t.rank == s.rank;
            if (shape_ok) for (0..s.rank) |d| {
                if (t.shape[d] != s.shape[d]) shape_ok = false;
            };
            if (s.rank == 0) shape_ok = t.rank == 1 and t.shape[0] >= cfg.kda_heads;
            if (t.dtype != want or !shape_ok) r.bad("{s}: {s} {any}, expected {s} {any}", .{ full, @tagName(t.dtype), t.shape[0..t.rank], @tagName(want), s.shape[0..s.rank] });
            if (t.start % 8 != 0) r.bad("{s} starts at {d}, not 8-byte aligned", .{ full, t.start });
            if (t.start % 16 != 0) align16 += 1;
            const b = t.bytes;
            if (std.mem.indexOf(u8, s.name, ".experts.") != null) g.experts += b else if (i == cfg.layers) g.head += b else if (std.mem.indexOf(u8, s.name, "self_attn.") != null) {
                if (cfg.kind(i) == .kda) g.kda += b else g.mla += b;
            } else if (std.mem.indexOf(u8, s.name, "block_sparse_moe.") != null) g.moe += b else if (std.mem.indexOf(u8, s.name, ".mlp.") != null) g.dense += b else g.layer_norms += b;
        }
    }
    std.debug.print("family: {d} tensors checked against the config (dtype, shape, 8-byte alignment); {d} start off a 16-byte boundary\n", .{ checked, align16 });
    return g;
}

/// E8M0 scales of `n` experts spread over the MoE layers: our folded-scale decode is exact for 2 <= e <= 252.
fn scales(io: std.Io, a: std.mem.Allocator, dir: []const u8, cfg: *const k3.config.Config, c: ckpt.Checkpoint, by_name: *const std.StringHashMapUnmanaged(u32), n: u32, r: *Report) !void {
    var hist: [256]u64 = @splat(0);
    var total: u64 = 0;
    for (0..n) |k| {
        const layer = 1 + @as(u32, @intCast(k * (cfg.layers - 1) / @max(n, 1)));
        const e: u32 = @intCast((k * 389) % cfg.experts);
        for ([_][]const u8{ "w1", "w2", "w3" }) |w| {
            const name = try std.fmt.allocPrint(a, "language_model.model.layers.{d}.block_sparse_moe.experts.{d}.{s}.weight_scale", .{ layer, e, w });
            const t = c.tensors[by_name.get(name) orelse continue];
            const file = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ dir, c.files[t.file].name }), .{});
            defer file.close(io);
            const buf = try a.alloc(u8, t.bytes);
            defer a.free(buf);
            _ = try file.readPositionalAll(io, buf, t.start);
            for (buf) |b| hist[b] += 1;
            total += buf.len;
        }
    }
    var lo: usize = 255;
    var hi: usize = 0;
    var outside: u64 = 0;
    for (hist, 0..) |h, e| if (h > 0) {
        lo = @min(lo, e);
        hi = @max(hi, e);
        if (e < 2 or e > 252) outside += h;
    };
    std.debug.print("scales: {d} E8M0 bytes from {d} experts, exponents {d}..{d} (2^{d}..2^{d}); {d} outside the exact range\n", .{ total, n, lo, hi, @as(i32, @intCast(lo)) - 127, @as(i32, @intCast(hi)) - 127, outside });
    if (outside > 0) r.bad("{d} scale bytes outside 2..252", .{outside});
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-k3-validate MODEL_DIR [SCALE_SAMPLES]\n", .{});
        std.process.exit(2);
    }
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = init.io;
    const dir = args[1];
    var r = Report{};
    const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
    const cfg = try k3.config.parse(a, cfg_text);
    const want = k3.config.Config{};
    inline for (@typeInfo(k3.config.Config).@"struct".field_names) |f| {
        if (!std.meta.eql(@field(cfg, f), @field(want, f))) r.bad("config {s} differs from the shape our kernels are built for", .{f});
    }
    const t0 = std.Io.Clock.awake.now(io);
    const c = try ckpt.load(io, a, dir);
    var total: u64 = 0;
    for (c.files) |f| total += f.size;
    std.debug.print("headers: {d} files, {d} bytes, {d} tensors read in {d} ms\n", .{ c.files.len, total, c.tensors.len, @divTrunc(t0.durationTo(std.Io.Clock.awake.now(io)).nanoseconds, std.time.ns_per_ms) });
    try coverage(io, a, dir, c, &r);
    try indexMatches(io, a, dir, c, &r);
    var by_name: std.StringHashMapUnmanaged(u32) = .empty;
    for (c.tensors, 0..) |t, i| try by_name.put(a, t.name, @intCast(i));
    const used = try a.alloc(bool, c.tensors.len);
    @memset(used, false);
    const g = try specs(a, &cfg, c, &by_name, used, &r);
    var vision: u64 = 0;
    var extras: usize = 0;
    for (c.tensors, used) |t, u| {
        if (u) continue;
        if (k3.shards.shortName(t.name) == null) {
            vision += t.bytes;
        } else {
            extras += 1;
            r.bad("{s} is in the checkpoint but our family never reads it", .{t.name});
        }
    }
    const gb = struct {
        fn f(x: u64) f64 {
            return @as(f64, @floatFromInt(x)) / 1e9;
        }
    }.f;
    std.debug.print("bytes (GB): KDA {d:.2}, MLA {d:.2}, MoE BF16 {d:.2}, dense {d:.2}, layer norms/residuals {d:.3}, embed+head+norms {d:.2}, experts {d:.2}, vision {d:.2}\n", .{ gb(g.kda), gb(g.mla), gb(g.moe), gb(g.dense), gb(g.layer_norms), gb(g.head), gb(g.experts), gb(vision) });
    const samples = if (args.len > 2) try std.fmt.parseInt(u32, args[2], 10) else 0;
    if (samples > 0) try scales(io, a, dir, &cfg, c, &by_name, samples, &r);
    std.debug.print("{s}: {d} problems, {d} unread language-model tensors\n", .{ if (r.problems == 0) "PASS" else "FAIL", r.problems, extras });
    if (r.problems != 0) std.process.exit(1);
}
