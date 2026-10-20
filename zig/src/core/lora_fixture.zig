//! Synthetic dense and shipped checkpoint folders and PEFT adapters for the merge tests: no model, no GPU.
const std = @import("std");
const se = @import("shard_edit.zig");
const host4 = @import("affine4_host.zig");
const lm = @import("lora_merge.zig");
const dir = @import("lora_merge_dir.zig");
const Io = std.Io;

pub const io = std.testing.io;
pub const base_config = "{\"model_type\":\"x\",\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}}";
pub const shard_names = [_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
pub const files = [_][]const u8{ "config.json", "model.safetensors.index.json", shard_names[0], shard_names[1] };
pub const module = "backbone.layers.0.mixer.in_proj";

pub const Scratch = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    root: []const u8,

    pub fn init() !Scratch {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        errdefer arena.deinit();
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const root = try tmp.dir.realPathFileAlloc(io, ".", arena.allocator());
        return .{ .arena = arena, .tmp = tmp, .root = root };
    }

    pub fn deinit(s: *Scratch) void {
        s.tmp.cleanup();
        s.arena.deinit();
    }

    pub fn path(s: *Scratch, parts: []const []const u8) ![]const u8 {
        return std.fs.path.join(s.arena.allocator(), &.{ s.root, std.fs.path.join(s.arena.allocator(), parts) catch unreachable });
    }

    pub fn read(s: *Scratch, folder: []const u8, name: []const u8) ![]const u8 {
        return Io.Dir.cwd().readFileAlloc(io, try s.path(&.{ folder, name }), s.arena.allocator(), .limited(1 << 24));
    }

    pub fn write(s: *Scratch, folder: []const u8, name: []const u8, data: []const u8) !void {
        const p = try s.path(&.{ folder, name });
        try Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(p).?);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = data });
    }

    /// The names in `folder` (relative to the scratch root), sorted.
    pub fn list(s: *Scratch, folder: []const u8) ![]const []const u8 {
        var names: std.ArrayList([]const u8) = .empty;
        var d = Io.Dir.cwd().openDir(io, try s.path(&.{folder}), .{ .iterate = true }) catch return &.{};
        defer d.close(io);
        var it = d.iterate();
        while (try it.next(io)) |e| try names.append(s.arena.allocator(), try s.arena.allocator().dupe(u8, e.name));
        std.mem.sort([]const u8, names.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        return names.items;
    }
};

/// The three leaves and the dense weight of a projection whose values are `vals` (each exactly a bf16).
pub fn leaves(a: std.mem.Allocator, name: []const u8, n: usize, k: usize, vals: []const f32, dense: *std.ArrayList(se.New), quantized: *std.ArrayList(se.New)) !void {
    const bf = try a.alloc(u16, n * k);
    for (bf, vals) |*h, v| h.* = host4.bf16of(v);
    const q: lm.Quant = .{ .words = try a.alloc(u32, n * k / 8), .scales = try a.alloc(u16, n * k / 64), .biases = try a.alloc(u16, n * k / 64) };
    if ((try lm.quantize(64, vals, q)) != null) return error.NonFinite;
    const g = try a.dupe(usize, &.{ n, k / 64 });
    try dense.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{name}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ n, k }), .bytes = std.mem.sliceAsBytes(bf) });
    try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{name}), .dtype = .u32, .shape = try a.dupe(usize, &.{ n, k / 8 }), .bytes = std.mem.sliceAsBytes(q.words) });
    try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{name}), .dtype = .bf16, .shape = g, .bytes = std.mem.sliceAsBytes(q.scales) });
    try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.biases", .{name}), .dtype = .bf16, .shape = g, .bytes = std.mem.sliceAsBytes(q.biases) });
}

/// Pseudo-random bf16-exact values.
pub fn randomValues(a: std.mem.Allocator, count: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const v = try a.alloc(f32, count);
    for (v) |*x| x.* = host4.f32of(host4.bf16of((prng.random().float(f32) - 0.5) * 0.2));
    return v;
}

/// The folders `dense` and `base` (shipped): layer 0 in_proj (8 x 128) in shard 1, layer 1 out_proj (8 x 64) in 2.
pub fn makeBase(s: *Scratch, config: []const u8) ![]const u8 {
    const a = s.arena.allocator();
    const base = try s.path(&.{"base"});
    try Io.Dir.cwd().createDirPath(io, base);
    var dense: std.ArrayList(se.New) = .empty;
    var first: std.ArrayList(se.New) = .empty;
    try first.append(a, .{ .name = "backbone.embeddings.weight", .dtype = .bf16, .shape = &.{ 2, 4 }, .bytes = "0123456789abcdef" });
    try leaves(a, module, 8, 128, try randomValues(a, 8 * 128, 1), &dense, &first);
    var second: std.ArrayList(se.New) = .empty;
    try leaves(a, "backbone.layers.1.mixer.out_proj", 8, 64, try randomValues(a, 8 * 64, 2), &dense, &second);
    try second.append(a, .{ .name = "backbone.layers.1.norm.weight", .dtype = .bf16, .shape = &.{4}, .bytes = "ABCDEFGH" });
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\"metadata\":{},\"weight_map\":{");
    for ([2][]const se.New{ first.items, second.items }, shard_names, 0..) |tensors, shard, i| {
        try se.create(std.testing.allocator, try s.path(&.{ "base", shard }), &.{.{ "format", "mlx" }}, tensors);
        for (tensors, 0..) |t, j| try index.print(a, "{s}\"{s}\":\"{s}\"", .{ if (i == 0 and j == 0) "" else ",", t.name, shard });
    }
    try index.appendSlice(a, "}}");
    try s.write("base", "model.safetensors.index.json", index.items);
    try s.write("base", "config.json", config);
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"dense"}));
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    return base;
}

/// A one-projection pair of folders `dense` and `base` (one shard, no index) around `vals` (n x k, k a multiple of 64).
pub fn makeSingle(s: *Scratch, n: usize, k: usize, vals: []const f32) ![]const u8 {
    const a = s.arena.allocator();
    var dense: std.ArrayList(se.New) = .empty;
    var quantized: std.ArrayList(se.New) = .empty;
    try leaves(a, module, n, k, vals, &dense, &quantized);
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"dense"}));
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"base"}));
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    try se.create(std.testing.allocator, try s.path(&.{ "base", "model.safetensors" }), &.{.{ "format", "mlx" }}, quantized.items);
    try s.write("base", "config.json", base_config);
    return try s.path(&.{"base"});
}

pub const Kind = enum { linear, stacked, plain, dense_absent };

pub const Entry = struct {
    dense: []const u8, // the module's name in the dense source
    shipped: []const u8, // and in the shipped checkpoint
    n: usize = 8,
    k: usize = 128,
    kind: Kind = .linear,
};

/// `dense` and `base` folders (one shard each, no index) holding `entries`, the shipped one under the shipped names.
pub fn makeNamed(s: *Scratch, entries: []const Entry) ![]const u8 {
    const a = s.arena.allocator();
    var dense: std.ArrayList(se.New) = .empty;
    var quantized: std.ArrayList(se.New) = .empty;
    for (entries, 0..) |e, i| {
        const zeros = try a.alloc(u8, 2 * e.n * e.k * 2);
        @memset(zeros, 0);
        switch (e.kind) {
            .linear, .dense_absent => {
                const before = dense.items.len;
                try leaves(a, e.shipped, e.n, e.k, try randomValues(a, e.n * e.k, 10 + i), &dense, &quantized);
                dense.items[before].name = try std.fmt.allocPrint(a, "{s}.weight", .{e.dense});
                if (e.kind == .dense_absent) _ = dense.pop();
            },
            .stacked => {
                try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{e.shipped}), .dtype = .u32, .shape = try a.dupe(usize, &.{ 2, e.n, e.k / 8 }), .bytes = zeros[0 .. 2 * e.n * e.k / 2] });
                try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{e.shipped}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ 2, e.n, e.k / 64 }), .bytes = zeros[0 .. 2 * e.n * e.k / 32] });
                try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.biases", .{e.shipped}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ 2, e.n, e.k / 64 }), .bytes = zeros[0 .. 2 * e.n * e.k / 32] });
                try dense.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{e.dense}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ 2, e.n, e.k }), .bytes = zeros[0 .. 2 * e.n * e.k * 2] });
            },
            .plain => {
                try quantized.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{e.shipped}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ e.n, e.k }), .bytes = zeros[0 .. e.n * e.k * 2] });
                try dense.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{e.dense}), .dtype = .bf16, .shape = try a.dupe(usize, &.{ e.n, e.k }), .bytes = zeros[0 .. e.n * e.k * 2] });
            },
        }
    }
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"dense"}));
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"base"}));
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    try se.create(std.testing.allocator, try s.path(&.{ "base", "model.safetensors" }), &.{.{ "format", "mlx" }}, quantized.items);
    try s.write("base", "config.json", base_config);
    return try s.path(&.{"base"});
}

pub const Spec = struct { module: []const u8 = module, n: usize = 8, k: usize = 128, r: usize = 2, zero: bool = false, neg_zero: bool = false, nan: bool = false, value: ?f32 = null };

pub const Adapter = struct { specs: []const Spec = &.{.{}}, alpha: []const u8 = "4", extra: []const u8 = "", names: ?[]const []const u8 = null, rank: []const u8 = "2" };

/// An adapter folder: A and B for each spec; `value` fills both factors, `names` replaces the tensor names.
pub fn makeAdapter(s: *Scratch, o: Adapter) ![]const u8 {
    const a = s.arena.allocator();
    const folder = try s.path(&.{"adapter"});
    try Io.Dir.cwd().createDirPath(io, folder);
    var prng = std.Random.DefaultPrng.init(5);
    var tensors: std.ArrayList(se.New) = .empty;
    for (o.specs, 0..) |sp, i| {
        const la = try a.alloc(f32, sp.r * sp.k);
        const lb = try a.alloc(f32, sp.n * sp.r);
        for (la) |*v| v.* = sp.value orelse (prng.random().float(f32) - 0.5) * 0.1;
        for (lb) |*v| v.* = if (sp.neg_zero) -0.0 else if (sp.zero) 0 else sp.value orelse (prng.random().float(f32) - 0.5) * 0.1;
        if (sp.nan) la[0] = std.math.nan(f32);
        const an = if (o.names) |nm| nm[2 * i] else try std.fmt.allocPrint(a, "base_model.model.{s}.lora_A.weight", .{sp.module});
        const bn = if (o.names) |nm| nm[2 * i + 1] else try std.fmt.allocPrint(a, "base_model.model.{s}.lora_B.weight", .{sp.module});
        try tensors.append(a, .{ .name = an, .dtype = .f32, .shape = try a.dupe(usize, &.{ sp.r, sp.k }), .bytes = std.mem.sliceAsBytes(la) });
        try tensors.append(a, .{ .name = bn, .dtype = .f32, .shape = try a.dupe(usize, &.{ sp.n, sp.r }), .bytes = std.mem.sliceAsBytes(lb) });
    }
    try se.create(std.testing.allocator, try s.path(&.{ "adapter", "adapter_model.safetensors" }), &.{}, tensors.items);
    try s.write("adapter", "adapter_config.json", try std.fmt.allocPrint(a, "{{\"peft_type\":\"LORA\",\"r\":{s},\"lora_alpha\":{s}{s}}}", .{ o.rank, o.alpha, o.extra }));
    return folder;
}

pub const Result = struct { code: u8, out: []const u8, err: []const u8 };

/// `merge` of the scratch folders `dense` and `base` into `merged`, with the messages it printed.
pub fn merge(s: *Scratch, base: []const u8, adapter: []const u8, msg: []u8) !Result {
    var out = Io.Writer.fixed(msg[0 .. msg.len / 2]);
    var err = Io.Writer.fixed(msg[msg.len / 2 ..]);
    const code = try dir.run(std.testing.allocator, io, &out, &err, try s.path(&.{"dense"}), adapter, try s.path(&.{"merged"}), base);
    return .{ .code = code, .out = out.buffered(), .err = err.buffered() };
}

/// The tf-merge-check gates on the scratch folders, with what they printed.
pub fn gates(s: *Scratch, base: []const u8, adapter: []const u8, msg: []u8) !struct { ok: bool, out: []const u8 } {
    var w = Io.Writer.fixed(msg);
    const ok = try dir.gate.check(std.testing.allocator, s.arena.allocator(), io, &w, try s.path(&.{"dense"}), adapter, try s.path(&.{"merged"}), base);
    return .{ .ok = ok, .out = w.buffered() };
}

/// `merge`'s default on the scratch folders: the adapter folded onto `base`'s own scales into `merged`.
pub fn fold(s: *Scratch, base: []const u8, adapter: []const u8, msg: []u8) !Result {
    var out = Io.Writer.fixed(msg[0 .. msg.len / 2]);
    var err = Io.Writer.fixed(msg[msg.len / 2 ..]);
    const code = try dir.run(std.testing.allocator, io, &out, &err, null, adapter, try s.path(&.{"merged"}), base);
    return .{ .code = code, .out = out.buffered(), .err = err.buffered() };
}

/// The fold's tf-merge-check gates on the scratch folders, with what they printed.
pub fn foldGates(s: *Scratch, base: []const u8, adapter: []const u8, msg: []u8) !struct { ok: bool, out: []const u8 } {
    var w = Io.Writer.fixed(msg);
    const ok = try dir.gate.check(std.testing.allocator, s.arena.allocator(), io, &w, null, adapter, try s.path(&.{"merged"}), base);
    return .{ .ok = ok, .out = w.buffered() };
}
