//! What a PEFT adapter asks of a checkpoint folder: each named projection checked against its stored layout.
const std = @import("std");
const st = @import("safetensors.zig");
const lm = @import("lora_merge.zig");
const sh = @import("lora_shards.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Refused = sh.Refused;
pub const refuse = sh.refuse;
pub const isRegular = sh.isRegular;
pub const Shards = sh.Shards;
pub const Job = sh.Job;
pub const quantConfig = sh.quantConfig;
pub const Layout = sh.Layout;
pub const Names = sh.Names;
pub const resolve = sh.resolve;
pub const layoutOf = sh.layoutOf;
pub const denseOf = sh.denseOf;
pub const quantOf = sh.quantOf;
pub const free = sh.free;

pub const Plan = struct {
    jobs: []const Job,
    r: usize,
    scale: f64,
    adapter: st.File,

    pub fn deinit(p: *Plan, io: Io) void {
        p.adapter.close(io);
    }
};

const Span = struct { begin: usize, end: usize };

fn lessSpan(_: void, a: Span, b: Span) bool {
    return a.begin < b.begin;
}

/// Refuses a shard whose tensor payload ranges overlap: a write to one tensor would change another.
fn checkDisjoint(w: *Io.Writer, s: *Shards, shard: []const u8) !void {
    const f = (try s.open(shard)) orelse return;
    const spans = try s.gpa.alloc(Span, f.names.count());
    defer s.gpa.free(spans);
    var n: usize = 0;
    for (f.names.values()) |e| {
        if (e.end == e.begin) continue;
        spans[n] = .{ .begin = e.begin, .end = e.end };
        n += 1;
    }
    std.mem.sort(Span, spans[0..n], {}, lessSpan);
    for (1..n) |i| if (spans[i].begin < spans[i - 1].end) return refuse(w, shard, "tensor payload ranges overlap: a write would change another tensor");
}

fn number(v: ?std.json.Value) ?f64 {
    return switch (v orelse return null) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn isOn(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

fn nonEmpty(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return switch (v) {
        .object => |m| m.count() > 0,
        .array => |a| a.items.len > 0,
        .null => false,
        else => true,
    };
}

/// The adapter's `r` and alpha / r, with every LoRA variant the merge does not do refused.
fn adapterConfig(arena: Allocator, io: Io, w: *Io.Writer, dir: []const u8) !struct { r: usize, scale: f64 } {
    const path = try std.fs.path.join(arena, &.{ dir, "adapter_config.json" });
    const c = (try sh.readObject(arena, io, w, path)) orelse return refuse(w, "adapter_config.json", "missing or not an object");
    if (c.get("peft_type")) |t| if (t != .string or !std.mem.eql(u8, t.string, "LORA")) return refuse(w, "adapter_config.json", "peft_type is not LORA");
    for ([_][]const u8{ "use_dora", "use_rslora", "fan_in_fan_out" }) |flag| if (isOn(c, flag)) return refuse(w, "adapter_config.json", flag);
    for ([_][]const u8{ "rank_pattern", "alpha_pattern", "modules_to_save" }) |key| if (nonEmpty(c, key)) return refuse(w, "adapter_config.json", key);
    if (c.get("bias")) |b| if (b != .string or !std.mem.eql(u8, b.string, "none")) return refuse(w, "adapter_config.json", "bias other than none");
    const r = number(c.get("r")) orelse return refuse(w, "adapter_config.json", "no r");
    const alpha = number(c.get("lora_alpha")) orelse return refuse(w, "adapter_config.json", "no lora_alpha");
    const usize_end = std.math.pow(f64, 2, @bitSizeOf(usize));
    if (!std.math.isFinite(r) or !std.math.isFinite(alpha) or r < 1 or r != @floor(r) or r >= usize_end or alpha < 0) return refuse(w, "adapter_config.json", "r or lora_alpha out of range");
    if (!std.math.isFinite(alpha / r)) return refuse(w, "adapter_config.json", "alpha / r is not finite");
    return .{ .r = @intFromFloat(r), .scale = alpha / r };
}

const Pair = struct { a: ?[]const u8 = null, b: ?[]const u8 = null };

/// The projections `adapter` names, each in the dense base and the shipped checkpoint, all checked before any write.
pub fn plan(gpa: Allocator, arena: Allocator, io: Io, w: *Io.Writer, dense: *Shards, shipped: *Shards, adapter: []const u8) !Plan {
    const cfg = try adapterConfig(arena, io, w, adapter);
    const quant = try quantConfig(arena, io, w, shipped.dir);
    const file_path = try std.fs.path.join(arena, &.{ adapter, "adapter_model.safetensors" });
    if (!try sh.requireRegular(io, w, file_path)) return refuse(w, "adapter_model.safetensors", "missing");
    var file = try st.File.open(gpa, io, file_path);
    errdefer file.close(io);
    var pairs: std.StringArrayHashMapUnmanaged(Pair) = .empty;
    for (file.names.keys()) |name| {
        const k = lm.parseAdapterKey(name) orelse return refuse(w, name, "not a lora_A or lora_B weight of base_model.model.<module>");
        const slot = try pairs.getOrPut(arena, k.module);
        if (!slot.found_existing) slot.value_ptr.* = .{};
        const held = switch (k.side) {
            .a => &slot.value_ptr.a,
            .b => &slot.value_ptr.b,
        };
        if (held.*) |other| return refuse(w, name, try std.fmt.allocPrint(arena, "duplicates {s}", .{other}));
        held.* = name;
    }
    if (pairs.count() == 0) return refuse(w, "adapter_model.safetensors", "names no tensors");
    var jobs: std.ArrayList(Job) = .empty;
    var checked: std.StringHashMapUnmanaged(void) = .empty;
    for (pairs.keys(), pairs.values()) |module, pair| {
        const a_name = pair.a orelse return refuse(w, module, "lora_A is missing");
        const b_name = pair.b orelse return refuse(w, module, "lora_B is missing");
        const names = try sh.resolve(arena, w, dense, shipped, module);
        const lay = try layoutOf(arena, w, shipped, quant, names.shipped);
        const shard = shipped.shardOf(try std.fmt.allocPrint(arena, "{s}.weight", .{names.shipped})).?;
        if (!(try checked.getOrPut(arena, shard)).found_existing) try checkDisjoint(w, shipped, shard);
        _ = try sh.denseTensor(w, dense, names.dense, lay.n, lay.k);
        const a = file.get(a_name).?;
        const b = file.get(b_name).?;
        _ = std.math.mul(usize, cfg.r, lay.k) catch return refuse(w, module, "adapter dimensions overflow");
        _ = std.math.mul(usize, lay.n, cfg.r) catch return refuse(w, module, "adapter dimensions overflow");
        if (a.rank != 2 or b.rank != 2 or a.dim(0) != cfg.r or a.dim(1) != lay.k or b.dim(0) != lay.n or b.dim(1) != cfg.r) return refuse(w, module, "adapter shapes do not fit the projection or r");
        for ([2]st.Tensor{ a, b }) |f| if (f.dtype != .f32 and f.dtype != .bf16 and f.dtype != .f16) return refuse(w, module, "adapter dtype is not f32, bf16 or f16");
        try jobs.append(arena, .{ .module = names.dense, .shipped = names.shipped, .a = a_name, .b = b_name, .n = lay.n, .k = lay.k, .group = lay.group, .dtype = lay.dtype });
    }
    return .{ .jobs = jobs.items, .r = cfg.r, .scale = cfg.scale, .adapter = file };
}

fn allFinite(v: []const f32) bool {
    for (v) |x| if (!std.math.isFinite(x)) return false;
    return true;
}

/// The adapter's factors for `j` as f32; `free` returns them.
pub fn factors(gpa: Allocator, w: *Io.Writer, p: *const Plan, j: Job) !struct { a: []f32, b: []f32 } {
    const a = try gpa.alloc(f32, p.r * j.k);
    errdefer gpa.free(a);
    const b = try gpa.alloc(f32, j.n * p.r);
    errdefer gpa.free(b);
    const ta = p.adapter.get(j.a).?;
    const tb = p.adapter.get(j.b).?;
    try lm.toF32(ta.dtype, ta.bytes, a);
    try lm.toF32(tb.dtype, tb.bytes, b);
    if (!allFinite(a) or !allFinite(b)) return refuse(w, j.module, "the adapter holds a non-finite value");
    return .{ .a = a, .b = b };
}
