//! The DSpark drafter's tensors (checkpoint mtp.0 .. mtp.N-1) on Metal: its stages, main_proj and norms, the heads.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const weights = @import("weights.zig");

const Config = host.config.Config;
const Pack = host.pack.Pack;
const Ref = weights.Ref;

/// What the drafter reads beside its blocks: main_proj (FP8, all taps of a row to the hidden size), the two norms.
pub const Weights = struct {
    a: std.mem.Allocator,
    /// one weights.Layer a stage, layer numbers `layers` ..
    stages: []weights.Layer,
    arena: mtl.Buffer,
    main_proj: weights.Fp8,
    /// fp32, as every norm on the device
    main_norm: Ref,
    norm: Ref,
    /// the Markov head's tables bf16 [vocab][rank]: embed (what the previous token selects), head (the candidate's)
    markov_embed: []u16,
    markov_head: []u16,
    /// the confidence head's projection [hidden + rank] widened to fp32, null when the release has none
    conf: ?[]f32,

    pub fn deinit(w: *Weights) void {
        for (w.stages) |*s| s.deinit();
        w.a.free(w.stages);
        w.arena.deinit();
        w.a.free(w.markov_embed);
        w.a.free(w.markov_head);
        if (w.conf) |v| w.a.free(v);
    }

    /// The host half of the heads as the family's chain takes them.
    pub fn markov(w: *const Weights, c: *const Config) host.dspark.Markov {
        return .{ .w1 = w.markov_embed, .w2 = w.markov_head, .rank = c.dspark_markov_rank, .conf = w.conf };
    }
};

fn blockOf(i: u32, buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "mtp.{d}", .{i});
}

fn spec(c: *const Config, suffix: []const u8) !host.names.NativeSpec {
    const n = try host.names.dsparkNatives(c);
    for (n.items()) |s| if (std.mem.eql(u8, s.suffix, suffix)) return s;
    return error.BadNativeTensor;
}

/// A bf16 (or fp16, rounded once) table of `count` values from the pack, as bf16 bits.
fn readHalf(a: std.mem.Allocator, io: std.Io, pack: *const Pack, name: []const u8, dims: []const usize) ![]u16 {
    const info = try pack.need(name);
    if (!(info.is(.bf16, dims) or info.is(.f16, dims))) return error.BadNativeTensor;
    var count: usize = 1;
    for (dims) |d| count *= d;
    const out = try a.alloc(u16, count);
    errdefer a.free(out);
    if (info.dtype == .bf16) {
        try pack.read(io, info, .{ .offset = 0, .len = info.nbytes }, std.mem.sliceAsBytes(out));
        return out;
    }
    const raw = try a.alloc(u16, count);
    defer a.free(raw);
    try pack.read(io, info, .{ .offset = 0, .len = info.nbytes }, std.mem.sliceAsBytes(raw));
    for (out, raw) |*o, r| o.* = weights.bf16Round(@floatCast(@as(f16, @bitCast(r))));
    return out;
}

/// The confidence head's projection as fp32 (bf16, fp16 or fp32 in the release), null when absent.
fn readConf(a: std.mem.Allocator, io: std.Io, pack: *const Pack, name: []const u8, n: usize) !?[]f32 {
    const info = pack.get(name) orelse return null;
    var count: usize = 1;
    for (info.dims()) |d| count *= d;
    if (count != n) return error.BadNativeTensor;
    const raw = try a.alloc(u8, @intCast(info.nbytes));
    defer a.free(raw);
    try pack.read(io, info, .{ .offset = 0, .len = info.nbytes }, raw);
    const out = try a.alloc(f32, n);
    errdefer a.free(out);
    for (out, 0..) |*o, i| o.* = switch (info.dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[2 * i ..][0..2], .little)))),
        else => return error.BadNativeTensor,
    };
    return out;
}

/// Every DSpark block and head of `pack`; the stages are the release's mtp.i in order.
pub fn load(a: std.mem.Allocator, io: std.Io, device: mtl.Device, pack: *const Pack, c: *const Config) !Weights {
    if (c.mtp_layers == 0 or c.dspark_targets.len == 0) return error.NoDspark;
    var w: Weights = undefined;
    w.a = a;
    w.stages = try a.alloc(weights.Layer, c.mtp_layers);
    errdefer a.free(w.stages);
    var made: usize = 0;
    errdefer for (w.stages[0..made]) |*s| s.deinit();
    for (w.stages, 0..) |*s, i| {
        s.* = try weights.load(a, io, device, pack, c, c.layers + @as(u32, @intCast(i)));
        made = i + 1;
    }
    var l: weights.Loader = .{ .a = a, .pack = pack };
    defer l.items.deinit(a);
    var first: [32]u8 = undefined;
    var last: [32]u8 = undefined;
    const b0 = try blockOf(0, &first);
    const bl = try blockOf(c.mtp_layers - 1, &last);
    const proj = try l.fp8(b0, host.names.dspark_main_proj, @as(u32, @intCast(c.dspark_targets.len)) * c.hidden, c.hidden);
    const mn = try l.native(b0, try spec(c, "main_norm.weight"));
    const nn = try l.native(bl, try spec(c, "norm.weight"));
    const arena = try device.buffer(@max(l.size, 256), mtl.ResourceOptions.shared);
    errdefer arena.deinit();
    try l.fill(io, arena.contents()[0..l.size]);
    w.arena = arena;
    w.main_proj = weights.fix(arena, proj);
    w.main_norm = mn.ref(arena);
    w.norm = nn.ref(arena);
    var nb: [96]u8 = undefined;
    const dims = [2]usize{ c.vocab, c.dspark_markov_rank };
    w.markov_embed = try readHalf(a, io, pack, try std.fmt.bufPrint(&nb, "{s}.markov_head.embed.weight", .{bl}), &dims);
    errdefer a.free(w.markov_embed);
    w.markov_head = try readHalf(a, io, pack, try std.fmt.bufPrint(&nb, "{s}.markov_head.head.weight", .{bl}), &dims);
    errdefer a.free(w.markov_head);
    w.conf = try readConf(a, io, pack, try std.fmt.bufPrint(&nb, "{s}.{s}", .{ bl, host.names.dspark_confidence }), c.hidden + c.dspark_markov_rank);
    return w;
}
