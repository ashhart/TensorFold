//! codec_net.zig's decode and encode over a recording binding, every call in order against a pinned list.
const std = @import("std");
const net = @import("codec_net.zig");

const decode_calls = @embedFile("fixtures/codec_decode_calls.txt");
const encode_calls = @embedFile("fixtures/codec_encode_calls.txt");

/// A named address range, so the call list reads `dec.1.wa` instead of a number.
const Label = struct { base: u64, size: u64, name: []const u8 };

/// The binding both schedules call (allocator, kernels, clip buffers), writing one line a call.
const Rec = struct {
    gpa: std.mem.Allocator,
    labels: std.ArrayList(Label) = .empty,
    out: std.ArrayList(u8) = .empty,
    next: u64 = 1 << 20,
    tmp: u64 = 0,
    kvs: usize = 0,
    kv_buf: [8][2]u64 = undefined,

    fn deinit(r: *Rec) void {
        for (r.labels.items) |l| r.gpa.free(l.name);
        r.labels.deinit(r.gpa);
        r.out.deinit(r.gpa);
    }

    fn named(r: *Rec, bytes: usize, comptime fmt: []const u8, args: anytype) !u64 {
        const base = r.next;
        r.next += std.mem.alignForward(u64, @max(bytes, 16), 256);
        const name = try std.fmt.allocPrint(r.gpa, fmt, args);
        errdefer r.gpa.free(name);
        try r.labels.append(r.gpa, .{ .base = base, .size = @max(bytes, 16), .name = name });
        return base;
    }

    /// Decoder.alloc's allocator; `nameTree` names the buffers once the struct holds them.
    pub fn alloc(r: *Rec, bytes: usize) !u64 {
        return r.named(bytes, "?", .{});
    }

    fn rename(r: *Rec, base: u64, path: []const u8) !void {
        for (r.labels.items) |*l| if (l.base == base) {
            r.gpa.free(l.name);
            l.name = try r.gpa.dupe(u8, path);
            return;
        };
        return error.UnknownBuffer;
    }

    fn arg(r: *Rec, v: anytype) !void {
        const T = @TypeOf(v);
        if (T == u64) return r.addr(v);
        if (T == net.Stage) {
            try r.addr(v.buf);
            return r.out.print(r.gpa, "/{d}/{d}", .{ v.margin, v.cols });
        }
        if (T == [2]u64) return for (v) |a| try r.addr(a);
        if (T == *const net.Weights) return r.out.appendSlice(r.gpa, " w");
        if (T == []const u8) return r.out.print(r.gpa, " {s}", .{v});
        if (T == bool) return r.out.appendSlice(r.gpa, if (v) " true" else " false");
        return r.out.print(r.gpa, " {d}", .{v});
    }

    fn addr(r: *Rec, a: u64) !void {
        if (a == 0) return r.out.appendSlice(r.gpa, " 0");
        for (r.labels.items) |l| if (a >= l.base and a < l.base + l.size) {
            if (a == l.base) return r.out.print(r.gpa, " {s}", .{l.name});
            return r.out.print(r.gpa, " {s}+{d}", .{ l.name, a - l.base });
        };
        return error.UnknownAddress;
    }

    fn log(r: *Rec, comptime name: []const u8, args: anytype) !void {
        try r.out.appendSlice(r.gpa, name);
        inline for (args) |a| try r.arg(a);
        try r.out.append(r.gpa, '\n');
    }

    // the kernels, as codec_net.zig's schedules call them on each backend's Run
    pub fn sgemm(r: *Rec, w: u64, o: usize, cin: usize, x: u64, ldx: usize, y: u64, ldy: usize, t: usize, beta: f32) !void {
        try r.log("sgemm", .{ w, o, cin, x, ldx, y, ldy, t, beta });
    }
    pub fn bias(r: *Rec, y: u64, b: u64, rows: usize, n: usize, tanh: bool) !void {
        try r.log("bias", .{ y, b, rows, n, tanh });
    }
    pub fn snake(r: *Rec, x: u64, y: u64, rows: usize, cols: usize, alpha: u64) !void {
        try r.log("snake", .{ x, y, rows, cols, alpha });
    }
    pub fn copy(r: *Rec, src: u64, dst: u64, n: usize) !void {
        try r.log("copy", .{ src, dst, n });
    }
    pub fn conv(r: *Rec, st: net.Stage, rows: usize, taps: u64, kk: usize, dil: usize, o: usize, out: u64, b: u64, tanh: bool) !void {
        try r.log("conv", .{ st, rows, taps, kk, dil, o, out, b, tanh });
    }
    pub fn rmsnorm(r: *Rec, x: u64, w: u64, y: u64, rows: usize) !void {
        try r.log("rmsnorm", .{ x, w, y, rows });
    }
    pub fn addGamma(r: *Rec, x: u64, y: u64, gamma: u64, rows: usize, n: usize) !void {
        try r.log("addGamma", .{ x, y, gamma, rows, n });
    }
    pub fn ropeKv(r: *Rec, qkv: u64, cache: [2]u64, pos0: usize, ring: usize, rows: usize) !void {
        try r.log("ropeKv", .{ qkv, cache, pos0, ring, rows });
    }
    pub fn attn(r: *Rec, qkv: u64, cache: [2]u64, out: u64, pos0: usize, window: usize, ring: usize, rows: usize) !void {
        try r.log("attn", .{ qkv, cache, out, pos0, window, ring, rows });
    }
    pub fn siluMul(r: *Rec, x: u64, y: u64, rows: usize) !void {
        try r.log("siluMul", .{ x, y, rows });
    }
    pub fn dwconv7(r: *Rec, src: u64, dw: u64, dw_b: u64, out: u64, rows: usize) !void {
        try r.log("dwconv7", .{ src, dw, dw_b, out, rows });
    }
    pub fn layernorm(r: *Rec, x: u64, w: u64, b: u64, y: u64, rows: usize) !void {
        try r.log("layernorm", .{ x, w, b, y, rows });
    }
    pub fn gelu(r: *Rec, x: u64, n: usize) !void {
        try r.log("gelu", .{ x, n });
    }
    pub fn vqGather(r: *Rec, frames: u64, table: u64, out: u64, n: usize) !void {
        try r.log("vqGather", .{ frames, table, out, n });
    }
    pub fn vqEncode(r: *Rec, x: u64, w: *const net.Weights, codes_out: u64, nf: usize) !void {
        try r.log("vqEncode", .{ x, w, codes_out, nf });
    }

    // the clip's buffers, as net.encode calls each backend's Clip
    pub fn stage(r: *Rec, buf: u64, margin: usize, cols: usize) !net.Stage {
        try r.log("stage", .{ buf, margin, cols });
        return .{ .buf = buf, .margin = margin, .cols = cols };
    }
    pub fn kv(r: *Rec, layers: usize, rows: usize) ![]const [2]u64 {
        defer r.kvs += 1;
        for (r.kv_buf[0..layers], 0..) |*p, l| p.* = .{
            try r.named(rows * net.D * 4, "kv{d}.{d}.k", .{ r.kvs, l }),
            try r.named(rows * net.D * 4, "kv{d}.{d}.v", .{ r.kvs, l }),
        };
        try r.log("kv", .{ layers, rows });
        return r.kv_buf[0..layers];
    }
    pub fn codes(r: *Rec, nf: usize) !u64 {
        const p = try r.named(nf * 10 * 4, "codes", .{});
        try r.log("codes", .{nf});
        return p;
    }
    pub fn probe(r: *Rec, name: []const u8, ptr: u64, rows: usize, cols: usize) !void {
        try r.log("probe", .{ name, ptr, rows, cols });
    }
};

/// Every u64 under `v` (structs, arrays, slices, set optionals): a fresh named buffer, or (rename) its existing name.
fn nameTree(r: *Rec, comptime T: type, v: *T, comptime rename: bool, path: []const u8) !void {
    if (T == u64) {
        if (rename) return r.rename(v.*, path);
        v.* = try r.named(16, "{s}", .{path});
        return;
    }
    var nb: [96]u8 = undefined;
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names, s.field_types) |name, F| {
            if (comptime std.mem.eql(u8, name, "host_layers")) continue;
            const sep = if (path.len == 0) "" else ".";
            try nameTree(r, F, &@field(v, name), rename, try std.fmt.bufPrint(&nb, "{s}{s}{s}", .{ path, sep, name }));
        },
        .array => for (v, 0..) |*e, i| try nameTree(r, @TypeOf(e.*), e, rename, try std.fmt.bufPrint(&nb, "{s}.{d}", .{ path, i })),
        .pointer => |p| if (p.size == .slice) for (v.*, 0..) |*e, i| {
            try nameTree(r, p.child, e, rename, try std.fmt.bufPrint(&nb, "{s}.{d}", .{ path, i }));
        },
        .optional => |o| if (v.*) |*x| try nameTree(r, o.child, x, rename, path),
        else => {},
    }
}

/// Weights shaped as Weights.load shapes them (the encoder's transformer in block 3 alone), each tensor named.
fn weights(r: *Rec, layers: *[20]net.TfLayer) !net.Weights {
    var w: net.Weights = undefined;
    w.host_layers = .empty;
    w.bytes = 0;
    w.post.layers = layers[0..8];
    w.pre.layers = layers[8..16];
    for (&w.dec, 0..) |*B, bi| {
        B.cin = @as(usize, 1536) >> @intCast(bi);
        B.cout = B.cin / 2;
        B.s = ([4]usize{ 8, 8, 4, 2 })[bi];
    }
    for (&w.enc, 0..) |*B, bi| {
        B.dim = @as(usize, 128) << @intCast(bi);
        B.cin = B.dim / 2;
        B.s = ([4]usize{ 2, 4, 8, 8 })[bi];
        B.tf = if (bi == 3) .{ .layers = layers[16..20], .norm = 0 } else null;
    }
    try nameTree(r, net.Weights, &w, false, "");
    return w;
}

/// `got` against the pinned list; on a difference, the first differing line and the whole new list to copy.
fn expectCalls(pinned: []const u8, got: []const u8) !void {
    if (std.mem.eql(u8, pinned, got)) return;
    var a = std.mem.splitScalar(u8, pinned, '\n');
    var b = std.mem.splitScalar(u8, got, '\n');
    var line: usize = 1;
    while (true) : (line += 1) {
        const x = a.next() orelse "<end>";
        const y = b.next() orelse "<end>";
        if (!std.mem.eql(u8, x, y)) {
            std.debug.print("call {d}: pinned `{s}`, got `{s}`\n--- new list ---\n{s}--- end ---\n", .{ line, x, y, got });
            return error.ScheduleChanged;
        }
    }
}

test "codec schedule: decode's call list over two chunks is pinned" {
    var r: Rec = .{ .gpa = std.testing.allocator };
    defer r.deinit();
    var layers: [20]net.TfLayer = undefined;
    const w = try weights(&r, &layers);
    const o: net.Options = .{ .max_chunk = 2, .ring = 4 };
    var s = try net.Decoder.alloc(&w, o, &r);
    try nameTree(&r, net.Decoder, &s, true, "");
    r.tmp = s.carry_tmp;
    const frames = try r.named(3 * 11 * 4, "frames", .{});
    const audio = try r.named(3 * net.samples_per_frame * 4, "audio", .{});
    try net.decode(&r, &w, &s, o, frames, 3, audio);
    try std.testing.expectEqual(@as(usize, 3), s.pos);
    try expectCalls(decode_calls, r.out.items);
}

test "codec schedule: encode's call list is pinned" {
    var r: Rec = .{ .gpa = std.testing.allocator };
    defer r.deinit();
    var layers: [20]net.TfLayer = undefined;
    const w = try weights(&r, &layers);
    r.tmp = try r.named(16, "carry_tmp", .{});
    // cuda/codec.zig's clip buffers: the widest activation (2 frames x 2048 rows x 64) and the stages' margins
    const big = (2 * net.samples_per_frame * 64 + 64 * 1024) * 4;
    var sc: [4]u64 = undefined;
    for (&sc, 0..) |*p, i| p.* = try r.named(big, "sc{d}", .{i});
    const stbuf = try r.named(big, "stbuf", .{});
    try net.encode(&r, &r, &w, sc, stbuf, 2);
    try expectCalls(encode_calls, r.out.items);
}
