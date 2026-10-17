//! Independent FP64 fixtures at the routed 35B geometry and its MTP head's served layouts.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
pub const Mode = enum { router, topk, combine, norm, attention, tree };
pub const Arg = union(enum) { buffer: usize, int: c_int, long: c_longlong, float: f32 };
pub const Elem = enum { f32, bf16, i32 };
pub const Buffer = struct { bytes: []const u8, output: bool = false };
pub const Result = struct {
    buffer: usize,
    elem: Elem,
    want: []f64,
    bound: []f64,
    pub fn accepts(self: Result, bytes: []const u8) bool {
        const width: usize = if (self.elem == .bf16) 2 else 4;
        if (bytes.len != width * self.want.len) return false;
        for (self.want, self.bound, 0..) |w, b, i| {
            const got: f64 = switch (self.elem) {
                .bf16 => c.read(.bf16, bytes, i),
                .f32 => c.read(.f32, bytes, i),
                .i32 => @floatFromInt(std.mem.readInt(i32, bytes[4 * i ..][0..4], .little)),
            };
            if (!std.math.isFinite(got) or @abs(got - w) > b) return false;
        }
        return true;
    }
};
pub const Fixture = struct {
    a: std.mem.Allocator,
    mode: Mode,
    grid: [3]u32,
    threads: u32,
    buffers: std.ArrayList(Buffer) = .empty,
    args: std.ArrayList(Arg) = .empty,
    results: std.ArrayList(Result) = .empty,
    fn input(f: *Fixture, values: anytype) !usize {
        const at = f.buffers.items.len;
        try f.buffers.append(f.a, .{ .bytes = std.mem.sliceAsBytes(values) });
        return at;
    }
    fn output(f: *Fixture, elem: Elem, n: usize) !usize {
        const at = f.buffers.items.len;
        const bytes = try f.a.alloc(u8, n * @as(usize, if (elem == .bf16) 2 else 4));
        @memset(bytes, 0xff);
        try f.buffers.append(f.a, .{ .bytes = bytes, .output = true });
        try f.results.append(f.a, .{ .buffer = at, .elem = elem, .want = try f.a.alloc(f64, n), .bound = try f.a.alloc(f64, n) });
        return at;
    }
    fn ptr(f: *Fixture, at: usize) !void {
        try f.args.append(f.a, .{ .buffer = at });
    }
    fn int(f: *Fixture, v: usize) !void {
        try f.args.append(f.a, .{ .int = @intCast(v) });
    }
    fn long(f: *Fixture, v: usize) !void {
        try f.args.append(f.a, .{ .long = @intCast(v) });
    }
    fn float(f: *Fixture, v: f32) !void {
        try f.args.append(f.a, .{ .float = v });
    }
};
pub fn symbol(mode: Mode) [:0]const u8 {
    return switch (mode) {
        .router => "tf_moe_router",
        .topk => "tf_moe_topk",
        .combine => "tf_moe_combine",
        .norm => "tf_q36m_norm_rope",
        .attention => "tf_q36m_attn",
        .tree => "tf_q36m_attn_tree",
    };
}
fn bf(v: f64) u16 {
    return c.toBf16(@floatCast(v));
}
fn val(v: u16) f64 {
    return c.fromBf16(v);
}
fn data(a: std.mem.Allocator, n: usize, salt: usize) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out, 0..) |*v, i| v.* = bf(@as(f64, @floatFromInt((i * 13 + salt) % 23)) / 32 - 0.25);
    return out;
}
fn linearBound(depth: usize, mag: f64, w: f64, output: Elem) f64 {
    const work = c.gamma(@floatFromInt(depth), c.unit(.f32)) * mag;
    return work + (if (output == .bf16) c.unit(.bf16) else c.unit(.f32)) * (@abs(w) + work);
}
pub fn make(a: std.mem.Allocator, mode: Mode, rows: usize, variant: bool) !Fixture {
    var f = Fixture{ .a = a, .mode = mode, .grid = .{ @intCast(rows), 1, 1 }, .threads = 32 };
    switch (mode) {
        .router => try router(&f, rows, variant),
        .topk => try topk(&f, rows),
        .combine => try combine(&f, rows, variant),
        .norm => try norm(&f, rows, variant),
        .attention, .tree => try attention(&f, rows, variant),
    }
    return f;
}
fn router(f: *Fixture, rows: usize, padded: bool) !void {
    const dims = 2048;
    const experts = 257;
    const stride = dims + @as(usize, if (padded) 16 else 0);
    const x = try data(f.a, rows * stride, 7);
    const w = try data(f.a, experts * dims, 11);
    const xp = try f.input(x);
    const wp = try f.input(w);
    const op = try f.output(.f32, rows * experts);
    const result = f.results.items[0];
    for (0..rows) |r| for (0..experts) |e| {
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..dims) |d| {
            const term = val(x[r * stride + d]) * val(w[e * dims + d]);
            sum += term;
            mag += @abs(term);
        }
        const at = r * experts + e;
        result.want[at] = sum;
        result.bound[at] = linearBound(dims, mag, sum, .f32);
    };
    try f.ptr(xp);
    try f.int(stride);
    try f.ptr(wp);
    try f.ptr(op);
    try f.int(rows);
    try f.int(dims);
    try f.int(experts);
    f.grid = .{ @intCast((rows + 15) / 16), 17, 1 };
    f.threads = 256;
}
fn topk(f: *Fixture, rows: usize) !void {
    const logits = try f.a.alloc(f32, rows * 257);
    for (logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 17) % 31)) / 8 - 2;
    const lp = try f.input(logits);
    const ip = try f.output(.i32, rows * 9);
    const wp = try f.output(.f32, rows * 9);
    const ids = f.results.items[0];
    const weights = f.results.items[1];
    for (0..rows) |r| {
        var order: [256]usize = undefined;
        for (&order, 0..) |*v, i| v.* = i;
        const Sort = struct {
            fn less(l: []const f32, x: usize, y: usize) bool {
                return l[x] > l[y] or (l[x] == l[y] and x < y);
            }
        };
        const l: []const f32 = logits[r * 257 ..][0..257];
        std.mem.sort(usize, &order, l, Sort.less);
        var total: f64 = 0;
        for (order[0..8]) |id| total += @exp(@as(f64, l[id]) - l[order[0]]);
        for (order[0..8], 0..) |id, k| {
            const at = r * 9 + k;
            ids.want[at] = @floatFromInt(id);
            ids.bound[at] = 0;
            const weight = @exp(@as(f64, l[id]) - l[order[0]]) / total;
            weights.want[at] = weight;
            weights.bound[at] = (c.unit(.bf16) + 32 * c.unit(.f32)) * weight;
        }
        const at = r * 9 + 8;
        ids.want[at] = 256;
        ids.bound[at] = 0;
        const weight = 1 / (1 + @exp(-val(bf(l[256]))));
        weights.want[at] = weight;
        weights.bound[at] = (c.unit(.bf16) + 16 * c.unit(.f32)) * weight;
    }
    try f.ptr(lp);
    try f.ptr(ip);
    try f.ptr(wp);
    try f.int(256);
    try f.int(8);
}
fn combine(f: *Fixture, rows: usize, fp32: bool) !void {
    const dims = 2048;
    const slots = 9;
    const y = try data(f.a, rows * slots * dims, 5);
    const yf = try f.a.alloc(f32, y.len);
    for (yf, y, 0..) |*v, bits, i| v.* = @floatCast(val(bits) + @as(f64, @floatFromInt(i % 7)) / 4096);
    const weights = try f.a.alloc(f32, rows * slots);
    for (weights, 0..) |*v, i| v.* = @floatCast(val(bf(@as(f64, @floatFromInt(1 + i % slots)) / 16)));
    const yp = if (fp32) try f.input(yf) else try f.input(y);
    const wp = try f.input(weights);
    const op = try f.output(.bf16, rows * dims);
    const result = f.results.items[0];
    for (0..rows) |r| for (0..dims) |d| {
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..slots) |s| {
            const yi = (r * slots + s) * dims + d;
            const term = @as(f64, weights[r * slots + s]) * (if (fp32) @as(f64, yf[yi]) else val(y[yi]));
            sum += term;
            mag += @abs(term);
        }
        const at = r * dims + d;
        result.want[at] = sum;
        result.bound[at] = linearBound(slots, mag, sum, .bf16);
    };
    try f.ptr(yp);
    try f.int(@intFromBool(fp32));
    try f.ptr(wp);
    try f.ptr(op);
    try f.int(slots);
    try f.int(dims);
    f.grid[1] = 8;
    f.threads = 256;
}
fn norm(f: *Fixture, rows: usize, query: bool) !void {
    const heads: usize = if (query) 16 else 2;
    const in_head: usize = if (query) 512 else 256;
    const stride = heads * in_head;
    const x = try data(f.a, rows * stride, 19);
    const w = try data(f.a, 256, 9);
    const xp = try f.input(x);
    const wp = try f.input(w);
    const op = try f.output(.bf16, rows * heads * 256);
    const result = f.results.items[0];
    const pos0: usize = if (query) 131071 else 7;
    const step: usize = if (query) 0 else 1;
    for (0..rows) |r| for (0..heads) |h| {
        const row = x[r * stride + h * in_head ..][0..256];
        var ss: f64 = 0;
        for (row) |v| ss += val(v) * val(v);
        const inv = 1 / @sqrt(ss / 256 + 1e-6);
        var n: [256]f64 = undefined;
        for (&n, row, w) |*v, bits, weight| v.* = val(bits) * inv * val(weight);
        for (n, 0..) |v, d| {
            var want = v;
            var mag = @abs(v);
            if (d < 64) {
                const j = d % 32;
                const partner = n[(d + 32) % 64];
                const angle = @as(f64, @floatFromInt(pos0 + r * step)) * std.math.pow(f64, 10000000, -@as(f64, @floatFromInt(j)) / 32);
                want = v * @cos(angle) + (if (d < 32) -partner else partner) * @sin(angle);
                mag += @abs(partner);
            }
            const at = (r * heads + h) * 256 + d;
            result.want[at] = want;
            // Two bf16 roundings; large-position powf/sincosf angle error is bounded separately by its input scale.
            const angle_error: f64 = if (d < 64) @as(f64, @floatFromInt(pos0 + r * step)) * 8 * c.unit(.f32) else 0;
            result.bound[at] = (2 * c.unit(.bf16) + 64 * c.unit(.f32) + angle_error) * mag + 0x1p-126;
        }
    };
    try f.ptr(xp);
    try f.ptr(wp);
    try f.ptr(op);
    try f.int(heads);
    try f.int(stride);
    try f.int(in_head);
    try f.long(pos0);
    try f.int(step);
    try f.float(1e-6);
    try f.float(10000000);
    f.grid[0] = @intCast(rows * heads);
}
fn attention(f: *Fixture, rows: usize, wrapped: bool) !void {
    const heads = 16;
    const kv = 2;
    const width = kv * 256;
    const capacity: usize = if (wrapped) 32768 else 7;
    const begin: usize = if (wrapped) capacity - 3 else 0;
    const end = begin + 6;
    const q = try data(f.a, rows * heads * 256, 3);
    const gate = try data(f.a, rows * heads * 512, 8);
    const kb = try data(f.a, rows * width, 10);
    const vb = try data(f.a, rows * width, 17);
    const rk = try data(f.a, capacity * width, 13);
    const rv = try data(f.a, capacity * width, 15);
    const lk = try data(f.a, 64 * width, 2);
    const lv = try data(f.a, 64 * width, 21);
    const anc = try f.a.alloc(i32, rows * 65);
    @memset(anc, -1);
    for (0..rows) |r| {
        anc[r * 65] = 2;
        anc[r * 65 + 1] = 3;
        anc[r * 65 + 2] = 1;
    }
    const qp = try f.input(q);
    const gp = try f.input(gate);
    const kp = try f.input(kb);
    const vp = try f.input(vb);
    const lkp = try f.input(lk);
    const lvp = try f.input(lv);
    const ap = try f.input(anc);
    const rkp = try f.input(rk);
    const rvp = try f.input(rv);
    const op = try f.output(.bf16, rows * heads * 256);
    const result = f.results.items[0];
    for (0..rows) |r| for (0..heads) |h| {
        const kvh = h / (heads / kv);
        const count = 6 + (if (f.mode == .tree) @as(usize, 3) else r + 1);
        const scores = try f.a.alloc(f64, count);
        const values = try f.a.alloc([]const u16, count);
        for (scores, values, 0..) |*score, *value, j| {
            var key: []const u16 = undefined;
            if (j < 6) {
                const at = ((begin + j) % capacity) * width + kvh * 256;
                key = rk[at..][0..256];
                value.* = rv[at..][0..256];
            } else if (f.mode == .tree and j < 8) {
                const at = @as(usize, @intCast(anc[r * 65 + 1 + j - 6])) * width + kvh * 256;
                key = lk[at..][0..256];
                value.* = lv[at..][0..256];
            } else {
                const at = (if (f.mode == .tree) r else j - 6) * width + kvh * 256;
                key = kb[at..][0..256];
                value.* = vb[at..][0..256];
            }
            var dot: f64 = 0;
            for (q[(r * heads + h) * 256 ..][0..256], key) |x, k| dot += val(x) * val(k);
            score.* = dot / 16;
        }
        const maximum = std.mem.max(f64, scores);
        var total: f64 = 0;
        for (scores) |*s| {
            s.* = @exp(s.* - maximum);
            total += s.*;
        }
        for (0..256) |d| {
            var sum: f64 = 0;
            var mag: f64 = 0;
            for (scores, values) |s, v| {
                sum += s * val(v[d]);
                mag = @max(mag, @abs(val(v[d])));
            }
            const g = 1 / (1 + @exp(-val(gate[r * heads * 512 + h * 512 + 256 + d])));
            const at = (r * heads + h) * 256 + d;
            result.want[at] = sum / total * g;
            result.bound[at] = (2 * c.unit(.bf16) + c.gamma(@floatFromInt(256 + 8 * count), c.unit(.f32))) * mag;
        }
    };
    try f.ptr(qp);
    try f.ptr(gp);
    try f.int(heads * 512);
    try f.ptr(kp);
    try f.ptr(vp);
    if (f.mode == .tree) {
        try f.ptr(lkp);
        try f.ptr(lvp);
        try f.ptr(ap);
        try f.int(65);
    }
    try f.ptr(rkp);
    try f.ptr(rvp);
    try f.int(heads);
    try f.int(kv);
    try f.int(capacity);
    try f.long(begin);
    try f.long(end);
    try f.ptr(op);
    f.grid[1] = heads;
}
test "FP64 cases reject poisoned, finite-corrupted and missing served outputs" {
    for (std.enums.values(Mode)) |mode| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const f = try make(arena.allocator(), mode, 1, true);
        for (f.results.items) |result| {
            const bytes = try arena.allocator().dupe(u8, f.buffers.items[result.buffer].bytes);
            try std.testing.expect(!result.accepts(bytes));
            const width: usize = if (result.elem == .bf16) 2 else 4;
            for (result.want, 0..) |want, i| switch (result.elem) {
                .bf16 => std.mem.writeInt(u16, bytes[i * width ..][0..2], bf(want), .little),
                .f32 => std.mem.writeInt(u32, bytes[i * width ..][0..4], @bitCast(@as(f32, @floatCast(want))), .little),
                .i32 => std.mem.writeInt(i32, bytes[i * width ..][0..4], @intFromFloat(want), .little),
            };
            try std.testing.expect(result.accepts(bytes));
            @memset(bytes[0..width], 0x40);
            try std.testing.expect(!result.accepts(bytes));
            try std.testing.expect(!result.accepts(bytes[0 .. bytes.len - 1]));
        }
    }
}
test "top-k tie breaks to lower IDs and shared sigmoid is independent" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const f = try make(arena.allocator(), .topk, 2, false);
    try std.testing.expectEqual(@as(f64, 20), f.results.items[0].want[0]);
    try std.testing.expectEqual(@as(f64, 256), f.results.items[0].want[8]);
}
