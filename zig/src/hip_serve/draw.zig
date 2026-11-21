//! Tokens from logits rows on the device: argmax per greedy row, top_k + MARGIN candidates per sampled row.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const sample = @import("sample.zig");

/// Most candidates a row downloads; wider requests (top_k 0, or past this) download the row and draw on the host.
pub const max_candidates = 512;

/// One row's draw: its sampling (null is greedy), the position that keys it, the grammar's allowed bits, a host copy.
pub const Request = struct {
    sampling: ?lanes.Sampling,
    position: u64,
    /// Structured output: only tokens whose bit is set (empty: every token).
    mask: []const u32 = &.{},
    /// The row's logits copied here (the prompt's last row, kept for a masked first draw).
    keep: ?[]u16 = null,
    /// The checkpoint's multiplicative penalty over prompt and reply tokens, applied once per seen token in FP32.
    repetition: f32 = 1,
    /// A checkpoint-forced final EOS, after its ordinary score processors.
    forced: ?u32 = null,
    history: []const u32 = &.{},
    /// OpenAI penalties count reply tokens only; history may also contain prompt tokens.
    reply: []const u32 = &.{},
    penalties: lanes.sampling.Penalties = .{},
    logit_bias: []const lanes.logit_bias.Entry = &.{},
};

/// Whether a row is drawn on the host from the whole row.
fn whole(r: Request) bool {
    return r.mask.len > 0 or r.keep != null or r.repetition != 1 or r.penalties.active() or r.logit_bias.len > 0 or r.forced != null;
}

fn greedy(s: ?lanes.Sampling) bool {
    return if (s) |x| x.temperature <= 0.0 else true;
}

/// Candidates a sampled row asks for, or 0 when it takes the whole row.
fn candidates(s: lanes.Sampling, vocab: usize) usize {
    if (s.top_k == 0) return 0;
    const n = @min(vocab, @as(usize, s.top_k) + sample.MARGIN);
    return if (n > max_candidates) 0 else n;
}

fn view(comptime T: type, bytes: []u8) []T {
    return std.mem.bytesAsSlice(T, @as([]align(@alignOf(T)) u8, @alignCast(bytes)));
}

pub const Drawer = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    dtype: sample.Dtype,
    vocab: usize,
    rows: usize,
    /// Per row: the candidate count, the argmax, then (rows, stride) candidate ids and value patterns.
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    row: hip.HostBuffer,

    /// Device bytes a drawer of `rows` rows holds: each row's argmax and its sampling candidates.
    pub fn deviceBytes(rows: usize) usize {
        return rows * (8 + max_candidates * 6);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, dtype: sample.Dtype, vocab: usize, rows: usize) !Drawer {
        const bytes = deviceBytes(rows);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        var dev = try hip.DeviceBuffer.alloc(d, bytes);
        errdefer dev.free();
        return .{ .gpa = gpa, .d = d, .dtype = dtype, .vocab = vocab, .rows = rows, .host = host, .dev = dev, .row = try hip.HostBuffer.alloc(d, vocab * 2) };
    }

    pub fn deinit(w: *Drawer) void {
        w.row.free();
        w.dev.free();
        w.host.free();
    }

    /// Row r's offsets into the scratch of `stride` candidates a row, as byte counts: ks, argmax, ids, values.
    fn layout(w: *const Drawer, stride: usize) struct { arg: usize, ids: usize, vals: usize, total: usize } {
        const arg = w.rows * 4;
        const ids = arg + w.rows * 4;
        const vals = ids + w.rows * stride * 4;
        return .{ .arg = arg, .ids = ids, .vals = vals, .total = vals + w.rows * stride * 2 };
    }

    /// Where the argmax of a logits block's first rows goes on the device (a graph can hold the launch).
    pub fn argmaxAt(w: *const Drawer) u64 {
        return w.dev.base() + w.layout(0).arg;
    }

    /// `out[r]` the token of row r of `logits` per `reqs[r]`, stream synchronized; `argmaxed`: argmax is at `argmaxAt`.
    pub fn draw(w: *Drawer, o: hip.ops.Ops, stream: hip.Stream, logits: hip.ops.Tensor, reqs: []const Request, out: []u32, argmaxed: bool) !void {
        const rows = reqs.len;
        if (rows > w.rows) return error.WindowTooWide;
        const ks = w.host.slice(i32)[0..rows];
        var stride: usize = 0;
        var any_greedy = false;
        for (reqs, ks) |r, *k| {
            k.* = 0;
            if (whole(r)) continue;
            if (greedy(r.sampling)) {
                any_greedy = true;
            } else {
                const n = candidates(r.sampling.?, w.vocab);
                k.* = @intCast(n);
                stride = @max(stride, n);
            }
        }
        const at = w.layout(stride);
        const base = w.dev.base();
        if (any_greedy and !argmaxed) try o.argmaxRows(logits, rows, w.vocab, base + at.arg);
        if (stride > 0) {
            try hip.raw.upload(w.dev, 0, std.mem.sliceAsBytes(ks), stream.handle);
            try o.topkRows(logits, rows, w.vocab, base, stride, base + at.ids, base + at.vals);
        }
        const host = w.host.bytes;
        if (any_greedy) try hip.raw.download(w.dev, at.arg, host[at.arg..][0 .. rows * 4], stream.handle);
        if (stride > 0) try hip.raw.download(w.dev, at.ids, host[at.ids..at.total], stream.handle);
        try stream.synchronize();
        const arg = view(i32, host[at.arg..][0 .. rows * 4]);
        for (reqs, 0..) |r, i| {
            if (whole(r)) {
                out[i] = try w.drawWhole(stream, logits, i, r);
                continue;
            }
            if (greedy(r.sampling)) {
                out[i] = @intCast(arg[i]);
                continue;
            }
            const s = r.sampling.?;
            const n: usize = @intCast(ks[i]);
            if (n == 0) {
                const bytes = w.vocab * 2;
                try hip.runtime.check(w.d.api.hipMemcpyDtoHAsync(w.row.bytes.ptr, @ptrFromInt(logits.ptr + i * bytes), bytes, stream.handle));
                try stream.synchronize();
                out[i] = try sample.draw(w.gpa, w.row.slice(u16)[0..w.vocab], w.dtype, s, r.position);
                continue;
            }
            const ids = view(i32, host[at.ids + i * stride * 4 ..][0 .. n * 4]);
            const vals = view(u16, host[at.vals + i * stride * 2 ..][0 .. n * 2]);
            var values: [max_candidates]f64 = undefined;
            var wide: [max_candidates]u64 = undefined;
            for (0..n) |j| {
                values[j] = sample.widen(vals, w.dtype, j);
                wide[j] = @intCast(ids[j]);
            }
            out[i] = @intCast(try lanes.sampling.choose(w.gpa, values[0..n], wide[0..n], r.position, s));
        }
    }

    /// Row `i` downloaded whole: kept if asked, masked if asked, then drawn as an unmasked whole row is.
    fn drawWhole(w: *Drawer, stream: hip.Stream, logits: hip.ops.Tensor, i: usize, r: Request) !u32 {
        const bytes = w.vocab * 2;
        try hip.runtime.check(w.d.api.hipMemcpyDtoHAsync(w.row.bytes.ptr, @ptrFromInt(logits.ptr + i * bytes), bytes, stream.handle));
        try stream.synchronize();
        const row = w.row.slice(u16)[0..w.vocab];
        if (r.keep) |k| @memcpy(k, row);
        return w.drawHost(row, r);
    }

    /// A host row's token per `r`: its mask applied in place, then greedy or the keyed draw.
    pub fn drawHost(w: *const Drawer, row: []u16, r: Request) !u32 {
        if (r.forced) |id| {
            if (id >= row.len) return error.BadForcedToken;
            if (lanes.logit_bias.find(r.logit_bias, id)) |b| if (b.ban) return error.NoAllowedToken;
            if (r.mask.len > 0 and (id / 32 >= r.mask.len or (r.mask[id / 32] >> @intCast(id % 32)) & 1 == 0)) return error.NoAllowedToken;
            return id;
        }
        if (r.mask.len > 0) sample.mask(row, w.dtype, r.mask);
        if (r.repetition == 1 and !r.penalties.active() and r.logit_bias.len == 0) return sample.draw(w.gpa, row, w.dtype, r.sampling, r.position);
        const values = try w.gpa.alloc(f32, row.len);
        defer w.gpa.free(values);
        for (values, 0..) |*v, i| v.* = sample.widen(row, w.dtype, i);
        try process(w.gpa, values, r);
        return sample.drawWide(w.gpa, values, r.sampling, r.position);
    }
};

/// HF repetition processing widens raw scores before FP32 multiply/divide; it must not round back to BF16.
pub fn process(a: std.mem.Allocator, values: []f32, r: Request) !void {
    if (!(r.repetition > 0) or !std.math.isFinite(r.repetition)) return error.BadRepetitionPenalty;
    const counts = try a.alloc(u32, values.len);
    defer a.free(counts);
    @memset(counts, 0);
    if (r.repetition != 1) {
        for (r.history) |id| {
            if (id >= values.len) return error.BadHistoryToken;
            counts[id] = 1;
        }
        for (values, counts) |*v, seen| if (seen != 0) {
            v.* = if (v.* < 0) v.* * r.repetition else v.* / r.repetition;
        };
    }
    if (r.penalties.active()) {
        @memset(counts, 0);
        for (r.reply) |id| {
            if (id >= values.len) return error.BadHistoryToken;
            counts[id] += 1;
        }
        for (values, counts) |*v, count| if (count != 0) {
            v.* -= r.penalties.of(count);
        };
    }
    try lanes.logit_bias.apply(values, r.logit_bias);
    if (r.logit_bias.len > 0) {
        var allowed = false;
        for (values) |v| if (std.math.isFinite(v)) {
            allowed = true;
            break;
        };
        if (!allowed) return error.NoAllowedToken;
    }
}

test "repetition keeps FP32 ranking and counts prompt tokens once, OpenAI penalties count only reply tokens" {
    var values = [_]f32{ 1.5, 1.3671875, -2, 4 };
    try process(std.testing.allocator, &values, .{ .sampling = null, .position = 0, .repetition = 1.1, .history = &.{ 0, 0, 2 }, .reply = &.{ 3, 3 }, .penalties = .{ .presence = 1, .frequency = 0.5 } });
    try std.testing.expectEqual(@as(f32, 1.5) / @as(f32, 1.1), values[0]);
    try std.testing.expectEqual(@as(f32, -2) * @as(f32, 1.1), values[2]);
    try std.testing.expectEqual(@as(f32, 2), values[3]);
    try std.testing.expectEqual(@as(u32, 1), try sample.drawWide(std.testing.allocator, values[0..2], null, 0));
    try std.testing.expectError(error.BadHistoryToken, process(std.testing.allocator, &values, .{ .sampling = null, .position = 0, .repetition = 1.1, .history = &.{4} }));
}

test "logit_bias draws ban a token at every seed and shift greedy picks without changing a solo row" {
    const a = std.testing.allocator;
    const drawer: Drawer = .{ .gpa = a, .d = undefined, .dtype = .bf16, .vocab = 3, .rows = 1, .host = undefined, .dev = undefined, .row = undefined };
    var words = [_]u16{ 0x4120, 0x4110, 0 };
    const bias = [_]lanes.logit_bias.Entry{ .{ .token = 0, .value = -100, .ban = true }, .{ .token = 2, .value = 20 } };
    try std.testing.expectEqual(@as(u32, 2), try drawer.drawHost(&words, .{ .sampling = null, .position = 0, .logit_bias = &bias }));
    for (0..128) |seed| {
        try std.testing.expect(try drawer.drawHost(&words, .{ .sampling = .{ .seed = seed, .top_k = 0, .top_p = 1 }, .position = seed, .logit_bias = &bias }) != 0);
    }
    try std.testing.expectEqual(@as(u32, 0), try drawer.drawHost(&words, .{ .sampling = null, .position = 0 }));
    try std.testing.expectError(error.NoAllowedToken, drawer.drawHost(&words, .{ .sampling = null, .position = 0, .logit_bias = &bias, .forced = 0 }));
}
