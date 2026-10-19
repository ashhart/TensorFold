//! Living Weights' pocket beside each dense MLP: fp32 SwiGLU neurons a layer, added to the MLP's output in every pass.
const std = @import("std");
const cuda = @import("cuda");
const to = @import("cuda_train_ops.zig");

/// A layer's neurons on the device: fp32 [cap, hidden] gate rows, up rows and outputs; the first `used` live.
pub const Rows = struct { gate: cuda.DeviceBuffer, up: cuda.DeviceBuffer, down: cuda.DeviceBuffer, cap: usize, used: usize = 0 };

/// What a taught folder holds a layer: `n` neurons' gate rows, up rows and outputs, fp32 [n, hidden] each.
pub const Host = struct { gate: []f32, up: []f32, down: []f32, n: usize };

/// Every layer's pocket, and the kernel that adds it (loaded with the first).
pub const Pockets = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    hidden: usize,
    layers: []?Rows = &.{},
    k: ?to.PocketKernels = null,
    rows: ?cuda.DeviceBuffer = null, // f32 a row of a call, for an untaught stream's zeros (made with the first neuron)

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, hidden: usize, layers: usize) !Pockets {
        const slots = try gpa.alloc(?Rows, layers);
        @memset(slots, null);
        return .{ .gpa = gpa, .d = d, .hidden = hidden, .layers = slots };
    }

    pub fn deinit(p: *Pockets) void {
        for (p.layers) |*slot| if (slot.*) |*r| {
            r.gate.free();
            r.up.free();
            r.down.free();
            slot.* = null;
        };
        if (p.k) |*k| k.deinit();
        if (p.rows) |*b| b.free();
        p.gpa.free(p.layers);
        p.* = undefined;
    }

    /// Layer `layer`'s neurons (fp32 [n, hidden] gate, up, down rows; the first `used` live), after the stream's work.
    pub fn set(p: *Pockets, s: cuda.Stream, layer: usize, gate: []const f32, up: []const f32, down: []const f32, used: usize) !void {
        if (layer >= p.layers.len or p.hidden == 0 or gate.len % p.hidden != 0) return error.UnexpectedTensor;
        const n = gate.len / p.hidden;
        if (n == 0 or n > to.max_pocket or up.len != gate.len or down.len != gate.len or used > n) return error.UnexpectedTensor;
        if (p.k == null) p.k = try to.PocketKernels.load(p.d);
        if (p.rows == null) p.rows = try cuda.DeviceBuffer.alloc(p.d, 4096 * 4);
        if (p.layers[layer]) |r| if (r.cap != n) return error.PocketOfAnotherSize;
        try s.synchronize();
        var g = try staged(p.d, gate);
        errdefer g.free();
        var u = try staged(p.d, up);
        errdefer u.free();
        const down_buffer = try staged(p.d, down);
        var prior = p.layers[layer];
        p.layers[layer] = .{ .gate = g, .up = u, .down = down_buffer, .cap = n, .used = used };
        if (prior) |*old| {
            old.gate.free();
            old.up.free();
            old.down.free();
        }
    }

    /// Download failure stays an error; a loaded fact can never become an apparently empty pocket.
    pub fn read(p: *const Pockets, s: cuda.Stream, layer: usize, gate: []f32, up: []f32, down: []f32) !usize {
        if (layer >= p.layers.len) return error.UnexpectedTensor;
        const r = p.layers[layer] orelse return 0;
        if (gate.len != r.cap * p.hidden or up.len != gate.len or down.len != gate.len) return error.PocketOfAnotherSize;
        try s.synchronize();
        try r.gate.download(0, std.mem.sliceAsBytes(gate));
        try r.up.download(0, std.mem.sliceAsBytes(up));
        try r.down.download(0, std.mem.sliceAsBytes(down));
        return r.used;
    }

    pub fn uploadScales(p: *Pockets, s: cuda.Stream, scales: []const f32) !void {
        if (scales.len > 4096) return error.UnexpectedTensor;
        try s.synchronize();
        const next = try staged(p.d, scales);
        if (p.rows) |*old| old.free();
        p.rows = next;
    }

    /// Layer `layer`'s live neurons as the kernels take them, or null.
    pub fn of(p: *const Pockets, layer: usize) ?to.Pocket {
        const r = p.layers[layer] orelse return null;
        if (r.used == 0) return null;
        return .{ .gate = r.gate.ptr, .up = r.up.ptr, .down = r.down.ptr, .n = r.used };
    }

    /// Room for a call's row scales (0 before the first neuron).
    pub fn rowsBuffer(p: *const Pockets) u64 {
        return if (p.rows) |b| b.ptr else 0;
    }

    /// Whether any layer has a live neuron.
    pub fn any(p: *const Pockets) bool {
        for (p.layers) |slot| if (slot) |r| if (r.used > 0) return true;
        return false;
    }

    /// out [rows, hidden] bf16 += layer `layer`'s pocket on y [rows, hidden] bf16 (nothing without one).
    pub fn add(p: *const Pockets, s: cuda.Stream, layer: usize, y: u64, out: u64, rows: usize, taught: u64) !void {
        const pk = p.of(layer) orelse return;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(@as(c_int, @intCast(p.hidden)));
        for ([_]u64{ pk.gate, pk.up, pk.down }) |v| a.add(v);
        a.add(@as(c_int, @intCast(pk.n)));
        a.add(out);
        a.add(taught);
        a.add(@as(c_int, 0));
        try cuda.launch.launch(p.k.?.rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// A taught folder's pockets as weights.zig read them, each up to its last neuron with an up row; how many live.
    pub fn loadAll(p: *Pockets, s: cuda.Stream, host: []const ?Host) !usize {
        var learned: usize = 0;
        var off: std.ArrayList([2]usize) = .empty;
        defer off.deinit(p.gpa);
        if (@import("sys").c.getenv("TF_Q27_POCKET_OFF")) |list| {
            const all = std.mem.eql(u8, std.mem.span(list), "all");
            if (all) for (host, 0..) |slot, l| if (slot) |h| for (0..h.n) |j| try off.append(p.gpa, .{ l, j });
            var it = std.mem.tokenizeAny(u8, if (all) "" else std.mem.span(list), ", ");
            while (it.next()) |pair| {
                const c = std.mem.indexOfScalar(u8, pair, ':') orelse return error.UnexpectedTensor;
                try off.append(p.gpa, .{ try std.fmt.parseInt(usize, pair[0..c], 10), try std.fmt.parseInt(usize, pair[c + 1 ..], 10) });
            }
            std.log.info("qwen27: pocket diagnostic: {d} neurons switched off as loaded", .{off.items.len});
        }
        for (host, 0..) |slot, l| if (slot) |h| {
            var used: usize = 0;
            for (0..h.n) |j| if (std.mem.indexOfNone(f32, h.up[j * p.hidden ..][0..p.hidden], &.{0}) != null) {
                used = j + 1;
            };
            for (off.items) |o| if (o[0] == l and o[1] < h.n) @memset(h.down[o[1] * p.hidden ..][0..p.hidden], 0);
            try p.set(s, l, h.gate, h.up, h.down, used);
            learned += used;
        };
        return learned;
    }
};

/// A diagnostic (TF_Q27_POCKET_DECOUPLE): every neuron's output projected off what each neuron at a later layer reads.
pub fn decouple(gpa: std.mem.Allocator, hidden: usize, host: []const ?Host, norms: []const []const f32) !f64 {
    const D = hidden;
    var basis: std.ArrayList(f64) = .empty; // orthonormal rows, the read directions of every layer above
    defer basis.deinit(gpa);
    const r = try gpa.alloc(f64, D);
    defer gpa.free(r);
    var before: f64 = 0;
    var removed: f64 = 0;
    var l = host.len;
    while (l > 0) {
        l -= 1;
        const h = host[l] orelse continue;
        const rows = basis.items.len / D;
        // this layer's outputs off the read directions of the layers above
        for (0..h.n) |j| {
            const g = h.gate[j * D ..][0..D];
            if (std.mem.indexOfNone(f32, g, &.{0}) == null) continue;
            const d = h.down[j * D ..][0..D];
            var sq: f64 = 0;
            for (d) |v| sq += @as(f64, v) * v;
            before += sq;
            for (0..rows) |k| {
                const b = basis.items[k * D ..][0..D];
                var dot: f64 = 0;
                for (d, b) |v, x| dot += @as(f64, v) * x;
                for (d, b) |*v, x| v.* -= @floatCast(dot * x);
            }
            var after: f64 = 0;
            for (d) |v| after += @as(f64, v) * v;
            removed += sq - after;
        }
        // then this layer's read directions join the basis for the layers below
        for (0..h.n) |j| {
            const g = h.gate[j * D ..][0..D];
            if (std.mem.indexOfNone(f32, g, &.{0}) == null) continue;
            for (r, g, norms[l]) |*o, x, w| o.* = @as(f64, x) * w;
            for (0..basis.items.len / D) |k| {
                const b = basis.items[k * D ..][0..D];
                var dot: f64 = 0;
                for (r, b) |v, x| dot += v * x;
                for (r, b) |*v, x| v.* -= dot * x;
            }
            var n: f64 = 0;
            for (r) |v| n += v * v;
            n = @sqrt(n);
            if (n < 1e-9) continue;
            for (r) |*v| v.* /= n;
            try basis.appendSlice(gpa, r);
        }
    }
    return if (before > 0) removed / before else 0;
}

test {
    _ = @import("cuda_pocket_test.zig");
}

// Pinned synchronous copies finish before any source storage is released, including failed calls.
fn staged(d: *const cuda.Driver, values: []const f32) !cuda.DeviceBuffer {
    const bytes = std.mem.sliceAsBytes(values);
    var host = try cuda.HostBuffer.alloc(d, bytes.len);
    defer host.free();
    @memcpy(host.bytes, bytes);
    var out = try cuda.DeviceBuffer.alloc(d, bytes.len);
    errdefer out.free();
    try d.check(d.api.cuMemcpyHtoD_v2(out.ptr, host.bytes.ptr, bytes.len), "pocket staged upload");
    return out;
}
