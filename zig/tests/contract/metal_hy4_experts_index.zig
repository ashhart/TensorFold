//! tensor_dispatch.metal under the contract: the Q/KV/gate row split and the two expert job tables, each exact.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const bank = @import("metal_hy4_experts_bank.zig");

const c = contract.spec;
const g = contract.gates;
const fl = h.fl;

/// The three kernels, as metal_tensor_qkv and metal_tensor_experts load them.
pub const Which = enum { split, active, group };

pub fn function(w: Which) []const u8 {
    return switch (w) {
        .split => "tf_hy4_qkv_split",
        .active => "tf_hy4_active_experts",
        .group => "tf_hy4_group_pairs",
    };
}

/// A planted fault in tensor_dispatch.metal's text, red at `gate`.
pub const Fault = struct { label: []const u8, what: []const u8, gate: g.Gate, edit: fl.Edit };

pub fn text(a: std.mem.Allocator, fault: ?Fault) ![]u8 {
    return fl.edited(a, sources.hy4_flash_tensor_dispatch, if (fault) |f| f.edit else null);
}

/// The fused input projection's columns: q_lora_rank, then the absorbed KV row, then a gate a value element.
pub const Split = struct { q: usize, kv: usize, value: usize, heads: [2]usize };

const rules = [_][]const u8{
    "a bit-preserving copy: each row's q, kv and gate columns hold the input's three ranges bit for bit, NaN payloads and signed zeros included",
    "each active expert's offset is the exclusive sum of the earlier ones' member counts (members up to the first -1 or 16), every later job's the pair count; order lists each active expert's members in their own order",
    "a counting sort of the pairs by expert: offsets the exclusive prefix counts over every expert, order each expert's pairs in pair order inside its own range, ids the identity",
};

pub const Index = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    fault: ?Fault,
    cols: Split,
    experts: usize,
    topk: usize,
    pipe: mtl.Pipeline,
    served: [4]c.Shape = undefined,
    nserved: usize = 0,
    base_buf: [96]u8 = undefined,
    label: [256]u8 = undefined,
    msg: [200]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    input: []u16 = &.{},
    route: ?bank.Route = null,
    members: []i32 = &.{},
    ins: [2]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, fault: ?Fault, cols: Split, experts: usize, topk: usize) !Index {
        const body = try text(a, fault);
        defer a.free(body);
        const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var t: Index = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .fault = fault, .cols = cols, .experts = experts, .topk = topk, .pipe = try mtl.Pipeline.init(device, lib, function(which), false) };
        const width = [2]usize{ cols.q + cols.kv + cols.heads[0] * cols.value, cols.q + cols.kv + cols.heads[1] * cols.value };
        const most = h.hy.metal_routed.max_rows;
        switch (which) {
            // a window and a prompt chunk at each attention's heads (one row reads the projection in place)
            .split => for ([_]usize{ h.window_rows, h.prompt_rows, h.window_rows, h.prompt_rows }, 0..) |rows, i| {
                t.served[i] = .{ .rows = rows, .n = width[i / 2], .k = 1 };
            },
            // router windows of 2 to 16 rows over every expert; prompt chunks past a window, 17 rows to 512
            .active, .group => for ([_]usize{ h.window_rows, most, 2 }, [_]usize{ h.prompt_rows, most + 1, 100 }, 0..) |w, p, i| {
                t.served[i] = .{ .rows = if (which == .active) w else p, .n = experts, .k = 1 };
            },
        }
        t.nserved = if (which == .split) 4 else 3;
        return t;
    }

    pub fn deinit(t: *Index) void {
        t.drop();
        t.pipe.deinit();
        t.rig.deinit();
    }

    fn drop(t: *Index) void {
        t.a.free(t.input);
        t.a.free(t.members);
        t.input = &.{};
        t.members = &.{};
        if (t.route) |r| r.deinit();
        t.route = null;
        t.rig.reset();
    }

    pub fn baseName(which: Which, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "metal hy4 tensor_dispatch {s}", .{function(which)}) catch unreachable;
    }

    pub fn kernel(t: *Index) g.Kernel {
        t.name = fl.named(baseName(t.which, &t.base_buf), if (t.fault) |f| f.what else null, &t.label);
        const block: c.Tile = if (t.which == .group) .{ .size = 64, .step = 1, .min = h.hy.metal_routed.max_rows + 1 } else .{};
        return .{
            .spec = .{
                .name = t.name,
                .accum = .none,
                .depth = fl.noSums,
                .out = .raw,
                .nan = .{ .stated = rules[@backingInt(t.which)] },
                .tiles = .{ block, .{}, .{} },
                .served = t.served[0..t.nserved],
            },
            .ctx = t,
            .vt = &vt,
            .control = if (t.fault) |f| f.gate else null,
        };
    }
    const vt = g.hooks(Index);

    fn gateCols(t: *const Index) usize {
        return t.shape.n - t.cols.q - t.cols.kv;
    }

    pub fn prepare(t: *Index, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        t.drop();
        t.shape = shape;
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (t.which) {
            .split => {
                if (rows < 2 or shape.n <= t.cols.q + t.cols.kv or t.gateCols() % t.cols.value != 0) return false;
                t.input = try t.a.alloc(u16, rows * shape.n);
                for (t.input) |*v| v.* = c.toBf16(rng.float(f32) * 8 - 4);
                if (fill == .nonfinite) {
                    // NaNs with payloads and signs, both infinities and -0, at the ranges' edges and inside them
                    const at = [_]usize{ 0, t.cols.q - 1, t.cols.q, t.cols.q + t.cols.kv - 1, t.cols.q + t.cols.kv, shape.n - 1, 7 };
                    const bits = [_]u16{ 0x7fc1, 0xffa5, 0x7f80, 0xff80, 0x8000, 0x7f81, 0xffff };
                    for (0..rows) |r| for (at, 0..) |col, i| {
                        t.input[r * shape.n + col] = bits[(i + r) % bits.len];
                    };
                }
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.input));
                t.outs[0] = try t.rig.output(rows * t.cols.q * 2);
                t.outs[1] = try t.rig.output(rows * t.cols.kv * 2);
                t.outs[2] = try t.rig.output(rows * t.gateCols() * 2);
            },
            .active => {
                if (rows < 2 or rows > h.hy.metal_routed.max_rows or shape.n != t.experts) return false;
                t.route = try bank.Route.init(t.a, rows, t.topk, t.experts, t.experts, .window, seed);
                const r = t.route.?;
                const slots = h.hy.metal_routed.max_rows;
                // the router's members: each active expert's pairs, -1 after; later rows stale from earlier windows
                t.members = try t.a.alloc(i32, t.experts * slots);
                for (t.members) |*m| m.* = @intCast(rng.uintLessThan(usize, r.pairs()));
                var active: i32 = 0;
                for (0..r.pairs()) |u| {
                    const first: usize = @intCast(r.offsets[u]);
                    if (first == r.pairs()) break;
                    const end: usize = if (u + 1 < r.pairs()) @intCast(r.offsets[u + 1]) else r.pairs();
                    for (t.members[u * slots ..][0..slots], 0..) |*m, i| m.* = if (first + i < end) r.order[first + i] else -1;
                    active += 1;
                }
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.members));
                t.ins[1] = try t.rig.input(std.mem.asBytes(&active));
                t.outs[0] = try t.rig.output(r.pairs() * 4);
                t.outs[1] = try t.rig.output(r.pairs() * 4);
            },
            .group => {
                // the gates' small calls take fewer rows than a prompt chunk does; the grouping is the same
                if (rows == 0 or rows > h.prompt_rows or shape.n != t.experts) return false;
                t.route = try bank.Route.init(t.a, rows, t.topk, t.experts, t.experts, .prompt, seed);
                const r = t.route.?;
                t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(r.pick));
                t.outs[0] = try t.rig.output(t.experts * 4);
                t.outs[1] = try t.rig.output(r.pairs() * 4);
                t.outs[2] = try t.rig.output(t.experts * 4);
            },
        }
        return true;
    }

    pub fn run(t: *Index) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        const rows: u32 = @intCast(t.shape.rows);
        switch (t.which) {
            // metal_tensor_qkv.Kernels.split's launch (Metal's uint3 takes uint4's 16 bytes)
            .split => {
                for ([_]rig_.Ref{ t.ins[0], t.outs[0], t.outs[1], t.outs[2] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setValue([4]u32{ @intCast(t.cols.q), @intCast(t.cols.kv), @intCast(t.gateCols()), 0 }, 4);
                e.dispatchThreads(mtl.Size.of(t.shape.n, rows, 1), mtl.Size.of(128, 1, 1));
            },
            // metal_tensor_experts.Kernels.prepare's
            .active => {
                for ([_]rig_.Ref{ t.ins[0], t.ins[1], t.outs[0], t.outs[1] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setValue([2]u32{ rows * @as(u32, @intCast(t.topk)), h.hy.metal_routed.max_rows }, 4);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            },
            // metal_tensor_experts.Kernels.group's
            .group => {
                for ([_]rig_.Ref{ t.ins[0], t.outs[0], t.outs[1], t.outs[2] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                const pairs = rows * @as(u32, @intCast(t.topk));
                e.setValue([4]u32{ pairs, @intCast(t.experts), pairs / rows, 0 }, 4);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Index) usize {
        const out = t.rig.outputBytes() catch return 0;
        return out.len;
    }

    pub fn reference(_: *Index, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    fn ints(t: *Index, at: usize, n: usize) []align(1) const i32 {
        return std.mem.bytesAsSlice(i32, t.rig.bytes(t.outs[at], n * 4));
    }

    pub fn check(t: *Index, _: g.Gate) !?[]const u8 {
        switch (t.which) {
            .split => {
                const n = t.shape.n;
                const widths = [3]usize{ t.cols.q, t.cols.kv, t.gateCols() };
                var first: usize = 0;
                for (widths, 0..) |w, part| {
                    const got = std.mem.bytesAsSlice(u16, t.rig.bytes(t.outs[part], t.shape.rows * w * 2));
                    for (0..t.shape.rows) |r| for (0..w) |col| {
                        const want = t.input[r * n + first + col];
                        if (got[r * w + col] != want) return try std.fmt.bufPrint(&t.msg, "row {d} column {d} of output {d} is 0x{x:0>4} where the input holds 0x{x:0>4}", .{ r, col, part, got[r * w + col], want });
                    };
                    first += w;
                }
            },
            .active => {
                const r = t.route.?;
                const offs = t.ints(0, r.pairs());
                const order = t.ints(1, r.pairs());
                for (offs, r.offsets, 0..) |got, want, u| if (got != want) return try std.fmt.bufPrint(&t.msg, "job {d}'s offset is {d} where the members give {d}", .{ u, got, want });
                for (order, r.order, 0..) |got, want, i| if (got != want) return try std.fmt.bufPrint(&t.msg, "order[{d}] is pair {d} where the members give {d}", .{ i, got, want });
            },
            .group => {
                const r = t.route.?;
                const offs = t.ints(0, t.experts);
                const order = t.ints(1, r.pairs());
                const ids = t.ints(2, t.experts);
                for (offs, r.offsets, 0..) |got, want, e| if (got != want) return try std.fmt.bufPrint(&t.msg, "expert {d}'s offset is {d} where the host counts {d}", .{ e, got, want });
                for (order, r.order, 0..) |got, want, i| if (got != want) return try std.fmt.bufPrint(&t.msg, "order[{d}] is pair {d} where the host's stable sort gives {d}", .{ i, got, want });
                for (ids, 0..) |got, e| if (got != e) return try std.fmt.bufPrint(&t.msg, "ids[{d}] is {d}, not the identity", .{ e, got });
            },
        }
        return null;
    }
};
