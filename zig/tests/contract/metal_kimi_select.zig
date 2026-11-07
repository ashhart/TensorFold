//! K3.1's experimental radix top-k against an independent comparison sort, including exceptional values and ties.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("kimi_index_ref.zig");
const c = contract.spec;
const g = contract.gates;
const ix = k3.index;

fn noSums(_: c.Shape) usize {
    return 0;
}
const shapes = [_]c.Shape{
    .{ .rows = 8, .n = 2048, .k = 2051 },  .{ .rows = 1, .n = 1, .k = 1 },
    .{ .rows = 3, .n = 127, .k = 127 },    .{ .rows = 2, .n = 2047, .k = 2047 },
    .{ .rows = 1, .n = 2048, .k = 2048 },  .{ .rows = 2, .n = 2048, .k = 2049 },
    .{ .rows = 1, .n = 2048, .k = 65536 },
};

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    control: bool,
    shape: c.Shape = undefined,
    scores: rig_.Ref = undefined,
    meta: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [192]u8 = undefined,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Probe {
        const text = if (control) try std.mem.replaceOwned(u8, a, ix.source, "isnan(v)) return 0u", "isnan(v)) return 0xffffffffu") else try a.dupe(u8, ix.source);
        defer a.free(text);
        if (control and std.mem.eql(u8, text, ix.source)) return error.ControlSourceChanged;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .control = control, .pipe = try rig_.pipeline(device, text, "k3_index_topk") };
    }
    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.pipe.deinit();
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "kimi index control: NaN selected first" else "kimi index select k3_index_topk";
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = .{ .name = name(p.control), .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "NaNs last; signed zeros tie; numeric ties prefer low position; causal selected list is chronological" }, .served = &shapes }, .ctx = p, .vt = &vt, .control = if (p.control) .nonfinite else null };
    }

    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > 256 or shape.k < 1 or shape.k > 65536 or shape.n != @min(ix.topk, shape.k)) return false;
        p.rig.reset();
        p.shape = shape;
        const scores = try p.a.alloc(f32, shape.rows * (shape.k + 3));
        defer p.a.free(scores);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // A small alphabet forces ties through every radix byte, with signed zeros kept at the cutoff.
        for (scores, 0..) |*v, i| v.* = if (i % 5 == 0) -0.0 else @as(f32, @floatFromInt(rng.uintLessThan(u32, 5))) - 2;
        for (0..shape.rows) |r| {
            const row = scores[r * (shape.k + 3) ..][0 .. shape.k + 3];
            @memset(row[shape.k..], std.math.inf(f32));
            if (r % 3 == 1) @memset(row[0..shape.k], 0);
            if (fill == .nonfinite) {
                row[0] = std.math.nan(f32);
                if (shape.k > 2) {
                    row[1] = std.math.inf(f32);
                    row[2] = -std.math.inf(f32);
                }
            }
        }
        p.scores = try p.rig.input(std.mem.sliceAsBytes(scores));
        const meta = try p.a.alloc(ix.Row, shape.rows);
        defer p.a.free(meta);
        for (meta, 0..) |*row, r| row.* = .{ .slot = 0, .pos = @intCast(shape.k - 1), .list = @intCast(r * ix.topk) };
        // nkeys is a documented mutable output in the metadata buffer, not an immutable input.
        p.meta = try p.rig.scratch(meta.len * @sizeOf(ix.Row));
        @memcpy(p.rig.bytes(p.meta, meta.len * @sizeOf(ix.Row)), std.mem.sliceAsBytes(meta));
        p.out = try p.rig.output(shape.rows * ix.topk * 4);
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * ix.topk * 4;
    }
    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.scores.buf, p.scores.off, 0);
        pass.e.setBuffer(p.meta.buf, p.meta.off, 1);
        pass.e.setValue(ix.Args{ .rows = @intCast(p.shape.rows), .slot_keys = @intCast(p.shape.k + 3), .score_stride = @intCast(p.shape.k + 3) }, 2);
        pass.e.setBuffer(p.out.buf, p.out.off, 3);
        pass.e.dispatchGroups(mtl.Size.of(p.shape.rows, 1, 1), mtl.Size.of(1024, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const scores: []f32 = @alignCast(std.mem.bytesAsSlice(f32, p.rig.bytes(p.scores, p.shape.rows * (p.shape.k + 3) * 4)));
        const got: []u32 = @alignCast(std.mem.bytesAsSlice(u32, p.rig.bytes(p.out, p.count())));
        const meta: []ix.Row = @alignCast(std.mem.bytesAsSlice(ix.Row, p.rig.bytes(p.meta, p.shape.rows * @sizeOf(ix.Row))));
        const want = try p.a.alloc(u32, p.shape.n);
        defer p.a.free(want);
        for (0..p.shape.rows) |r| {
            if (meta[r].nkeys != p.shape.n or meta[r].slot != 0 or meta[r].pos != p.shape.k - 1 or meta[r].list != r * ix.topk) return "selected count or immutable row metadata differs";
            try ref.selected(p.a, scores[r * (p.shape.k + 3) ..][0..p.shape.k], want);
            if (!std.mem.eql(u32, want, got[r * ix.topk ..][0..want.len])) return try std.fmt.bufPrint(&p.msg, "row {d}: selected positions differ from comparison sort", .{r});
            for (got[r * ix.topk + want.len ..][0 .. ix.topk - want.len]) |v| if (v != 0xffffffff) return "selection wrote past the visible key count";
        }
        return null;
    }
};
