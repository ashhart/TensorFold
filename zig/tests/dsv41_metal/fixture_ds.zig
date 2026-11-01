//! The tiny model with its DSpark drafter: the rig, the drafter's weights and pass, and random main rows by position.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const ref = @import("ref.zig");

pub const Ds = struct {
    a: std.mem.Allocator,
    r: *fixture.Rig,
    w: pkg.dspark.Weights,
    pass: *pkg.dspark.Pass,

    /// The rig of `fixture.dsparkConfig`, the drafter's weights read back from the same image, a pass of `slots` slots.
    pub fn init(a: std.mem.Allocator, io: std.Io, limit: u32, seed: u64, slots: u32) !*Ds {
        const d = try a.create(Ds);
        errdefer a.destroy(d);
        d.a = a;
        d.r = try fixture.Rig.initWith(a, io, fixture.dsparkConfig(), limit, seed);
        errdefer d.r.deinit();
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const img = try fixture.image(a, &d.r.c, seed);
        defer a.free(img);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = img });
        const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
        defer a.free(dir);
        var pack = try host.pack.Pack.open(a, io, dir);
        defer pack.deinit();
        d.w = try pkg.dspark.loadWeights(a, io, d.r.device, &pack, &d.r.c);
        errdefer d.w.deinit();
        d.pass = try pkg.dspark.Pass.init(a, d.r.device, d.r.queue, &d.r.c, &d.r.p, &d.r.s, &d.r.tables, &d.r.top.?, &d.w, .{ .slots = slots, .candidates = d.r.c.vocab });
        return d;
    }

    pub fn deinit(d: *Ds) void {
        d.pass.deinit();
        d.w.deinit();
        d.r.deinit();
        d.a.destroy(d);
    }

    /// Main rows [n][targets * hidden] bf16 for positions start ..: a row depends on its position and `seed` only.
    pub fn taps(d: *const Ds, start: u64, n: usize, seed: u64) !mtl.Buffer {
        const k = d.r.c.dspark_targets.len * d.r.c.hidden;
        const buf = try d.r.device.buffer(n * k * 2, mtl.ResourceOptions.shared);
        const v = buf.slice(u16, n * k);
        for (0..n) |i| {
            var prng = std.Random.DefaultPrng.init(seed *% 1000003 +% start + i);
            for (v[i * k ..][0..k]) |*x| x.* = ref.toBf((prng.random().float(f32) * 2 - 1) * 1.5);
        }
        return buf;
    }

    /// `n` committed rows from `start` into `slot`'s rings (the production ingest), one call.
    pub fn ingest(d: *Ds, slot: u32, start: u64, n: usize, seed: u64) !void {
        const buf = try d.taps(start, n, seed);
        defer buf.deinit();
        const rows = try d.a.alloc(u32, n);
        defer d.a.free(rows);
        for (rows, 0..) |*r, i| r.* = @intCast(i);
        try d.pass.ingest(slot, start, .{ .buf = buf, .rows = @intCast(n) }, rows);
    }

    /// The ask of a slot whose pending token `anchor` sits at `start`.
    pub fn ask(slot: u32, anchor: u32, start: u64, params: ?host.dspark.Params) pkg.dspark.Ask {
        return .{ .slot = slot, .anchor = anchor, .start = start, .params = params orelse .{ .pos0 = start + 1 } };
    }
};

/// One ask's results in memory of their own.
pub const Out = struct {
    drafts: []u32,
    conf: []f32,
    cand: []i32,
    base: []f32,

    pub fn init(a: std.mem.Allocator, c: *const host.config.Config) !Out {
        const n: usize = c.dspark_block;
        return .{ .drafts = try a.alloc(u32, n), .conf = try a.alloc(f32, n), .cand = try a.alloc(i32, n * c.vocab), .base = try a.alloc(f32, n * c.vocab) };
    }

    pub fn deinit(o: Out, a: std.mem.Allocator) void {
        a.free(o.drafts);
        a.free(o.conf);
        a.free(o.cand);
        a.free(o.base);
    }

    pub fn proposal(o: Out) pkg.dspark.Proposal {
        return .{ .drafts = o.drafts, .conf = o.conf, .cand = o.cand, .base = o.base };
    }

    /// 0 when every value of the two is bit-equal, else the count of values that differ.
    pub fn diff(x: Out, y: Out) usize {
        var n: usize = 0;
        for (x.drafts, y.drafts) |p, q| n += @intFromBool(p != q);
        for (x.conf, y.conf) |p, q| n += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
        for (x.cand, y.cand) |p, q| n += @intFromBool(p != q);
        for (x.base, y.base) |p, q| n += @intFromBool(@as(u32, @bitCast(p)) != @as(u32, @bitCast(q)));
        return n;
    }
};
