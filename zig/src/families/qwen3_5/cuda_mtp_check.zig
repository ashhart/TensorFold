//! GPU regressions for the MTP drafter: absorbs in chunks, a tree's lanes against their paths chained, a poisoned ring.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Mtp = @import("cuda_mtp.zig").Mtp;
const max_rows = @import("cuda_engine.zig").max_rows;

/// Siblings 0 and 1 at depth 0, six siblings 3 to 8 at depth 1, lane 9 at depth 2 under 3 and 0; each has a child.
const parents = [_]i32{ -1, -1, -1, 0, 0, 0, 0, 1, 1, 3, 4, 5, 6, 7, 8, 9 };
const ranks = [_]u8{ 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
const depths = [_]u8{ 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3 };
const internal = [_]usize{ 0, 1, 3, 4, 5, 6, 7, 8, 9 };

/// A lane's rows from a pass: its normed output, its key and its value.
const Rows = struct { state: []u8, k: []u8, v: []u8 };

/// Absorbs `ids` from the target's tap ring (`tap`, filled by forwards over them), then checks the tree twice.
pub fn run(gpa: std.mem.Allocator, d: *Mtp, tap: u64, ids: []const u32) !bool {
    const e = d.e;
    const g = e.g;
    d.trees = true;
    const ctx = try d.create();
    defer d.drop(ctx);
    try d.absorb(ctx, tap, 0, ids.len - 1, null, ids[1..]);
    const pos0 = ctx.end;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // one absorb over several prompt chunks against the same rows absorbed a chunk a call
    const apart = try d.create();
    defer d.drop(apart);
    var from: usize = 0;
    while (from < pos0) : (from += max_rows) {
        const n = @min(max_rows, pos0 - from);
        try d.absorb(apart, tap, from, n, null, ids[from + 1 ..][0..n]);
    }
    const ring = 2 * d.capacity * d.kvWidth() * 2;
    const whole = try a.alloc(u8, ring + g.hidden * 2);
    const parts = try a.alloc(u8, ring + g.hidden * 2);
    try download(d, whole[0..ring], ctx.ring.ptr);
    try download(d, whole[ring..], ctx.last.ptr);
    try download(d, parts[0..ring], apart.ring.ptr);
    try download(d, parts[ring..], apart.last.ptr);
    var ok = std.mem.eql(u8, whole, parts) and std.mem.eql(u32, &ctx.cands, &apart.cands);
    std.debug.print("{s} {d} rows absorbed at once vs a chunk of {d} a call: ring, state and candidates\n", .{ if (ok) "PASS" else "FAIL", pos0, max_rows });
    var p = parents;
    var r = ranks;
    var dp = depths;
    const shape: lanes.shape.Shape = .{ .parents = &p, .ranks = &r, .depths = &dp, .expected = 0 };
    const hidden = g.hidden * 2;
    const kw = d.kvWidth() * 2;

    try d.holdTree(ctx, pos0, &shape);
    const tree = try lanesRows(a, d, hidden, kw);
    const held = ctx.held;

    // the ring at and past the committed end poisoned: a tree pass reads none of it, so nothing may change
    const junk = try a.alloc(u8, 16 * kw);
    @memset(junk, 0x7f);
    for (0..16) |i| {
        const slot = (pos0 + i) % d.capacity;
        try upload(d, d.ringK(ctx) + slot * kw, junk[0..kw]);
        try upload(d, d.ringV(ctx) + slot * kw, junk[0..kw]);
    }
    try d.holdTree(ctx, pos0, &shape);
    const poisoned = try lanesRows(a, d, hidden, kw);
    ok = report("poisoned ring suffix, tree rows", poisoned, tree, &internal) and ok;
    if (!std.mem.eql(u32, &ctx.held, &held)) {
        std.debug.print("FAIL poisoned ring suffix: the lanes' tokens changed\n", .{});
        ok = false;
    }

    // each internal lane's path chained from the newest absorbed row, one row a slot, against the tree's rows
    var chain: Rows = .{ .state = try a.alloc(u8, tree.state.len), .k = try a.alloc(u8, tree.k.len), .v = try a.alloc(u8, tree.v.len) };
    for (internal) |lane| {
        var path: [4]usize = undefined;
        var n: usize = 0;
        var at: i32 = @intCast(lane);
        while (at >= 0) : (at = parents[@intCast(at)]) {
            path[n] = @intCast(at);
            n += 1;
        }
        std.mem.reverse(usize, path[0..n]);
        for (path[0..n], 0..) |step, k| {
            const src = if (k == 0) ctx.last.ptr else d.x;
            try d.scratch.copyFrom(d.hin - d.scratch.ptr, src, hidden, e.stream.handle);
            try d.putIds(&.{held[step]});
            try d.layer(ctx, 1, pos0 + k, .ring);
        }
        try download(d, chain.state[lane * hidden ..][0..hidden], d.x);
        try download(d, chain.k[lane * kw ..][0..kw], d.kb);
        try download(d, chain.v[lane * kw ..][0..kw], d.vb);
    }
    ok = report("six siblings a depth, tree rows vs each lane's path chained", tree, chain, &internal) and ok;
    return ok;
}

fn lanesRows(a: std.mem.Allocator, d: *Mtp, hidden: usize, kw: usize) !Rows {
    const n = parents.len;
    const out: Rows = .{ .state = try a.alloc(u8, n * hidden), .k = try a.alloc(u8, n * kw), .v = try a.alloc(u8, n * kw) };
    try download(d, out.state, d.lane_state);
    try download(d, out.k, d.lane_k);
    try download(d, out.v, d.lane_v);
    return out;
}

/// PASS when every internal lane's state, key and value bytes are equal; else the first lane that differs.
fn report(what: []const u8, got: Rows, want: Rows, lanes_: []const usize) bool {
    const hidden = got.state.len / parents.len;
    const kw = got.k.len / parents.len;
    for (lanes_) |l| {
        const same = std.mem.eql(u8, got.state[l * hidden ..][0..hidden], want.state[l * hidden ..][0..hidden]) and
            std.mem.eql(u8, got.k[l * kw ..][0..kw], want.k[l * kw ..][0..kw]) and std.mem.eql(u8, got.v[l * kw ..][0..kw], want.v[l * kw ..][0..kw]);
        if (!same) {
            std.debug.print("FAIL {s}: lane {d} (depth {d}) differs\n", .{ what, l, depths[l] });
            return false;
        }
    }
    std.debug.print("PASS {s}: {d} lanes byte-equal\n", .{ what, lanes_.len });
    return true;
}

fn upload(d: *Mtp, dst: u64, bytes: []const u8) !void {
    const e = d.e;
    try cuda.copy_source.enqueue(e.d, dst, bytes, e.stream);
    try e.stream.synchronize();
}

fn download(d: *Mtp, out: []u8, src: u64) !void {
    const e = d.e;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, src, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
}
