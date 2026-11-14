//! GLM exchange copies and token selection under both committed and stale receive epochs.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const source = @import("kernel_sources").glm_ep;
const width = 4096;
const words = width / 2;
const topk = 8;
const vocab_part = 154880 / 2;
const epoch = 0x1_0000_0003;
const stale_epoch = epoch - (1 << 32);
const poison: u32 = 0xd3d3d3d3;
const absent = std.math.maxInt(u32);

const Which = enum {
    ep_pack,
    ep_unpack,
    ep_rsend,
    ep_amax_send,
    ep_amax_merge,

    fn name(w: Which) []const u8 {
        return switch (w) {
            inline else => |kind| "metal glm ep " ++ @tagName(kind),
        };
    }

    fn threads(w: Which) u32 {
        return switch (w) {
            .ep_pack, .ep_unpack, .ep_rsend => 256,
            .ep_amax_send, .ep_amax_merge => 64,
        };
    }

    fn spec(w: Which) c.Spec {
        return .{
            .name = w.name(),
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .tiles = .{ .{ .size = w.maxRows() }, .{}, .{} },
            .nan = .{ .stated = switch (w) {
                .ep_pack, .ep_rsend => "the matching epoch copies every bit; a stale epoch leaves the send slot untouched",
                .ep_unpack => "mapped picks copy every bit when ready or become zero when stale; other picks are untouched",
                .ep_amax_send => "a selected BF16 value widens to FP32; a missing pick sends minus infinity and no id",
                .ep_amax_merge => "the peer wins a larger value or a lower id on a tie; a stale epoch writes token zero",
            } },
            .served = switch (w) {
                .ep_pack, .ep_unpack => &.{
                    .{ .rows = 16, .n = words, .k = topk }, .{ .rows = 1, .n = 0, .k = topk },
                },
                .ep_rsend => &.{
                    .{ .rows = 4096, .n = width, .k = 1 }, .{ .rows = 16, .n = width, .k = 1 },
                },
                .ep_amax_send => &.{
                    .{ .rows = 64, .n = vocab_part, .k = 1 }, .{ .rows = 16, .n = vocab_part - 64, .k = 1 },
                    .{ .rows = 64, .n = 0, .k = 1 },
                },
                .ep_amax_merge => &.{.{ .rows = 64, .n = 2, .k = 1 }},
            },
        };
    }

    fn maxRows(w: Which) usize {
        return switch (w) {
            .ep_pack, .ep_unpack => 16,
            .ep_rsend => 4096,
            .ep_amax_send, .ep_amax_merge => 64,
        };
    }

    fn countsEntries(w: Which) bool {
        return w == .ep_rsend or w == .ep_amax_send;
    }

    fn planted(w: Which, a: std.mem.Allocator) ![]u8 {
        const edit = switch (w) {
            .ep_pack, .ep_rsend, .ep_amax_send => .{ "if (previous[0] != expected) return;", "if (uint(previous[0]) != uint(expected)) return;" },
            .ep_unpack => .{ "ready[0] == x", "uint(ready[0]) == uint(x)" },
            .ep_amax_merge => .{ "ready[0] != x", "uint(ready[0]) != uint(x)" },
        };
        const begin = std.mem.indexOf(u8, source, @tagName(w)) orelse return error.ControlEditMissing;
        const end = std.mem.indexOf(u8, source[begin..], "\n}") orelse return error.ControlEditMissing;
        const at = std.mem.indexOf(u8, source[begin..][0..end], edit[0]) orelse return error.ControlEditMissing;
        return std.mem.concat(a, u8, &.{ source[0 .. begin + at], edit[1], source[begin + at + edit[0].len ..] });
    }
};

fn noSums(_: c.Shape) usize {
    return 0;
}

fn wordsAt(rig: *const rig_.Rig, ref: rig_.Ref, n: usize) []u32 {
    return @ptrCast(@alignCast(rig.bytes(ref, n * @sizeOf(u32))));
}

fn halvesAt(rig: *const rig_.Rig, ref: rig_.Ref, n: usize) []u16 {
    return @ptrCast(@alignCast(rig.bytes(ref, n * @sizeOf(u16))));
}

const Slot = struct {
    stamp: rig_.Ref,
    out: rig_.Ref,
    count: ?rig_.Ref = null,
};

const Probe = struct {
    rig: rig_.Rig,
    which: Which,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    control: bool,
    values: rig_.Ref = undefined,
    picks: rig_.Ref = undefined,
    counts: rig_.Ref = undefined,
    slots: [2]Slot = undefined,
    entries: usize = 0,
    message: [120]u8 = undefined,

    const vt = g.hooks(Probe);

    fn kernel(p: *Probe) g.Kernel {
        var desc = p.which.spec();
        if (p.control) {
            desc.name = switch (p.which) {
                inline else => |kind| "control: metal glm ep " ++ @tagName(kind) ++ " ignores high epoch bits",
            };
        }
        return .{ .spec = desc, .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null };
    }

    pub fn count(p: *Probe) usize {
        const shape = p.shape;
        return switch (p.which) {
            .ep_pack, .ep_unpack => shape.rows * topk * words,
            .ep_rsend => shape.rows * width,
            .ep_amax_send => 2 * 64,
            .ep_amax_merge => 64,
        };
    }

    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    fn route(p: *Probe, fill: g.Fill, rng: std.Random) !void {
        const picks = p.shape.rows * topk;
        p.entries = if (p.shape.n == 0) 0 else if (p.shape.rows == 16) picks else picks * 2 / 3;
        const order = try p.rig.a.alloc(u32, picks);
        defer p.rig.a.free(order);
        @memset(order, absent);
        for (order[0..p.entries], 0..) |*at, i| at.* = @intCast(i * picks / p.entries);
        p.picks = try p.rig.input(std.mem.sliceAsBytes(order));
        const owned: i32 = @intCast(p.entries);
        const other: i32 = @intCast(picks - p.entries);
        const counts: [2]i32 = if (p.which == .ep_pack) .{ owned, other } else .{ other, owned };
        p.counts = try p.rig.input(std.mem.asBytes(&counts));
        try p.payload(fill, rng, true);
    }

    fn payload(p: *Probe, fill: g.Fill, rng: std.Random, packed_words: bool) !void {
        const a = p.rig.a;
        const data = try a.alloc(u32, p.count());
        defer a.free(data);
        for (data) |*v| v.* = if (packed_words) rng.int(u32) & 0xbfff_bfff else rng.int(u32) & 0xbfff_ffff;
        if (fill == .nonfinite) {
            data[0..3].* = if (packed_words) .{ 0x7fc1_ff80, 0x7f80_8000, 0xff80_7fc2 } else .{ 0x7fc0_0081, 0xff80_0000, 0x7f80_0000 };
        }
        p.values = try p.rig.input(std.mem.sliceAsBytes(data));
    }

    fn logits(p: *Probe, fill: g.Fill, rng: std.Random) !void {
        const a = p.rig.a;
        const n = p.shape.n;
        const buf = try a.alloc(u16, @max(1, p.shape.rows * n));
        defer a.free(buf);
        @memset(buf, 0x3f80);
        var chosen: [64]u32 = @splat(absent);
        if (n != 0) for (chosen[0..p.shape.rows], 0..) |*id, r| {
            if (r % 8 == 7) continue;
            id.* = if (r == 0) @intCast(n - 1) else rng.uintLessThan(u32, @intCast(n));
            buf[r * n + id.*] = if (fill == .nonfinite) switch (r) {
                0 => 0x7fc1,
                1 => 0x7f80,
                2 => 0xff80,
                3 => 0x8000,
                else => 0x4000,
            } else 0x4000 | @as(u16, rng.int(u7));
        };
        p.values = try p.rig.input(std.mem.sliceAsBytes(buf));
        p.picks = try p.rig.input(std.mem.asBytes(&chosen));
    }

    fn candidates(p: *Probe, fill: g.Fill, rng: std.Random) !void {
        var mine: [128]u32 = undefined;
        var peer: [128]u32 = undefined;
        for (0..64) |r| {
            const ma = &mine[2 * r];
            const mb = &peer[2 * r];
            const a: f32 = rng.float(f32) + 1;
            const b: f32 = rng.float(f32) - 1;
            ma.* = @bitCast(if (r % 4 == 0) b else a);
            mb.* = @bitCast(if (r % 4 == 0) a else if (r % 4 == 1) b else a);
            mine[2 * r + 1] = if (r % 4 == 3) 19 else 41;
            peer[2 * r + 1] = if (r % 4 == 3) 41 else 19;
        }
        if (fill == .nonfinite) {
            mine[0..10].* = .{ 0x7fc0_0018, 41, 0x7f80_0000, 10, 0xff80_0000, absent, 0x8000_0000, 41, 0x3f80_0000, 41 };
            peer[0..10].* = .{ 0x7f80_0000, 19, 0x7f80_0000, 9, 0xff80_0000, absent, 0, 19, 0x7fc0_0081, 19 };
        }
        p.values = try p.rig.input(std.mem.asBytes(&mine));
        p.picks = try p.rig.input(std.mem.asBytes(&peer));
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > p.which.maxRows()) return false;
        p.rig.reset();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        switch (p.which) {
            .ep_pack, .ep_unpack => try p.route(fill, rng),
            .ep_rsend => try p.payload(fill, rng, false),
            .ep_amax_send => try p.logits(fill, rng),
            .ep_amax_merge => try p.candidates(fill, rng),
        }
        for (&p.slots, [_]u64{ epoch, stale_epoch }) |*slot, seen_epoch| {
            slot.* = .{
                .stamp = try p.rig.input(std.mem.asBytes(&seen_epoch)),
                .out = try p.rig.output(p.count() * 4),
                .count = if (p.which.countsEntries()) try p.rig.output(4) else null,
            };
            @memset(wordsAt(&p.rig, slot.out, p.count()), poison);
            if (slot.count) |at| wordsAt(&p.rig, at, 1)[0] = poison;
        }
        return true;
    }

    fn bind(e: mtl.ComputeEncoder, at: usize, refs: []const rig_.Ref) void {
        for (refs, at..) |r, i| e.setBuffer(r.buf, r.off, i);
    }

    fn stamp(e: mtl.ComputeEncoder, at: usize, slot: Slot) void {
        bind(e, at, &.{slot.stamp});
        e.setValue(@as(u64, epoch), at + 1);
    }

    fn offset(p: *const Probe) u32 {
        return if (p.shape.rows % 2 == 0) vocab_part else 0;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.shape.rows);
        const group = mtl.Size.of(p.which.threads(), 1, 1);
        e.setPipeline(p.pipe);
        for (p.slots) |slot| {
            switch (p.which) {
                .ep_pack => {
                    bind(e, 0, &.{ p.values, p.picks, p.counts, slot.out });
                    stamp(e, 4, slot);
                    e.dispatchGroups(mtl.Size.of(1, rows * topk, 1), group);
                },
                .ep_unpack => {
                    stamp(e, 0, slot);
                    bind(e, 3, &.{ p.values, p.picks, p.counts, slot.out });
                    e.dispatchGroups(mtl.Size.of(1, rows * topk, 1), group);
                },
                .ep_rsend => {
                    bind(e, 0, &.{p.values});
                    e.setValue(@as(i32, @intCast(rows)), 1);
                    bind(e, 2, &.{ slot.out, slot.count.? });
                    stamp(e, 4, slot);
                    e.dispatchThreads(mtl.Size.of(rows * width, 1, 1), group);
                },
                .ep_amax_send => {
                    bind(e, 0, &.{ p.values, p.picks });
                    e.setValue([4]u32{ rows, @intCast(p.shape.n), p.offset(), 0 }, 2);
                    bind(e, 3, &.{ slot.out, slot.count.? });
                    stamp(e, 5, slot);
                    e.dispatchThreads(group, group);
                },
                .ep_amax_merge => {
                    stamp(e, 0, slot);
                    bind(e, 3, &.{ p.values, p.picks });
                    e.setValue(rows, 5);
                    bind(e, 6, &.{slot.out});
                    e.dispatchThreads(group, group);
                },
            }
        }
        try p.rig.finish(pass);
    }

    fn checkRoute(p: *Probe, out: []const u32, ready: bool) ?usize {
        const picked = wordsAt(&p.rig, p.picks, p.shape.rows * topk);
        const data = wordsAt(&p.rig, p.values, p.count());
        var unpacked: usize = 0;
        for (0..picked.len) |row| {
            if (p.which == .ep_pack) {
                const source_row = if (ready and row < p.entries) picked[row] else absent;
                if (checkWords(out, data, row, source_row, poison)) |at| return at;
            } else {
                const mapped = unpacked < p.entries and picked[unpacked] == row;
                const source_row = if (mapped and ready) @as(u32, @intCast(unpacked)) else absent;
                if (checkWords(out, data, row, source_row, if (mapped) 0 else poison)) |at| return at;
                if (mapped) unpacked += 1;
            }
        }
        return null;
    }

    fn checkWords(out: []const u32, data: []const u32, row: usize, source_row: u32, empty: u32) ?usize {
        const start = row * words;
        for (out[start..][0..words], 0..) |got, col| {
            const want = if (source_row == absent) empty else data[@as(usize, source_row) * words + col];
            if (got != want) return start + col;
        }
        return null;
    }

    fn checkPartial(p: *Probe, out: []const u32, ready: bool) ?usize {
        const data = wordsAt(&p.rig, p.values, p.count());
        for (out, data, 0..) |got, bits, i| {
            if (got != if (ready) bits else poison) return i;
        }
        return null;
    }

    fn checkLogits(p: *Probe, out: []const u32, ready: bool) ?usize {
        const n = p.shape.n;
        const data = halvesAt(&p.rig, p.values, @max(1, p.shape.rows * n));
        const ids = wordsAt(&p.rig, p.picks, 64);
        for (ids, 0..) |id, row| {
            const wrote = ready and row < p.shape.rows;
            const has = wrote and id < n;
            const want = if (!wrote) poison else if (has) @as(u32, data[row * n + id]) << 16 else 0xff80_0000;
            const got = out[2 * row];
            const nan = std.math.isNan(@as(f32, @bitCast(want)));
            if (if (nan) !std.math.isNan(@as(f32, @bitCast(got))) else got != want) return 2 * row;
            const want_id = if (!wrote) poison else if (has) p.offset() + id else absent;
            if (out[2 * row + 1] != want_id) return 2 * row + 1;
        }
        return null;
    }

    fn checkPick(p: *Probe, out: []const u32, ready: bool) ?usize {
        const mine = wordsAt(&p.rig, p.values, 128);
        const peer = wordsAt(&p.rig, p.picks, 128);
        for (out, 0..) |got, r| {
            const local_value: f64 = @as(f32, @bitCast(mine[2 * r]));
            const peer_value: f64 = @as(f32, @bitCast(peer[2 * r]));
            const peer_id = peer[2 * r + 1];
            const local_id = mine[2 * r + 1];
            const winner = if (peer_value > local_value or (peer_value == local_value and peer_id < local_id)) peer_id else local_id;
            if (got != if (r >= p.shape.rows) poison else if (!ready) 0 else winner) return r;
        }
        return null;
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        if (!try p.rig.intact()) return "a launch wrote outside its buffers";
        for (p.slots, [_]bool{ true, false }) |slot, ready| {
            const out = wordsAt(&p.rig, slot.out, p.count());
            const mismatch = switch (p.which) {
                .ep_pack, .ep_unpack => p.checkRoute(out, ready),
                .ep_rsend => p.checkPartial(out, ready),
                .ep_amax_send => p.checkLogits(out, ready),
                .ep_amax_merge => p.checkPick(out, ready),
            };
            const when = if (ready) "matching" else "stale";
            if (mismatch) |at| return try std.fmt.bufPrint(&p.message, "{s} epoch differs at output word {d}", .{ when, at });
            if (slot.count) |ref| {
                const want: u32 = if (!ready) poison else if (p.which == .ep_rsend) @intCast(p.shape.rows * 2) else 1;
                if (wordsAt(&p.rig, ref, 1)[0] != want) return try std.fmt.bufPrint(&p.message, "{s} epoch wrote an incorrect entry count", .{when});
            }
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (std.enums.values(Which)) |which| {
            if (only) |needle| if (std.mem.indexOf(u8, which.name(), needle) == null) continue;
            try s.add(which, false);
            try s.add(which, true);
        }
    }

    fn add(s: *Suite, which: Which, control: bool) !void {
        try s.probes.ensureUnusedCapacity(s.a, 1);
        try s.kernels.ensureUnusedCapacity(s.a, 1);
        const text = if (control) try which.planted(s.a) else source;
        defer if (control) s.a.free(text);
        const pipe = try rig_.pipeline(s.device, text, @tagName(which));
        errdefer pipe.deinit();
        if (pipe.maxThreads() < which.threads() or rig_.tgMemory(pipe) != 0) return error.UnexpectedLaunchLimits;
        const p = try s.a.create(Probe);
        p.* = .{ .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .which = which, .pipe = pipe, .control = control };
        s.probes.appendAssumeCapacity(p);
        s.kernels.appendAssumeCapacity(p.kernel());
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.rig.deinit();
            p.pipe.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
