//! Matching and stale full-width epochs exercise every Flash Next pair producer and consumer guard.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig = @import("rig.zig");
const suite = @import("metal_flashnext_tp.zig");
const D = 2560;
const WIDE = 4 * D;
const poison: u32 = 0xd3d3d3d3;

pub const Kind = enum { tp_copy, tp_plain, tp_merge, tp_pick, tp_branch, tp_argmax_part, tp_ids };

pub fn name(comptime kind: Kind, control: bool) []const u8 {
    const normal = "metal flashnext tp " ++ @tagName(kind) ++ " epochs";
    return if (control) normal ++ " low-word control" else normal;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

fn bind(e: mtl.ComputeEncoder, first: usize, refs: []const rig.Ref) void {
    for (refs, first..) |ref, at| e.setBuffer(ref.buf, ref.off, at);
}

pub const Probe = struct {
    const vt = g.hooks(Probe);
    rig: rig.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    rows: usize = 0,
    nonfinite: bool = false,
    data: rig.Ref = undefined,
    peer: rig.Ref = undefined,
    extra: rig.Ref = undefined,
    h: rig.Ref = undefined,
    inj: rig.Ref = undefined,
    count_ref: rig.Ref = undefined,
    ready: [2]rig.Ref = undefined,
    out: [2]rig.Ref = undefined,
    ssp: [2]rig.Ref = undefined,

    pub fn kernel(p: *Probe, label: []const u8, control: bool) g.Kernel {
        return .{ .spec = .{
            .name = label,
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            .nan = .{ .stated = "matching epochs preserve the operation; stale full-width epochs leave output untouched" },
            .served = &.{ .{ .rows = 1, .n = D, .k = 1 }, .{ .rows = 2, .n = D, .k = 1 }, .{ .rows = 16, .n = D, .k = 1 } },
        }, .ctx = p, .vt = &vt, .control = if (control) .shapes else null };
    }

    fn input(p: *Probe, comptime T: type, n: usize, v: T) !rig.Ref {
        const data = try p.rig.a.alloc(T, n);
        defer p.rig.a.free(data);
        @memset(data, v);
        return p.rig.input(std.mem.sliceAsBytes(data));
    }

    fn bytes(p: *const Probe) usize {
        return switch (p.kind) {
            .tp_plain, .tp_merge => p.rows * WIDE * 2,
            .tp_branch => p.rows * D * 2,
            .tp_argmax_part => p.rows * 8,
            .tp_pick => p.rows * 4,
            .tp_ids => 4,
            .tp_copy => p.rows * D * 4,
        };
    }

    pub fn count(p: *const Probe) usize {
        return p.bytes() / 4;
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, _: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16) return false;
        p.rig.reset();
        p.rows = shape.rows;
        p.nonfinite = fill == .nonfinite;
        const row_word: i32 = @intCast(p.rows);
        p.count_ref = try p.rig.input(std.mem.asBytes(&row_word));
        switch (p.kind) {
            .tp_copy => {
                p.data = try p.input(u32, p.rows * D, 0x7fc0_0001);
            },
            .tp_plain => {
                p.data = try p.input(f32, p.rows * D, 1);
                p.peer = try p.input(f32, p.rows * D, 2);
                if (p.nonfinite) {
                    const peer = p.words(p.peer, p.rows * D);
                    peer[0..3].* = .{ 0x7fc0_0001, 0x7f80_0000, 0xff80_0000 };
                }
            },
            .tp_merge => {
                p.data = try p.input(u16, p.rows * D, 0x3f80);
                p.peer = try p.input(u16, p.rows * D, 0x4000);
                if (p.nonfinite) {
                    const peer: []u16 = @ptrCast(@alignCast(p.rig.bytes(p.peer, p.rows * D * 2)));
                    peer[0..3].* = .{ 0x7fc1, 0x7f80, 0xff80 };
                }
            },
            .tp_branch => {
                p.data = try p.input(u16, p.rows * 11 * D, 0x3f80);
                p.extra = try p.input(f32, p.rows * 10, 1);
                p.peer = try p.input(f32, p.rows * 513, 0);
            },
            .tp_argmax_part => {
                p.data = try p.input(u16, p.rows * D, 0x3f80);
                const values: []u16 = @ptrCast(@alignCast(p.rig.bytes(p.data, p.rows * D * 2)));
                for (0..p.rows) |r| values[r * D + r + 7] = 0x4080;
            },
            .tp_pick => {
                p.data = try p.input(u32, p.rows * 2, 11);
                p.peer = try p.input(u32, p.rows * 2, 22);
                for (0..p.rows) |r| {
                    p.words(p.data, p.rows * 2)[2 * r] = 0x3f80_0000;
                    p.words(p.peer, p.rows * 2)[2 * r] = 0x4000_0000;
                }
            },
            .tp_ids => {
                p.data = try p.input(u32, 1, 1);
                const ids = [_]u32{ 101, 202, 303 };
                p.peer = try p.rig.input(std.mem.asBytes(&ids));
            },
        }
        if (p.kind == .tp_plain or p.kind == .tp_merge) {
            p.h = try p.input(u16, p.rows * WIDE, 0x3f80);
            p.inj = try p.input(u16, p.rows * 4, 0x3f80);
        }
        for (0..2) |slot| {
            const epoch = if (slot == 0) suite.epoch else suite.stale;
            p.ready[slot] = try p.rig.input(std.mem.asBytes(&epoch));
            p.out[slot] = try p.rig.output(p.bytes());
            @memset(p.words(p.out[slot], p.count()), poison);
            if (p.kind == .tp_plain or p.kind == .tp_merge) {
                p.ssp[slot] = try p.rig.output(p.rows * 40 * 4);
                @memset(p.words(p.ssp[slot], p.rows * 40), poison);
            }
        }
        return true;
    }

    fn words(p: *const Probe, ref: rig.Ref, n: usize) []u32 {
        return @ptrCast(@alignCast(p.rig.bytes(ref, n * 4)));
    }

    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.rows);
        for (0..2) |slot| {
            @memset(p.words(p.out[slot], p.count()), poison);
            e.setPipeline(p.pipe);
            switch (p.kind) {
                .tp_copy => {
                    bind(e, 0, &.{p.ready[slot]});
                    e.setValue(suite.epoch, 1);
                    bind(e, 2, &.{ p.data, p.out[slot] });
                    e.setValue(rows * D, 4);
                    e.dispatchThreads(mtl.Size.of(rows * D, 1, 1), mtl.Size.of(256, 1, 1));
                },
                .tp_plain, .tp_merge => {
                    bind(e, 0, &.{p.ready[slot]});
                    e.setValue(suite.epoch, 1);
                    bind(e, 2, &.{p.count_ref});
                    if (p.kind == .tp_plain) {
                        bind(e, 3, &.{ p.data, p.peer });
                        e.setValue(@as(u32, 0), 5);
                        bind(e, 6, &.{ p.h, p.inj });
                    } else {
                        bind(e, 3, &.{ p.h, p.inj, p.data, p.peer });
                        e.setValue([4]u32{ 0, 0, 0, 0 }, 7);
                    }
                    bind(e, 8, &.{ p.out[slot], p.ssp[slot] });
                    e.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
                },
                .tp_pick => {
                    bind(e, 0, &.{p.ready[slot]});
                    e.setValue(suite.epoch, 1);
                    bind(e, 2, &.{ p.count_ref, p.data, p.peer, p.out[slot] });
                    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
                },
                .tp_branch => {
                    bind(e, 0, &.{ p.data, p.extra, p.peer, p.out[slot] });
                    e.setValue([2]u32{ 0, rows }, 4);
                    bind(e, 5, &.{p.ready[slot]});
                    e.setValue(suite.epoch, 6);
                    e.dispatchThreads(mtl.Size.of(D, rows, 1), mtl.Size.of(256, 1, 1));
                },
                .tp_argmax_part => {
                    bind(e, 0, &.{ p.data, p.out[slot] });
                    e.setValue([4]u32{ D, 0, D, 0 }, 2);
                    bind(e, 3, &.{p.ready[slot]});
                    e.setValue(suite.epoch, 4);
                    e.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
                },
                .tp_ids => {
                    bind(e, 0, &.{ p.data, p.peer, p.out[slot], p.ready[slot] });
                    e.setValue(suite.epoch, 4);
                    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
                },
            }
        }
        try p.rig.finish(pass);
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        if (!std.mem.allEqual(u32, p.words(p.out[1], p.count()), poison)) return "a stale epoch consumed old payload or reused an active send slot";
        if (p.kind == .tp_plain or p.kind == .tp_merge) {
            if (!std.mem.allEqual(u32, p.words(p.ssp[1], p.rows * 40), poison)) return "a stale epoch changed partial sums";
        }
        const out = p.words(p.out[0], p.count());
        if (p.kind == .tp_plain or p.kind == .tp_merge or p.kind == .tp_branch) {
            const halves: []const u16 = @ptrCast(@alignCast(p.rig.bytes(p.out[0], p.bytes())));
            const want: u16 = switch (p.kind) {
                .tp_plain => 0x4080,
                .tp_merge => 0x4040,
                else => 0x4128,
            };
            for (halves, 0..) |got, i| {
                const col = i % D;
                const special = p.nonfinite and p.kind != .tp_branch and i < WIDE and col < 3;
                if (special) {
                    const class = c.classOf(c.fromBf16(got));
                    const expected: c.Class = switch (col) {
                        0 => .nan,
                        1 => .pos_inf,
                        else => .neg_inf,
                    };
                    if (class != expected) return "a matching epoch lost a non-finite payload class";
                } else if (got != want) return "a matching epoch changed the stream update or branch";
            }
        } else for (out, 0..) |got, i| {
            const want: u32 = switch (p.kind) {
                .tp_copy => 0x7fc0_0001,
                .tp_pick => 22,
                .tp_ids => 202,
                .tp_argmax_part => if (i % 2 == 0) 0x4080_0000 else @intCast(i / 2 + 7),
                else => unreachable,
            };
            if (got != want) return "a matching epoch changed copied bytes or selected token";
        }
        return null;
    }
};
