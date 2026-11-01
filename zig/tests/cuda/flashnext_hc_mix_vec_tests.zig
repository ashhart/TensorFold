//! HC vector mix alone against dense_mix8: guarded inputs and rows, repeated bytes, solo rows and event timing.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const guarded = @import("contract_rig.zig");
const bf16 = @import("flashnext_rig.zig").bf16;
const d = 2560;
const ss = 4;
const wide = d * ss;
const most = 4218;
const tested_rows = [_]usize{ 1, 64, 65, 511, 512, 513, 1536, most };

const Context = struct { gpu: check.Gpu, s: cuda.Stream, old: cuda.Function, fused: flash.hc_fused.Kernels };

const Span = struct {
    base: u64,
    ptr: u64,
    pitch: usize,
    rows: usize,

    fn output(r: *guarded.Rig, rows: usize, pitch: usize) !Span {
        const base = try r.output((rows + 2) * pitch);
        return .{ .base = base, .ptr = base + pitch, .pitch = pitch, .rows = rows };
    }

    fn edge(self: Span, r: *guarded.Rig) !void {
        for (try r.bytes(self.base, self.pitch)) |v| try check.expect(v == 0xff, "HC mix prefix row", .{});
        for (try r.bytes(self.ptr + self.rows * self.pitch, self.pitch)) |v| try check.expect(v == 0xff, "HC mix suffix row", .{});
    }
};

const Output = struct {
    mixed: Span,
    sums: Span,

    fn init(r: *guarded.Rig, rows: usize) !Output {
        return .{ .mixed = try Span.output(r, rows, d * 2), .sums = try Span.output(r, rows, d / 32 * 4) };
    }

    fn at(self: Output, row: usize) Output {
        var out = self;
        out.mixed.ptr += row * self.mixed.pitch;
        out.sums.ptr += row * self.sums.pitch;
        return out;
    }

    fn edges(self: Output, r: *guarded.Rig) !void {
        try self.mixed.edge(r);
        try self.sums.edge(r);
    }
};

const Fixture = struct {
    ctx: Context,
    r: *guarded.Rig,
    rows: usize,
    up: u64,
    norm: u64,
    old: Output,
    new: Output,

    fn init(ctx: Context, r: *guarded.Rig, a: std.mem.Allocator, rows: usize, special: bool) !Fixture {
        var prng = std.Random.DefaultPrng.init(0x48434d00);
        const rng = prng.random();
        const up = try a.alloc(u16, (rows + 2) * wide);
        const norm = try a.alloc(u16, up.len);
        for (up, norm) |*u, *n| {
            u.* = bf16(rng.float(f32) * 8 - 4);
            n.* = bf16(rng.float(f32) * 4 - 2);
        }
        if (special) {
            const plants = [_]u16{ 0x7fc1, 0xffc2, 0x7f80, 0xff80, 0x0001, 0x8001, 0, 0x8000 };
            for (plants, 0..) |value, row| {
                up[(row + 1) * wide] = value;
                norm[(row + 1) * wide + d + 33] = value;
            }
        }
        return .{
            .ctx = ctx,
            .r = r,
            .rows = rows,
            .up = try r.input(std.mem.sliceAsBytes(up)) + wide * 2,
            .norm = try r.input(std.mem.sliceAsBytes(norm)) + wide * 2,
            .old = try Output.init(r, rows),
            .new = try Output.init(r, rows),
        };
    }

    fn baseline(self: *Fixture, out: Output, up: u64, norm: u64, rows: usize) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ up, norm, out.mixed.ptr, out.sums.ptr }) |v| args.add(v);
        args.add(@as(c_int, d));
        args.add(@as(c_int, ss));
        try cuda.launch.launch(self.ctx.old, .{ .grid = .{ .x = @intCast(rows), .y = d / 32 / 8 }, .block = .{ .x = 256 } }, self.ctx.s, &args);
    }

    fn launch(self: *Fixture, new: bool) !void {
        if (new) return self.ctx.fused.mix(self.ctx.s, self.up, self.norm, self.new.mixed.ptr, self.new.sums.ptr, self.rows);
        try self.baseline(self.old, self.up, self.norm, self.rows);
    }

    fn compare(self: *Fixture) !void {
        try same(self.r, "HC vector mixed rows equal dense_mix8", self.old.mixed.ptr, self.new.mixed.ptr, self.rows * d * 2);
        try same(self.r, "HC vector group sums equal dense_mix8", self.old.sums.ptr, self.new.sums.ptr, self.rows * (d / 32) * 4);
        try self.old.edges(self.r);
        try self.new.edges(self.r);
        try check.expect(try self.r.intact(), "HC vector mix allocation guards", .{});
    }

    fn solo(self: *Fixture) !void {
        for (0..self.rows) |row|
            try self.baseline(self.old.at(row), self.up + row * wide * 2, self.norm + row * wide * 2, 1);
        try self.compare();
    }

    fn time(self: *Fixture, new: bool, begin: cuda.Event, end: cuda.Event) !f64 {
        try begin.record(self.ctx.s);
        if (new) {
            try self.ctx.fused.mix(self.ctx.s, self.up, self.norm, self.old.mixed.ptr, self.old.sums.ptr, self.rows);
        } else try self.baseline(self.old, self.up, self.norm, self.rows);
        try end.record(self.ctx.s);
        try end.synchronize();
        return @as(f64, try cuda.Event.elapsedMs(begin, end));
    }
};

fn same(r: *guarded.Rig, what: []const u8, old: u64, new: u64, n: usize) !void {
    const bytes = try r.gpu.gpa.dupe(u8, try r.bytes(new, n));
    defer r.gpu.gpa.free(bytes);
    try check.sameBytes(what, bytes, try r.bytes(old, n));
}

fn testOne(ctx: Context, rows: usize, special: bool) !void {
    var arena = std.heap.ArenaAllocator.init(ctx.gpu.gpa);
    defer arena.deinit();
    var r: guarded.Rig = .{ .gpu = ctx.gpu, .s = ctx.s };
    defer r.deinit();
    var fixture = try Fixture.init(ctx, &r, arena.allocator(), rows, special);
    const before = try ctx.gpu.gpa.dupe(u8, try r.inputBytes());
    defer ctx.gpu.gpa.free(before);
    try fixture.launch(false);
    for (0..if (rows == most or special) @as(usize, 5) else 2) |_| {
        try fixture.launch(true);
        try fixture.compare();
    }
    try fixture.solo();
    try check.sameBytes("HC vector mix inputs unchanged", before, try r.inputBytes());
    check.pass("HC vector mix M={d} special={}: dense_mix8 bytes, repeats, solo rows, guards and unchanged inputs", .{ rows, special });
}

fn benchOne(ctx: Context, rows: usize) !void {
    var arena = std.heap.ArenaAllocator.init(ctx.gpu.gpa);
    defer arena.deinit();
    var r: guarded.Rig = .{ .gpu = ctx.gpu, .s = ctx.s };
    defer r.deinit();
    var fixture = try Fixture.init(ctx, &r, arena.allocator(), rows, false);
    try fixture.launch(false);
    try fixture.launch(true);
    try fixture.compare();
    var begin = try cuda.Event.init(ctx.gpu.d, true);
    defer begin.deinit();
    var end = try cuda.Event.init(ctx.gpu.d, true);
    defer end.deinit();
    var ms: [2][9]f64 = undefined;
    for (0..11) |round| for (0..2) |order| {
        const which = (round + order) % 2;
        const elapsed = try fixture.time(which == 1, begin, end);
        if (round >= 2) ms[which][round - 2] = elapsed;
    };
    const old = check.median(&ms[0]);
    const new = check.median(&ms[1]);
    std.debug.print("BENCH HC vector mix M={d}: dense_mix8={d:.4} vector={d:.4} saved={d:.4} ms\n", .{ rows, old, new, old - new });
    try fixture.launch(false);
    try fixture.launch(true);
    try fixture.compare();
}

pub fn run(gpu: check.Gpu, timing: bool) !void {
    return driver(gpu, timing, most);
}

pub fn bench(gpu: check.Gpu, rows: usize) !void {
    return driver(gpu, true, rows);
}

fn driver(gpu: check.Gpu, timing: bool, bench_rows: usize) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var hc = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_hc);
    defer hc.unload();
    const ctx: Context = .{ .gpu = gpu, .s = s, .old = try hc.function("fn_hc_dense_mix8"), .fused = try flash.hc_fused.Kernels.resolve(hc) };
    if (timing) return benchOne(ctx, bench_rows);
    for (tested_rows) |rows| try testOne(ctx, rows, false);
    try testOne(ctx, 65, true);
}
