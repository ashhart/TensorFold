//! Shared-down FP32 stores into slot 10 equal the contiguous product followed by scatter, with GPU event timing.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const bf16 = @import("flashnext_rig.zig").bf16;
const d = 2560;
const width = 640;
const slots = 11;
const most = 4218;
const row_counts = [_]usize{ 1, 16, 17, 32, 64, 65, 1536, most };
const bench_rows = [_]usize{ 64, 256, 512, 1024, 1536, most };

/// A whole shared down: the lane or the BF16 prompt tile into a contiguous result then the scatter, or direct.
const Way = enum { lane, prompt, direct };

const Fixture = struct {
    r: Rig,
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    glue: flash.Glue,
    shared: cuda.qmmf.Weight,
    shared_x: u64,
    shared_out: u64,
    shared_slots: [2]u64,

    fn init(gpu: check.Gpu) !Fixture {
        const a = gpu.gpa;
        const s = try cuda.Stream.init(gpu.d, false);
        errdefer {
            var stream = s;
            stream.deinit();
        }
        var r: Rig = .{ .gpu = gpu, .s = s };
        errdefer r.deinit();
        var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
        errdefer lane.unload();
        var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
        errdefer prompt.unload();
        var glue = try flash.Glue.load(gpu.d);
        errdefer glue.deinit();
        var prng = std.Random.DefaultPrng.init(0x5a4ed);
        const rng = prng.random();
        const x = try a.alloc(u16, most * width);
        defer a.free(x);
        for (x) |*v| v.* = bf16(rng.float(f32) - 0.5);
        const weights = try a.alloc(u16, d * width);
        defer a.free(weights);
        const tiled = try a.alloc(u16, weights.len);
        defer a.free(tiled);
        for (weights) |*v| v.* = bf16((rng.float(f32) - 0.5) * 0.1);
        cuda.qmmf.packBf16(tiled, weights, d, width);
        var result: Fixture = .{
            .r = undefined,
            .lane = lane,
            .prompt = prompt,
            .glue = glue,
            .shared = .{ .mode = .bf16, .codes = try r.input(std.mem.sliceAsBytes(tiled)), .scales = 0, .scale = 0.8123, .n = d, .k = width, .npad = d },
            .shared_x = try r.input(std.mem.sliceAsBytes(x)),
            .shared_out = try r.scratch(most * d * 4),
            .shared_slots = .{ try r.output(most * slots * d * 4), try r.output(most * slots * d * 4) },
        };
        result.r = r;
        return result;
    }

    fn deinit(f: *Fixture) void {
        f.r.deinit();
        f.r.s.deinit();
        f.glue.deinit();
        f.prompt.unload();
        f.lane.unload();
    }

    fn sharedDown(f: *Fixture, rows: usize, way: Way) !void {
        switch (way) {
            .direct => try f.lane.matmulF32Strided(f.r.s, f.shared_x, width, rows, f.shared, f.shared_slots[1] + 10 * d * 4, slots * d),
            .lane, .prompt => {
                if (way == .lane) try f.lane.matmulF32(f.r.s, f.shared_x, width, rows, f.shared, f.shared_out, 0) else try f.prompt.matmulOut(f.r.s, f.shared_x, width, rows, f.shared, f.shared_out, .f32);
                try f.glue.sharedScatter(f.r.s, f.shared_out, f.shared_slots[if (way == .lane) 0 else 1], rows, d, slots);
            },
        }
    }

    fn poison(f: *Fixture, ptr: u64, bytes: usize) !void {
        const b: cuda.DeviceBuffer = .{ .d = f.r.gpu.d, .ptr = ptr, .len = bytes };
        try b.fill8(0xff, f.r.s.handle);
    }

    fn compare(f: *Fixture, rows: usize) !void {
        const a = f.r.gpu.gpa;
        try f.poison(f.shared_slots[0], most * slots * d * 4);
        try f.sharedDown(rows, .lane);
        const want_shared = try a.dupe(u8, try f.r.bytes(f.shared_slots[0], most * slots * d * 4));
        defer a.free(want_shared);
        for (0..5) |_| {
            try f.poison(f.shared_slots[1], most * slots * d * 4);
            try f.sharedDown(rows, .direct);
            try check.sameBytes("slot 10 and untouched routed slots", try f.r.bytes(f.shared_slots[1], want_shared.len), want_shared);
        }
        // the prompt tile that main's nativeMatmul picks from prompt_rows rows writes the lane's bytes too
        try f.poison(f.shared_slots[1], most * slots * d * 4);
        try f.sharedDown(rows, .prompt);
        try check.sameBytes("prompt tile then scatter", try f.r.bytes(f.shared_slots[1], want_shared.len), want_shared);
        try check.expect(try f.r.intact(), "Shared down buffer guards", .{});
        check.pass("moetile rows={d}: direct and prompt tile equal the lane in all slots, five repeats, guards intact", .{rows});
    }

    fn time(f: *Fixture, rows: usize, way: Way) !f64 {
        var start = try cuda.Event.init(f.r.gpu.d, true);
        defer start.deinit();
        var end = try cuda.Event.init(f.r.gpu.d, true);
        defer end.deinit();
        for (0..3) |_| {
            try f.sharedDown(rows, way);
        }
        try start.record(f.r.s);
        for (0..10) |_| {
            try f.sharedDown(rows, way);
        }
        try end.record(f.r.s);
        try end.synchronize();
        return @as(f64, try cuda.Event.elapsedMs(start, end)) / 10;
    }
};

pub fn run(gpu: check.Gpu) !void {
    var f = try Fixture.init(gpu);
    defer f.deinit();
    for (row_counts) |rows| try f.compare(rows);
}

pub fn bench(gpu: check.Gpu) !void {
    var f = try Fixture.init(gpu);
    defer f.deinit();
    for (bench_rows) |rows| {
        try f.compare(rows);
        // six rounds, each way's turn rotating within a round
        var totals = [_]f64{ 0, 0, 0 };
        for (0..6) |round| for (0..3) |j| {
            const i = (round + j) % 3;
            totals[i] += try f.time(rows, @fromBackingInt(@intCast(i)));
        };
        for (&totals) |*t| t.* /= 6;
        check.pass("moetile rows={d}: lane+scatter {d:.4} ms, prompt tile+scatter {d:.4} ms, direct slot {d:.4} ms", .{ rows, totals[0], totals[1], totals[2] });
    }
}

test "Shared down registrations cover four exact geometries and replacement hooks" {
    const records = @import("core").contract.registry.all_cuda;
    const names = cuda.qmmf_shared.symbols;
    const shared = [_]u32{ 24576, 40960, 49152, 65536 };
    for (names, shared) |name, smem| {
        const record = for (records) |record| {
            if (std.mem.eql(u8, name, record.launch.symbol)) break record;
        } else return error.MissingMoETileRecord;
        try std.testing.expectEqual(smem, record.launch.dynamic);
        try std.testing.expectEqual(@as(u32, 128), record.launch.threads);
        try std.testing.expect(!record.geometry_pending);
    }
    var lane: cuda.qmmf.Lane = undefined;
    for (0..4) |i| {
        var p: @import("contract_qmmf_shared.zig").Product = .{ .rig = undefined, .lane = &lane, .index = i };
        const kernel = p.kernel();
        try std.testing.expect(kernel.spec.replaces != null and kernel.vt.replaced != null);
    }
}
