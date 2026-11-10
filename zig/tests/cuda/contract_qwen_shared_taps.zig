//! Shared prompt taps through the actual helper and CUDA tap/gather kernels, with guarded caller-owned buffers.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const core = @import("core");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const helper = qwen.prompt_taps;
const c = core.contract.spec;
const taps = 5;
const max_rows = 8192;
const shared_rows = 2112;
const max_positions = 20000;
const absent = std.math.maxInt(u32);
const Seg = qwen.cache.Seg;
const Selected = extern struct { n: c_int, rows: [64]c_int };
const Cache = struct { tap: ?cuda.DeviceBuffer, len: usize = 0, tap_epoch: u64 = 1, tap_begin: u64 = 0, tap_end: u64 = 0 };
const Part = struct { cache: *Cache, rows: usize };
const Engine = struct {
    n_taps: usize = taps,
    tap_capacity: usize = helper.private_rows,
    prompt_taps: helper.State,
    g: struct { hidden: usize },
    stream: cuda.Stream,
};

const Probe = struct {
    rig: Rig,
    e: Engine,
    tap: cuda.Function,
    gather: cuda.Function,
    selected: cuda.Function,
    h: []u16,
    r: []u16,
    input: [2]u64,
    caches: [2]Cache,
    origin: [2][max_positions]u32 = @splat(@splat(absent)),
    out: u64,
    saved: u64,
    segs: cuda.DeviceBuffer,
    params: cuda.DeviceBuffer,
    map: cuda.DeviceBuffer,

    fn view(gpu: check.Gpu, ptr: u64, len: usize) cuda.DeviceBuffer {
        return .{ .d = gpu.d, .ptr = ptr, .len = len, .borrowed = true };
    }
    fn init(gpu: check.Gpu, stream: cuda.Stream, d: usize, exceptional: bool, functions: [3]cuda.Function) !Probe {
        var self: Probe = undefined;
        self.rig = .{ .gpu = gpu, .s = stream };
        errdefer self.rig.deinit();
        self.h = try gpu.gpa.alloc(u16, (max_rows + 32) * d);
        errdefer gpu.gpa.free(self.h);
        self.r = try gpu.gpa.alloc(u16, self.h.len);
        errdefer gpu.gpa.free(self.r);
        for (self.h, self.r, 0..) |*h, *r, i| {
            h.* = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 31) % 8192)) - 4096)) / 256);
            r.* = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast((i * 29 + 7) % 4096)) - 2048)) / 512);
            if (exceptional) switch (i % d) {
                0 => h.* = 0x7fc1,
                1 => h.* = 0x7f80,
                2 => h.* = 0xff80,
                3 => {
                    h.* = 0x7f80;
                    r.* = 0xff80;
                },
                else => {},
            };
        }
        self.input = .{ try self.rig.input(std.mem.sliceAsBytes(self.h)), try self.rig.input(std.mem.sliceAsBytes(self.r)) };
        const pitch = taps * d * 2;
        const shared = try self.rig.scratch(shared_rows * pitch);
        self.e = .{ .g = .{ .hidden = d }, .stream = stream, .prompt_taps = .{
            .buffer = view(gpu, shared, shared_rows * pitch),
            .rows = shared_rows,
            .row_bytes = pitch,
        } };
        for (&self.caches) |*cache| cache.* = .{ .tap = view(gpu, try self.rig.scratch(helper.private_rows * pitch), helper.private_rows * pitch) };
        self.origin = @splat(@splat(absent));
        self.out = try self.rig.output(helper.private_rows * pitch);
        self.saved = try self.rig.scratch(helper.private_rows * pitch);
        self.segs = view(gpu, try self.rig.scratch(2 * @sizeOf(Seg)), 2 * @sizeOf(Seg));
        self.params = view(gpu, try self.rig.scratch(3 * 4), 3 * 4);
        self.map = view(gpu, try self.rig.scratch((max_rows + 7) * 4), (max_rows + 7) * 4);
        self.tap = functions[0];
        self.gather = functions[1];
        self.selected = functions[2];
        return self;
    }
    fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.rig.gpu.gpa.free(self.h);
        self.rig.gpu.gpa.free(self.r);
    }
    fn launch(self: *Probe, f: cuda.Function, rows: usize, columns: usize, values: anytype) !void {
        var args: cuda.Args = .{};
        inline for (values) |value| args.add(value);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(columns) }, .block = .{ .x = 256 } }, self.e.stream, &args);
    }
    fn frame(self: *Probe, which: usize, rows: usize, prompt: bool, phase: usize) !void {
        const cache = &self.caches[which];
        const first = cache.len;
        const row0: usize = if (prompt) 0 else 7;
        const parts = [_]Part{.{ .cache = cache, .rows = rows }};
        var segs: [2]Seg = @splat(std.mem.zeroes(Seg));
        segs[1].tap = cache.tap.?.ptr;
        segs[1].tap_rows = helper.private_rows;
        segs[1].len = @intCast(first);
        segs[1].rows = @intCast(rows);
        segs[1].row0 = @intCast(row0);
        segs[1].taps = taps;
        try helper.begin(&self.e, &parts, segs[1..], prompt);
        errdefer helper.abort(&self.e, &parts);
        const a = self.rig.gpu.gpa;
        const map = try a.alloc(c_int, rows + row0);
        defer a.free(map);
        @memset(map, 1);
        @memset(map[0..row0], 0);
        try self.segs.uploadAsync(0, std.mem.sliceAsBytes(&segs), self.e.stream.handle);
        try self.map.uploadAsync(0, std.mem.sliceAsBytes(map), self.e.stream.handle);
        const params = [_]c_int{ 0, 0, @intCast(rows + row0) };
        try self.params.uploadAsync(0, std.mem.sliceAsBytes(&params), self.e.stream.handle);
        const d = self.e.g.hidden;
        for (0..taps) |tap| try self.launch(self.tap, rows + row0, (d + 2047) / 2048, .{
            self.input[0] + (phase + tap) * d * 2, self.input[1] + (phase + tap) * d * 2,
            self.segs.ptr,                         self.map.ptr,
            @as(c_int, @intCast(tap)),             @as(c_int, @intCast(d)),
            self.params.ptr,
        });
        try helper.publish(&self.e, &parts, prompt);
        for (0..rows) |r| self.origin[which][first + r] = @intCast(r + row0 + phase);
        cache.len += rows;
    }
    fn fetch(self: *Probe, which: usize, start: usize, requested: usize, read_start: usize, count: usize, selected: ?[]const u32) !u64 {
        const src = (try helper.read(&self.e, &self.caches[which], start, requested, 2048)) orelse return error.MissingSharedTapView;
        const cap: c_int = @intCast(helper.modulus(&self.e, src));
        const pitch = taps * self.e.g.hidden;
        try view(self.rig.gpu, self.out, helper.private_rows * pitch * 2).fill8(0xff, self.e.stream.handle);
        if (selected) |indices| {
            var choice: Selected = .{ .n = @intCast(indices.len), .rows = @splat(0) };
            for (indices, 0..) |index, i| choice.rows[i] = @intCast(index);
            try self.launch(self.selected, count, 1, .{ src, cap, @as(c_longlong, @intCast(read_start)), choice, @as(c_int, @intCast(pitch)), self.out });
        } else try self.launch(self.gather, count, 1, .{ src, cap, @as(c_longlong, @intCast(read_start)), @as(c_int, @intCast(pitch)), self.out });
        const bytes = try self.rig.bytes(self.out, helper.private_rows * pitch * 2);
        const got = std.mem.bytesAsSlice(u16, bytes);
        for (0..count) |row| {
            const position = read_start + if (selected) |indices| indices[row] else row;
            const origin = self.origin[which][position];
            try check.expect(origin != absent, "shared tap oracle has no row {d}", .{position});
            for (0..taps) |tap| for (0..self.e.g.hidden) |col| {
                const source = (origin + tap) * self.e.g.hidden + col;
                const sum: f32 = @floatCast(@as(f64, c.fromBf16(self.h[source])) + @as(f64, c.fromBf16(self.r[source])));
                const want = c.toBf16(sum);
                const actual = got[(row * taps + tap) * self.e.g.hidden + col];
                const ok = if (std.math.isNan(c.fromBf16(want))) std.math.isNan(c.fromBf16(actual)) else actual == want;
                try check.expect(ok, "shared tap row {d} tap {d} col {d}:0x{x:0>4} expected0x{x:0>4}", .{ position, tap, col, actual, want });
            };
        }
        for (got[count * pitch ..]) |word| try check.expect(word == 0xffff, "shared tap gather touched padding", .{});
        return std.hash.Wyhash.hash(0, bytes);
    }
    fn refused(self: *Probe, which: usize, start: usize, count: usize) !void {
        _ = helper.read(&self.e, &self.caches[which], start, count, 2048) catch |err| {
            if (err == error.RowsNotHeld) return;
            return err;
        };
        return error.StaleSharedTapReadAccepted;
    }
    fn saveSharedTail(self: *Probe, end: usize) !void {
        try self.launch(self.gather, 65, 1, .{
            self.e.prompt_taps.buffer.?.ptr,              @as(c_int, shared_rows), @as(c_longlong, @intCast(end - 65)),
            @as(c_int, @intCast(taps * self.e.g.hidden)), self.saved,
        });
    }
    fn sameAsShared(self: *Probe) !void {
        const bytes = 65 * taps * self.e.g.hidden * 2;
        const saved = try self.rig.gpu.gpa.dupe(u8, try self.rig.bytes(self.saved, bytes));
        defer self.rig.gpu.gpa.free(saved);
        try check.sameBytes("private tap tail vs shared BF16 words", try self.rig.bytes(self.out, bytes), saved);
    }
    fn round(self: *Probe) ![7]u64 {
        self.e.prompt_taps.held = null;
        self.e.prompt_taps.pending = null;
        self.e.prompt_taps.active = false;
        self.origin = @splat(@splat(absent));
        for (&self.caches) |*cache| {
            helper.reset(cache);
            cache.len = 0;
        }
        var trace: [7]u64 = undefined;
        try self.frame(0, 65, true, 0);
        trace[0] = try self.fetch(0, 0, 64, 0, 64, null);
        const a_start = self.caches[0].len;
        try self.frame(0, max_rows, true, 1);
        try self.saveSharedTail(self.caches[0].len);
        self.caches[1].len = 2090;
        try self.frame(1, max_rows, true, 3); // no GPU wait between A's copy and B overwriting the global ring
        const a_end = self.caches[0].len;
        const b_end = self.caches[1].len;
        trace[1] = try self.fetch(0, a_end - 65, 65, a_end - 65, 65, null);
        try self.sameAsShared();
        try self.refused(0, a_start, max_rows);
        trace[2] = try self.fetch(1, 2090, max_rows, b_end - 1 - 2048, 65, null);
        trace[3] = try self.fetch(1, 2090, max_rows, b_end - 65, 65, null);
        helper.invalidate(&self.e, &self.caches[1]);
        try self.refused(1, b_end - 1, 1);
        const verify = self.caches[0].len;
        try self.frame(0, 64, false, 5);
        const indices = [_]u32{ 0, 7, 31, 63 };
        trace[4] = try self.fetch(0, verify, 64, verify, indices.len, &indices);
        self.caches[0].len = verify + 1;
        try self.frame(0, 1, false, 6);
        try self.refused(0, verify, 1);
        trace[5] = try self.fetch(0, verify + 1, 1, verify + 1, 1, null);
        try self.frame(0, 17, true, 7);
        const failed = self.caches[0].len - 17;
        const parts = [_]Part{.{ .cache = &self.caches[0], .rows = 17 }};
        helper.abort(&self.e, &parts); // a later forward readback failed after publication
        try self.refused(0, failed, 17);
        const recover = self.caches[0].len;
        try self.frame(0, 65, true, 8);
        trace[6] = try self.fetch(0, recover, 64, recover, 64, null);
        return trace;
    }
};

pub fn run(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var model = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer model.unload();
    var draft = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_draft);
    defer draft.unload();
    const functions = [3]cuda.Function{ try model.function("tf_q35_tap"), try draft.function("tf_q35d_gather_taps"), try draft.function("tf_q35d_gather_rows") };
    for ([_]usize{ 256, 5120 }) |d| for ([_]bool{ false, true }) |exceptional| {
        var probe = try Probe.init(gpu, stream, d, exceptional, functions);
        defer probe.deinit();
        const before = try gpu.gpa.dupe(u8, try probe.rig.inputBytes());
        defer gpu.gpa.free(before);
        const first = try probe.round();
        for (1..5) |_| {
            const current = try probe.round();
            try check.expect(std.mem.eql(u64, &first, &current), "shared tap repeat changed BF16 output bytes", .{});
        }
        try check.sameBytes("shared tap immutable inputs", try probe.rig.inputBytes(), before);
        try check.expect(try probe.rig.intact(), "shared tap guard corrupted", .{});
        check.pass("shared prompt taps width{d} exceptional{}:7 checks x5 repeats, FP64/BF16 exact, guards and owners", .{ d, exceptional });
    };
    std.debug.print("SUMMARY shared prompt taps:4 fixtures,140 checked gathers,0 failures\n", .{});
}
