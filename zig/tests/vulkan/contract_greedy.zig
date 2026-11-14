// Vulkan greedy contract: one composed draw checks non-finite refusal and lowest-index ties against host greedyRef.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const fx = @import("stage5_fixture.zig");

const c = contract.spec;
const g = contract.gates;

// The shell reserves 256 bytes before data, matching Vulkan's maximum minStorageBufferOffsetAlignment.
pub const guard: usize = fx.at * 2;
const pattern: u8 = 0xa5;

// One workgroup folds a row: max_chunks mirrors greedy.zig's max_chunks; fx.chunk mirrors its block_elements.
const max_chunks: usize = 256;

/// A row the combine folds in its single workgroup cannot exceed.
const max_row: usize = max_chunks * fx.chunk;

// Refused rows retain this reset index, so unwritten output cannot pass as token 0.
pub const refused: u32 = std.math.maxInt(u32);

pub const rule = "any non-finite anywhere in the row refuses it, winner or not; the largest value wins; " ++
    "exact ties (including +0/-0) go to the lowest index";

fn flat(_: c.Shape) usize {
    return 0;
}

// Served vocab and draft-window rows cover one row at 2 elements and chunk edges 4094, 4096, 4098.
pub fn spec() c.Spec {
    return .{
        .name = "vulkan greedy draw: finite_logits, argreduce_chunk, argreduce_combine (bf16)",
        .accum = .none,
        .depth = flat,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = fx.chunk, .step = 2, .min = 2 }, .{} },
        .served = &.{
            .{ .rows = 1, .n = fx.vocab, .k = 0 },
            .{ .rows = 4, .n = fx.vocab, .k = 0 },
            .{ .rows = 1, .n = max_row, .k = 0 },
        },
    };
}

/// The control's registration: it claims the same rule and must fail it.
pub fn controlSpec() c.Spec {
    return .{
        .name = "control: fmax row max (drops a NaN)",
        .accum = .none,
        .depth = flat,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = spec().tiles,
        .served = spec().served,
    };
}

// Input data sits between downloaded guard spans for repeat checks of unchanged inputs and intact guards.
pub const Rig = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    timeout: u64,
    // prepare owns the guarded shell; views copy its handle but do not deinitialize it.
    shell: ?vk.Buffer = null,
    data_bytes: usize = 0,
    seen: std.ArrayList(u8) = .empty,
    whole: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.seen.deinit(r.a);
        r.whole.deinit(r.a);
        r.outs.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        if (r.shell) |b| {
            var s = b;
            s.deinit();
            r.shell = null;
        }
        r.outs.clearRetainingCapacity();
    }

    // Upload the pattern with data; fresh device memory is zero, not the pattern required by intact().
    pub fn shellFor(r: *Rig, data: []const u8) ![]u8 {
        const all = try r.a.alloc(u8, data.len + 2 * guard);
        errdefer r.a.free(all);
        @memset(all, pattern);
        @memcpy(all[guard..][0..data.len], data);
        return all;
    }

    /// Uploads `data` between two guard spans; the shell's data window starts at `guard`.
    pub fn input(r: *Rig, data: []const u8) !vk.Buffer {
        const all = try r.shellFor(data);
        defer r.a.free(all);
        var b = try vk.Buffer.init(r.dev, .device, all.len);
        errdefer b.deinit();
        try vk.orExit(vk.upload(r.cmds, b, 0, r.staging, all, r.timeout));
        r.shell = b;
        r.data_bytes = data.len;
        return b;
    }

    /// Every input's data window now, on the host: a run must leave it as prepared.
    pub fn inputBytes(r: *Rig) ![]const u8 {
        const b = r.shell orelse return error.Invalid;
        try r.seen.resize(r.a, r.data_bytes);
        try vk.orExit(vk.download(r.cmds, b, guard, r.staging, r.seen.items, r.timeout));
        return r.seen.items;
    }

    /// Every guard of the shell still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        const b = r.shell orelse return error.Invalid;
        try r.whole.resize(r.a, r.data_bytes + 2 * guard);
        try vk.orExit(vk.download(r.cmds, b, 0, r.staging, r.whole.items, r.timeout));
        for (r.whole.items, 0..) |v, i| {
            if (i >= guard and i < guard + r.data_bytes) continue;
            if (v != pattern) return false;
        }
        return true;
    }

    /// The bytes the last run wrote, on the host.
    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.outs.items;
    }

    // Draw outputs use separate internal result and partial buffers; the control owns its maxima allocation.
    pub fn overlap(_: *Rig) bool {
        return false;
    }
};

// Same shapes, fill, and check for draw and control; only run differs, so the control fails the named gate.
pub const Greedy = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: Rig,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    timeout: u64,
    shaders: vk.greedy.Shaders,
    /// Present turns the registration into the control.
    control_spirv: ?[]align(4) const u8,
    greedy: ?vk.Greedy = null,
    pipe: ?vk.dispatch.Pipeline = null,
    maxima: ?vk.Buffer = null,
    shape: c.Shape = undefined,
    shell: vk.Buffer = undefined,
    views: []vk.tensor.View = &.{},
    bits: []u16 = &.{},
    msg: [192]u8 = undefined,

    pub fn init(
        a: std.mem.Allocator,
        dev: *const vk.Device,
        cmds: *vk.Commands,
        staging: vk.Buffer,
        timeout: u64,
        shaders: vk.greedy.Shaders,
        control_spirv: ?[]align(4) const u8,
    ) Greedy {
        return .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .timeout = timeout },
            .dev = dev,
            .cmds = cmds,
            .staging = staging,
            .timeout = timeout,
            .shaders = shaders,
            .control_spirv = control_spirv,
        };
    }

    pub fn deinit(self: *Greedy) void {
        self.dropDevices();
        self.rig.deinit();
        self.arena.deinit();
    }

    fn dropDevices(self: *Greedy) void {
        if (self.greedy) |*x| x.deinit();
        self.greedy = null;
        if (self.pipe) |*p| p.deinit();
        self.pipe = null;
        if (self.maxima) |*m| m.deinit();
        self.maxima = null;
    }

    const Arm = enum { draw, control };

    fn arm(self: *const Greedy) Arm {
        return if (self.control_spirv == null) .draw else .control;
    }

    pub fn kernel(self: *Greedy) g.Kernel {
        return .{
            .spec = if (self.arm() == .draw) spec() else controlSpec(),
            .ctx = self,
            .vt = &vt,
            .control = if (self.arm() == .draw) null else .nonfinite,
        };
    }

    const vt = g.hooks(Greedy);

    pub fn prepare(self: *Greedy, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n = shape.n;
        const rows = shape.rows;
        if (rows == 0 or n == 0) return false;
        if (n % 2 != 0) return false; // an odd bf16 count has no legal view: the word-padding rule refuses
        if (rows > 64 or n > max_row) return false;
        self.dropDevices();
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        self.bits = try a.alloc(u16, rows * n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..rows) |r| fillRow(self.bits[r * n ..][0..n], rng);
        if (fill == .nonfinite) plantNonFinite(self.bits, rows, n);
        const data = try a.alloc(u8, rows * n * 2);
        for (self.bits, 0..) |b, i| std.mem.writeInt(u16, data[2 * i ..][0..2], b, .little);
        self.shell = try self.rig.input(data);
        self.views = try a.alloc(vk.tensor.View, rows);
        for (self.views, 0..) |*v, r|
            v.* = try vk.tensor.View.init(&self.shell, .bf16, guard + r * n * 2, n);
        switch (self.arm()) {
            .draw => self.greedy = try vk.Greedy.init(self.dev, self.shaders, n),
            .control => {
                self.pipe = try vk.dispatch.Pipeline.init(self.dev, self.control_spirv.?, .{
                    .bindings = 2,
                    .push_bytes = 4, // the control's push block: one uint, the row length
                    .writes = 1 << 1,
                });
                self.maxima = try vk.Buffer.init(self.dev, .device, rows * 4);
            },
        }
        try self.rig.outs.resize(self.a, rows * 4);
        @memset(self.rig.outs.items, 0xff); // an unwritten row reads as the refused marker
        return true;
    }

    pub fn run(self: *Greedy) !void {
        const n = self.shape.n;
        const rows = self.shape.rows;
        switch (self.arm()) {
            .draw => {
                const greedy = &self.greedy.?;
                for (0..rows) |r| {
                    const tok: u32 = greedy.select(self.cmds, self.staging, self.views[r], self.timeout) catch |e| switch (e) {
                        // The draw rule's refusal; check names the row and its evidence.
                        error.NonFinite => refused,
                        else => return e,
                    };
                    std.mem.writeInt(u32, self.rig.outs.items[4 * r ..][0..4], tok, .little);
                }
            },
            .control => {
                const pipe = &self.pipe.?;
                const maxima = &self.maxima.?;
                try pipe.bind(&.{
                    .{ .buffer = &self.shell, .offset = guard, .len = rows * n * 2 },
                    .{ .buffer = maxima, .len = rows * 4 },
                });
                var push: [4]u8 align(4) = undefined;
                std.mem.writeInt(u32, &push, @intCast(n), .little);
                const cmd = try self.cmds.begin();
                try pipe.record(cmd, &push, .{ @intCast(rows), 1, 1 });
                try self.cmds.submit(self.timeout);
                try vk.orExit(vk.download(self.cmds, maxima.*, 0, self.staging, self.rig.outs.items, self.timeout));
            },
        }
    }

    pub fn count(self: *Greedy) usize {
        return self.shape.rows * 4;
    }

    /// Raw outputs check themselves; the float64 slots stay empty.
    pub fn reference(_: *Greedy, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    // A planted row refuses at its first offender; a finite row matches greedyRef's token.
    pub fn check(self: *Greedy, at: g.Gate) !?[]const u8 {
        const n = self.shape.n;
        const rows = self.shape.rows;
        switch (at) {
            .nonfinite => for (0..rows) |r| {
                const row = self.bits[r * n ..][0..n];
                if (!rowHasNonFinite(row))
                    return try std.fmt.bufPrint(&self.msg, "row {d}: the planted fill holds no non-finite, so the gate proves nothing", .{r});
                const bad = fx.firstBadBits(row);
                const got = std.mem.readInt(u32, self.rig.outs.items[4 * r ..][0..4], .little);
                if (got != refused)
                    return try std.fmt.bufPrint(&self.msg, "row {d} did not refuse: it reported {d} where the first non-finite sits at {d}", .{ r, got, bad });
            },
            .shapes => {
                const wide = try self.arena.allocator().alloc(f32, n);
                for (0..rows) |r| {
                    const row = self.bits[r * n ..][0..n];
                    for (row, 0..) |b, i| wide[i] = c.fromBf16(b);
                    const want = vk.greedy.greedyRef(wide) catch |e|
                        return try std.fmt.bufPrint(&self.msg, "row {d}: the reference refused a finite row: {s}", .{ r, @errorName(e) });
                    const got = std.mem.readInt(u32, self.rig.outs.items[4 * r ..][0..4], .little);
                    if (got == refused)
                        return try std.fmt.bufPrint(&self.msg, "row {d} refused a finite row", .{r});
                    if (got != want)
                        return try std.fmt.bufPrint(&self.msg, "row {d} picks {d}, greedyRef gives {d}", .{ r, got, want });
                }
            },
            else => {},
        }
        return null;
    }
};

// Eighth-step logits provide 256 values and common ties, testing lowest-index selection across chunk folds.
fn fillRow(data: []u16, rng: std.Random) void {
    for (data) |*d| d.* = c.toBf16(@as(f32, @floatFromInt(rng.uintLessThan(u32, 256))) / 8 - 16);
}

// Every row: a finite would-be winner, then a losing NaN, +inf and -inf; every third row loses its winner to a NaN too.
fn plantNonFinite(bits: []u16, rows: usize, n: usize) void {
    for (0..rows) |r| {
        const row = bits[r * n ..][0..n];
        row[n / 2] = c.toBf16(5.0);
        row[0] = 0xff80; // -inf
        row[n / 3] = 0x7fc0; // NaN
        row[n - 1] = 0x7f80; // +inf
        if (r % 3 == 0) row[n / 2] = 0x7fc0;
    }
}

/// The finite scan's own predicate: exponent all ones is non-finite bf16.
pub fn rowHasNonFinite(row: []const u16) bool {
    for (row) |b| if ((b & 0x7fff) >= 0x7f80) return true;
    return false;
}

const testing = std.testing;

test "the spec serves the vocab, a draft window and the max_chunks edge; the chunk edges ride one row" {
    const s = spec();
    try testing.expect(s.nan.stated.len > 0);
    try testing.expectEqual(fx.vocab, s.served[0].n);
    try testing.expectEqual(@as(usize, 4), s.served[1].rows);
    try testing.expectEqual(max_row, s.served[2].n);
    try testing.expectEqual(@as(usize, 1_048_576), max_row); // 256 chunks of 4096
    const list = try c.shapes(testing.allocator, s);
    defer testing.allocator.free(list);
    try testing.expectEqual(@as(usize, 7), list.len);
    try testing.expectEqual(fx.vocab, list[0].n);
    try testing.expectEqual(@as(usize, 4), list[1].rows);
    try testing.expectEqual(max_row, list[2].n);
    for (list[3..], [_]usize{ 2, fx.chunk - 2, fx.chunk, fx.chunk + 2 }) |x, want| {
        try testing.expectEqual(@as(usize, 1), x.rows);
        try testing.expectEqual(want, x.n);
    }
}

test "the eighths fill is bf16-exact and tie-rich, and greedyRef and the fixture reference agree" {
    const a = testing.allocator;
    const n = 512; // more values than the fill's 256 classes, so ties must exist
    const bits = try a.alloc(u16, n);
    defer a.free(bits);
    var prng = std.Random.DefaultPrng.init(0x6b63);
    fillRow(bits, prng.random());
    var distinct: usize = 0;
    for (bits, 0..) |b, at| {
        const v = c.fromBf16(b);
        const k = (v + 16) * 8; // the fill's eighth class: an exact small integer
        try testing.expect(k == @trunc(k) and k >= 0 and k < 256);
        if (std.mem.indexOfScalar(u16, bits[0..at], b) == null) distinct += 1;
    }
    try testing.expect(distinct < n);
    for (0..256) |k| // every drawn class round-trips bf16 exactly
        try testing.expectEqual(@as(f32, @as(f32, @floatFromInt(k)) / 8 - 16), c.fromBf16(c.toBf16(@as(f32, @floatFromInt(k)) / 8 - 16)));
    const wide = try a.alloc(f32, n);
    defer a.free(wide);
    for (bits, 0..) |b, i| wide[i] = c.fromBf16(b);
    const want = try vk.greedy.greedyRef(wide);
    try testing.expectEqual(try fx.referenceBits(bits), want);
    var best = -std.math.inf(f32);
    for (wide) |v| best = @max(best, v);
    try testing.expectEqual(@as(u32, @intCast(std.mem.indexOfScalar(f32, wide, best).?)), want); // the tie's lowest index
}

test "the nonfinite plant covers every row, keeps the loser rows' winner, and names the first offender" {
    const a = testing.allocator;
    const rows = 3;
    const n = 4096;
    const bits = try a.alloc(u16, rows * n);
    defer a.free(bits);
    plantNonFinite(bits, rows, n);
    for (0..rows) |r| {
        const row = bits[r * n ..][0..n];
        try testing.expect(rowHasNonFinite(row));
        try testing.expectEqual(@as(usize, 0), fx.firstBadBits(row)); // the losing -inf at index 0
        try testing.expectEqual(@as(u16, 0xff80), row[0]);
        try testing.expectEqual(@as(u16, 0x7fc0), row[n / 3]);
        try testing.expectEqual(@as(u16, 0x7f80), row[n - 1]);
        if (r % 3 != 0) try testing.expectEqual(@as(f32, 5.0), c.fromBf16(row[n / 2]));
    }
}

test "the refused marker is the word the draw's own decode refuses to pass" {
    try testing.expectEqual(std.math.maxInt(u32), refused);
    try testing.expectError(error.VulkanFailed, vk.greedy.decode(4, .{ .finite_flag = 0, .first_bad = refused, .index = refused }));
    try testing.expectError(error.NonFinite, vk.greedy.decode(4, .{ .finite_flag = 1, .first_bad = 1, .index = 0 }));
}

test "the input shell uploads the pattern intact() checks; fresh device memory reads zeros" {
    var rig = Rig{ .a = testing.allocator, .dev = undefined, .cmds = undefined, .staging = undefined, .timeout = 0 };
    defer rig.deinit();
    const data = [_]u8{ 1, 2, 3, 4 };
    const all = try rig.shellFor(&data);
    defer testing.allocator.free(all);
    try testing.expectEqual(data.len + 2 * guard, all.len);
    for (all[0..guard]) |v| try testing.expectEqual(pattern, v);
    for (all[guard + data.len ..]) |v| try testing.expectEqual(pattern, v);
    try testing.expectEqualSlices(u8, &data, all[guard..][0..data.len]);
    // intact() skips data between patterned spans; without uploading the pattern, the repeat gate would fail.
    try testing.expect(all[0] == pattern and all[guard - 1] == pattern);
}
