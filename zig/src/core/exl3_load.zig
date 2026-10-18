//! EXL3 checkpoint groups: every `{prefix}.trellis` and its parts, admitted from the safetensors headers (format.py).
const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const st = @import("safetensors.zig");

pub const Codebook = enum { inst3, mcg, mul1 };

/// format.py's MARKERS: a scalar int32 tensor per codebook, whose value must be the magic.
pub const Marker = struct { name: []const u8, codebook: Codebook, value: u32 };
pub const markers = [_]Marker{
    .{ .name = "mcg", .codebook = .mcg, .value = 0xCBAC1FED },
    .{ .name = "mul1", .codebook = .mul1, .value = 0x83DCD12D },
};

/// One admitted group: a trellis int16 [K/16, N/16, 16 * bits], its scales, and the optional bias.
pub const Group = struct {
    prefix: []const u8,
    halves: u16, // 2 * bits; odd halves are the mul1 half steps
    codebook: Codebook,
    k: usize,
    n: usize,
    suh: bool, // fp16 scales; false: the packed sign words su and sv
    bias: bool,
};

/// Counts of the admitted groups, by halves and for the routed experts' down projections.
pub const Summary = struct {
    groups: usize = 0,
    inst3: usize = 0,
    mcg: usize = 0,
    mul1: usize = 0,
    suh_svh: usize = 0,
    su_sv: usize = 0,
    bias: usize = 0,
    bits: [17]usize = @splat(0), // by halves, 2 to 16
    experts_down: [17]usize = @splat(0), // *.experts.*.down_proj by halves

    fn bump(s: *Summary, g: Group) void {
        s.groups += 1;
        switch (g.codebook) {
            .inst3 => s.inst3 += 1,
            .mcg => s.mcg += 1,
            .mul1 => s.mul1 += 1,
        }
        if (g.suh) s.suh_svh += 1 else s.su_sv += 1;
        if (g.bias) s.bias += 1;
        s.bits[g.halves] += 1;
        if (std.mem.endsWith(u8, g.prefix, ".down_proj") and std.mem.indexOf(u8, g.prefix, ".experts.") != null) s.experts_down[g.halves] += 1;
    }
};

/// One group's tensors: the trellis words, its scales, and the optional bias.
pub const Parts = struct {
    trellis: st.Tensor,
    suh: st.Tensor,
    svh: st.Tensor,
    bias: ?st.Tensor = null,
    suh_packed: bool = false, // su/sv sign words instead of fp16 scales
};

pub const Scan = struct {
    gpa: std.mem.Allocator,
    ck: checkpoint.Checkpoint,
    groups: []Group,
    summary: Summary = .{},
    bad: usize = 0, // trellis tensors refused: another format's tensors, a broken or half-written group

    pub fn deinit(s: *Scan) void {
        for (s.groups) |g| s.gpa.free(g.prefix);
        s.gpa.free(s.groups);
        s.ck.close();
    }

    /// The group a tensor prefix names, if any.
    pub fn find(s: *const Scan, prefix: []const u8) ?usize {
        for (s.groups, 0..) |g, i| if (std.mem.eql(u8, g.prefix, prefix)) return i;
        return null;
    }

    /// One group's tensors, their layouts checked on the way out.
    pub fn parts(s: *Scan, a: std.mem.Allocator, gi: usize) !Parts {
        const g = s.groups[gi];
        const trellis = try s.ck.get(try std.mem.concat(a, u8, &.{ g.prefix, ".trellis" }));
        if (!trellis.is(.i16, &.{ g.k / 16, g.n / 16, @as(usize, g.halves) * 8 })) return error.BadTrellis;
        const su_name = if (g.suh) "suh" else "su";
        const sv_name = if (g.suh) "svh" else "sv";
        const suh = try s.ck.get(try std.mem.concat(a, u8, &.{ g.prefix, ".", su_name }));
        const svh = try s.ck.get(try std.mem.concat(a, u8, &.{ g.prefix, ".", sv_name }));
        const in_shape: []const usize = if (g.suh) &.{g.k} else &.{g.k / 16};
        const out_shape: []const usize = if (g.suh) &.{g.n} else &.{g.n / 16};
        const dtype: st.DType = if (g.suh) .f16 else .i16;
        if (!suh.is(dtype, in_shape) or !svh.is(dtype, out_shape)) return error.BadScales;
        var bias: ?st.Tensor = null;
        if (g.bias) {
            bias = try s.ck.get(try std.mem.concat(a, u8, &.{ g.prefix, ".bias" }));
            if (!bias.?.is(.f16, &.{g.n})) return error.BadBias;
        }
        return .{ .trellis = trellis, .suh = suh, .svh = svh, .bias = bias, .suh_packed = !g.suh };
    }
};

/// Every trellis tensor of `dir`'s shards, admitted or counted as refused.
pub fn scan(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Scan {
    var ck = try checkpoint.Checkpoint.openModel(gpa, io, dir);
    errdefer ck.close();
    var groups: std.ArrayList(Group) = .empty;
    errdefer {
        for (groups.items) |g| gpa.free(g.prefix);
        groups.deinit(gpa);
    }
    var summary: Summary = .{};
    var bad: usize = 0;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for (ck.files.items) |*f| for (f.names.keys(), f.names.values()) |name, e| {
        if (!std.mem.endsWith(u8, name, ".trellis")) continue;
        const prefix = name[0 .. name.len - ".trellis".len];
        if (admit(a, &ck, prefix, e)) |g| {
            try groups.ensureUnusedCapacity(gpa, 1);
            const owned = try gpa.dupe(u8, g.prefix);
            errdefer gpa.free(owned);
            groups.appendAssumeCapacity(.{ .prefix = owned, .halves = g.halves, .codebook = g.codebook, .k = g.k, .n = g.n, .suh = g.suh, .bias = g.bias });
            summary.bump(g);
        } else |err| {
            if (err == error.OutOfMemory) return err;
            bad += 1;
        }
    };
    return .{ .gpa = gpa, .ck = ck, .groups = try groups.toOwnedSlice(gpa), .summary = summary, .bad = bad };
}

/// parse_group's admission: the trellis layout, the codebook marker's value, and the group's parts.
fn admit(a: std.mem.Allocator, ck: *checkpoint.Checkpoint, prefix: []const u8, e: st.Entry) !Group {
    if (e.dtype != .i16 or e.rank != 3) return error.Refuse;
    const halves = e.shape[2] / 8; // the trellis's last dim is 16 * bits
    if (e.shape[2] % 8 != 0 or halves < 2 or halves > 16) return error.Refuse;
    const k = e.shape[0] * 16;
    const n = e.shape[1] * 16;
    if (k % 128 != 0 or n % 128 != 0) return error.Refuse;
    var codebook: Codebook = .inst3;
    var marker: ?Marker = null;
    for (markers) |m| if (ck.has(try std.mem.concat(a, u8, &.{ prefix, ".", m.name }))) {
        if (marker != null) return error.Refuse; // both codebooks' markers: not a loadable group
        marker = m;
        codebook = m.codebook;
    };
    if (halves % 2 == 1 and codebook != .mul1) return error.Refuse; // odd halves are the mul1 half steps
    if (marker) |m| {
        const t = try ck.get(try std.mem.concat(a, u8, &.{ prefix, ".", m.name }));
        if (t.bytes.len != 4 or std.mem.readInt(u32, t.bytes[0..4], .little) != m.value) return error.Refuse;
    }
    const su = try scale(a, ck, prefix, "suh", k, "su", k / 16);
    const sv = try scale(a, ck, prefix, "svh", n, "sv", n / 16);
    if (su == null or sv == null or su.? != sv.?) return error.Refuse; // one fp16, one packed: a half-written group
    const bias = try scale(a, ck, prefix, "bias", n, "", 0);
    return .{ .prefix = prefix, .halves = @intCast(halves), .codebook = codebook, .k = k, .n = n, .suh = su.? == .f16, .bias = bias != null };
}

const Scales = enum { f16, words }; // fp16 per row, or the packed sign words

/// One part's admission: the fp16 name, or the packed-words fallback; a missing input scale (or fallback) is refused.
fn scale(a: std.mem.Allocator, ck: *checkpoint.Checkpoint, prefix: []const u8, name: []const u8, len: usize, packed_name: []const u8, packed_len: usize) !?Scales {
    const full = try std.mem.concat(a, u8, &.{ prefix, ".", name });
    if (!ck.has(full)) {
        if (packed_name.len == 0) return null; // an optional part
        const packed_full = try std.mem.concat(a, u8, &.{ prefix, ".", packed_name });
        if (!ck.has(packed_full)) return error.Refuse;
        const p = try ck.get(packed_full);
        if (p.dtype != .i16 or p.numel() != packed_len) return error.Refuse;
        return .words;
    }
    const t = try ck.get(full);
    if (t.dtype != .f16 or t.numel() != len) return error.Refuse;
    return .f16;
}

test "a hand-built safetensors file admits its one group and refuses the scaleless one" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // one full group (trellis [8, 8, 32] i16, fp16 scales, the mul1 marker) and one without scales
    const trellis: [4096]u8 = @splat(0);
    const scales: [256]u8 = @splat(0);
    const marker = [_]u8{ 0x2D, 0xD1, 0xDC, 0x83 }; // 0x83DCD12D little-endian
    const Entry = st.Entry;
    const entries = [_]Entry{
        .{ .dtype = .i16, .rank = 3, .shape = .{ 8, 8, 32, 1, 1 }, .begin = 0, .end = 4096 },
        .{ .dtype = .f16, .rank = 1, .shape = .{ 128, 1, 1, 1, 1 }, .begin = 4096, .end = 4352 },
        .{ .dtype = .f16, .rank = 1, .shape = .{ 128, 1, 1, 1, 1 }, .begin = 4352, .end = 4608 },
        .{ .dtype = .i32, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .begin = 4608, .end = 4612 },
        .{ .dtype = .i16, .rank = 3, .shape = .{ 8, 8, 32, 1, 1 }, .begin = 4612, .end = 4612 + 4096 },
    };
    const names = [_][]const u8{ "a.trellis", "a.suh", "a.svh", "a.mul1", "b.trellis" };
    const dts = [_][]const u8{ "I16", "F16", "F16", "I32", "I16" };
    const data = try std.mem.concat(gpa, u8, &.{ &trellis, &scales, &scales, &marker, &trellis });
    defer gpa.free(data);
    var header: std.ArrayListUnmanaged(u8) = .empty;
    defer header.deinit(gpa);
    try header.appendSlice(gpa, "{");
    for (names, entries, dts, 0..) |name, e, dt, i| {
        if (i != 0) try header.appendSlice(gpa, ",");
        try header.print(gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, dt });
        for (e.shape[0..e.rank], 0..) |d, j| {
            if (j != 0) try header.appendSlice(gpa, ",");
            try header.print(gpa, "{d}", .{d});
        }
        try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ e.begin, e.end });
    }
    try header.appendSlice(gpa, "}");
    var file: std.ArrayListUnmanaged(u8) = .empty;
    defer file.deinit(gpa);
    try file.appendNTimes(gpa, 0, 8);
    std.mem.writeInt(u64, file.items[0..8], header.items.len, .little);
    try file.appendSlice(gpa, header.items);
    try file.appendSlice(gpa, data);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);

    var s = try scan(gpa, io, dir);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 1), s.bad);
    try std.testing.expectEqual(@as(usize, 1), s.summary.mul1);
    try std.testing.expectEqual(@as(usize, 1), s.summary.suh_svh);
    try std.testing.expectEqual(@as(usize, 1), s.summary.bits[4]); // a 32-word trellis is 4 halves
    const gi = s.find("a") orelse return error.TestUnexpectedResult;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const p = try s.parts(arena.allocator(), gi);
    try std.testing.expectEqual(@as(usize, 128), p.trellis.dim(0) * 16);
    try std.testing.expect(!p.suh_packed);
    try std.testing.expectEqual(@as(usize, 128), p.svh.numel());
}

test "packed sign planes the scan admits load as the parts" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // one full group (trellis [8, 8, 32] i16, fp16 scales, the mul1 marker) and one without scales
    const trellis: [4096]u8 = @splat(0);
    const scales: [16]u8 = @splat(0);
    const marker = [_]u8{ 0x2D, 0xD1, 0xDC, 0x83 }; // 0x83DCD12D little-endian
    const Entry = st.Entry;
    const entries = [_]Entry{
        .{ .dtype = .i16, .rank = 3, .shape = .{ 8, 8, 32, 1, 1 }, .begin = 0, .end = 4096 },
        .{ .dtype = .i16, .rank = 1, .shape = .{ 8, 1, 1, 1, 1 }, .begin = 4096, .end = 4112 },
        .{ .dtype = .i16, .rank = 1, .shape = .{ 8, 1, 1, 1, 1 }, .begin = 4112, .end = 4128 },
        .{ .dtype = .i32, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .begin = 4128, .end = 4132 },
        .{ .dtype = .i16, .rank = 3, .shape = .{ 8, 8, 32, 1, 1 }, .begin = 4132, .end = 4132 + 4096 },
    };
    const names = [_][]const u8{ "a.trellis", "a.su", "a.sv", "a.mul1", "b.trellis" };
    const dts = [_][]const u8{ "I16", "I16", "I16", "I32", "I16" };
    const data = try std.mem.concat(gpa, u8, &.{ &trellis, &scales, &scales, &marker, &trellis });
    defer gpa.free(data);
    var header: std.ArrayListUnmanaged(u8) = .empty;
    defer header.deinit(gpa);
    try header.appendSlice(gpa, "{");
    for (names, entries, dts, 0..) |name, e, dt, i| {
        if (i != 0) try header.appendSlice(gpa, ",");
        try header.print(gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, dt });
        for (e.shape[0..e.rank], 0..) |d, j| {
            if (j != 0) try header.appendSlice(gpa, ",");
            try header.print(gpa, "{d}", .{d});
        }
        try header.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ e.begin, e.end });
    }
    try header.appendSlice(gpa, "}");
    var file: std.ArrayListUnmanaged(u8) = .empty;
    defer file.deinit(gpa);
    try file.appendNTimes(gpa, 0, 8);
    std.mem.writeInt(u64, file.items[0..8], header.items.len, .little);
    try file.appendSlice(gpa, header.items);
    try file.appendSlice(gpa, data);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);

    var s = try scan(gpa, io, dir);
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 1), s.bad);
    try std.testing.expectEqual(@as(usize, 1), s.summary.mul1);
    try std.testing.expectEqual(@as(usize, 1), s.summary.su_sv);
    try std.testing.expectEqual(@as(usize, 1), s.summary.bits[4]); // a 32-word trellis is 4 halves
    const gi = s.find("a") orelse return error.TestUnexpectedResult;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const p = try s.parts(arena.allocator(), gi);
    try std.testing.expectEqual(@as(usize, 128), p.trellis.dim(0) * 16);
    try std.testing.expect(p.suh_packed);
    try std.testing.expectEqual(@as(usize, 8), p.svh.numel());
}

fn scanFault(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    var s = try scan(a, io, dir);
    defer s.deinit();
}

test "a scan that fails an allocation frees every group prefix it held" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"a.trellis\":{\"dtype\":\"I16\",\"shape\":[8,8,32],\"data_offsets\":[0,4096]},\"a.su\":{\"dtype\":\"I16\",\"shape\":[8],\"data_offsets\":[4096,4112]},\"a.sv\":{\"dtype\":\"I16\",\"shape\":[8],\"data_offsets\":[4112,4128]},\"b.trellis\":{\"dtype\":\"I16\",\"shape\":[8,8,32],\"data_offsets\":[4128,8224]},\"b.su\":{\"dtype\":\"I16\",\"shape\":[8],\"data_offsets\":[8224,8240]},\"b.sv\":{\"dtype\":\"I16\",\"shape\":[8],\"data_offsets\":[8240,8256]}}";
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    try bytes.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, bytes.items[0..8], header.len, .little);
    try bytes.appendSlice(a, header);
    try bytes.appendNTimes(a, 0, 8256);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes.items });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var admitted = try scan(a, io, dir);
    defer admitted.deinit();
    try std.testing.expectEqual(@as(usize, 2), admitted.groups.len);
    try std.testing.checkAllAllocationFailures(a, scanFault, .{ io, dir });
}
