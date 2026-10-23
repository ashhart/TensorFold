//! One GPU: production padding/copy calls and the unchanged fused core against a direct 32-head launch.
const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const heads = @import("prefill_heads.zig");
const tc = @import("triton_call.zig");
const Buffer = struct { dev: cuda.DeviceBuffer, host: []u8 };
const Case = struct { total: i64, rows: i64, row0: i64 = 0 };
const Receipt = struct { total: i64, rows: i64, row0: i64, sha256: [64]u8 };

const Memory = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    buffers: std.StringArrayHashMapUnmanaged(Buffer) = .empty,

    fn close(m: *Memory) void {
        m.stream.synchronize() catch std.process.exit(1);
        for (m.buffers.values()) |*b| b.dev.free();
    }
    fn add(m: *Memory, name: []const u8, len: usize, poison: u8) !void {
        if (m.buffers.contains(name)) return;
        const host = try m.a.alloc(u8, len + 32);
        @memset(host, 0xA5);
        @memset(host[16..][0..len], poison);
        var dev = try cuda.DeviceBuffer.alloc(m.d, host.len);
        errdefer dev.free();
        try m.buffers.put(m.a, name, .{ .dev = dev, .host = host });
    }
    fn data(m: *Memory, name: []const u8) []u8 {
        const host = m.buffers.get(name).?.host;
        return host[16 .. host.len - 16];
    }
    pub fn tensorAddr(m: *Memory, t: calls.Tensor) !u64 {
        const name = switch (t.role) {
            .buf => |s| s,
            .weight => |s| s,
            .empty => return 0,
        };
        return (m.buffers.get(name) orelse return error.Unbound).dev.ptr + 16 + @as(u64, @intCast(t.offset));
    }
    fn address(ctx: *anyopaque, t: calls.Tensor) anyerror!u64 {
        const m: *Memory = @ptrCast(@alignCast(ctx));
        return m.tensorAddr(t);
    }
    fn launch(m: *Memory, set: *const cuda.aot.Set, c: *const calls.Call) !void {
        var specs: std.ArrayList(cuda.aot.Spec) = .empty;
        for (set.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", c.name)) try specs.append(m.a, v.spec);
        const split = try tc.split(m.a, c, specs.items, .{ .ctx = m, .of = address });
        try set.run(m.stream, c.name, .{ @intCast(c.grid[0]), @intCast(c.grid[1]), @intCast(c.grid[2]) }, split.args, split.consts);
    }
};

fn word(bytes: []u8, i: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
}
fn bf16(bytes: []u8, i: usize, value: f32) void {
    std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @import("core").contract.spec.toBf16(value), .little);
}
fn nameOf(t: calls.Tensor) []const u8 {
    return switch (t.role) {
        .buf => |s| s,
        .weight => |s| s,
        .empty => "",
    };
}

fn one(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, set: *const cuda.aot.Set, shape: Case) !Receipt {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var m: Memory = .{ .a = a, .d = d, .stream = stream };
    defer m.close();
    const cfg: @import("config.zig").Config = .{};
    var widths: block.Widths = .{};
    var e: block.Emitter = .{ .a = a, .cfg = &cfg, .w = &widths, .o = .{ .world = 4 }, .n = shape.total, .start = 128 };
    const q = try e.buf("L.q", .{}, .bf16, &.{ shape.total, 8192 });
    const padded = try heads.query(&e, q, 512);
    const out = try e.buf("L.o", .{}, .bf16, &.{ 2, shape.total, 4096 });
    const kv = try e.buf("L.kv", .{}, .bf16, &.{ shape.total, 512 });
    e.row0 = shape.row0;
    e.n = shape.rows;
    var seg: bp.Seg = .{ .e = &e, .staged = false, .ring = 256 };
    try seg.core(0, padded, kv, .none, false, false, out, shape.total);
    var selected: std.ArrayList(calls.Call) = .empty;
    var fused: ?calls.Call = null;
    var sizes: std.StringArrayHashMapUnmanaged(usize) = .empty;
    for (e.out.items) |c| {
        if (!std.mem.startsWith(u8, c.name, "glue.pf_heads_") and !std.mem.eql(u8, c.name, "_fused")) continue;
        try selected.append(a, c);
        if (c.triton) fused = c;
        for (c.args) |x| if (x.arg == .t) {
            const t = x.arg.t;
            const size: usize = @intCast(@import("buffers.zig").extent(t));
            const entry = try sizes.getOrPut(a, nameOf(t));
            if (!entry.found_existing) entry.value_ptr.* = 0;
            entry.value_ptr.* = @max(entry.value_ptr.*, size);
        };
    }
    for (sizes.keys(), sizes.values()) |name, size| try m.add(name, size, 0);
    // The real output is poisoned differently from the direct output, so two unwritten outputs cannot agree.
    try m.add("ref.q", @intCast(shape.total * 32768), 0);
    try m.add("ref.sink", 128, 0);
    try m.add("ref.out", @intCast(4 * shape.total * 8192), 0x5A);
    @memset(m.data("L.o"), 0xA5);
    @memset(m.data("L.pf.o32"), 0xA5);
    const real_q = m.data("L.q");
    for (0..real_q.len / 2) |i| bf16(real_q, i, @as(f32, @floatFromInt(i % 29)) / 32 - 0.4);
    const total: usize = @intCast(shape.total);
    for (0..total) |r| @memcpy(m.data("ref.q")[r * 32768 ..][0..16384], real_q[r * 16384 ..][0..16384]);
    for (0..16) |h| word(m.data("L0.attn.sink"), h, @as(f32, @floatFromInt(h)) / 16 - 2);
    @memcpy(m.data("ref.sink")[0..64], m.data("L0.attn.sink"));
    const v = m.data("s.L0.swa.v");
    for (0..v.len / 576) |r| {
        @memset(v[r * 576 ..][0..448], 0x30);
        for (0..64) |j| bf16(v[r * 576 + 448 ..], j, @as(f32, @floatFromInt((j + r) % 13)) / 32);
    }
    @memset(m.data("s.L0.swa.s"), 127);
    const cs = m.data("s.rope.main");
    for (0..cs.len / 256) |r| for (0..32) |j| word(cs, r * 64 + j, 1);
    std.mem.writeInt(i32, m.data("w.pos")[0..4], 128, .little);
    for (m.buffers.values()) |b| try b.dev.upload(0, b.host);
    for (selected.items) |*c| {
        if (c.triton) try m.launch(set, c) else if (!try heads.glue(&m, c)) return error.NoBinding;
    }
    var direct = fused orelse return error.NoFused;
    const args = try a.dupe(calls.Named, direct.args);
    for (args) |*x| {
        if (std.mem.eql(u8, x.name, "Q")) x.arg.t.role = .{ .buf = "ref.q" };
        if (std.mem.eql(u8, x.name, "SINK")) x.arg.t.role = .{ .buf = "ref.sink" };
        if (std.mem.eql(u8, x.name, "OUT")) x.arg.t.role = .{ .buf = "ref.out" };
    }
    direct.args = args;
    try m.launch(set, &direct);
    try stream.synchronize();
    for (m.buffers.values()) |b| {
        try b.dev.download(0, b.host);
        if (!std.mem.allEqual(u8, b.host[0..16], 0xA5) or !std.mem.allEqual(u8, b.host[b.host.len - 16 ..], 0xA5)) return error.GuardChanged;
    }
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    const actual = m.data("L.o");
    const reference = m.data("ref.out");
    const row0: usize = @intCast(shape.row0);
    const rows: usize = @intCast(shape.rows);
    for (0..2) |g| for (0..total) |r| {
        const at = (g * total + r) * 8192;
        const bytes = actual[at..][0..8192];
        if (r >= row0 and r < row0 + rows) {
            if (!std.mem.eql(u8, bytes, reference[at..][0..8192])) return error.DifferentHeads;
            hash.update(bytes);
        } else if (!std.mem.allEqual(u8, bytes, 0xA5)) return error.WrongRows;
    };
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return .{ .total = shape.total, .rows = shape.rows, .row0 = shape.row0, .sha256 = std.fmt.bytesToHex(digest, .lower) };
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedAotAndOutputDirectory;
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(init.io, args[2]);
    const output = try std.fs.path.join(init.arena.allocator(), &.{ args[2], "summary.json" });
    try cwd.writeFile(init.io, .{ .sub_path = output, .data = "{\"equal\":false,\"state\":\"running\"}\n" });
    errdefer cwd.writeFile(init.io, .{ .sub_path = output, .data = "{\"equal\":false,\"state\":\"failed\"}\n" }) catch {};
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&d, false);
    defer stream.deinit();
    var set = try cuda.aot.Set.load(init.gpa, init.io, &d, ctx.device, args[1]);
    defer set.deinit();
    const shapes = [_]Case{ .{ .total = 1, .rows = 1 }, .{ .total = 16, .rows = 16 }, .{ .total = 17, .rows = 17 }, .{ .total = 33, .rows = 33 }, .{ .total = 129, .rows = 129 }, .{ .total = 50, .rows = 17, .row0 = 33 } };
    var receipts: [shapes.len]Receipt = undefined;
    for (shapes, &receipts) |shape, *receipt| receipt.* = try one(init.gpa, &d, stream, &set, shape);
    const text = try std.json.Stringify.valueAlloc(init.arena.allocator(), .{ .sm = try ctx.capability(), .equal = true, .cases = receipts }, .{});
    try cwd.writeFile(init.io, .{ .sub_path = output, .data = text });
    std.debug.print("PASS padded prefill heads: {d} cases, real bytes and row guards equal\n", .{shapes.len});
    return 0;
}
