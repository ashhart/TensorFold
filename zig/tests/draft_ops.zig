//! Compare borrowed core BF16 primitive outputs with tiny pinned MLX fixtures.
const std = @import("std");
const mtl = @import("metal");
const core = @import("draft_ops");
const Pin = struct { name: []const u8, bytes: usize, sha256: []const u8 };
const Case = struct { name: []const u8, kind: []const u8, params: std.json.Value, files: std.json.Value };
const Manifest = struct { schema: []const u8, cases: []const Case };
const Owned = struct {
    a: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    dir: []const u8,
    buffers: std.ArrayList(mtl.Buffer) = .empty,
    fn payload(o: *Owned, c: Case, field: []const u8) ![]const u8 {
        const p = try std.json.parseFromValue(Pin, o.a, c.files.object.get(field) orelse return error.BadFixture, .{});
        defer p.deinit();
        const pin = p.value;
        if (std.mem.indexOfScalar(u8, pin.name, '/') != null) return error.BadFixture;
        const path = try std.fs.path.join(o.a, &.{ o.dir, pin.name });
        const data = try std.Io.Dir.cwd().readFileAlloc(o.io, path, o.a, .limited(8 << 20));
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(data, &digest, .{});
        if (data.len != pin.bytes or !std.mem.eql(u8, pin.sha256, &std.fmt.bytesToHex(digest, .lower))) return error.BadFixture;
        return data;
    }
    fn get(o: *Owned, c: Case, field: []const u8) !core.Ref {
        const data = try o.payload(c, field);
        const buffer = try o.device.buffer(data.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        o.buffers.append(o.a, buffer) catch |err| {
            buffer.deinit();
            return err;
        };
        @memcpy(buffer.contents()[0..data.len], data);
        return .{ .buf = buffer };
    }
    fn deinit(o: *Owned) void {
        for (o.buffers.items) |buffer| buffer.deinit();
        o.buffers.deinit(o.a);
    }
};
fn args(comptime T: type, a: std.mem.Allocator, json_value: std.json.Value) !T {
    return (try std.json.parseFromValue(T, a, json_value, .{})).value;
}
fn value(bytes: []const u8, index: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[index * 2 ..][0..2], .little)) << 16);
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const cli = try init.minimal.args.toSlice(a);
    if (cli.len != 3 or !std.mem.eql(u8, cli[1], "--fixtures")) return error.BadOptions;
    const path = try std.fs.path.join(a, &.{ cli[2], "manifest.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 20));
    const manifest = (try std.json.parseFromSlice(Manifest, a, text, .{})).value;
    if (!std.mem.eql(u8, manifest.schema, "tf-draft-ops-v1")) return error.BadFixture;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const ops = try core.Ops.init(device);
    defer ops.deinit();
    var errors: usize = 0;
    var total: usize = 0;
    for (manifest.cases) |c| {
        var owned = Owned{ .a = a, .io = init.io, .device = device, .dir = cli[2] };
        defer owned.deinit();
        const expected = try owned.payload(c, "expected");
        const out = try device.buffer(expected.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer out.deinit();
        @memset(out.contents()[0..expected.len], 0xa5);
        const y = core.Ref{ .buf = out };
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        var isolated: ?mtl.Buffer = null;
        defer if (isolated) |buffer| buffer.deinit();
        if (std.mem.eql(u8, c.kind, "linear")) {
            const p = try args(core.Linear, a, c.params);
            const x = try owned.get(c, "x");
            const weight = try owned.get(c, "weight");
            try ops.linear(e, x, weight, y, p);
            isolated = try device.buffer(expected.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            var single = p;
            single.rows = 1;
            for (0..p.rows) |row| try ops.linear(e, .{ .buf = x.buf, .off = row * p.x_stride * 2 }, weight, .{ .buf = isolated.?, .off = row * p.y_stride * 2 }, single);
        } else if (std.mem.eql(u8, c.kind, "norm")) try ops.norm(e, try owned.get(c, "x"), try owned.get(c, "weight"), y, try args(core.Norm, a, c.params)) else if (std.mem.eql(u8, c.kind, "conv")) {
            const p = try args(core.Conv, a, c.params);
            try ops.conv(e, try owned.get(c, "x"), try owned.get(c, "dynamic"), try owned.get(c, "base"), if (p.residual == 1) try owned.get(c, "residual") else null, y, p);
        } else if (std.mem.eql(u8, c.kind, "rope")) try ops.rope(e, try owned.get(c, "x"), try owned.get(c, "positions"), y, try args(core.Rope, a, c.params)) else if (std.mem.eql(u8, c.kind, "swiglu")) try ops.swiglu(e, try owned.get(c, "gate"), try owned.get(c, "up"), y, try args(core.Activation, a, c.params)) else return error.BadFixture;
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailure;
        const actual = out.contents()[0..expected.len];
        if (isolated) |buffer| if (!std.mem.eql(u8, actual, buffer.contents()[0..expected.len])) return error.RowInvariantFailed;
        var unequal: usize = 0;
        var maximum: f32 = 0;
        for (0..expected.len / 2) |i| {
            if (!std.mem.eql(u8, expected[i * 2 ..][0..2], actual[i * 2 ..][0..2])) unequal += 1;
            maximum = @max(maximum, @abs(value(expected, i) - value(actual, i)));
        }
        const result = try std.fs.path.join(a, &.{ cli[2], try std.fmt.allocPrint(a, "{s}.native.bin", .{c.name}) });
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = result, .data = actual });
        std.debug.print("{s}: {d}/{d} unequal BF16, max_abs {d}\n", .{ c.name, unequal, expected.len / 2, maximum });
        errors += unequal;
        total += expected.len / 2;
    }
    std.debug.print("draft primitives: {d} BF16 values, {d} unequal\n", .{ total, errors });
    if (errors > 0) return error.PrimitiveBytesDiffer;
}
