//! The reference capture's Triton cubins (aot/aot.json, by hash) loaded one by one; launches packed as Triton does.

const std = @import("std");
const cuda = @import("cuda");
const Value = std.json.Value;

const checked = @import("capture_args.zig");
pub const Param = checked.Param;

pub const Kernel = struct {
    kernel: cuda.triton.Kernel,
    params: []const Param,
    global_scratch: u32,
    global_align: u32,
};

const Spec = struct {
    hash: []const u8,
    name: []const u8,
    num_warps: u32,
    num_ctas: u32 = 1,
    shared: u32 = 0,
    global_scratch: u32 = 0,
    global_align: u32 = 1,
    profile_scratch: u32 = 0,
    pdl: bool = false,
    params: []struct { name: []const u8, type: []const u8 },
};
pub const Set = struct {
    arena: std.heap.ArenaAllocator,
    kernels: std.StringHashMapUnmanaged(Kernel) = .empty,
    failed: std.ArrayList([2][]const u8) = .empty,
    pub fn deinit(s: *Set) void {
        var it = s.kernels.valueIterator();
        while (it.next()) |*k| k.*.kernel.unload();
        s.arena.deinit();
        s.* = undefined;
    }
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: cuda.abi.Device, dir: []const u8) !Set {
        var s: Set = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.deinit();
        const a = s.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "aot.json" }), a, .limited(1 << 26));
        const root = try std.json.parseFromSliceLeaky(struct { kernels: []Spec }, a, text, .{ .ignore_unknown_fields = true });
        if (root.kernels.len > 65536) return error.BadManifest;
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        try seen.ensureTotalCapacity(a, @intCast(root.kernels.len));
        for (root.kernels) |spec| {
            if (spec.hash.len == 0 or spec.hash.len > 128 or spec.name.len == 0 or spec.num_warps == 0 or spec.num_warps > 32 or
                spec.num_ctas == 0 or spec.num_ctas > 16 or !std.math.isPowerOfTwo(spec.num_ctas) or
                spec.shared > 101376 or spec.global_align == 0 or !std.math.isPowerOfTwo(spec.global_align) or
                spec.params.len > cuda.Args.max_args - 2) return error.BadManifest;
            for (spec.hash) |ch| if (!std.ascii.isHex(ch)) return error.BadManifest;
            if (!checked.identifier(spec.name)) return error.BadManifest;
            for (spec.params, 0..) |p, i| {
                if (!checked.identifier(p.name)) return error.BadManifest;
                _ = checked.typeOf(p.type) catch return error.BadManifest;
                for (spec.params[0..i]) |prior| if (std.mem.eql(u8, p.name, prior.name)) return error.BadManifest;
            }
            const entry = seen.getOrPutAssumeCapacity(spec.hash);
            if (entry.found_existing) return error.DuplicateKernel;
        }
        try s.kernels.ensureTotalCapacity(a, @intCast(root.kernels.len));
        try s.failed.ensureTotalCapacity(a, root.kernels.len);
        for (root.kernels) |spec| {
            const params = try a.alloc(Param, spec.params.len);
            for (spec.params, params) |source, *dest| dest.* = .{ .name = source.name, .ty = try checked.typeOf(source.type) };
            const name_z = try a.dupeSentinel(u8, spec.name, 0);
            const file = try std.fmt.allocPrint(a, "{s}/cubins/{s}.cubin", .{ dir, spec.hash });
            const cubin = std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null) catch |e| {
                if (e == error.OutOfMemory) return e;
                s.failed.appendAssumeCapacity(.{ spec.hash, @errorName(e) });
                continue;
            };
            defer gpa.free(cubin);
            const meta: cuda.triton.Meta = .{ .name = spec.name, .num_warps = spec.num_warps, .num_ctas = spec.num_ctas, .shared = spec.shared, .global_scratch_size = spec.global_scratch, .global_scratch_align = spec.global_align, .profile_scratch_size = spec.profile_scratch, .launch_pdl = spec.pdl };
            const kernel = cuda.triton.Kernel.load(d, device, cubin, meta, name_z) catch |e| {
                s.failed.appendAssumeCapacity(.{ spec.hash, @errorName(e) });
                continue;
            };
            // All fallible allocations precede the module owner. Publication cannot replace an existing owner.
            s.kernels.putAssumeCapacity(spec.hash, .{ .kernel = kernel, .params = params, .global_scratch = spec.global_scratch, .global_align = spec.global_align });
        }
        return s;
    }
    /// Captured launches join the same contract schema. Strings are borrowed until Set.deinit.
    pub fn declarations(s: *const Set, a: std.mem.Allocator) ![]@import("core").contract.registry.CudaRecord {
        const R = @import("core").contract.registry.CudaRecord;
        const out = try a.alloc(R, s.kernels.count());
        var it = s.kernels.iterator();
        var i: usize = 0;
        while (it.next()) |entry| {
            const m = entry.value_ptr.kernel.meta;
            out[i] = .{ .launch = .{ .name = m.name, .symbol = m.name, .image = entry.key_ptr.*, .threads = 32 * m.num_warps, .dynamic = m.shared, .cluster = m.num_ctas, .opt_in = m.shared > 49152, .min_sm = 121 }, .usage = @splat(.{ .regs = 0, .static = 0 }) };
            i += 1;
        }
        return out;
    }
};

/// A captured scalar's bits as the parameter's C type (Triton's ty_to_cpp: i1 is int8, floats by their bit patterns).
fn scalar(args: *cuda.Args, ty: []const u8, v: Value) !void {
    return checked.scalar(args, try checked.typeOf(ty), v);
}
pub const Resolve = checked.Resolve;
pub const Packed = checked.Packed;
pub fn pack(k: *const Kernel, op: std.json.ObjectMap, ctx: *anyopaque, resolve: Resolve) !Packed {
    return checked.pack(k.params, op, ctx, resolve);
}
pub fn launch(k: *const Kernel, op: std.json.ObjectMap, stream: cuda.Stream, ctx: *anyopaque, resolve: Resolve, scratch: u64) !void {
    const need = try scratchBytes(k, op);
    if (need > 0 and (scratch == 0 or scratch % k.global_align != 0)) return error.InvalidScratch;
    var packed_op = try pack(k, op, ctx, resolve);
    try k.kernel.launchOn(packed_op.grid, stream, &packed_op.args, .{ .global = scratch }, &.{});
}
pub fn scratchBytes(k: *const Kernel, op: std.json.ObjectMap) !usize {
    const g = try checked.grid(op);
    if (k.global_scratch == 0) return 0;
    var size = try std.math.mul(usize, g.x, g.y);
    size = try std.math.mul(usize, size, g.z);
    size = try std.math.mul(usize, size, k.kernel.meta.num_ctas);
    return std.math.mul(usize, size, k.global_scratch);
}

fn dummyResolve(_: *anyopaque, _: Value) !?u64 {
    return 16;
}
test "capture operation rejects malformed grids scalars and argument identities before launch" {
    const a = std.testing.allocator;
    const dummy: Kernel = .{ .kernel = undefined, .params = &.{}, .global_scratch = 0, .global_align = 1 };
    const bad_ops = [_][]const u8{
        "{}",                                                              "{\"grid\":[],\"args\":[]}",        "{\"grid\":[1,1],\"args\":[]}",
        "{\"grid\":[1,1,1,1],\"args\":[]}",                                "{\"grid\":[0,1,1],\"args\":[]}",   "{\"grid\":[-1,1,1],\"args\":[]}",
        "{\"grid\":[4294967296,1,1],\"args\":[]}",                         "{\"grid\":[1.0,1,1],\"args\":[]}", "{\"grid\":[1,1,1],\"args\":[{}]}",
        "{\"grid\":[1,1,1],\"args\":[{\"name\":\"x\"},{\"name\":\"x\"}]}",
    };
    for (bad_ops) |text| {
        var json = try std.json.parseFromSlice(Value, a, text, .{});
        defer json.deinit();
        if (pack(&dummy, json.value.object, undefined, dummyResolve)) |_| return error.TestUnexpectedResult else |_| {}
    }
    const cases = [_][]const u8{
        "{}",                              "{\"t\":\"int\",\"v\":128}",                 "{\"t\":\"int\",\"v\":-129}",
        "{\"t\":\"float\",\"f32\":\"x\"}", "{\"t\":\"float\",\"f32\":\"0x100000000\"}", "{\"t\":\"float\",\"f64\":\"0x3\"}",
    };
    for (cases, 0..) |text, i| {
        var json = try std.json.parseFromSlice(Value, a, text, .{});
        defer json.deinit();
        var args: cuda.Args = .{};
        if (scalar(&args, if (i < 3) "i8" else if (i == 5) "fp64" else "fp32", json.value)) |_| return error.TestUnexpectedResult else |_| {}
    }
}
test "capture scratch rejects overflow and validates grids even when scratch is zero" {
    var json = try std.json.parseFromSlice(Value, std.testing.allocator, "{\"grid\":[2147483647,65535,65535]}", .{});
    defer json.deinit();
    const k: Kernel = .{ .kernel = .{ .module = undefined, .function = undefined, .meta = .{ .name = "k", .num_warps = 4, .num_ctas = 16, .global_scratch_size = 4294967295 } }, .params = &.{}, .global_scratch = 4294967295, .global_align = 1 };
    try std.testing.expectError(error.Overflow, scratchBytes(&k, json.value.object));
    var bad = try std.json.parseFromSlice(Value, std.testing.allocator, "{\"grid\":[],\"args\":[]}", .{});
    defer bad.deinit();
    if (pack(&k, bad.value.object, undefined, dummyResolve)) |_| return error.TestUnexpectedResult else |_| {}
}

test "capture scalar boundaries and exact float bit encodings pack without a GPU" {
    const C = struct { ty: []const u8, json: []const u8, bytes: usize, bits: u64 };
    const cases = [_]C{
        .{ .ty = "i1", .json = "{\"t\":\"bool\",\"v\":true}", .bytes = 1, .bits = 1 },
        .{ .ty = "i8", .json = "{\"t\":\"int\",\"v\":-128}", .bytes = 1, .bits = 128 },
        .{ .ty = "u8", .json = "{\"t\":\"int\",\"v\":255}", .bytes = 1, .bits = 255 },
        .{ .ty = "i16", .json = "{\"t\":\"int\",\"v\":-32768}", .bytes = 2, .bits = 32768 },
        .{ .ty = "u16", .json = "{\"t\":\"int\",\"v\":65535}", .bytes = 2, .bits = 65535 },
        .{ .ty = "i32", .json = "{\"t\":\"int\",\"v\":-2147483648}", .bytes = 4, .bits = 2147483648 },
        .{ .ty = "u32", .json = "{\"t\":\"int\",\"v\":4294967295}", .bytes = 4, .bits = 4294967295 },
        .{ .ty = "i64", .json = "{\"t\":\"int\",\"v\":-9223372036854775808}", .bytes = 8, .bits = 0x8000000000000000 },
        .{ .ty = "u64", .json = "{\"t\":\"int\",\"v\":18446744073709551615}", .bytes = 8, .bits = 0xffffffffffffffff },
        .{ .ty = "fp32", .json = "{\"t\":\"float\",\"f32\":\"0x3fa00000\"}", .bytes = 4, .bits = 0x3fa00000 },
        .{ .ty = "fp64", .json = "{\"t\":\"float\",\"f64\":\"0x8000000000000000\"}", .bytes = 8, .bits = 0x8000000000000000 },
    };
    for (cases) |c| {
        var json = try std.json.parseFromSlice(Value, std.testing.allocator, c.json, .{ .parse_numbers = false });
        defer json.deinit();
        var args: cuda.Args = .{};
        try scalar(&args, c.ty, json.value);
        try std.testing.expectEqual(c.bytes, args.sizes[0]);
        for (args.storage[0..c.bytes], 0..) |byte, i| try std.testing.expectEqual(@as(u8, @truncate(c.bits >> @intCast(i * 8))), byte);
    }
    const jsons = [_]struct { []const u8, []const u8 }{
        .{ "i1", "{\"t\":\"int\",\"v\":2}" },   .{ "u8", "{\"t\":\"int\",\"v\":-1}" },
        .{ "u64", "{\"t\":\"int\",\"v\":-1}" }, .{ "u32", "{\"t\":\"int\",\"v\":4294967296}" },
    };
    for (jsons) |c| {
        var json = try std.json.parseFromSlice(Value, std.testing.allocator, c[1], .{});
        defer json.deinit();
        var args: cuda.Args = .{};
        try std.testing.expectError(error.IntRange, scalar(&args, c[0], json.value));
    }
    var op = try std.json.parseFromSlice(Value, std.testing.allocator, "{\"grid\":[1,2,3],\"args\":[{\"name\":\"x\",\"t\":\"int\",\"v\":16}]}", .{});
    defer op.deinit();
    const k: Kernel = .{ .kernel = undefined, .params = &.{.{ .name = "x", .ty = .pointer }}, .global_scratch = 0, .global_align = 1 };
    const packed_op = try pack(&k, op.value.object, undefined, dummyResolve);
    try std.testing.expectEqual(@as(?u64, 16), packed_op.args.integer(0));
    try std.testing.expectEqual(@as(usize, 0), try scratchBytes(&k, op.value.object));
}
