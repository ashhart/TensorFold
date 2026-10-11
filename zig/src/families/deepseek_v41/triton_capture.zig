//! The reference capture's Triton cubins (aot/aot.json, by hash) loaded one by one; launches packed as Triton does.

const std = @import("std");
const cuda = @import("cuda");
const Value = std.json.Value;

pub const Param = struct { name: []const u8, ty: []const u8 };

pub const Kernel = struct {
    kernel: cuda.triton.Kernel,
    params: []Param,
    global_scratch: u32,
    global_align: u32,
};

pub const Set = struct {
    arena: std.heap.ArenaAllocator,
    kernels: std.StringHashMapUnmanaged(Kernel) = .empty,
    /// hashes whose cubin did not load, with the reason
    failed: std.ArrayList([2][]const u8) = .empty,

    pub fn deinit(s: *Set) void {
        var it = s.kernels.valueIterator();
        while (it.next()) |k| k.kernel.unload();
        s.arena.deinit();
        s.* = undefined;
    }

    /// Every kernel `dir`/aot.json lists; one that does not load is recorded in `failed`, the rest stay usable.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: cuda.abi.Device, dir: []const u8) !Set {
        var s: Set = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.deinit();
        const a = s.arena.allocator();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "aot.json" }), a, .limited(1 << 26));
        const root = try std.json.parseFromSliceLeaky(Value, a, text, .{});
        for (root.object.get("kernels").?.array.items) |kv| {
            const k = kv.object;
            const hash = k.get("hash").?.string;
            const name = k.get("name").?.string;
            const file = try std.fmt.allocPrint(a, "{s}/cubins/{s}.cubin", .{ dir, hash });
            const cubin = std.Io.Dir.cwd().readFileAllocOptions(io, file, gpa, .limited(1 << 26), .@"16", null) catch |e| {
                try s.failed.append(a, .{ hash, @errorName(e) });
                continue;
            };
            defer gpa.free(cubin);
            const meta: cuda.triton.Meta = .{
                .name = name,
                .num_warps = @intCast(k.get("num_warps").?.integer),
                .num_ctas = @intCast(k.get("num_ctas").?.integer),
                .shared = @intCast(k.get("shared").?.integer),
                .global_scratch_size = @intCast(k.get("global_scratch").?.integer),
                .global_scratch_align = @intCast(k.get("global_align").?.integer),
                .profile_scratch_size = @intCast(k.get("profile_scratch").?.integer),
                .launch_pdl = k.get("pdl").?.bool,
            };
            const kernel = cuda.triton.Kernel.load(d, device, cubin, meta, try a.dupeSentinel(u8, name, 0)) catch |e| {
                try s.failed.append(a, .{ hash, @errorName(e) });
                continue;
            };
            const ps = k.get("params").?.array.items;
            const params = try a.alloc(Param, ps.len);
            for (ps, params) |p, *out| out.* = .{ .name = p.object.get("name").?.string, .ty = p.object.get("type").?.string };
            try s.kernels.put(a, hash, .{ .kernel = kernel, .params = params, .global_scratch = meta.global_scratch_size, .global_align = meta.global_scratch_align });
        }
        return s;
    }
};

/// A captured scalar's bits as the parameter's C type (Triton's ty_to_cpp: i1 is int8, floats by their bit patterns).
fn scalar(args: *cuda.Args, ty: []const u8, v: Value) !void {
    const T = enum { i1, i8, i16, i32, i64, u8, u16, u32, u64, fp32, fp64 };
    const t = std.meta.stringToEnum(T, ty) orelse return error.UnsupportedScalar;
    const o = v.object;
    const kind = o.get("t").?.string;
    if (t == .fp32 or t == .fp64) {
        if (!std.mem.eql(u8, kind, "float")) return error.WrongScalar;
        const bits = o.get(if (t == .fp32) "f32" else "f64").?.string;
        const x = try std.fmt.parseInt(u64, bits[2..], 16);
        if (t == .fp32) args.add(@as(u32, @intCast(x))) else args.add(x);
        return;
    }
    const n: i64 = if (std.mem.eql(u8, kind, "bool")) @intFromBool(o.get("v").?.bool) else if (std.mem.eql(u8, kind, "int")) o.get("v").?.integer else return error.WrongScalar;
    switch (t) {
        .i1, .i8 => args.add(@as(i8, @intCast(n))),
        .u8 => args.add(@as(u8, @intCast(n))),
        .i16 => args.add(@as(i16, @intCast(n))),
        .u16 => args.add(@as(u16, @intCast(n))),
        .i32 => args.add(@as(i32, @intCast(n))),
        .u32 => args.add(@as(u32, @intCast(n))),
        .i64 => args.add(n),
        .u64 => args.add(@as(u64, @bitCast(n))),
        .fp32, .fp64 => unreachable,
    }
}

/// The address of a tensor argument (`addr`), the int it carries, or null for None.
pub const Resolve = *const fn (ctx: *anyopaque, arg: Value) anyerror!?u64;

/// Packs `op`'s named arguments in runtime order and launches it on the captured grid (global scratch: `scratch`).
pub fn launch(k: *const Kernel, op: std.json.ObjectMap, stream: cuda.Stream, ctx: *anyopaque, resolve: Resolve, scratch: u64) !void {
    var args: cuda.Args = .{};
    const given = op.get("args").?.array.items;
    for (k.params) |p| {
        const arg = for (given) |g| {
            if (std.mem.eql(u8, g.object.get("name").?.string, p.name)) break g;
        } else return error.MissingArgument;
        if (p.ty[0] == '*') {
            args.add(@as(u64, (try resolve(ctx, arg)) orelse 0));
        } else try scalar(&args, p.ty, arg);
    }
    const g = op.get("grid").?.array.items;
    const grid: cuda.Dim3 = .{ .x = @intCast(g[0].integer), .y = @intCast(g[1].integer), .z = @intCast(g[2].integer) };
    try k.kernel.launchOn(grid, stream, &args, .{ .global = scratch }, &.{});
}

/// Global scratch bytes a launch of `op` needs (Triton: blocks x CTAs x the kernel's size).
pub fn scratchBytes(k: *const Kernel, op: std.json.ObjectMap) usize {
    if (k.global_scratch == 0) return 0;
    const g = op.get("grid").?.array.items;
    return k.kernel.globalScratchBytes(.{ .x = @intCast(g[0].integer), .y = @intCast(g[1].integer), .z = @intCast(g[2].integer) });
}
