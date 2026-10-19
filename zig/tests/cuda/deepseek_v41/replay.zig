//! Replays one recorded extension call (dsv41_capture.py) on its recorded bytes; every viewed storage bit-compared.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const Fixture = @import("../fixture.zig").Fixture;

const replay = dsv41.replay;
const Gpu = check.Gpu;
const Value = std.json.Value;

pub const Outcome = enum { exact, skipped };

const Storage = struct { name: []const u8, buf: ?cuda.DeviceBuffer, len: usize };

const Ctx = struct {
    gpa: std.mem.Allocator,
    a: std.mem.Allocator, // the case's arena: argument trees
    storages: []Storage,
    extra: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn ptrOf(c: *Ctx, name: []const u8) !u64 {
        for (c.storages) |s| if (std.mem.eql(u8, s.name, name)) return if (s.buf) |b| b.ptr else 0;
        return error.UnknownStorage;
    }

    fn dtype(s: []const u8) !replay.DType {
        const map = .{ .{ "float16", .f16 }, .{ "bfloat16", .bf16 }, .{ "float32", .f32 }, .{ "float64", .f64 }, .{ "int8", .i8 }, .{ "uint8", .u8 }, .{ "int16", .i16 }, .{ "int32", .i32 }, .{ "int64", .i64 }, .{ "bool", .bool } };
        inline for (map) |m| if (std.mem.eql(u8, s, m[0])) return m[1];
        return error.UnknownDType;
    }

    fn ints(c: *Ctx, v: Value) ![]const i64 {
        const items = v.array.items;
        const out = try c.a.alloc(i64, items.len);
        for (items, out) |x, *o| o.* = x.integer;
        return out;
    }

    /// dsv41_capture.py's argument node -> a replay Arg.
    fn arg(c: *Ctx, v: Value) !replay.Arg {
        const o = v.object;
        if (o.get("t")) |t| {
            const shape = try c.ints(o.get("shape").?);
            var numel: i64 = 1;
            for (shape) |d| numel *= d;
            // an empty tensor's pointer can precede its storage: pass null, as launchers expect for numel 0
            const raw = o.get("off").?.integer;
            const base = try c.ptrOf(t.string);
            const ptr: u64 = if (base == 0 or raw < 0 or numel == 0) 0 else base + @as(u64, @intCast(raw));
            return .{ .tensor = .{ .ptr = ptr, .dtype = try dtype(o.get("dtype").?.string), .shape = shape, .stride = try c.ints(o.get("stride").?) } };
        }
        if (o.get("i")) |x| return .{ .int = x.integer };
        if (o.get("f")) |x| return .{ .float = switch (x) {
            .float => |f| f,
            .integer => |n| @floatFromInt(n),
            else => return error.WrongType,
        } };
        if (o.get("b")) |x| return .{ .boolean = x.bool };
        if (o.get("n") != null) return .none;
        if (o.get("l")) |l| {
            const items = l.array.items;
            const out = try c.a.alloc(replay.Arg, items.len);
            for (items, out) |x, *y| y.* = try c.arg(x);
            return .{ .list = out };
        }
        return error.WrongType;
    }

    /// A zeroed device buffer of `n` bytes owned by the case.
    fn zeros(c: *Ctx, gpu: Gpu, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(gpu.d, @max(n, 16));
        try b.fill8(0, null);
        try c.extra.append(c.gpa, b);
        return b.ptr;
    }
};

fn upload(gpu: Gpu, fx: Fixture, storages: []Storage) !void {
    var name_buf: [64]u8 = undefined;
    for (storages) |s| {
        if (s.buf) |b| {
            const host = try fx.bytes(try std.fmt.bufPrint(&name_buf, "{s}_before", .{s.name}));
            defer gpu.gpa.free(host);
            try b.upload(0, host);
        }
    }
}

/// Every storage against the recorded after-call bytes, except `skip` (a ticket / counter replaced by a variant).
fn compare(gpu: Gpu, fx: Fixture, storages: []Storage, what: []const u8, skip: ?[]const u8) !void {
    var name_buf: [64]u8 = undefined;
    for (storages) |s| {
        const b = s.buf orelse continue;
        if (skip) |k| if (std.mem.eql(u8, k, s.name)) continue;
        const want = try fx.bytes(try std.fmt.bufPrint(&name_buf, "{s}_after", .{s.name}));
        defer gpu.gpa.free(want);
        const got = try gpu.gpa.alloc(u8, want.len);
        defer gpu.gpa.free(got);
        try b.download(0, got);
        const label = try std.fmt.allocPrint(gpu.gpa, "{s} storage {s}", .{ what, s.name });
        defer gpu.gpa.free(label);
        try check.sameBytes(label, got, want);
    }
}

fn storageOf(v: Value) ?[]const u8 {
    const t = v.object.get("t") orelse return null;
    return t.string;
}

pub fn run(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !Outcome {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const call = fx.parsed.value.object.get("call") orelse return error.MissingField;
    const ext = call.object.get("ext").?.string;
    const func = call.object.get("func").?.string;

    // the storages, sized by their recorded bytes
    const n: usize = @intCast(try fx.int("storages"));
    const storages = try a.alloc(Storage, n);
    var made: usize = 0;
    defer for (storages[0..made]) |*s| if (s.buf) |*b| b.free();
    const arrays = fx.parsed.value.object.get("arrays").?.object;
    for (storages, 0..) |*s, i| {
        s.name = try std.fmt.allocPrint(a, "s{d}", .{i});
        const ent = arrays.get(try std.fmt.allocPrint(a, "s{d}_before", .{i})).?.object;
        s.len = @intCast(ent.get("shape").?.array.items[0].integer);
        s.buf = if (s.len == 0) null else try cuda.DeviceBuffer.alloc(gpu.d, s.len);
        made += 1;
    }
    var c: Ctx = .{ .gpa = gpu.gpa, .a = a, .storages = storages };
    defer {
        for (c.extra.items) |*b| b.free();
        c.extra.deinit(gpu.gpa);
    }
    const jargs = call.object.get("args").?.array.items;
    const args = try a.alloc(replay.Arg, jargs.len);
    for (jargs, args) |v, *o| o.* = try c.arg(v);
    var kws: std.ArrayList(replay.Kw) = .empty;
    var it = call.object.get("kwargs").?.object.iterator();
    while (it.next()) |e| try kws.append(a, .{ .name = e.key_ptr.*, .value = try c.arg(e.value_ptr.*) });

    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const scratch_buf = try cuda.DeviceBuffer.alloc(gpu.d, 64 << 20);
    var scratch_owned = scratch_buf;
    defer scratch_owned.free();
    var scratch: replay.Scratch = .{ .ptr = scratch_buf.ptr, .len = scratch_buf.len };

    try upload(gpu, fx, storages);
    if (!try replay.call(k, stream, ext, func, args, kws.items, &scratch)) return .skipped;
    try stream.synchronize();
    try compare(gpu, fx, storages, func, null);

    var variants: usize = 0;
    // R1 variants of a pre-R1 call (the twin's kernel, today's launch): the same bits by contract
    if (std.mem.eql(u8, ext, "tf_dsv41_attn_cuda_v1") and std.mem.eql(u8, func, "attn") and args.len == 23) {
        const ticket_name = storageOf(jargs[18]).?;
        const q = args[0].tensor;
        const hb: usize = @intCast(@divTrunc(q.shape[1], 16));
        for ([_]i64{ 1, 4 }) |p| {
            const v = try a.alloc(replay.Arg, 24);
            @memcpy(v[0..23], args);
            var t = args[18].tensor;
            t.ptr = try c.zeros(gpu, @as(usize, @intCast(q.shape[0])) * hb * dsv41.ops.attn.MAXP * 4);
            v[18] = .{ .tensor = t };
            v[23] = .{ .int = p };
            try upload(gpu, fx, storages);
            if (!try replay.call(k, stream, ext, func, v, kws.items, &scratch)) return error.VariantNotBound;
            try stream.synchronize();
            try compare(gpu, fx, storages, if (p == 1) "attn split 1" else "attn split 4", ticket_name);
            variants += 1;
        }
    }
    if (std.mem.eql(u8, ext, "tf_dsv41_mhc_cuda_v1") and std.mem.eql(u8, func, "run") and args.len == 24) {
        const cnt_name = storageOf(jargs[17]).?;
        const mode = args[19].int;
        const opre = args[14].tensor;
        var opre_n: i64 = 1;
        for (opre.shape) |d| opre_n *= d;
        // the deferred form only where the boundary computes the next site's coefficients (MIX: modes 0-2)
        const forms: []const bool = if (mode <= 2 and opre_n > 0) &.{ false, true } else &.{false};
        for (forms) |deferred| {
            const v = try a.alloc(replay.Arg, 26);
            @memcpy(v[0..24], args);
            var t = args[17].tensor;
            t.ptr = try c.zeros(gpu, 16); // the tail's 3 self-resetting ints
            v[17] = .{ .tensor = t };
            v[24] = .{ .int = 0 }; // spin 0: the parallel tail, no waiting
            v[25] = .{ .int = @intFromBool(deferred) };
            try upload(gpu, fx, storages);
            if (!try replay.call(k, stream, ext, func, v, kws.items, &scratch)) return error.VariantNotBound;
            if (deferred) {
                // coef_kernel: (part, base, scale, opre, opost, ocomb, R, eps, hc_eps, post_alpha, iters)
                const cv = [_]replay.Arg{ args[9], args[7], args[8], args[14], args[15], args[16], args[18], args[20], args[21], args[22], args[23] };
                if (!try replay.call(k, stream, ext, "coef", &cv, &.{}, &scratch)) return error.VariantNotBound;
            }
            try stream.synchronize();
            try compare(gpu, fx, storages, if (deferred) "mhc tail + defer" else "mhc tail", cnt_name);
            variants += 1;
        }
    }
    const test_name = if (call.object.get("test")) |t| t.string else "";
    check.pass("BITEXACT replay {s}.{s} ({d} storages, {d} R1 variants) from {s}", .{ ext, func, n, variants, test_name });
    return .exact;
}
