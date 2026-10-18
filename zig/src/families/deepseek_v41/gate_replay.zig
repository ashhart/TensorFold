//! The capture replay: each captured launch re-issued on our buffers, compared bit for bit, alone and chained.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const load = @import("cuda_weights.zig");
const tri = @import("triton_capture.zig");
const rp = dk.replay;
const Value = std.json.Value;
const Hex = [64]u8;

pub const Status = enum { equal, differ, skipped, failed };

/// One op's outcome in one mode.
pub const Outcome = struct {
    status: Status,
    why: []const u8 = "",
    /// buffers that differ, and the first one's differing bytes and first differing offset
    differ: u32 = 0,
    diff_bytes: u64 = 0,
    first: u64 = 0,
    /// chained mode: buffers fed from our own earlier outputs
    chained_in: u32 = 0,
    /// per buffer: "id:fresh|chained:equal|differ N@first" (r.gpa-owned, the caller frees it)
    detail: []u8 = &.{},
};

const Live = struct { buf: cuda.DeviceBuffer, after: Hex, used: u64 };

pub const Replay = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: ?*const dk.Kernels,
    weights: *const load.Weights,
    triton: *const tri.Set,
    dir: []const u8,
    /// chained mode: our buffer of each captured storage and the capture's state it should hold
    live: std.StringHashMapUnmanaged(Live) = .empty,
    live_bytes: usize = 0,
    live_cap: usize = 24 << 30,
    tick: u64 = 0,
    /// where differing buffers are saved (null: not saved)
    diff_dir: ?[]const u8 = null,
    diff_saved: usize = 0,

    pub fn deinit(r: *Replay) void {
        var it = r.live.iterator();
        while (it.next()) |e| {
            e.value_ptr.buf.free();
            r.gpa.free(e.key_ptr.*);
        }
        r.live.deinit(r.gpa);
    }

    fn blob(r: *Replay, a: std.mem.Allocator, hex: []const u8, len: usize) ![]u8 {
        const path = try std.fmt.allocPrint(a, "{s}/blobs/{s}/{s}.bin", .{ r.dir, hex[0..2], hex });
        const data = try std.Io.Dir.cwd().readFileAlloc(r.io, path, a, .limited(1 << 36));
        if (data.len != len) return error.BlobSize;
        return data;
    }

    /// The op's buffers as device addresses (index = buffer id), loaded with the capture's inputs unless chained.
    const Bound = struct { addr: []u64, own: []cuda.DeviceBuffer, hex_after: []Hex, fresh: []bool, has_reloc: []bool };

    fn bind(r: *Replay, a: std.mem.Allocator, op: std.json.ObjectMap, chained: bool, out: *Outcome) !Bound {
        const bufs = op.get("buffers").?.array.items;
        const relocs = op.get("relocs").?.array.items;
        var b: Bound = .{ .addr = try a.alloc(u64, bufs.len), .own = try a.alloc(cuda.DeviceBuffer, bufs.len), .hex_after = try a.alloc(Hex, bufs.len), .fresh = &.{}, .has_reloc = &.{} };
        @memset(b.own, .{ .d = r.d, .ptr = 0, .len = 0 });
        errdefer for (b.own) |*x| if (x.ptr != 0) x.free();
        var has_reloc = try a.alloc(bool, bufs.len);
        @memset(has_reloc, false);
        for (relocs) |rl| has_reloc[@intCast(rl.object.get("buf").?.integer)] = true;
        var fresh = try a.alloc(bool, bufs.len);
        b.fresh = fresh;
        b.has_reloc = has_reloc;
        r.tick += 1;
        for (bufs) |bv| {
            const o = bv.object;
            const id: usize = @intCast(o.get("id").?.integer);
            const n: usize = @intCast(o.get("nbytes").?.integer);
            const before = o.get("before").?;
            @memcpy(&b.hex_after[id], o.get("after").?.string[0..64]);
            fresh[id] = n > 0;
            if (n == 0) { // an empty tensor: a null pointer, as the launchers expect
                b.addr[id] = 0;
                continue;
            }
            if (!chained) {
                b.own[id] = try cuda.DeviceBuffer.alloc(r.d, @max(n, 1));
                b.addr[id] = b.own[id].ptr;
                continue;
            }
            const key = o.get("key").?.string;
            if (r.live.getPtr(key)) |lv| {
                lv.used = r.tick;
                b.addr[id] = lv.buf.ptr;
                if (before == .string and !has_reloc[id] and std.mem.eql(u8, &lv.after, before.string[0..64])) {
                    fresh[id] = false;
                    out.chained_in += 1;
                }
                continue;
            }
            if (r.live_bytes + n > r.live_cap) r.evict();
            const buf = try cuda.DeviceBuffer.alloc(r.d, @max(n, 1));
            try r.live.put(r.gpa, try r.gpa.dupe(u8, key), .{ .buf = buf, .after = @splat('0'), .used = r.tick });
            r.live_bytes += n;
            b.addr[id] = buf.ptr;
        }
        // the capture's bytes into every buffer not chained, pointer tables relocated to our addresses
        for (bufs) |bv| {
            const o = bv.object;
            const id: usize = @intCast(o.get("id").?.integer);
            if (!fresh[id]) continue;
            const n: usize = @intCast(o.get("nbytes").?.integer);
            const before = o.get("before").?;
            const host = if (before == .string) try r.blob(a, before.string, n) else try a.alloc(u8, n);
            if (before != .string) @memset(host, 0);
            try r.patch(relocs, id, b.addr, host);
            try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = b.addr[id], .len = @max(n, 1) }, 0, host);
        }
        // a pageable upload may return before its DMA lands: every byte is on the device before the launch
        try r.d.synchronize();
        return b;
    }

    /// Buffer `id`'s device addresses (pointer tables) rewritten to our weights' and buffers' addresses.
    fn patch(r: *Replay, relocs: []const Value, id: usize, addr: []const u64, host: []u8) !void {
        for (relocs) |rl| {
            const x = rl.object;
            if (@as(usize, @intCast(x.get("buf").?.integer)) != id) continue;
            const at: usize = @intCast(x.get("at").?.integer);
            // an int64 the capture could not place (a sort key's high word looks like an address): left as captured
            if (x.get("unresolved") != null) continue;
            const delta: u64 = @bitCast(x.get("delta").?.integer);
            const base = if (x.get("weight")) |w| (r.weights.get(w.string) orelse return error.MissingWeight).ptr else addr[@intCast(x.get("target").?.integer)];
            std.mem.writeInt(u64, host[at..][0..8], base + delta, .little);
        }
    }

    /// Frees chained buffers not used by the current op (the simplest bound on what the chain keeps).
    fn evict(r: *Replay) void {
        var it = r.live.iterator();
        var drop: std.ArrayList([]const u8) = .empty;
        defer drop.deinit(r.gpa);
        while (it.next()) |e| if (e.value_ptr.used != r.tick) drop.append(r.gpa, e.key_ptr.*) catch {};
        for (drop.items) |k| {
            const kv = r.live.fetchRemove(k).?;
            r.live_bytes -= kv.value.buf.len;
            var buf = kv.value.buf;
            buf.free();
            r.gpa.free(kv.key);
        }
    }

    const Ctx = struct { r: *Replay, b: *const Bound };

    fn resolveFn(ctx: *anyopaque, arg: Value) anyerror!?u64 {
        const c: *Ctx = @ptrCast(@alignCast(ctx));
        return c.r.address(c.b, arg);
    }

    fn address(r: *Replay, b: *const Bound, arg: Value) !?u64 {
        const o = arg.object;
        const t = o.get("t").?.string;
        if (std.mem.eql(u8, t, "none")) return null;
        if (std.mem.eql(u8, t, "int")) return @bitCast(o.get("v").?.integer);
        if (!std.mem.eql(u8, t, "tensor")) return error.WrongArgument;
        if (o.get("host") != null) return error.HostTensor;
        // an empty tensor's data pointer can be 0 inside a live storage: a null pointer, as for numel 0
        const raw = o.get("offset").?.integer;
        var numel: i64 = 1;
        for (o.get("shape").?.array.items) |d| numel = std.math.mul(i64, numel, d.integer) catch return error.BadOperation;
        if (raw < 0 or numel == 0) return 0;
        const off: u64 = @intCast(raw);
        if (o.get("weight")) |w| {
            const weight = r.weights.get(w.string) orelse return error.MissingWeight;
            if (off >= weight.len) return error.BadOperation;
            return std.math.add(u64, weight.ptr, off) catch error.BadOperation;
        }
        const id = std.math.cast(usize, o.get("buf").?.integer) orelse return error.BadOperation;
        if (id >= b.addr.len) return error.BadOperation;
        return std.math.add(u64, b.addr[id], off) catch error.BadOperation;
    }

    fn extArg(r: *Replay, a: std.mem.Allocator, b: *const Bound, v: Value) !rp.Arg {
        const o = v.object;
        const t = o.get("t").?.string;
        if (std.mem.eql(u8, t, "none")) return .none;
        if (std.mem.eql(u8, t, "bool")) return .{ .boolean = o.get("v").?.bool };
        if (std.mem.eql(u8, t, "float")) return .{ .float = @bitCast(try std.fmt.parseInt(u64, o.get("f64").?.string[2..], 16)) };
        if (std.mem.eql(u8, t, "int")) {
            if (o.get("ptr")) |p| return .{ .int = @bitCast((r.weights.get(p.object.get("weight").?.string) orelse return error.MissingWeight).ptr + @as(u64, @bitCast(p.object.get("delta").?.integer))) };
            return .{ .int = o.get("v").?.integer };
        }
        if (std.mem.eql(u8, t, "list")) {
            const items = o.get("items").?.array.items;
            const out = try a.alloc(rp.Arg, items.len);
            for (items, out) |x, *y| y.* = try r.extArg(a, b, x);
            return .{ .list = out };
        }
        if (!std.mem.eql(u8, t, "tensor")) return error.WrongArgument;
        const dtype = std.meta.stringToEnum(enum { float16, bfloat16, float32, float64, int8, uint8, int16, int32, int64, bool }, o.get("dtype").?.string) orelse return error.UnsupportedDType;
        const map = [_]rp.DType{ .f16, .bf16, .f32, .f64, .i8, .u8, .i16, .i32, .i64, .bool };
        const shape = o.get("shape").?.array.items;
        const stride = o.get("stride").?.array.items;
        const s = try a.alloc(i64, shape.len);
        const st = try a.alloc(i64, stride.len);
        for (shape, s) |x, *y| y.* = x.integer;
        for (stride, st) |x, *y| y.* = x.integer;
        return .{ .tensor = .{ .ptr = (try r.address(b, v)).?, .dtype = map[@backingInt(dtype)], .shape = s, .stride = st } };
    }

    /// Re-issues one op in one mode and compares its buffers with the capture's.
    pub fn run(r: *Replay, op: std.json.ObjectMap, chained: bool) Outcome {
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        var out: Outcome = .{ .status = .equal };
        r.issue(arena.allocator(), op, chained, &out) catch |e| {
            out.why = @errorName(e);
            const skips = [_][]const u8{ "Prefetch", "NoBinding", "NoKernels", "KernelNotLoaded", "HostTensor", "UnsupportedDType", "UnsupportedScalar", "UnresolvedPointer", "MissingWeight" };
            out.status = .failed;
            for (skips) |s| if (std.mem.eql(u8, s, out.why)) {
                out.status = .skipped;
            };
        };
        return out;
    }

    fn issue(r: *Replay, a: std.mem.Allocator, op: std.json.ObjectMap, chained: bool, out: *Outcome) !void {
        const b = try r.bind(a, op, chained, out);
        defer for (b.own) |*x| if (x.ptr != 0) x.free();
        var ctx: Ctx = .{ .r = r, .b = &b };
        const kind = op.get("kind").?.string;
        if (std.mem.eql(u8, kind, "triton")) {
            const k = r.triton.kernels.getPtr(op.get("hash").?.string) orelse return error.KernelNotLoaded;
            const need = try tri.scratchBytes(k, op);
            var scratch = try cuda.DeviceBuffer.alloc(r.d, need);
            defer scratch.free();
            try tri.launch(k, op, r.stream, &ctx, resolveFn, scratch.ptr);
        } else {
            // the L2 prefetchers only warm the cache, and the paced one can stall replayed out of its round
            const ext = op.get("ext").?.string;
            if (std.mem.startsWith(u8, ext, "tf_dsv41_l2pace") or std.mem.startsWith(u8, ext, "tf_dsv41_l2pf")) return error.Prefetch;
            const ks = r.kernels orelse return error.NoKernels;
            const given = op.get("args").?.array.items;
            const args = try a.alloc(rp.Arg, given.len);
            for (given, args) |g, *x| x.* = try r.extArg(a, &b, g);
            var kws: std.ArrayList(rp.Kw) = .empty;
            var it = op.get("kwargs").?.object.iterator();
            while (it.next()) |e| try kws.append(a, .{ .name = e.key_ptr.*, .value = try r.extArg(a, &b, e.value_ptr.*) });
            if (op.get("ret")) |ret| if (ret == .object) try kws.append(a, .{ .name = "__ret__", .value = try r.extArg(a, &b, ret) });
            var sbuf = try cuda.DeviceBuffer.alloc(r.d, 64 << 20);
            defer sbuf.free();
            var scratch: rp.Scratch = .{ .ptr = sbuf.ptr, .len = sbuf.len };
            if (!try rp.call(ks, r.stream, op.get("ext").?.string, op.get("func").?.string, args, kws.items, &scratch)) return error.NoBinding;
        }
        try r.stream.synchronize();
        try r.compare(a, op, &b, chained, out);
    }

    fn compare(r: *Replay, a: std.mem.Allocator, op: std.json.ObjectMap, b: *const Bound, chained: bool, out: *Outcome) !void {
        var detail: std.Io.Writer.Allocating = .init(r.gpa);
        defer detail.deinit();
        const relocs = op.get("relocs").?.array.items;
        for (op.get("buffers").?.array.items) |bv| {
            const o = bv.object;
            const id: usize = @intCast(o.get("id").?.integer);
            const n: usize = @intCast(o.get("nbytes").?.integer);
            if (n == 0) continue;
            const got = try a.alloc(u8, n);
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = b.addr[id], .len = @max(n, 1) }, 0, got);
            if (chained) r.live.getPtr(o.get("key").?.string).?.after = b.hex_after[id];
            // a pointer table holds our addresses where the capture holds its own: relocate the expected bytes
            var same: bool = undefined;
            var want: ?[]u8 = null;
            if (b.has_reloc[id]) {
                want = try r.blob(a, &b.hex_after[id], n);
                try r.patch(relocs, id, b.addr, want.?);
                same = std.mem.eql(u8, got, want.?);
            } else {
                var sha: [32]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(got, &sha, .{});
                same = std.mem.eql(u8, &std.fmt.bytesToHex(sha, .lower), &b.hex_after[id]);
            }
            const how = if (b.fresh[id]) "fresh" else "chained";
            if (same) {
                try detail.writer.print("{d}:{s}:equal ", .{ id, how });
                continue;
            }
            out.status = .differ;
            out.differ += 1;
            const w = want orelse try r.blob(a, &b.hex_after[id], n);
            var count: u64 = 0;
            var first: ?usize = null;
            for (got, w, 0..) |x, y, i| if (x != y) {
                count += 1;
                if (first == null) first = i;
            };
            if (out.differ == 1) {
                out.diff_bytes = count;
                out.first = first orelse 0;
            }
            try detail.writer.print("{d}:{s}:differ {d}@{d} ", .{ id, how, count, first orelse 0 });
            r.saveDiff(op, id, chained, got, w);
        }
        out.detail = try r.gpa.dupe(u8, detail.written());
    }

    /// Our bytes and the capture's of a differing buffer (up to 4 MB each, 400 MB in all) into `diff_dir`.
    fn saveDiff(r: *Replay, op: std.json.ObjectMap, id: usize, chained: bool, got: []const u8, want: []const u8) void {
        const dir = r.diff_dir orelse return;
        if (got.len > 4 << 20 or r.diff_saved + 2 * got.len > 400 << 20) return;
        r.diff_saved += 2 * got.len;
        var buf: [Io_max_path]u8 = undefined;
        const seq = op.get("seq").?.integer;
        const mode = if (chained) "chained" else "alone";
        inline for (.{ .{ "ours", 0 }, .{ "want", 1 } }) |kind| {
            const path = std.fmt.bufPrint(&buf, "{s}/{d}-{s}-{d}-{s}.bin", .{ dir, seq, mode, id, kind[0] }) catch return;
            std.Io.Dir.cwd().writeFile(r.io, .{ .sub_path = path, .data = if (kind[1] == 0) got else want }) catch {};
        }
    }
};

const Io_max_path = 1024;

/// Our loader's digests against the capture's weights.json: (equal, differ, missing here).
pub fn checkWeights(gpa: std.mem.Allocator, io: std.Io, w: *const load.Weights, dir: []const u8, log: *std.Io.Writer) ![3]usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.json" }), a, .limited(1 << 28));
    const root = try std.json.parseFromSliceLeaky(Value, a, text, .{});
    var n: [3]usize = @splat(0);
    var it = root.object.iterator();
    while (it.next()) |e| {
        const want = e.value_ptr.object.get("sha256").?.string;
        const got = w.get(e.key_ptr.*) orelse {
            n[2] += 1;
            try log.print("{{\"weight\": \"{s}\", \"status\": \"missing\"}}\n", .{e.key_ptr.*});
            continue;
        };
        const hex = std.fmt.bytesToHex(got.sha256, .lower);
        const same = std.mem.eql(u8, &hex, want);
        n[if (same) 0 else 1] += 1;
        if (!same) try log.print("{{\"weight\": \"{s}\", \"status\": \"differ\", \"bytes\": {d}}}\n", .{ e.key_ptr.*, got.len });
    }
    return n;
}
