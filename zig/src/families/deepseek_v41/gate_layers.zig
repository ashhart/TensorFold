//! The block-forward check on the GPU: each captured decode window's calls on our buffers, compared bit for bit.

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const check = @import("calls_check.zig");
const load = @import("cuda_weights.zig");
const tri = @import("triton_capture.zig");
const forms = @import("forms.zig");
const plan = @import("plan.zig");
const Config = @import("config.zig").Config;
const rp = dk.replay;
const Value = std.json.Value;

const Buf = struct { dev: cuda.DeviceBuffer, live: bool = false, after: [64]u8 = @splat('0') };

pub const Counts = struct {
    calls: usize = 0,
    equal: usize = 0,
    differ: usize = 0,
    skipped: usize = 0,
    failed: usize = 0,
    seeded: usize = 0,
    reseeded: usize = 0,
    prepared_equal: usize = 0,
    prepared_differ: usize = 0,
};

pub const Forward = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const dk.Kernels,
    weights: *const load.Weights,
    triton: *const tri.Set,
    plan: *const plan.Plan,
    dir: []const u8,
    bufs: std.StringHashMapUnmanaged(Buf) = .empty,
    counts: Counts = .{},
    /// roles re-seeded for glue, and how often
    glue: std.StringArrayHashMapUnmanaged(u32) = .empty,

    /// Every buffer seeded again from the capture at its next first touch (after a window stopped on structure).
    pub fn unseat(f: *Forward) void {
        var it = f.bufs.valueIterator();
        while (it.next()) |b| b.live = false;
    }

    pub fn close(f: *Forward) !void {
        try f.stream.synchronize();
        f.deinit();
    }
    pub fn deinit(f: *Forward) void {
        var it = f.bufs.iterator();
        while (it.next()) |e| {
            e.value_ptr.dev.free();
            f.gpa.free(e.key_ptr.*);
        }
        f.bufs.deinit(f.gpa);
        for (f.glue.keys()) |k| f.gpa.free(k);
        f.glue.deinit(f.gpa);
    }

    fn blob(f: *Forward, a: std.mem.Allocator, hex: []const u8, len: usize) ![]u8 {
        const path = try std.fmt.allocPrint(a, "{s}/blobs/{s}/{s}.bin", .{ f.dir, hex[0..2], hex });
        const data = try std.Io.Dir.cwd().readFileAlloc(f.io, path, a, .limited(1 << 36));
        if (data.len != len) return error.BlobSize;
        return data;
    }

    /// Scope start: window and layer roles are seeded again at their next touch.
    fn begin(f: *Forward, b: calls.Begin) void {
        if (b == .none) return;
        var it = f.bufs.iterator();
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            if (std.mem.startsWith(u8, k, "s.")) continue;
            if (b == .layer and std.mem.startsWith(u8, k, "w.")) continue;
            e.value_ptr.live = false;
        }
    }

    /// Our buffer of `role`, at least `n` bytes (a new or larger one when needed: never while live).
    fn buffer(f: *Forward, role: []const u8, n: usize) !*Buf {
        const g = try f.bufs.getOrPut(f.gpa, role);
        if (!g.found_existing) {
            g.key_ptr.* = try f.gpa.dupe(u8, role);
            g.value_ptr.* = .{ .dev = try cuda.DeviceBuffer.alloc(f.d, @max(n, 1)) };
        } else if (g.value_ptr.dev.len < n) {
            if (g.value_ptr.live) return error.RoleGrew;
            g.value_ptr.dev.free();
            g.value_ptr.dev = try cuda.DeviceBuffer.alloc(f.d, n);
        }
        return g.value_ptr;
    }

    const Bound = struct { role: []const u8, id: i64, o: std.json.ObjectMap };

    /// Pointer tables: the capture's addresses rewritten to our weights' and roles' (the reloc's target buffer).
    fn patch(f: *Forward, relocs: []const Value, id: i64, bound: []const Bound, host: []u8) !void {
        for (relocs) |rl| {
            const x = rl.object;
            if (x.get("buf").?.integer != id or x.get("unresolved") != null) continue;
            const at: usize = @intCast(x.get("at").?.integer);
            const delta: u64 = @bitCast(x.get("delta").?.integer);
            const base = if (x.get("weight")) |w| (f.weights.get(w.string) orelse return error.MissingWeight).ptr else blk: {
                const t = x.get("target").?.integer;
                for (bound) |b| if (b.id == t) break :blk f.bufs.get(b.role).?.dev.ptr;
                return error.UnboundTarget;
            };
            std.mem.writeInt(u64, host[at..][0..8], base + delta, .little);
        }
    }

    fn addr(f: *Forward, t: calls.Tensor) !u64 {
        var numel: i64 = 1;
        for (t.shape) |x| numel *= x;
        if (numel == 0) return 0;
        const off: u64 = @intCast(t.offset);
        return switch (t.role) {
            .weight => |w| (f.weights.get(w) orelse return error.MissingWeight).ptr + off,
            .buf => |r| (f.bufs.get(r) orelse return error.Unbound).dev.ptr + off,
            .empty => error.EmptyRole,
        };
    }

    fn extArg(f: *Forward, a: std.mem.Allocator, x: calls.Arg) !rp.Arg {
        return switch (x) {
            .i => |v| .{ .int = v },
            .f => |v| .{ .float = v },
            .b => |v| .{ .boolean = v },
            .none => .none,
            .list => |items| blk: {
                const out = try a.alloc(rp.Arg, items.len);
                for (items, out) |y, *z| z.* = try f.extArg(a, y);
                break :blk .{ .list = out };
            },
            .t, .opaque_table => |t| .{ .tensor = .{ .ptr = try f.addr(t), .dtype = std.meta.stringToEnum(rp.DType, @tagName(t.dt)).?, .shape = t.shape, .stride = t.stride } },
        };
    }

    const Ctx = struct { f: *Forward, c: *const calls.Call };

    /// A Triton argument's address from our call's argument of the same name (the capture's op picks the variant).
    fn resolveFn(ctx: *anyopaque, arg: Value) anyerror!?u64 {
        const c: *Ctx = @ptrCast(@alignCast(ctx));
        const name = arg.object.get("name").?.string;
        for (c.c.args) |x| if (std.mem.eql(u8, x.name, name)) return switch (x.arg) {
            .none => null,
            .i => |v| @bitCast(v),
            .t => |t| try c.f.addr(t),
            else => error.WrongArgument,
        };
        return error.MissingArgument;
    }

    /// Issues one call (op: the capture's launch it matched) and compares the buffers it touches.
    pub fn run(f: *Forward, c: *const calls.Call, op: std.json.ObjectMap, log: *std.Io.Writer, where: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(f.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        f.begin(c.begin);
        f.counts.calls += 1;
        if (std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) { // L2 prefetch: no output any call reads
            f.counts.skipped += 1;
            try log.print("{{\"at\": \"{s}\", \"name\": \"{s}\", \"status\": \"skipped\"}}\n", .{ where, c.name });
            return;
        }
        var bs: std.ArrayList(check.Binding) = .empty;
        for (c.args, op.get("args").?.array.items) |x, y| try check.bindings(a, x.arg, y, &bs);
        const bufs = op.get("buffers").?.array.items;
        const relocs = op.get("relocs").?.array.items;
        var bound: std.ArrayList(Bound) = .empty;
        for (bs.items) |b| {
            const seen = for (bound.items) |x| {
                if (std.mem.eql(u8, x.role, b.role)) break true;
            } else false;
            if (seen) continue;
            const o = for (bufs) |v| {
                if (v.object.get("id").?.integer == b.id) break v.object;
            } else return error.NoBuffer;
            try bound.append(a, .{ .role = b.role, .id = b.id, .o = o });
        }
        // allocate first (pointer tables point at roles), then seed
        for (bound.items) |b| _ = try f.buffer(b.role, @intCast(b.o.get("nbytes").?.integer));
        var seeds: u32 = 0;
        for (bound.items) |b| {
            const n: usize = @intCast(b.o.get("nbytes").?.integer);
            const buf = f.bufs.getPtr(b.role).?;
            const before = b.o.get("before").?;
            const has_reloc = for (relocs) |rl| {
                if (rl.object.get("buf").?.integer == b.id) break true;
            } else false;
            const fresh = !buf.live;
            const glue = buf.live and before == .string and !std.mem.eql(u8, &buf.after, before.string[0..64]);
            if (!fresh and !glue) continue;
            const host = if (before == .string) try f.blob(a, before.string, n) else blk: {
                const z = try a.alloc(u8, n);
                @memset(z, 0);
                break :blk z;
            };
            if (has_reloc) try f.patch(relocs, b.id, bound.items, host);
            if (fresh) {
                f.counts.seeded += 1;
                // our forms against the reference's (pointer tables after the capture's addresses became ours)
                try f.prepared(a, b.role, host, log);
            } else {
                f.counts.reseeded += 1;
                const g = try f.glue.getOrPut(f.gpa, b.role);
                if (!g.found_existing) {
                    g.key_ptr.* = try f.gpa.dupe(u8, b.role);
                    g.value_ptr.* = 0;
                }
                g.value_ptr.* += 1;
            }
            try cuda.DeviceBuffer.upload(.{ .d = f.d, .ptr = buf.dev.ptr, .len = @max(n, 1) }, 0, host);
            buf.live = true;
            seeds += 1;
        }
        try f.d.synchronize();
        f.launch(a, c, op) catch |e| {
            f.counts.failed += 1;
            try log.print("{{\"at\": \"{s}\", \"name\": \"{s}\", \"status\": \"failed\", \"why\": \"{t}\"}}\n", .{ where, c.name, e });
            return;
        };
        try f.stream.synchronize();
        var differ: std.ArrayList(u8) = .empty;
        for (bound.items) |b| {
            const n: usize = @intCast(b.o.get("nbytes").?.integer);
            const buf = f.bufs.getPtr(b.role).?;
            const after = b.o.get("after").?.string;
            @memcpy(&buf.after, after[0..64]);
            const got = try a.alloc(u8, n);
            try cuda.DeviceBuffer.download(.{ .d = f.d, .ptr = buf.dev.ptr, .len = @max(n, 1) }, 0, got);
            const has_reloc = for (relocs) |rl| {
                if (rl.object.get("buf").?.integer == b.id) break true;
            } else false;
            const same = if (has_reloc) blk: {
                const want = try f.blob(a, after, n);
                try f.patch(relocs, b.id, bound.items, want);
                break :blk std.mem.eql(u8, got, want);
            } else blk: {
                var sha: [32]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(got, &sha, .{});
                break :blk std.mem.eql(u8, &std.fmt.bytesToHex(sha, .lower), after[0..64]);
            };
            if (!same) try differ.print(a, "{s} ", .{b.role});
        }
        if (differ.items.len == 0) f.counts.equal += 1 else f.counts.differ += 1;
        try log.print("{{\"at\": \"{s}\", \"name\": \"{s}\", \"status\": \"{s}\", \"seeded\": {d}, \"differ\": \"{s}\"}}\n", .{ where, c.name, if (differ.items.len == 0) "equal" else "differ", seeds, differ.items });
    }

    fn launch(f: *Forward, a: std.mem.Allocator, c: *const calls.Call, op: std.json.ObjectMap) !void {
        if (c.triton) {
            const k = f.triton.kernels.getPtr(op.get("hash").?.string) orelse return error.KernelNotLoaded;
            var scratch = try cuda.DeviceBuffer.alloc(f.d, @max(try tri.scratchBytes(k, op), 1));
            defer scratch.free();
            var ctx: Ctx = .{ .f = f, .c = c };
            try tri.launch(k, op, f.stream, &ctx, resolveFn, scratch.ptr);
            return;
        }
        const dot = std.mem.lastIndexOfScalar(u8, c.name, '.').?;
        const args = try a.alloc(rp.Arg, c.args.len);
        for (c.args, args) |x, *y| y.* = try f.extArg(a, x.arg);
        var sbuf = try cuda.DeviceBuffer.alloc(f.d, 64 << 20);
        defer sbuf.free();
        var scratch: rp.Scratch = .{ .ptr = sbuf.ptr, .len = sbuf.len };
        if (!try rp.call(f.kernels, f.stream, c.name[0..dot], c.name[dot + 1 ..], args, &.{}, &scratch)) return error.NoBinding;
    }

    fn prepared(f: *Forward, a: std.mem.Allocator, role: []const u8, seed: []const u8, log: *std.Io.Writer) !void {
        if (!std.mem.startsWith(u8, role, "s.")) return;
        const src: forms.Source = .{ .d = f.d, .weights = f.weights, .plan = f.plan };
        const ours = forms.build(a, &src, role) catch |e| {
            f.counts.prepared_differ += 1;
            try log.print("{{\"prepared\": \"{s}\", \"status\": \"failed\", \"why\": \"{t}\"}}\n", .{ role, e });
            return;
        } orelse return;
        // the seed is the whole storage; ours is its head (a view of a larger buffer keeps the rest)
        const same = ours.len <= seed.len and std.mem.eql(u8, ours, seed[0..ours.len]);
        if (same) f.counts.prepared_equal += 1 else f.counts.prepared_differ += 1;
        try log.print("{{\"prepared\": \"{s}\", \"status\": \"{s}\", \"bytes\": {d}, \"seed_bytes\": {d}}}\n", .{ role, if (same) "equal" else "differ", ours.len, seed.len });
    }
};

/// The host check, then the GPU forward over every captured window; writes forward.jsonl and returns the counts.
pub fn runAll(f: *Forward, out_dir: []const u8, experts: []const u8) !Counts {
    var arena = std.heap.ArenaAllocator.init(f.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(f.io, try std.fs.path.join(a, &.{ out_dir, "forward.jsonl" }), .{});
    defer file.close(f.io);
    var fbuf: [1 << 16]u8 = undefined;
    var fw = file.writer(f.io, &fbuf);
    const log = &fw.interface;
    const listing = try cwd.readFileAlloc(f.io, try std.fs.path.join(a, &.{ f.dir, "weights.json" }), a, .limited(1 << 28));
    const w = try check.widthsFromCapture(a, listing, experts);
    const cfg: Config = .{};
    const ws = try check.windows(a, f.io, f.dir);
    var wg = w;
    try check.gmWidths(a, &wg, ws);
    var opts = try check.optionsOf(a, f.io, f.dir);
    opts.gm_v2 = check.gmMode(ws);
    opts.r1_sig = !opts.r1 and try check.r1SigOf(a, ws);
    var last_set: []const u8 = "";
    var roles: calls.Roles = .{ .a = a };
    for (ws) |win| {
        if (!std.mem.eql(u8, last_set, win.set)) { // each set ran in its own engine: every role seeded again
            var it = f.bufs.valueIterator();
            while (it.next()) |b| b.live = false;
            roles = .{ .a = a };
        }
        last_set = win.set;
        // a prefill segment (p<n>): block_prefill.zig's launches
        const cs = try calls.launches(a, if (win.prefill)
            try @import("block_prefill.zig").emitPrefill(a, &cfg, &wg, opts, win.layers, win.n, win.start, true)
        else
            try block.emit(a, &cfg, &wg, opts, win.layers, win.n, win.start, true));
        const before = f.counts;
        // a window that does not match the capture's structure stops there and every role is seeded again
        for (cs[0..@min(cs.len, win.ops.len)], win.ops[0..@min(cs.len, win.ops.len)], 0..) |*c, op, i| {
            roles.begin(c.begin);
            if (try calls.check(a, c, op, &roles)) |why| {
                std.debug.print("structure: {s} {s} #{d}: {s}\n", .{ win.set, win.phase, i, why });
                f.counts.failed += win.ops.len - i;
                f.unseat();
                break;
            }
            try f.run(c, op, log, try std.fmt.allocPrint(a, "{s} {s} #{d}", .{ win.set, win.phase, i }));
        } else if (cs.len != win.ops.len) {
            std.debug.print("structure: {s} {s}: {d} calls vs captured {d}\n", .{ win.set, win.phase, cs.len, win.ops.len });
            f.counts.failed += 1;
            f.unseat();
        }
        try log.flush();
        std.debug.print("forward {s} {s}: {d} calls, {d} equal, {d} differ, {d} skipped, {d} failed\n", .{ win.set, win.phase, f.counts.calls - before.calls, f.counts.equal - before.equal, f.counts.differ - before.differ, f.counts.skipped - before.skipped, f.counts.failed - before.failed });
    }
    return f.counts;
}

/// The GPU forward over a capture's set P (DSpark's ingest and passes), as runAll; no set P runs nothing.
pub fn runDraft(f: *Forward, out_dir: []const u8, experts: []const u8) !Counts {
    const ds = @import("dspark_emit.zig");
    var arena = std.heap.ArenaAllocator.init(f.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(f.io, try std.fs.path.join(a, &.{ out_dir, "forward-dspark.jsonl" }), .{});
    defer file.close(f.io);
    var fbuf: [1 << 16]u8 = undefined;
    var fw = file.writer(f.io, &fbuf);
    const log = &fw.interface;
    const listing = try cwd.readFileAlloc(f.io, try std.fs.path.join(a, &.{ f.dir, "weights.json" }), a, .limited(1 << 28));
    const w = try check.widthsFromCapture(a, listing, experts);
    const cfg: Config = .{};
    var it = f.bufs.valueIterator();
    while (it.next()) |b| b.live = false;
    var roles: calls.Roles = .{ .a = a };
    var dopts = try check.optionsOf(a, f.io, f.dir);
    const dphases = try ds.phases(a, f.io, f.dir, &cfg);
    dopts.r1_sig = !dopts.r1 and check.r1Sig(try opsOf(a, dphases));
    for (dphases) |p| {
        const cs = try ds.phaseCalls(a, &cfg, &w, dopts, p);
        const before = f.counts;
        for (cs[0..@min(cs.len, p.ops.len)], p.ops[0..@min(cs.len, p.ops.len)], 0..) |*c, op, i| {
            roles.begin(c.begin);
            if (try calls.check(a, c, op, &roles)) |why| {
                std.debug.print("structure: P {s} #{d}: {s}\n", .{ p.name, i, why });
                f.counts.failed += p.ops.len - i;
                f.unseat();
                break;
            }
            try f.run(c, op, log, try std.fmt.allocPrint(a, "P {s} #{d}", .{ p.name, i }));
        }
        try log.flush();
        std.debug.print("forward P {s}: {d} calls, {d} equal, {d} differ, {d} skipped, {d} failed\n", .{ p.name, f.counts.calls - before.calls, f.counts.equal - before.equal, f.counts.differ - before.differ, f.counts.skipped - before.skipped, f.counts.failed - before.failed });
    }
    return f.counts;
}

fn opsOf(a: std.mem.Allocator, ps: anytype) ![]const []std.json.ObjectMap {
    const lists = try a.alloc([]std.json.ObjectMap, ps.len);
    for (lists, ps) |*l, p| l.* = p.ops;
    return lists;
}
