//! The call executor: block.zig's calls issued on our own buffers, the plan bound once (persistent roles + one arena).

const std = @import("std");
const cuda = @import("cuda");
const dk = @import("dsv41_kernels");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const load = @import("cuda_weights.zig");
const tc = @import("triton_call.zig");
const rp = dk.replay;

/// A call timer behind the runner (profile.zig's Profile.timer); its methods forward to the timer's own.
pub const Timer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        begin: *const fn (ptr: *anyopaque) void,
        beginPrefill: *const fn (ptr: *anyopaque) void,
        prefillRows: *const fn (ptr: *anyopaque, rows: u64, tail: bool) void,
        before: *const fn (ptr: *anyopaque, s: cuda.Stream, c: *const calls.Call) anyerror!void,
        after: *const fn (ptr: *anyopaque, s: cuda.Stream) anyerror!void,
        end: *const fn (ptr: *anyopaque, s: cuda.Stream) anyerror!void,
        close: *const fn (ptr: *anyopaque, io: std.Io) void,
    };

    pub fn begin(t: Timer) void {
        t.vtable.begin(t.ptr);
    }
    pub fn beginPrefill(t: Timer) void {
        t.vtable.beginPrefill(t.ptr);
    }
    pub fn prefillRows(t: Timer, rows: u64, tail: bool) void {
        t.vtable.prefillRows(t.ptr, rows, tail);
    }
    pub fn before(t: Timer, s: cuda.Stream, c: *const calls.Call) !void {
        return t.vtable.before(t.ptr, s, c);
    }
    pub fn after(t: Timer, s: cuda.Stream) !void {
        return t.vtable.after(t.ptr, s);
    }
    pub fn end(t: Timer, s: cuda.Stream) !void {
        return t.vtable.end(t.ptr, s);
    }
    pub fn close(t: Timer, io: std.Io) void {
        t.vtable.close(t.ptr, io);
    }
};

pub const Runner = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const dk.Kernels,
    triton: *const cuda.aot.Set,
    weights: *const load.Weights,
    /// role -> device address
    addrs: std.StringHashMapUnmanaged(u64) = .empty,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    scratch: ?cuda.DeviceBuffer = null,
    /// the forward's glue steps (forward.zig: positions, exchanges over the Collective, Engram rows, kit roundings)
    glue: ?Glue = null,
    /// TF_DSV41_PROFILE: the eager decode windows' calls timed (profile.zig's Profile, set at engine open)
    prof: ?Timer = null,
    /// TF_DSV41_L2PF (l2pf.zig): L2 prefetch launches on a side stream forked at their site; null: skipped
    l2pf: ?*Side = null,
    /// rows of the window being issued (0: unknown): a window wider than `Side.rows` skips every prefetch site
    rows: u32 = 0,
    /// TF_DSV41_BRANCHES (branches.zig): the side stream of calls tagged `side`; null: every call on main
    br: ?*Side = null,
    /// TF_DSV41_MHC_DEFER (branches.zig): the stream of mHC coefficient launches, joined by the next mHC call
    dfr: ?*Side = null,
    /// TF_DSV41_DRAFT_OVERLAP: the event a row window records after its last "w.taps" writer, for the next window only
    taps_mark: ?TapsMark = null,
    /// Runner.guard: the programs (programKey) whose every Triton call the AOT set resolved
    verified: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// Runner.guard: each Triton function's runtime parameters (built at the first window)
    params: ?Params = null,

    pub const TapsMark = struct { ev: cuda.Event, armed: bool = false, recorded: u64 = 0 };

    /// The L2 prefetcher on the device: the side stream, the fork / join events, whether a fork is open.
    pub const Side = struct {
        stream: cuda.Stream,
        fork: cuda.Event,
        join: cuda.Event,
        /// TF_DSV41_L2PF_ROWS (16): wider windows skip every site
        rows: u32 = 16,
        open: bool = false,
        launches: u64 = 0,
        skipped_rows: u64 = 0,
        /// TF_DSV41_PF_OVERLAP_SITE (branches only): `mark` m records marks[m - 1]; `wait` m makes main wait for it
        marks: [max_marks]?cuda.Event = @splat(null),
    };

    pub const max_marks = 8;

    pub const Glue = struct {
        ctx: *anyopaque,
        run: *const fn (ctx: *anyopaque, r: *Runner, c: *const calls.Call) anyerror!void,
    };

    pub fn deinit(r: *Runner) void {
        for (r.owned.items) |*b| b.free();
        r.owned.deinit(r.gpa);
        if (r.scratch) |*s| s.free();
        var it = r.addrs.keyIterator();
        while (it.next()) |k| r.gpa.free(k.*);
        r.addrs.deinit(r.gpa);
        r.verified.deinit(r.gpa);
        if (r.params) |*p| freeParams(r.gpa, p);
    }

    /// A persistent role whose memory another owner holds (the KV pool's tensors and tables): bind leaves it.
    pub fn external(r: *Runner, role: []const u8, addr: u64) !void {
        // Triton specializes pointers on 16-byte alignment: a misaligned base fails here by name, not later
        if (addr % 16 != 0) {
            std.log.scoped(.dsv41).err("role {s} bound at 0x{x}: not 16-byte aligned", .{ role, addr });
            return error.MisalignedRole;
        }
        if (r.addrs.getPtr(role)) |value| {
            value.* = addr;
            return;
        }
        const key = try r.gpa.dupe(u8, role);
        errdefer r.gpa.free(key);
        try r.addrs.put(r.gpa, key, addr);
    }

    pub const External = struct { role: []const u8, addr: u64 };
    pub fn externals(r: *Runner, bindings: []const External) !void {
        const keys = try r.gpa.alloc(?[]u8, bindings.len);
        defer r.gpa.free(keys);
        @memset(keys, null);
        defer for (keys) |key| if (key) |k| r.gpa.free(k);
        try r.addrs.ensureUnusedCapacity(r.gpa, @intCast(bindings.len));
        for (bindings, keys) |b, *key| {
            if (b.addr % 16 != 0) return error.MisalignedRole;
            if (!r.addrs.contains(b.role)) key.* = try r.gpa.dupe(u8, b.role);
        }
        for (bindings, keys) |b, *key| {
            if (r.addrs.getPtr(b.role)) |value| value.* = b.addr else {
                r.addrs.putAssumeCapacity(key.*.?, b.addr);
                key.* = null;
            }
        }
    }
    /// An external role's address gone (an on-demand role released): a later use fails as error.Unbound.
    pub fn forget(r: *Runner, role: []const u8) void {
        if (r.addrs.fetchRemove(role)) |kv| r.gpa.free(kv.key);
    }

    /// Allocates every role of `plan`: persistent roles one buffer each (but external ones), the rest one zeroed arena.
    pub fn bind(r: *Runner, plan: *const buffers.Plan) !void {
        var arena_bytes: u64 = 0;
        for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
            if (buffers.scopeOf(k) == .persistent and r.addrs.contains(k)) continue;
            if (buffers.scopeOf(k) == .persistent) {
                const b = try cuda.DeviceBuffer.alloc(r.d, v);
                try b.fill8(0, null);
                try r.owned.append(r.gpa, b);
                try r.addrs.put(r.gpa, try r.gpa.dupe(u8, k), b.ptr);
            } else arena_bytes += std.mem.alignForward(u64, v, 256);
        }
        const arena = try cuda.DeviceBuffer.alloc(r.d, arena_bytes);
        try arena.fill8(0, null);
        try r.owned.append(r.gpa, arena);
        var at: u64 = 0;
        for (plan.sizes.keys(), plan.sizes.values()) |k, v| {
            if (buffers.scopeOf(k) == .persistent) continue;
            try r.addrs.put(r.gpa, try r.gpa.dupe(u8, k), arena.ptr + at);
            at += std.mem.alignForward(u64, v, 256);
        }
        r.scratch = try cuda.DeviceBuffer.alloc(r.d, 64 << 20);
    }

    /// A role's device address (state.zig and the glue write through it).
    pub fn addressOf(r: *const Runner, role: []const u8) ?u64 {
        return r.addrs.get(role);
    }

    pub fn tensorAddr(r: *const Runner, t: calls.Tensor) !u64 {
        var numel: i64 = 1;
        for (t.shape) |n| numel *= n;
        if (numel == 0) return 0;
        const off: u64 = @intCast(t.offset);
        return switch (t.role) {
            .weight => |w| (r.weights.get(w) orelse return error.MissingWeight).ptr + off,
            .buf => |b| (r.addrs.get(b) orelse return error.Unbound) + off,
            .empty => error.EmptyRole,
        };
    }

    /// The first Triton call of `cs` the AOT set has no variant for (16-aligned tensors), null when every one has.
    pub fn missingTriton(r: *const Runner, cs: []const calls.Call) !?[]const u8 {
        return missingIn(r.gpa, r.triton, cs);
    }

    /// Before a window's first launch: each Triton call of `cs` has a variant, else MissingTritonVariant (AOT guard).
    pub fn guard(r: *Runner, cs: []const calls.Call) !void {
        if (!guardOn()) return;
        if (r.params == null) r.params = try paramsOf(r.gpa, r.triton);
        const k = programKey(cs, &r.params.?);
        if (r.verified.contains(k)) return;
        if (try missingIn(r.gpa, r.triton, cs)) |name| {
            std.log.err("aot guard: no Triton variant of {s} in this window of {d} calls: nothing issued, its request fails", .{ name, cs.len });
            return error.MissingTritonVariant;
        }
        try r.verified.put(r.gpa, k, {});
    }

    fn addrOf(ctx: *anyopaque, t: calls.Tensor) anyerror!u64 {
        const r: *const Runner = @ptrCast(@alignCast(ctx));
        return r.tensorAddr(t);
    }

    fn extArg(r: *const Runner, a: std.mem.Allocator, x: calls.Arg) !rp.Arg {
        return switch (x) {
            .i => |v| .{ .int = v },
            .f => |v| .{ .float = v },
            .b => |v| .{ .boolean = v },
            .none => .none,
            .list => |items| blk: {
                const out = try a.alloc(rp.Arg, items.len);
                for (items, out) |y, *z| z.* = try r.extArg(a, y);
                break :blk .{ .list = out };
            },
            .t, .opaque_table => |t| .{ .tensor = .{ .ptr = try r.tensorAddr(t), .dtype = std.meta.stringToEnum(rp.DType, @tagName(t.dt)).?, .shape = t.shape, .stride = t.stride } },
        };
    }

    /// Issues one call on the runner's stream; L2 prefetchers go to the side stream (skipped without TF_DSV41_L2PF).
    pub fn issue(r: *Runner, c: *const calls.Call) !void {
        if (r.br) |b| {
            if (c.join) try r.joinBranch();
            if (c.wait > 0) {
                // the main stream after the side's work up to mark `wait` only (the fork stays open)
                if (!b.open) return error.BranchNotForked;
                try r.stream.wait(b.marks[c.wait - 1] orelse return error.NoBranchMark);
            }
            if (c.fork) {
                try b.fork.record(r.stream);
                try b.stream.wait(b.fork);
                b.open = true;
                b.launches += 1;
            }
            if (c.side) {
                if (!b.open) return error.BranchNotForked;
                // the call and anything it runs (a glue step's launches) on the side stream
                const main = r.stream;
                r.stream = b.stream;
                defer r.stream = main;
                try r.issueOn(b.stream, c, r.scratch);
                if (c.mark > 0) try (b.marks[c.mark - 1] orelse return error.NoBranchMark).record(b.stream);
                return;
            }
        }
        if (r.dfr) |d| {
            if (c.defer_join) try r.joinDeferred();
            if (c.defer_side) {
                try d.fork.record(r.stream);
                try d.stream.wait(d.fork);
                d.open = true;
                d.launches += 1;
                const main = r.stream;
                r.stream = d.stream;
                defer r.stream = main;
                return r.issueOn(d.stream, c, r.scratch);
            }
        }
        if (std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) {
            const p = r.l2pf orelse return;
            if (r.rows > p.rows) {
                p.skipped_rows += 1;
                return;
            }
            try p.fork.record(r.stream);
            try p.stream.wait(p.fork);
            p.open = true;
            p.launches += 1;
            return r.issueOn(p.stream, c, r.scratch);
        }
        return r.issueOn(r.stream, c, r.scratch);
    }

    /// The main stream waits for the deferred coefficients' stream (mhc.await_coefs).
    fn joinDeferred(r: *Runner) !void {
        const d = r.dfr orelse return;
        if (!d.open) return;
        d.open = false;
        try d.join.record(d.stream);
        try r.stream.wait(d.join);
    }

    /// The main stream waits for the branches' side stream (its open fork, if any).
    fn joinBranch(r: *Runner) !void {
        const b = r.br orelse return;
        if (!b.open) return;
        b.open = false;
        try b.join.record(b.stream);
        try r.stream.wait(b.join);
    }

    /// The main stream waits for the side stream's prefetches, so a captured graph holds no open fork.
    pub fn join(r: *Runner) !void {
        try r.joinBranch();
        try r.joinDeferred();
        const p = r.l2pf orelse return;
        if (!p.open) return;
        p.open = false;
        try p.join.record(p.stream);
        try r.stream.wait(p.join);
    }

    fn dropForks(r: *Runner) void {
        if (r.l2pf) |p| p.open = false;
        if (r.br) |b| b.open = false;
        if (r.dfr) |d| d.open = false;
    }

    fn issueOn(r: *Runner, stream: cuda.Stream, c: *const calls.Call, scratch_buf: ?cuda.DeviceBuffer) !void {
        if (c.glue) {
            const g = r.glue orelse return error.NoGlue;
            return g.run(g.ctx, r, c);
        }
        var arena = std.heap.ArenaAllocator.init(r.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        if (c.triton) {
            var mine: std.ArrayList(cuda.aot.Spec) = .empty;
            for (r.triton.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", c.name)) try mine.append(a, v.spec);
            const sp = try tc.split(a, c, mine.items, .{ .ctx = @constCast(r), .of = addrOf });
            const g = c.grid;
            return r.triton.run(stream, c.name, .{ @intCast(g[0]), @intCast(g[1]), @intCast(g[2]) }, sp.args, sp.consts);
        }
        const dot = std.mem.lastIndexOfScalar(u8, c.name, '.') orelse return error.BadName;
        const args = try a.alloc(rp.Arg, c.args.len);
        for (c.args, args) |x, *y| y.* = try r.extArg(a, x.arg);
        var scratch: rp.Scratch = .{ .ptr = scratch_buf.?.ptr, .len = scratch_buf.?.len };
        if (!try rp.call(r.kernels, stream, c.name[0..dot], c.name[dot + 1 ..], args, &.{}, &scratch)) return error.NoBinding;
    }

    /// Issues a window's calls in order (TF_DSV41_SYNC_EACH=1: synced after every call, a failing call named).
    pub fn window(r: *Runner, cs: []const calls.Call) !void {
        try r.guard(cs);
        if (syncEach()) return r.windowChecked(cs);
        errdefer r.dropForks(); // a failed window (or capture) leaves no fork for the next one to join
        const mark = r.markAt(cs);
        for (cs, 0..) |*c, i| {
            try r.issue(c);
            if (i == mark) try r.recordMark(c);
        }
        try r.join();
    }

    /// Issues `cs` on `stream` instead of the runner's (a drafter program beside a row window); nothing joins.
    pub fn windowOn(r: *Runner, stream: cuda.Stream, cs: []const calls.Call) !void {
        try r.guard(cs);
        const main = r.stream;
        r.stream = stream;
        defer r.stream = main;
        for (cs) |*c| {
            if (c.side or c.fork or c.join or c.defer_side or c.defer_join or c.glue or std.mem.startsWith(u8, c.name, "tf_dsv41_l2p")) return error.NotASideProgram;
            try r.issueOn(stream, c, r.scratch);
        }
    }

    /// The index of the call after which an armed window records the taps mark; null: not armed, or no taps writer.
    fn markAt(r: *Runner, cs: []const calls.Call) ?usize {
        const m = r.taps_mark orelse return null;
        if (!m.armed) return null;
        return lastTaps(cs);
    }

    /// The mark after call `c` on its stream: an external record node in a capture, a plain record when eager.
    fn recordMark(r: *Runner, c: *const calls.Call) !void {
        const m = &r.taps_mark.?;
        const s = if (c.side and r.br != null) r.br.?.stream else if (c.defer_side and r.dfr != null) r.dfr.?.stream else r.stream;
        if (try cuda.graph.captureStatus(s) != .none) try m.ev.recordExternal(s) else try m.ev.record(s);
        m.recorded += 1;
    }

    fn windowChecked(r: *Runner, cs: []const calls.Call) !void {
        const mark = r.markAt(cs);
        for (cs, 0..) |*c, i| {
            r.issue(c) catch |e| {
                std.log.err("sync-each: call {d}/{d} {s} failed to issue ({t})", .{ i, cs.len, c.name, e });
                return e;
            };
            if (i == mark) try r.recordMark(c);
            r.stream.synchronize() catch |e| {
                std.log.err("sync-each: call {d}/{d} {s} faulted ({t})", .{ i, cs.len, c.name, e });
                return e;
            };
        }
        try r.join();
    }

    /// An eager decode window, each call timed when TF_DSV41_PROFILE is set (never inside a graph capture).
    pub fn timedWindow(r: *Runner, cs: []const calls.Call) !void {
        const p = r.prof orelse return r.window(cs);
        try r.guard(cs);
        if (syncEach()) return r.windowChecked(cs);
        p.begin();
        errdefer r.dropForks();
        const mark = r.markAt(cs);
        for (cs, 0..) |*c, i| {
            try p.before(r.stream, c);
            try r.issue(c);
            if (i == mark) try r.recordMark(c);
            try p.after(r.stream);
        }
        try r.join();
        try p.end(r.stream);
    }
};

/// The last call of `cs` with a "w.taps" tensor argument; null when none does.
pub fn lastTaps(cs: []const calls.Call) ?usize {
    var at: ?usize = null;
    for (cs, 0..) |c, i| for (c.args) |x| switch (x.arg) {
        .t => |t| if (t.role == .buf and std.mem.eql(u8, t.role.buf, "w.taps")) {
            at = i;
        },
        else => {},
    };
    return at;
}

var sync_each: ?bool = null;
/// The first Triton call of `cs` that `set` has no variant for, every role's base 16-aligned (Runner.missingTriton).
pub fn missingIn(gpa: std.mem.Allocator, set: *const cuda.aot.Set, cs: []const calls.Call) !?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for (cs) |*c| if (c.triton) {
        var mine: std.ArrayList(cuda.aot.Spec) = .empty;
        for (set.variants) |v| if (std.mem.eql(u8, v.spec.@"fn", c.name)) try mine.append(a, v.spec);
        const sp = tc.split(a, c, mine.items, .{ .ctx = undefined, .of = alignedOf }) catch return c.name;
        _ = set.find(c.name, sp.args, sp.consts) catch return c.name; // error.MissingTritonVariant
    };
    return null;
}

/// A tensor's address with its role's base 16-aligned (aot-needs' rule); an empty tensor or role is 0 (tensorAddr's).
fn alignedOf(_: *anyopaque, t: calls.Tensor) anyerror!u64 {
    return if (emptyTensor(t)) 0 else (1 << 20) + @as(u64, @intCast(t.offset));
}

fn emptyTensor(t: calls.Tensor) bool {
    var numel: i64 = 1;
    for (t.shape) |n| numel *= n;
    return numel == 0 or t.role == .empty;
}

/// Each Triton function's runtime parameter names over its variants (tc.split: other ints are constexprs).
pub const Params = std.StringHashMapUnmanaged([]const []const u8);

pub fn paramsOf(gpa: std.mem.Allocator, set: *const cuda.aot.Set) !Params {
    var specs: std.ArrayList(cuda.aot.Spec) = .empty;
    defer specs.deinit(gpa);
    for (set.variants) |v| try specs.append(gpa, v.spec);
    return paramsOfSpecs(gpa, specs.items);
}

pub fn paramsOfSpecs(gpa: std.mem.Allocator, specs: []const cuda.aot.Spec) !Params {
    var lists: std.StringArrayHashMapUnmanaged(std.ArrayList([]const u8)) = .empty;
    defer {
        for (lists.values()) |*l| l.deinit(gpa);
        lists.deinit(gpa);
    }
    for (specs) |sp| {
        const got = try lists.getOrPut(gpa, sp.@"fn");
        if (!got.found_existing) got.value_ptr.* = .empty;
        for (sp.params) |q| if (!named(got.value_ptr.items, q.name)) try got.value_ptr.append(gpa, q.name);
    }
    var p: Params = .empty;
    errdefer freeParams(gpa, &p);
    for (lists.keys(), lists.values()) |k, *l| try p.put(gpa, k, try l.toOwnedSlice(gpa));
    return p;
}

pub fn freeParams(gpa: std.mem.Allocator, p: *Params) void {
    var it = p.valueIterator();
    while (it.next()) |v| gpa.free(v.*);
    p.deinit(gpa);
}

/// A program's Triton lookups hashed as `find` tells them apart (launch ints by class, tensors by dtype and alignment).
pub fn programKey(cs: []const calls.Call, params: *const Params) u64 {
    var h = std.hash.Wyhash.init(0x5d41);
    for (cs) |*c| if (c.triton) {
        h.update(c.name);
        const ps = params.get(c.name) orelse &.{};
        for (c.args) |x| {
            h.update(x.name);
            if (x.arg == .i and named(ps, x.name)) {
                const v = x.arg.i;
                h.update(&.{ @intFromBool(v == 1), @intFromBool(@mod(v, 16) == 0), @intFromBool(std.math.cast(i32, v) != null) });
            } else hashArg(&h, x.arg);
        }
        h.update("|");
    };
    return h.final();
}

fn named(ps: []const []const u8, name: []const u8) bool {
    for (ps) |p| if (std.mem.eql(u8, p, name)) return true;
    return false;
}

fn hashArg(h: *std.hash.Wyhash, x: calls.Arg) void {
    h.update(&.{@backingInt(std.meta.activeTag(x))});
    switch (x) {
        .t, .opaque_table => |t| {
            h.update(@tagName(t.dt));
            h.update(&.{ @intFromBool(emptyTensor(t)), @intFromBool(@mod(t.offset, 16) == 0) });
        },
        .i => |v| h.update(std.mem.asBytes(&v)),
        .f => |v| h.update(std.mem.asBytes(&v)),
        .b => |v| h.update(&.{@intFromBool(v)}),
        .none => {},
        .list => |items| for (items) |y| hashArg(h, y),
    }
}

var guard_on: ?bool = null;

fn guardOn() bool {
    if (guard_on) |v| return v;
    const v = if (std.c.getenv("TF_DSV41_AOT_GUARD")) |x| !std.mem.eql(u8, std.mem.span(x), "0") else true;
    guard_on = v;
    return v;
}

fn syncEach() bool {
    if (sync_each) |v| return v;
    const v = if (std.c.getenv("TF_DSV41_SYNC_EACH")) |x| std.mem.eql(u8, std.mem.span(x), "1") else false;
    sync_each = v;
    return v;
}
