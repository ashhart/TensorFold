//! Production GPU seams and host relay over a copying endpoint, including slot reuse and terminal wake.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const g = contract.gates;
const c = contract.spec;
const tp = @import("tensorfold").flashnext_replay.Tp2;
const Handoff = tp.handoff_module.Handoff;
const rig = @import("rig.zig");
const suite = @import("metal_flashnext_tp.zig");
pub const name = "metal flashnext tp loopback handoff and reused receive slot";
const poison: u32 = 0xd3d3d3d3;
const old: u32 = 0x11111111;

fn noSums(_: c.Shape) usize {
    return 0;
}

const Endpoint = struct {
    win: []u8,
    expected: [2][]const u8,
    outputs: [2][]const u8,
    pre: []const u32,
    calls: usize = 0,
    ordered: bool = true,

    pub fn writeSignalFromUntil(e: *Endpoint, _: u32, local: usize, remote: usize, bytes: usize, flag: usize, value: u64, _: u64) !void {
        if (e.calls >= 2) return error.ExtraRelay;
        if (!std.mem.eql(u8, e.expected[e.calls], e.win[local..][0..bytes])) e.ordered = false;
        if (!std.mem.allEqual(u32, e.pre, old)) e.ordered = false;
        if (!std.mem.allEqual(u8, e.outputs[e.calls], 0xd3)) e.ordered = false;
        @memcpy(e.win[remote..][0..bytes], e.win[local..][0..bytes]);
        @atomicStore(u64, @as(*u64, @ptrCast(@alignCast(e.win.ptr + flag))), value, .release);
        e.calls += 1;
    }

    pub fn flushPeerUntil(_: *Endpoint, _: u32, _: u64) !void {}
    pub fn setDeadline(_: *Endpoint, _: u64) void {}
};

const Loop = struct {
    handoff: *Handoff,
    stop: *std.atomic.Value(bool),
    failed: *std.atomic.Value(bool),
    queued: std.atomic.Value(u64) = .init(2),
    submitted: std.atomic.Value(u64) = .init(2),
    jobs: [4]tp.QueueJob = @splat(.{}),
    ep: *Endpoint,
    rd: *Endpoint,
    rank: u32 = 0,
    peer: u32 = 1,
    trace: bool = false,
    local: bool = false,
    stats: bool = false,

    pub fn recordStats(_: *Loop, _: u64, _: tp.QueueJob, _: u64) void {}

    pub fn word64(l: *Loop, off: usize) *u64 {
        return @ptrCast(@alignCast(l.ep.win.ptr + off));
    }

    pub fn sendNow(_: *Loop, _: anytype, _: usize, _: u64) !void {
        return error.UnexpectedSendJob;
    }
};

pub const Probe = struct {
    const vt = g.hooks(Probe);
    rig: rig.Rig,
    pipe: mtl.Pipeline,
    words: usize = 0,
    source: [2]rig.Ref = undefined,
    out: [2]rig.Ref = undefined,
    terminal: rig.Ref = undefined,
    pre: rig.Ref = undefined,
    window: rig.Ref = undefined,
    order_ok: bool = false,
    terminal_ok: bool = false,

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = .{
            .name = name,
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            .nan = .{ .stated = "the relay observes GPU production before arrival; reused and terminal slots never expose old bytes" },
            .served = &.{ .{ .rows = 1, .n = 2560, .k = 1 }, .{ .rows = 16, .n = 2560, .k = 1 } },
        }, .ctx = p, .vt = &vt };
    }

    pub fn count(p: *const Probe) usize {
        return 3 * p.words;
    }

    fn data(p: *const Probe, ref: rig.Ref) []u8 {
        return p.rig.bytes(ref, p.words * 4);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16) return false;
        p.rig.reset();
        p.words = shape.rows * 2560;
        const values = try p.rig.a.alloc(u32, p.words);
        defer p.rig.a.free(values);
        var prng = std.Random.DefaultPrng.init(seed);
        for (&p.source) |*src| {
            for (values) |*v| v.* = prng.random().int(u32);
            src.* = try p.rig.input(std.mem.sliceAsBytes(values));
        }
        for (&p.out) |*out| out.* = try p.rig.output(p.words * 4);
        p.terminal = try p.rig.output(p.words * 4);
        p.pre = try p.rig.scratch(p.words * 4);
        p.window = try p.rig.scratch(2 * p.words * 4 + 256);
        return true;
    }

    pub fn reference(_: *Probe, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    fn copy(p: *Probe, e: mtl.ComputeEncoder, h: *Handoff, epoch: u64, src: rig.Ref, dst: rig.Ref) void {
        e.setPipeline(p.pipe);
        h.bind(e, epoch, 0);
        e.setBuffer(src.buf, src.off, 2);
        e.setBuffer(dst.buf, dst.off, 3);
        e.setValue(@as(u32, @intCast(p.words)), 4);
        e.dispatchThreads(mtl.Size.of(p.words, 1, 1), mtl.Size.of(256, 1, 1));
    }

    pub fn run(p: *Probe) !void {
        for (p.out) |ref| @memset(p.data(ref), 0xd3);
        @memset(p.data(p.terminal), 0xd3);
        const send = p.window;
        const recv: rig.Ref = .{ .buf = send.buf, .off = send.off + p.words * 4 + 64 };
        const flag = recv.off + p.words * 4 + 64;
        @memset(@as([]u32, @ptrCast(@alignCast(p.data(recv)))), old);
        @memset(p.data(p.pre), 0);
        var failed: std.atomic.Value(bool) = .init(false);
        var stop: std.atomic.Value(bool) = .init(false);
        var handoff = try Handoff.init(p.rig.device, &failed, &stop, 0);
        defer handoff.deinit();
        var endpoint: Endpoint = .{
            .win = send.buf.contents()[0 .. flag + 8],
            .expected = .{ p.data(p.source[0]), p.data(p.source[1]) },
            .outputs = .{ p.data(p.out[0]), p.data(p.out[1]) },
            .pre = @ptrCast(@alignCast(p.data(p.pre))),
        };
        var relay: Loop = .{ .handoff = &handoff, .failed = &failed, .stop = &stop, .ep = &endpoint, .rd = &endpoint };
        for (1..3) |seq| relay.jobs[seq] = .{
            .kind = .reduce,
            .local = send.off,
            .remote = recv.off,
            .bytes = p.words * 4,
            .flag = flag,
            .value = seq,
        };
        const thread = try std.Thread.spawn(.{}, tp.relay_module.service, .{&relay});
        var joined = false;
        defer if (!joined) {
            stop.store(true, .release);
            thread.join();
        };
        const pass = p.rig.begin();
        var e = pass.e;
        p.copy(e, &handoff, 0, recv, p.pre);
        p.copy(e, &handoff, 0, p.source[0], send);
        handoff.seam(&e, 1);
        p.copy(e, &handoff, 1, recv, p.out[0]);
        p.copy(e, &handoff, 1, p.source[1], send);
        handoff.seam(&e, 2);
        p.copy(e, &handoff, 2, recv, p.out[1]);
        try p.rig.finish(.{ .pool = pass.pool, .cb = pass.cb, .e = e });
        p.order_ok = endpoint.ordered and endpoint.calls == 2 and !failed.load(.acquire);
        stop.store(true, .release);
        thread.join();
        joined = true;
        // The failure sentinel releases the real event wait, without publishing the matching full-width epoch.
        var terminal_failed: std.atomic.Value(bool) = .init(false);
        var terminal_stop: std.atomic.Value(bool) = .init(false);
        var terminal_handoff = try Handoff.init(p.rig.device, &terminal_failed, &terminal_stop, 0);
        defer terminal_handoff.deinit();
        terminal_handoff.ready.slice(u64, 1)[0] = suite.stale;
        const Terminal = struct {
            fn release(h: *Handoff) void {
                _ = h.posted.wait(suite.epoch, 1000);
                h.cancel();
            }
        };
        const terminal_thread = try std.Thread.spawn(.{}, Terminal.release, .{&terminal_handoff});
        defer terminal_thread.join();
        const last = p.rig.begin();
        e = last.e;
        terminal_handoff.seam(&e, suite.epoch);
        p.copy(e, &terminal_handoff, suite.epoch, recv, p.terminal);
        try p.rig.finish(.{ .pool = last.pool, .cb = last.cb, .e = e });
        p.terminal_ok = terminal_handoff.posted.value() == suite.epoch and terminal_handoff.ready.slice(u64, 1)[0] == suite.stale;
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        if (!p.order_ok) return "the relay observed a payload before production, or a consumer ran before arrival";
        for (p.out, p.source) |out, source| if (!std.mem.eql(u8, p.data(out), p.data(source)))
            return "a receive reused the previous exchange's bytes";
        if (!p.terminal_ok or !std.mem.allEqual(u8, p.data(p.terminal), 0xd3))
            return "terminal wake consumed a stale payload with the same low epoch bits";
        return null;
    }
};
