//! A step recorded as graphs cut at its collectives: the collectives run between the graphs' replays, never inside one.
const std = @import("std");
const abi = @import("abi.zig");
const graph = @import("graph.zig");
const collective = @import("collective.zig");
const Stream = @import("stream.zig").Stream;

/// One collective of a step, with the buffers it was recorded with (the graphs' addresses stay fixed too).
pub const Call = struct {
    what: enum { all_reduce, all_gather, broadcast },
    send: u64,
    recv: u64,
    n: usize,
    dtype: collective.Dtype,
    root: usize = 0,

    fn issue(c: Call, comm: collective.Comm, stream: abi.Stream) collective.Error!void {
        switch (c.what) {
            .all_reduce => try comm.allReduce(c.send, c.recv, c.n, c.dtype, stream),
            .all_gather => try comm.allGather(c.send, c.recv, c.n, c.dtype, stream),
            .broadcast => try comm.broadcast(c.send, c.recv, c.n, c.dtype, c.root, stream),
        }
    }
};

pub const Piece = union(enum) { graph: graph.Exec, call: Call };

/// A recorded step: its graphs and collectives in order, replayed on the stream it was recorded on.
pub const Split = struct {
    gpa: std.mem.Allocator,
    pieces: []Piece,
    comm: ?collective.Comm,

    pub fn deinit(s: *Split) void {
        for (s.pieces) |*p| switch (p.*) {
            .graph => |*x| x.deinit(),
            .call => {},
        };
        s.gpa.free(s.pieces);
        s.* = undefined;
    }

    pub fn launchOn(s: Split, stream: Stream) !void {
        for (s.pieces) |p| switch (p) {
            .graph => |x| try x.launchOn(stream),
            .call => |c| try c.issue(s.comm orelse return error.CollectiveFailed, stream.handle),
        };
    }
};

/// Records a step from `begin` to `finish`: launches go into graphs, cut wherever a collective goes through `comm()`.
pub const Recorder = struct {
    gpa: std.mem.Allocator,
    stream: Stream,
    /// The step's communicator (null: one rank, no collectives).
    inner: ?collective.Comm,
    pieces: std.ArrayList(Piece) = .empty,
    /// A cut failed: the step is not kept and its launches must run eagerly.
    failed: bool = false,
    capturing: bool = false,

    const vtable: collective.Comm.VTable = .{ .all_reduce = allReduce, .all_gather = allGather, .broadcast = broadcast, .close = close };

    pub fn begin(r: *Recorder) !void {
        try graph.beginCapture(r.stream, .thread_local);
        r.capturing = true;
    }

    /// The communicator the step uses while it records: each call ends a graph and starts the next.
    pub fn comm(r: *Recorder) ?collective.Comm {
        const c = r.inner orelse return null;
        return .{ .ptr = r, .vtable = &vtable, .rank = c.rank, .world = c.world };
    }

    /// The recorded step, or null when a cut failed or its pieces could not be kept; the recorder is empty after.
    pub fn finish(r: *Recorder) !?Split {
        if (!r.failed) r.closePiece() catch {
            r.failed = true;
        };
        if (r.capturing) {
            if (graph.endCapture(r.stream)) |g| {
                var x = g;
                x.deinit();
            } else |_| {}
            r.capturing = false;
        }
        if (r.failed) return r.discard();
        const pieces = r.pieces.toOwnedSlice(r.gpa) catch return r.discard();
        return .{ .gpa = r.gpa, .pieces = pieces, .comm = r.inner };
    }

    /// Frees what was recorded: the step runs eagerly instead.
    fn discard(r: *Recorder) ?Split {
        for (r.pieces.items) |*p| if (p.* == .graph) p.graph.deinit();
        r.pieces.deinit(r.gpa);
        r.pieces = .empty;
        return null;
    }

    fn closePiece(r: *Recorder) !void {
        var g = try graph.endCapture(r.stream);
        r.capturing = false;
        defer g.deinit();
        var x = try g.instantiate();
        errdefer x.deinit();
        try x.upload(r.stream);
        try r.pieces.append(r.gpa, .{ .graph = x });
    }

    fn cut(ptr: *anyopaque, call: Call) collective.Error!void {
        const r: *Recorder = @ptrCast(@alignCast(ptr));
        if (r.failed) return error.CollectiveFailed;
        r.closePiece() catch {
            r.failed = true;
            return error.CollectiveFailed;
        };
        r.pieces.append(r.gpa, .{ .call = call }) catch {
            r.failed = true;
            return error.CollectiveFailed;
        };
        r.begin() catch {
            r.failed = true;
            return error.CollectiveFailed;
        };
    }

    fn allReduce(ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: collective.Dtype, _: abi.Stream) collective.Error!void {
        return cut(ptr, .{ .what = .all_reduce, .send = send, .recv = recv, .n = n, .dtype = dtype });
    }

    fn allGather(ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: collective.Dtype, _: abi.Stream) collective.Error!void {
        return cut(ptr, .{ .what = .all_gather, .send = send, .recv = recv, .n = n, .dtype = dtype });
    }

    fn broadcast(ptr: *anyopaque, send: u64, recv: u64, n: usize, dtype: collective.Dtype, root: usize, _: abi.Stream) collective.Error!void {
        return cut(ptr, .{ .what = .broadcast, .send = send, .recv = recv, .n = n, .dtype = dtype, .root = root });
    }

    fn close(_: *anyopaque) void {}
};
