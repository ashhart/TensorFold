//! A stand-in CUDA driver and rank link for the EXL3 graph cache's host tests.

const std = @import("std");
const cuda = @import("cuda");
const collective = @import("../../cuda_collective.zig");
const wire = @import("../../cuda_rank_wire.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const Pending = gpending.Pending;

pub const GraphTestDriver = struct {
    pub var exec_drops: usize = 0;
    pub var graph_drops: usize = 0;
    pub var launches: usize = 0;
    pub var exec_updates: usize = 0;

    pub fn dropExec(_: cuda.abi.GraphExec) callconv(.c) c_int {
        exec_drops += 1;
        return 0;
    }

    pub fn dropGraph(_: cuda.abi.Graph) callconv(.c) c_int {
        graph_drops += 1;
        return 0;
    }

    pub fn instantiate(out: *cuda.abi.GraphExec, _: cuda.abi.Graph, _: u64) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }

    pub fn getContext(out: *?*anyopaque) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }

    pub fn noContext(_: *?*anyopaque) callconv(.c) c_int {
        return 1;
    }

    pub fn setContext(_: ?*anyopaque) callconv(.c) c_int {
        return 0;
    }

    pub fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuGraphExecDestroy = dropExec;
        d.api.cuGraphDestroy = dropGraph;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuMemGetInfo_v2 = memoryInfo;
        d.api.cuCtxSynchronize = synchronize;
        d.api.cuStreamSynchronize = synchronizeStream;
        d.api.cuStreamBeginCapture_v2 = beginCapture;
        d.api.cuStreamEndCapture = endCapture;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphExecUpdate_v2 = update;
        return d;
    }

    pub fn memoryInfo(free: *usize, total: *usize) callconv(.c) c_int {
        free.* = 100;
        total.* = 100;
        return 0;
    }

    pub fn synchronize() callconv(.c) c_int {
        return 0;
    }

    pub fn synchronizeStream(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }

    pub fn beginCapture(_: cuda.abi.Stream, _: cuda.abi.CaptureMode) callconv(.c) c_int {
        return 0;
    }

    pub fn endCapture(_: cuda.abi.Stream, out: *cuda.abi.Graph) callconv(.c) c_int {
        out.* = @ptrFromInt(2);
        return 0;
    }

    pub fn upload(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }

    pub fn update(_: cuda.abi.GraphExec, _: cuda.abi.Graph, info: *cuda.abi.ExecUpdateResultInfo) callconv(.c) c_int {
        exec_updates += 1;
        info.result = .success;
        return 0;
    }

    pub fn launch(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
        launches += 1;
        return 0;
    }

    pub fn abort(_: cuda.nccl.Comm) callconv(.c) cuda.nccl.Result {
        return 0;
    }

    pub fn control(link: wire.Link) collective.Collective {
        var c: collective.Collective = undefined;
        c.comm = @ptrFromInt(1);
        c.api.ncclCommAbort = abort;
        c.link = link;
        c.failed = false;
        c.timeout_ms = 1000;
        c.graph_step = 0;
        return c;
    }

    pub fn finishPending(pd: *Pending, driver_: *cuda.Driver) void {
        pd.exec = .{ .d = driver_, .handle = @ptrFromInt(1) };
        pd.done.store(true, .release);
    }
};
