//! tf-kv-split-test's kvsplit kernels (zig/kernels/cuda/deepseek_v41/kvsplit.cu) behind split.Kernels.

const cuda = @import("cuda");
const tp = @import("tp");
const split = @import("split.zig");

pub const DeviceKernels = struct {
    d: *const cuda.Driver,
    dense_fn: cuda.Function,
    gather_fn: cuda.Function,
    pack_fn: cuda.Function,
    /// dedup_mark, _flags, _pack, _tok (null: a fatbin without them)
    dedup_fns: ?[4]cuda.Function = null,
    /// hist_gather, hist_tok
    hist_fns: ?[2]cuda.Function = null,

    fn grid(words: u64) u32 {
        return @intCast(@max(1, @min((words + 255) / 256, 4096)));
    }

    pub fn kernels(k: *DeviceKernels) split.Kernels {
        return .{ .ptr = k, .vtable = &.{ .dense = dense, .gather = gather, .pack = pack, .dedup = dedup, .history = history } };
    }

    fn history(ptr: *anyopaque, a: split.HistArgs, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const fs = k.hist_fns orelse return error.Unsupported;
        for (fs, [2]u64{ @as(u64, a.m) * (a.row_bytes / 8), @as(u64, a.rows) * a.k }) |f, w| {
            if (w == 0) continue;
            var args: cuda.Args = .{};
            args.add(a);
            try cuda.launch.launch(f, .{ .grid = .{ .x = grid(w) }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
        }
    }

    fn dedup(ptr: *anyopaque, a: split.DedupArgs, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const fs = k.dedup_fns orelse return error.Unsupported;
        const n = @as(u64, a.p.rows) * a.p.k;
        if (n == 0) return;
        const b: u32 = @intCast((n + 255) / 256);
        for (fs, [4]u32{ grid(n), b, b, grid(n) }) |f, g| {
            var args: cuda.Args = .{};
            args.add(a);
            try cuda.launch.launch(f, .{ .grid = .{ .x = g }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
        }
    }

    fn dense(ptr: *anyopaque, a: split.DenseArgs, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const words = @as(u64, a.rows) * a.k * (a.row_bytes / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(k.dense_fn, .{ .grid = .{ .x = grid(words) }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
    }

    /// pack_kernel: one block of 256 threads a 256 entries (the kernel's kPackThreads), any world <= 8.
    fn pack(ptr: *anyopaque, a: split.PackArgs, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const n = @as(u64, a.rows) * a.k;
        if (n == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try cuda.launch.launch(k.pack_fn, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
    }

    fn gather(ptr: *anyopaque, base: u64, rb: u32, phys: u64, n: u32, send: u64, stream: tp.collective.Stream) anyerror!void {
        const k: *DeviceKernels = @ptrCast(@alignCast(ptr));
        const words = @as(u64, n) * (rb / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(base);
        args.add(rb);
        args.add(phys);
        args.add(n);
        args.add(send);
        try cuda.launch.launch(k.gather_fn, .{ .grid = .{ .x = grid(words) }, .block = .{ .x = 256 } }, .{ .d = k.d, .handle = stream }, &args);
    }
};
