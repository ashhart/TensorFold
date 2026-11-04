const std = @import("std");
const cuda = @import("cuda");
const mtp = @import("../../cuda_mtp.zig");
const graph = @import("../graph/cuda_exl3_graph.zig");
const prefill = @import("../graph/cuda_exl3_graph_prefill.zig");

pub fn Cases(comptime C: anytype) type {
    return struct {
        test "exl3: synchronous chain launch failure retains ownership until retirement" {
            const Mock = struct {
                var drops: usize = 0;

                fn instantiate(out: *cuda.abi.GraphExec, _: cuda.abi.Graph, _: u64) callconv(.c) c_int {
                    out.* = @ptrFromInt(1);
                    return 0;
                }

                fn upload(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
                    return 0;
                }

                fn synchronize(_: cuda.abi.Stream) callconv(.c) c_int {
                    return 0;
                }

                fn launch(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
                    return cuda.abi.error_not_ready;
                }

                fn destroy(_: cuda.abi.GraphExec) callconv(.c) c_int {
                    drops += 1;
                    return 0;
                }

                fn installAndLaunch(target: *?C.Entry, template: cuda.graph.Graph, entry: C.Entry) !void {
                    try C.installSync(target, template, entry);
                    try target.*.?.exec.launchOn(entry.stream.?);
                }
            };
            Mock.drops = 0;
            var driver: cuda.Driver = undefined;
            driver.api.cuGraphInstantiateWithFlags = Mock.instantiate;
            driver.api.cuGraphUpload = Mock.upload;
            driver.api.cuStreamSynchronize = Mock.synchronize;
            driver.api.cuGraphLaunch = Mock.launch;
            driver.api.cuGraphExecDestroy = Mock.destroy;
            const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(1) };
            try graph.memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
            const ticket = try graph.memory.admit(.chain, 0, 100);
            var ticket_owned = true;
            defer if (ticket_owned) graph.memory.release(ticket) catch unreachable;
            var entry: ?C.Entry = null;
            defer if (entry) |*cached| C.dropEntry(cached);
            const result = Mock.installAndLaunch(&entry, .{ .d = &driver, .handle = @ptrFromInt(1) }, .{ .key = undefined, .used = 1, .admission = ticket, .stream = stream });
            if (entry != null) ticket_owned = false;
            try std.testing.expectError(error.NotReady, result);
            try std.testing.expect(entry != null);
            try std.testing.expectEqual(@as(usize, 0), Mock.drops);
            try std.testing.expectEqual(@as(u64, 30), graph.memory.charged());
            C.dropEntry(&entry.?);
            entry = null;
            try std.testing.expectEqual(@as(usize, 1), Mock.drops);
            try std.testing.expectEqual(@as(u64, 0), graph.memory.charged());
            try std.testing.expectError(error.StaleGraphTicket, graph.memory.release(ticket));
        }

        test "exl3: draft destruction retires every reader before freeing buffers, preserving other owners" {
            const Mock = struct {
                var live: [13]bool = undefined;
                var freed_with_readers: bool = false;
                var frees: usize = 0;
                var allocations: usize = 0;

                fn allocate(out: *cuda.abi.DevicePtr, _: usize) callconv(.c) c_int {
                    allocations += 1;
                    out.* = 256 * allocations;
                    return 0;
                }

                fn free(_: cuda.abi.DevicePtr) callconv(.c) c_int {
                    for ([_]usize{ 1, 2, 5, 6, 7, 8, 12 }) |id| freed_with_readers = freed_with_readers or live[id];
                    frees += 1;
                    return 0;
                }

                fn synchronize(_: cuda.abi.Stream) callconv(.c) c_int {
                    return 0;
                }

                fn destroy(handle: cuda.abi.GraphExec) callconv(.c) c_int {
                    live[@intFromPtr(handle)] = false;
                    return 0;
                }
            };
            Mock.live = @splat(true);
            Mock.freed_with_readers = false;
            Mock.frees = 0;
            Mock.allocations = 0;
            var driver: cuda.Driver = undefined;
            driver.api.cuMemAlloc_v2 = Mock.allocate;
            driver.api.cuMemFree_v2 = Mock.free;
            driver.api.cuStreamSynchronize = Mock.synchronize;
            driver.api.cuGraphExecDestroy = Mock.destroy;
            var foreign_driver = driver;
            const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(1) };
            const foreign_stream: cuda.Stream = .{ .d = &foreign_driver, .handle = @ptrFromInt(2) };
            var buffers_owned = true;
            var kc = try cuda.DeviceBuffer.alloc(&driver, 4);
            defer if (buffers_owned) kc.free();
            var scores = try cuda.DeviceBuffer.alloc(&driver, 4);
            defer if (buffers_owned) scores.free();
            var logits = try cuda.DeviceBuffer.alloc(&driver, 4);
            defer if (buffers_owned) logits.free();
            var streams = try cuda.DeviceBuffer.alloc(&driver, 4);
            defer if (buffers_owned) streams.free();
            var spare = try cuda.DeviceBuffer.alloc(&driver, 4);
            defer if (buffers_owned) spare.free();
            const borrowed: cuda.DeviceBuffer = .{ .d = &driver, .ptr = 0, .len = 0, .borrowed = true };
            var window: mtp.Draft = .{
                .stream = stream,
                .layout = .of(.full),
                .gpa = std.testing.allocator,
                .kc = kc,
                .vc = borrowed,
                .ikc = borrowed,
                .pooled = borrowed,
                .cap = 1,
                .streams = streams,
                .rows_cap = mtp.max_rows,
                .out = borrowed,
                .logits = logits,
                .dlogits = scores,
                .best = borrowed,
                .saved = borrowed,
                .spare = @splat(borrowed),
                .host = &.{},
            };
            window.spare[0] = spare;
            for (graph.cache.entries) |entry| try std.testing.expect(entry == null);
            for (prefill.pentries) |entry| try std.testing.expect(entry == null);
            defer for (&graph.cache.entries) |*slot| if (slot.*) |*entry| {
                graph.cache.drop(entry);
                slot.* = null;
            };
            defer for (&prefill.pentries) |*slot| if (slot.*) |*entry| {
                prefill.dropPrefill(entry);
                slot.* = null;
            };
            for (0..5) |i| {
                var key = std.mem.zeroes(graph.Key);
                key.kind = if (i == 1) 1 else 2;
                key.output = if (i == 1 or i == 2 or i == 4) 2048 else window.logits.ptr;
                if (i == 1) key.seq[0][4] = window.spare[0].ptr;
                if (i == 4) key.keep_state = window.spare[0].ptr;
                const owner = if (i == 3) foreign_stream else stream;
                const id: usize = if (i < 2) 5 + i else if (i < 4) 7 + i else 12;
                graph.cache.entries[i] = .{ .key = key, .used = i, .exec = .{ .d = owner.d, .handle = @ptrFromInt(id) }, .stream = owner };
            }
            for (0..3) |i| {
                var key = std.mem.zeroes(prefill.PKey);
                if (i == 0) key.seq[2] = window.kc.ptr else key.seq[4] = window.spare[0].ptr;
                const owner = if (i == 2) foreign_stream else stream;
                const id: usize = if (i < 2) 7 + i else 11;
                prefill.pentries[i] = .{ .key = key, .used = i, .exec = .{ .d = owner.d, .handle = @ptrFromInt(id) }, .stream = owner };
            }
            for (C.entries) |entry| try std.testing.expect(entry == null);
            defer for (C.entries) |*slot| if (slot.*) |*entry| {
                C.dropEntry(entry);
                slot.* = null;
            };
            for (0..4) |i| {
                var key = std.mem.zeroes(C.Key);
                key.kc = if (i == 2) 4096 else kc.ptr;
                key.dlogits = if (i == 2) 4096 else window.dlogits.ptr;
                key.picks = i + 1;
                const owner = if (i == 3) foreign_stream else stream;
                C.entries[i] = .{ .key = key, .used = i, .exec = .{ .d = owner.d, .handle = @ptrFromInt(i + 1) }, .stream = owner };
            }
            window.deinit();
            buffers_owned = false;
            try std.testing.expect(!Mock.freed_with_readers);
            try std.testing.expectEqual(@as(usize, 5), Mock.frees);
            try std.testing.expect(!Mock.live[1] and !Mock.live[2]);
            try std.testing.expect(Mock.live[3] and Mock.live[4]);
            for ([_]usize{ 5, 6, 7, 8, 12 }) |id| try std.testing.expect(!Mock.live[id]);
            for ([_]usize{ 9, 10, 11 }) |id| try std.testing.expect(Mock.live[id]);
        }
    };
}
