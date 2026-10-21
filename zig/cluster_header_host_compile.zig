//! Compile the cluster adapter through the Metal engine's actual core namespace without GPU imports or execution.
const std = @import("std");
const core = @import("core");
const ck = @import("cluster_checkpoint");

export fn clusterHeaderContract(bytes: [*]const u8, len: usize) usize {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const tensors = ck.parseHeader(arena.allocator(), 0, bytes[0..len]) catch return 0;
    const header = core.safetensors.parseHeader(arena.allocator(), "{}", 0) catch return 0;
    return tensors.len + header.count();
}
