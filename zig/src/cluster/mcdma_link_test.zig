//! A model's MCDMA ranks and each node's half of every link from the file alone; refusals by name before opening.
const std = @import("std");
const fabric = @import("fabric");
const config = @import("config.zig");
const mcdma_link = @import("mcdma_link.zig");
const example = @import("config_test.zig").example;

const mib = 1 << 20;

fn parse(a: std.mem.Allocator, text: []const u8) !config.Config {
    var probs: config.Problems = .{ .a = a };
    return config.parse(a, text, &probs);
}

test "a disaggregated model's ranks are its prefill nodes then its decode nodes, whatever the file's node order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    var m = c.model("qwen-27b").?;
    const names = try mcdma_link.ranks(a, m);
    try std.testing.expectEqual(@as(usize, 2), names.len);
    try std.testing.expectEqualStrings("node-e", names[0]);
    try std.testing.expectEqualStrings("node-a", names[1]);
    m.prefill = &.{ "p1", "p2" };
    m.decode = &.{"d1"};
    const three = try mcdma_link.ranks(a, m);
    try std.testing.expectEqualStrings("p2", three[1]);
    try std.testing.expectEqualStrings("d1", three[2]);
    try std.testing.expectEqual(@as(?u32, 2), mcdma_link.rankOf(three, "d1"));
    // a converged model keeps its node list
    const k3 = c.model("kimi-k3").?;
    try std.testing.expectEqualSlices([]const u8, k3.nodes, try mcdma_link.ranks(a, k3));
    // reversing the file's node array changes no rank
    var swapped = c;
    swapped.nodes = try a.dupe(config.Node, c.nodes);
    std.mem.reverse(config.Node, swapped.nodes);
    var why: []const u8 = "";
    const s = try mcdma_link.spec(a, swapped, c.model("qwen-27b").?, "node-a", .{ .mac = true }, &why);
    try std.testing.expectEqual(@as(u32, 1), s.rank);
}

test "each end derives its half of the one declared link: same name, kind and path MTU, each end's peer port the other's port" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    const m = c.model("qwen-27b").?;
    var why: []const u8 = "";
    const mac = try mcdma_link.spec(a, c, m, "node-a", .{ .mac = true }, &why);
    const gb10 = try mcdma_link.spec(a, c, m, "node-e", .{ .mac = false }, &why);
    try std.testing.expectEqual(@as(u32, 1), mac.rank);
    try std.testing.expectEqual(@as(u32, 0), gb10.rank);
    try std.testing.expectEqualStrings("/opt/mcdma/libmcdma-fabric.dylib", mac.library);
    try std.testing.expectEqualStrings("/opt/mcdma/libmcdma-fabric.so", gb10.library);
    const x = mac.links[0];
    const y = gb10.links[0];
    try std.testing.expectEqual(@as(u32, 0), x.peer);
    try std.testing.expectEqual(@as(u32, 1), y.peer);
    try std.testing.expectEqualStrings("rdma_mcrdma0", x.device);
    try std.testing.expectEqualStrings("en7/192.0.2.31", x.via);
    try std.testing.expectEqualStrings("rocep1s0f1", y.device);
    try std.testing.expectEqual(@as(u16, 7490), x.port);
    try std.testing.expectEqual(@as(u16, 7491), x.peer_port);
    try std.testing.expectEqual(x.port, y.peer_port);
    try std.testing.expectEqual(y.port, x.peer_port);
    try std.testing.expectEqual(@as(c_int, 0), x.gid);
    try std.testing.expectEqual(@as(c_int, 1), y.gid);
    try std.testing.expectEqualStrings(x.name, y.name);
    try std.testing.expectEqual(@as(c_int, 1024), x.mtu);
    try std.testing.expectEqual(x.mtu, y.mtu);
    try std.testing.expectEqual(@as(?fabric.rdma.LinkKind, .cx5), x.kind);
    try std.testing.expectEqual(x.kind, y.kind);
    // one window layout on both ends, the endpoint's alignment, the layout's total
    for ([_]mcdma_link.Spec{ mac, gb10 }) |s| {
        try std.testing.expectEqual(@as(u32, 2), s.config.ranks);
        try std.testing.expectEqual(s.layout.total, s.config.window_bytes);
        try std.testing.expectEqual(@as(usize, 0), s.config.window_bytes % fabric.mcdma.alignment);
        try std.testing.expectEqual(@as(usize, 1), s.config.links.len);
    }
    try std.testing.expectEqual(mac.config.window_bytes, gb10.config.window_bytes);
    try std.testing.expectEqual(mac.config.staging_bytes, gb10.config.staging_bytes);
}

test "a node outside the model, a missing link, a node listed twice or no library is refused by name before anything opens" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    var m = c.model("qwen-27b").?;
    var why: []const u8 = "";
    try std.testing.expectError(error.NotARank, mcdma_link.spec(a, c, m, "node-b", .{}, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "node-b is not one of model qwen-27b's prefill or decode nodes") != null);
    m.decode = &.{ "node-a", "node-b" };
    try std.testing.expectError(error.NoLink, mcdma_link.spec(a, c, m, "node-a", .{}, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "no link between node-a and node-b") != null);
    m.decode = &.{ "node-a", "node-e" };
    try std.testing.expectError(error.ListedTwice, mcdma_link.spec(a, c, m, "node-a", .{}, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "node-e is listed twice") != null);
    var bare = c;
    bare.nodes = try a.dupe(config.Node, c.nodes);
    bare.nodes[0].library = null;
    try std.testing.expectError(error.NoLibrary, mcdma_link.spec(a, bare, c.model("qwen-27b").?, "node-a", .{}, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "node-a names no MCDMA library") != null);
    try std.testing.expect(mcdma_link.libraryError(error.NoLibrary) and !mcdma_link.libraryError(error.NoLink));
    var many: [17][]const u8 = undefined;
    for (&many, 0..) |*x, i| x.* = try std.fmt.allocPrint(a, "n{d}", .{i});
    m.decode = &many;
    try std.testing.expectError(error.TooManyRanks, mcdma_link.spec(a, c, m, "n3", .{}, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "18 ranks; the cluster's membership over MCDMA takes at most 16") != null);
}

test "a Mac's ConnectX window past the kext's 64 regions is refused by name; a Linux host or a Thunderbolt link is not held to it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 63 pieces of 4 MiB and the peer's region fit; one byte more does not
    try std.testing.expectEqual(@as(usize, 64), mcdma_link.regions(63 * 4 * mib - 2 * 16384, 16384, 2));
    try std.testing.expectEqual(@as(usize, 65), mcdma_link.regions(63 * 4 * mib - 2 * 16384 + 1, 16384, 2));
    const c = try parse(a, example);
    const m = c.model("qwen-27b").?;
    var why: []const u8 = "";
    const small = try mcdma_link.spec(a, c, m, "node-a", .{ .mac = true }, &why);
    try std.testing.expect(mcdma_link.regions(small.config.window_bytes, small.config.staging_bytes, 2) <= 4);
    const big: mcdma_link.Options = .{ .mac = true, .sizes = .{ .step_bytes = 64 * mib } };
    try std.testing.expectError(error.TooManyRegions, mcdma_link.spec(a, c, m, "node-a", big, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "the Mac's MCDMA kext registers at most 64 per ConnectX link") != null);
    var linux = big;
    linux.mac = false;
    _ = try mcdma_link.spec(a, c, m, "node-e", linux, &why);
    // a Thunderbolt link on a Mac registers through Apple's driver, not the kext
    var tb = c;
    tb.links = try a.dupe(config.Link, c.links);
    tb.links[0].kind = .tb5;
    tb.links[0].ends[0].device = "rdma_en3";
    _ = try mcdma_link.spec(a, tb, m, "node-a", big, &why);
    // no kind, but the kext's device name: a ConnectX link all the same
    tb.links[0].kind = null;
    tb.links[0].ends[0].device = "rdma_mcrdma1";
    try std.testing.expectError(error.TooManyRegions, mcdma_link.spec(a, tb, m, "node-a", big, &why));
}

test "an endpoint over a missing library fails as McdmaFabricUnavailable, which counts as the library not found" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    var why: []const u8 = "";
    var s = try mcdma_link.spec(a, c, c.model("qwen-27b").?, "node-a", .{}, &why);
    s.library = "/nonexistent/libmcdma-fabric.dylib";
    const err = mcdma_link.Open.init(std.testing.allocator, s);
    try std.testing.expectError(error.McdmaFabricUnavailable, err);
    try std.testing.expect(mcdma_link.libraryError(error.McdmaFabricUnavailable));
    try std.testing.expectError(error.FileNotFound, mcdma_link.findLibrary(std.testing.io, s.library));
}

test "the plan digest is the same from either end and file order, and changes with what both ends must read alike" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    const m = c.model("qwen-27b").?;
    const names = try mcdma_link.ranks(a, m);
    const d = mcdma_link.planDigest(c, m, names, .{});
    var other = c;
    other.nodes = try a.dupe(config.Node, c.nodes);
    std.mem.reverse(config.Node, other.nodes);
    other.links = try a.dupe(config.Link, c.links);
    std.mem.swap(config.End, &other.links[0].ends[0], &other.links[0].ends[1]);
    try std.testing.expectEqual(d, mcdma_link.planDigest(other, m, names, .{}));
    other.links[0].mtu = 4096;
    try std.testing.expect(mcdma_link.planDigest(other, m, names, .{}) != d);
    other.links[0].mtu = 1024;
    other.links[0].ends[0].port += 1;
    try std.testing.expect(mcdma_link.planDigest(other, m, names, .{}) != d);
    var swapped = m;
    swapped.prefill = m.decode;
    swapped.decode = m.prefill;
    try std.testing.expect(mcdma_link.planDigest(c, swapped, try mcdma_link.ranks(a, swapped), .{}) != d);
    var longer = m;
    longer.context *= 2;
    try std.testing.expect(mcdma_link.planDigest(c, longer, names, .{}) != d);
    try std.testing.expect(mcdma_link.planDigest(c, m, names, .{ .staging_bytes = 2 << 20 }) != d);
}

test "every node of a model checked as it will check itself: a Mac's window against the kext's regions even from a CUDA leader" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    const m = c.model("qwen-27b").?;
    var why: []const u8 = "";
    try mcdma_link.specAll(a, c, m, .{}, &why);
    // the GB10 (CUDA) alone passes these sizes; its Mac peer (node-a) does not, wherever the check runs
    const big: mcdma_link.Sizes = .{ .step_bytes = 64 * mib };
    _ = try mcdma_link.spec(a, c, m, "node-e", .{ .mac = false, .sizes = big }, &why);
    try std.testing.expectError(error.TooManyRegions, mcdma_link.specAll(a, c, m, big, &why));
    try std.testing.expect(std.mem.startsWith(u8, why, "node-a's window"));
    var bare = c;
    bare.nodes = try a.dupe(config.Node, c.nodes);
    bare.nodes[4].library = null;
    try std.testing.expectError(error.NoLibrary, mcdma_link.specAll(a, bare, m, .{}, &why));
    try std.testing.expect(std.mem.indexOf(u8, why, "node-e names no MCDMA library") != null);
}

test "the declared links among a model's ranks are its fake bring-up's cables, ports in file order; other links are left out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try parse(a, example);
    const m = c.model("qwen-27b").?;
    const es = try mcdma_link.cables(a, c, try mcdma_link.ranks(a, m));
    try std.testing.expectEqual(@as(usize, 1), es.len);
    // node-a (rank 1) is the link's first end, node-e (rank 0) its second
    try std.testing.expectEqual(@as(u16, 1), es[0].a);
    try std.testing.expectEqual(@as(u16, 0), es[0].b);
    try std.testing.expectEqual(@as(u8, 0), es[0].a_port);
    try std.testing.expectEqual(@as(u8, 0), es[0].b_port);
    try std.testing.expect(es[0].up and es[0].kind == .cx5);
    const k3 = c.model("kimi-k3").?;
    try std.testing.expectEqual(@as(usize, 0), (try mcdma_link.cables(a, c, try mcdma_link.ranks(a, k3))).len);
}
