//! Deterministic identity pinning and private row-staging regressions.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("sys");
const rows = @import("rows_store.zig");
const rowdir = @import("rows_dir.zig");
const Fake = @import("rows_fake.zig").Fake;
const t = std.testing;
const a = t.allocator;

fn options(root: []const u8) rows.Options {
    return .{ .root = root, .identity = 7, .cap = 1 << 30, .floor = 0, .daily = 1 << 30, .sync_writes = true };
}

fn path(tmp: *const t.TmpDir, sub: []const u8) ![:0]u8 {
    const resolved = try tmp.dir.realPathFileAlloc(t.io, sub, a);
    defer a.free(resolved);
    return a.dupeSentinel(u8, resolved, 0);
}

fn identityComponent(identity: u64) [16]u8 {
    var hash = std.hash.Wyhash.init(0x726f7773);
    hash.update(std.mem.asBytes(&identity));
    var out: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{x:0>16}", .{hash.final()}) catch unreachable;
    return out;
}

fn writeBlock(fd: c_int, name: rows.chain.Name, bytes: []const u8, nonce: u64) !void {
    var writer = try rows.file.Writer.openUnique(fd, &rows.chain.hex(name), ".blk", nonce);
    errdefer writer.abort();
    try writer.append(bytes);
    _ = try writer.finish(.{ .kind = @backingInt(rows.file.Kind.block), .payload = 0, .payload_hash = 0, .name = name }, &.{}, false);
}

fn expectBlock(fd: c_int, name: rows.chain.Name, bytes: []const u8) !void {
    var leaf_buf: [64]u8 = undefined;
    const leaf = try std.fmt.bufPrintSentinel(&leaf_buf, "{s}.blk", .{rows.chain.hex(name)}, 0);
    var opened = try rows.file.openAt(fd, leaf.ptr, .block, name);
    defer opened.close();
    const stage = try rows.file.stage(a, bytes.len);
    defer a.free(stage);
    try t.expectEqualSlices(u8, bytes, try rows.file.payload(&opened, stage));
}

fn leafPath(dir: [:0]const u8, stem: []const u8, ext: []const u8, part: ?u64) ![:0]u8 {
    return if (part) |nonce|
        std.fmt.allocPrintSentinel(a, "{s}/{s}{s}.{x}.part", .{ dir, stem, ext, nonce }, 0)
    else
        std.fmt.allocPrintSentinel(a, "{s}/{s}{s}", .{ dir, stem, ext }, 0);
}

fn expectFile(tmp: *const t.TmpDir, sub: []const u8, bytes: []const u8) !void {
    const actual = try tmp.dir.readFileAlloc(t.io, sub, a, .limited(128));
    defer a.free(actual);
    try t.expectEqualSlices(u8, bytes, actual);
}

test "rows_identity_symlink_is_refused_without_touching_target" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(t.io, "root");
    try tmp.dir.createDirPath(t.io, "foreign");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "foreign/stale.part", .data = "keep this staging file" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "foreign/sentinel", .data = "never change me" });
    const root = try path(&tmp, "root");
    defer a.free(root);
    const foreign = try path(&tmp, "foreign");
    defer a.free(foreign);
    const id = identityComponent(7);
    const alias = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ root, id }, 0);
    defer a.free(alias);
    try t.expectEqual(@as(c_int, 0), platform.c.symlink(foreign.ptr, alias.ptr));
    var fake: Fake = .{ .gpa = a };
    try t.expectError(error.RowsDir, rows.Rows.open(a, fake.snapshots(), options(root)));
    try expectFile(&tmp, "foreign/stale.part", "keep this staging file");
    try expectFile(&tmp, "foreign/sentinel", "never change me");
}

test "rows_unique_staging_refuses_a_link_without_touching_its_target" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(t.io, "store");
    try tmp.dir.writeFile(t.io, .{ .sub_path = "victim", .data = "original victim" });
    const store = try path(&tmp, "store");
    defer a.free(store);
    const victim = try path(&tmp, "victim");
    defer a.free(victim);
    const fd = try rowdir.openPath(store);
    defer rowdir.close(fd);
    const name: rows.chain.Name = @splat(0x51);
    const stem = rows.chain.hex(name);
    const staging = try leafPath(store, &stem, ".st", 0x2a);
    defer a.free(staging);
    try t.expectEqual(@as(c_int, 0), platform.c.symlink(victim.ptr, staging.ptr));
    try t.expectError(error.RowsWrite, rows.file.Writer.openUnique(fd, &stem, ".st", 0x2a));
    try expectFile(&tmp, "victim", "original victim");
}

test "rows_identity_descriptor_ignores_post_open_move_and_relink" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(t.io, "foreign");
    const root = try path(&tmp, ".");
    defer a.free(root);
    const foreign = try path(&tmp, "foreign");
    defer a.free(foreign);
    var fake: Fake = .{ .gpa = a };
    const r = try rows.Rows.open(a, fake.snapshots(), options(root));
    defer r.deinit();
    const foreign_fd = try rowdir.openPath(foreign);
    defer rowdir.close(foreign_fd);
    const old_name: rows.chain.Name = @splat(0x31);
    const new_name: rows.chain.Name = @splat(0x72);
    try writeBlock(r.dir_fd, old_name, "original row", 1);
    try writeBlock(foreign_fd, old_name, "foreign row", 1);
    const pinned = try std.fmt.allocPrintSentinel(a, "{s}-pinned", .{r.dir}, 0);
    defer a.free(pinned);
    try t.expectEqual(@as(c_int, 0), platform.c.rename(r.dir.ptr, pinned.ptr));
    try t.expectEqual(@as(c_int, 0), platform.c.symlink(foreign.ptr, r.dir.ptr));
    try expectBlock(r.dir_fd, old_name, "original row");
    try writeBlock(r.dir_fd, new_name, "written after relink", 2);
    try expectBlock(r.dir_fd, new_name, "written after relink");
    rows.unlinkName(r.dir_fd, old_name, ".blk");
    var old_leaf_buf: [64]u8 = undefined;
    const old_leaf = try std.fmt.bufPrintSentinel(&old_leaf_buf, "{s}.blk", .{rows.chain.hex(old_name)}, 0);
    try t.expectError(error.RowsMissing, rows.file.openAt(r.dir_fd, old_leaf.ptr, .block, old_name));
    try expectBlock(foreign_fd, old_name, "foreign row");
    const foreign_new = try leafPath(foreign, &rows.chain.hex(new_name), ".blk", null);
    defer a.free(foreign_new);
    try t.expect(platform.c.access(foreign_new.ptr, 0) != 0);
    const moved_new = try leafPath(pinned, &rows.chain.hex(new_name), ".blk", null);
    defer a.free(moved_new);
    try t.expectEqual(@as(c_int, 0), platform.c.access(moved_new.ptr, 0));
}
