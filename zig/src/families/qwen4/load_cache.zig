//! Embedding and PLE rows are CPU lookups at runtime; only the remaining checkpoint files are load-only.
const std = @import("std");
fn under(name: []const u8, table: []const u8) bool {
    return name.len > table.len and name[table.len] == '.' and std.mem.startsWith(u8, name, table);
}
fn live(file: anytype, model: anytype) bool {
    for (file.names.keys()) |name| {
        if (under(name, model.embedding)) return true;
        for (model.layers) |layer| if (layer.ple) |ple| {
            if (under(name, ple.table)) return true;
        };
    }
    return false;
}
pub fn release(ck: anytype, model: anytype) void {
    files: for (ck.files.items) |*file| {
        if (live(file, model)) continue;
        const id = file.file.stat(ck.io) catch continue;
        for (ck.files.items) |*held| if (live(held, model)) {
            const other = held.file.stat(ck.io) catch continue :files;
            if (id.inode == other.inode) continue :files;
        };
        file.release();
    }
}

test "Qwen4 cache: load-only files leave after reads, runtime lookup companions mixed shards and aliases stay" {
    const Fake = struct {
        names: struct {
            values: []const []const u8,
            pub fn keys(n: @This()) []const []const u8 {
                return n.values;
            }
        },
        file: struct {
            id: ?usize,
            pub fn stat(f: @This(), _: void) !struct { inode: usize } {
                return .{ .inode = f.id orelse return error.StatFailed };
            }
        },
        reads: u32 = 3,
        released: bool = false,
        pub fn release(f: *@This()) void {
            std.debug.assert(f.reads == 3);
            f.released = true;
        }
    };
    var files = [_]Fake{
        .{ .names = .{ .values = &.{"model.layers.0.mlp.gate_proj.weight"} }, .file = .{ .id = 1 } },
        .{ .names = .{ .values = &.{"model.embed_tokens.scales"} }, .file = .{ .id = 2 } },
        .{ .names = .{ .values = &.{ "model.layers.1.ple.table.biases", "model.norm.weight" } }, .file = .{ .id = 3 } },
        .{ .names = .{ .values = &.{"other"} }, .file = .{ .id = 2 } },
        .{ .names = .{ .values = &.{"other"} }, .file = .{ .id = null } },
    };
    var ck = .{ .io = {}, .files = .{ .items = &files } };
    const model = .{ .embedding = "model.embed_tokens", .layers = [_]struct { ple: ?struct { table: []const u8 } }{.{ .ple = .{ .table = "model.layers.1.ple.table" } }} };
    release(&ck, &model);
    try std.testing.expect(files[0].released);
    for (files[1..]) |f| try std.testing.expect(!f.released);
    files[0].released = false;
    files[1].file.id = null;
    release(&ck, &model);
    for (files) |f| try std.testing.expect(!f.released);
}
