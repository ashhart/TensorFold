//! The GLP control vector container as the outside world sees it: the reader's view of a file, and a rewrite of it.
const std = @import("std");
const glp = @import("glp");

const help =
    \\tf-glp-check read FILE OUT.json OUT.bin  the reader's view: the metadata, the layers and their numbers
    \\tf-glp-check rewrite FILE OUT.gguf [--mode project|add]
    \\                                    the file again through this writer, in the mode named
    \\
    \\The binary sidecar holds the directions in ascending layer order, each one's fp32 payload as it
    \\stands in the file, so another reader's numbers can be compared byte for byte.
;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help")) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(64 << 20));
    var why: glp.Why = .{};
    const vector = glp.read(a, bytes, &why) catch |err| {
        if (err == error.OutOfMemory) return err;
        const line = try std.fmt.allocPrint(a, "refused: {s}\n", .{why.text()});
        try std.Io.File.stdout().writeStreamingAll(io, line);
        return error.BadVector;
    };
    if (std.mem.eql(u8, args[1], "rewrite")) {
        // The mode may be changed on the way out: the same directions, applied as a bias instead.
        var mode = vector.mode;
        var i: usize = 4;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--mode")) {
                i += 1;
                if (i >= args.len) return error.MissingValue;
                if (std.mem.eql(u8, args[i], "add")) {
                    mode = .add;
                } else if (std.mem.eql(u8, args[i], "project")) {
                    mode = .project;
                } else return error.UnknownMode;
            } else return error.UnknownOption;
        }
        const out = try glp.write(a, .{
            .mode = mode,
            .alpha = vector.alpha,
            .hook_point = vector.hook_point,
            .derived_at = vector.derived_at,
            .base_model = vector.base_model,
            .base_revision = vector.base_revision,
            .method = vector.method,
            .contrast = vector.contrast,
            .publication = vector.publication,
            .layers = vector.layers,
        });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = out });
        return;
    }
    if (!std.mem.eql(u8, args[1], "read")) return error.UnknownCommand;

    // The directions as bytes, in the order the file holds them.
    var blob: std.ArrayList(u8) = .empty;
    for (vector.layers) |l| {
        for (l.dir) |v| try blob.appendSlice(a, &std.mem.toBytes(@as(u32, @bitCast(v))));
    }
    var ids: std.ArrayList(u32) = .empty;
    for (vector.layers) |l| try ids.append(a, l.id);
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .mode = @tagName(vector.mode),
        .alpha = vector.alpha,
        .hook_point = vector.hook_point,
        .derived_at = vector.derived_at,
        .base_model = vector.base_model,
        .base_revision = vector.base_revision,
        .method = vector.method,
        .contrast = vector.contrast,
        .width = vector.width,
        .sha256 = std.fmt.bytesToHex(vector.sha256, .lower),
        .transferred = vector.transferred,
        .name = vector.publication.name,
        .description = vector.publication.description,
        .author = vector.publication.author,
        .license = vector.publication.license,
        .model_hint = vector.publication.model_hint,
        .measured_effect = vector.publication.measured_effect,
        .created = vector.publication.created,
        .layers = ids.items,
        .bytes = blob.items.len,
    }, .{});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[3], .data = receipt });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = blob.items });
}
