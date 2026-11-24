//! The GLP control vector container as the outside world sees it: the reader's view of a file, and a rewrite of it.
const std = @import("std");
const glp = @import("glp");

const help =
    \\tf-glp-check read FILE OUT.json OUT.bin  the reader's view: the metadata, the layers and their numbers
    \\tf-glp-check rewrite FILE OUT.gguf [--repo ID] [--revision REV]
    \\                                    the file again through this writer
    \\
    \\--repo and --revision re-stamp the checkpoint the file names, for weights published under another
    \\name: the directions stay the same bytes, so only do it for a checkpoint with the same weights.
    \\
    \\The binary sidecar holds the directions in ascending layer order, each one's fp32 payload as it
    \\stands in the file, so another reader's numbers can be compared byte for byte.
;

/// Why these arguments cannot run, in the words the tool prints; null when they can.
pub fn usageProblem(args: []const []const u8) ?[]const u8 {
    if (args.len < 2) return "one of read or rewrite is required";
    const command = args[1];
    if (std.mem.eql(u8, command, "--help")) return null;
    if (std.mem.eql(u8, command, "read")) {
        return if (args.len == 5) null else "read takes FILE OUT.json OUT.bin";
    }
    if (!std.mem.eql(u8, command, "rewrite")) return "the command is read or rewrite";
    if (args.len < 4) return "rewrite takes FILE OUT.gguf";
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--repo") or std.mem.eql(u8, args[i], "--revision")) {
            i += 1;
            if (i >= args.len) return "the option needs a value";
        } else return "the option is --repo or --revision";
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help")) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    if (usageProblem(args)) |problem| {
        const line = try std.fmt.allocPrint(a, "tf-glp-check: {s}\n{s}\n", .{ problem, help });
        try std.Io.File.stdout().writeStreamingAll(io, line);
        std.process.exit(2);
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
        // The checkpoint the file names may be re-stamped: the same weights under another name.
        var base_model = vector.base_model;
        var base_revision = vector.base_revision;
        var i: usize = 4;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--repo")) {
                i += 1;
                if (i >= args.len) return error.MissingValue;
                base_model = args[i];
            } else if (std.mem.eql(u8, args[i], "--revision")) {
                i += 1;
                if (i >= args.len) return error.MissingValue;
                base_revision = args[i];
            } else return error.UnknownOption;
        }
        const out = try glp.write(a, .{
            .alpha = vector.alpha,
            .hook_point = vector.hook_point,
            .derived_at = vector.derived_at,
            .base_model = base_model,
            .base_revision = base_revision,
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
        .alpha = vector.alpha,
        .hook_point = vector.hook_point,
        .derived_at = vector.derived_at,
        .base_model = vector.base_model,
        .base_revision = vector.base_revision,
        .method = vector.method,
        .contrast = vector.contrast,
        .width = vector.width,
        .sha256 = std.fmt.bytesToHex(vector.sha256, .lower),
        .file_sha256 = std.fmt.bytesToHex(vector.file_sha256, .lower),
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

test "the checker refuses missing output arguments before reading a file" {
    try std.testing.expect(usageProblem(&.{"tf-glp-check"}) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "read" }) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "read", "f" }) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "rewrite", "f" }) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "bogus", "f", "o" }) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "read", "f", "o.json", "o.bin" }) == null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "rewrite", "f", "o.gguf" }) == null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "rewrite", "f", "o.gguf", "--repo", "id", "--revision", "rev" }) == null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "rewrite", "f", "o.gguf", "--repo" }) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "rewrite", "f", "o.gguf", "--bogus" }) != null);
    try std.testing.expect(usageProblem(&.{ "tf-glp-check", "--help" }) == null);
}
