//! Flash Next packs built from a checkpoint folder, then compared byte for byte with a Python dump.
const std = @import("std");
const tf = @import("tensorfold");
const pack = tf.flashnext_pack;

const usage =
    \\usage: tf-flashnext-pack MODEL_DIR OUT_DIR [--draft-vocab FILE] [--compare DUMP_DIR]
    \\
;

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("tf-flashnext-pack: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    var draft_vocab: ?[]const u8 = null;
    var compare: ?[]const u8 = null;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--draft-vocab")) {
            i += 1;
            if (i >= args.len) fail("--draft-vocab takes a file", .{});
            draft_vocab = args[i];
        } else if (std.mem.eql(u8, args[i], "--compare")) {
            i += 1;
            if (i >= args.len) fail("--compare takes the dump's folder", .{});
            compare = args[i];
        } else fail("unknown option {s}", .{args[i]});
    }
    try std.Io.Dir.cwd().createDirPath(init.io, args[2]);
    const report = try pack.build(gpa, init.io, args[1], args[2], draft_vocab);
    std.debug.print("{s}: {d} decode tensors, {d} mlx, {d} mtp mlx, norms around {s}, draft ids {d}\n", .{
        args[2],                                        report.decode_tensors, report.mlx_tensors, report.mtp_mlx_tensors,
        if (report.norms_around_one) "one" else "zero", report.draft_ids,
    });
    if (compare) |dump_dir| {
        var ok = true;
        for ([_][]const u8{ "pack.safetensors", "pack_mlx.safetensors", "pack_mtp_mlx.safetensors" }) |name| {
            const built = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ args[2], name });
            defer gpa.free(built);
            const reference = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dump_dir, name });
            defer gpa.free(reference);
            var out: std.Io.Writer.Allocating = .init(gpa);
            defer out.deinit();
            const same = pack.compareFile(gpa, init.io, built, reference, &out.writer) catch |e| {
                std.debug.print("{s}: cannot compare ({s})\n", .{ name, @errorName(e) });
                ok = false;
                continue;
            };
            std.debug.print("{s}", .{out.written()});
            ok = ok and same;
        }
        if (!ok) std.process.exit(1);
    }
}
