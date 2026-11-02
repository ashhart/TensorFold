//! Host-only parity runner for GGUF metadata, existing tokenizer.json and a JSON array of text cases.
const std = @import("std");
const gg = @import("src/core/gguf.zig");
const adapter = @import("src/core/gguf_tokenizer.zig");
const tokenizer = @import("tokenizer");
fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(128 << 20));
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4 and args.len != 5) return error.ExpectedGgufTokenizerJsonAndCasesJson;
    var file = try gg.File.open(a, io, args[1], .{});
    defer file.close(io);
    const d = &file.document;
    var g = try tokenizer.parse(a, try std.json.Stringify.valueAlloc(a, (try adapter.assets(a, d)).tokenizer, .{}));
    defer g.deinit();
    var j = try tokenizer.parse(a, try read(a, io, args[2]));
    defer j.deinit();
    if (args.len == 5) {
        const generation = try std.json.parseFromSliceLeaky(std.json.Value, a, try read(a, io, args[4]), .{});
        const expected = try tokenizer.eos.resolve(a, .null, generation, (try d.tokenizer()).tokens.?.len);
        const actual = try adapter.eosIds(a, d);
        if (actual.array.items.len != expected.len) return error.StopSetLengthMismatch;
        for (actual.array.items, expected) |id, want| if (id.integer != want) return error.StopSetMismatch;
        std.debug.print("GGUF_EOS_PARITY ids={any}\n", .{expected});
    }
    const cases = try std.json.parseFromSliceLeaky(std.json.Value, a, try read(a, io, args[3]), .{});
    if (cases != .array) return error.ExpectedCaseArray;
    var failures: usize = 0;
    for (cases.array.items, 0..) |case, i| {
        const text = if (case == .string) case.string else case.object.get("text").?.string;
        for ([_]bool{ false, true }) |plain| {
            const actual = if (plain) try g.encodePlain(a, text) else try g.encode(a, text);
            const expected = if (plain) try j.encodePlain(a, text) else try j.encode(a, text);
            if (!std.mem.eql(u32, actual, expected)) {
                std.debug.print("FAIL case={d} plain={} GGUF={any} JSON={any}\n", .{ i, plain, actual, expected });
                failures += 1;
            }
            if (!plain and case == .object) if (case.object.get("ids")) |ids| {
                if (ids.array.items.len != expected.len) return error.ExpectedIdsLength;
                for (ids.array.items, expected) |want, got| if (want.integer != got) return error.IndependentExpectedIdMismatch;
            };
        }
    }
    std.debug.print("TOKENIZER_PARITY cases={d} comparisons={d} failures={d}\n", .{ cases.array.items.len, 2 * cases.array.items.len, failures });
    if (failures != 0) return error.TokenizerParityFailed;
}
