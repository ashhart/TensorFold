//! A declared EOS plus named control spellings form a stop set without adding ordinary vocabulary words.
const std = @import("std");
pub fn read(doc: anytype, extra_controls: []const []const u8, out: []u32) !usize {
    const view = try doc.tokenizer();
    const words = view.tokens orelse return error.MissingStopVocabulary;
    const eos = view.eos orelse return error.MissingStopEosId;
    if (out.len == 0) return error.StopSetCapacity;
    if (eos >= words.len or eos > std.math.maxInt(u32)) return error.InvalidStopEosId;
    out[0] = @intCast(eos);
    var count: usize = 1;
    var it = words.iterator();
    var id: usize = 0;
    while (try it.next()) |word| : (id += 1) {
        var wanted = false;
        for (extra_controls) |text| wanted = wanted or std.mem.eql(u8, text, word.string);
        if (!wanted) continue;
        const types = view.types orelse return error.MissingStopTokenTypes;
        if (try (try types.at(id)).unsigned() != 3) return error.StopTokenIsNotControl;
        var found = false;
        for (out[0..count]) |old| found = found or old == id;
        if (found) continue;
        if (count == out.len or id > std.math.maxInt(u32)) return error.StopSetCapacity;
        out[count] = @intCast(id);
        count += 1;
    }
    return count;
}

test {
    _ = @import("gguf_stops_test.zig");
}
