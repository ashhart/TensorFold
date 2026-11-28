const std = @import("std");
const Rule = @import("environment.zig").Rule;
const Error = @import("environment.zig").Error;

pub fn equal(r: Rule, a: []const u8, b: []const u8) bool {
    return if (r.fold or r.ignore_case) std.ascii.eqlIgnoreCase(a, b) else std.mem.eql(u8, a, b);
}
pub fn wordSet(r: Rule, text: []const u8) Error!u64 {
    if (r.set_switch) {
        if (std.mem.eql(u8, text, "0")) return 0;
        if (std.mem.eql(u8, text, "1")) return (@as(u64, 1) << @intCast(r.words.len)) - 1;
    }
    var it = std.mem.tokenizeAny(u8, text, r.split);
    var bits: u64 = 0;
    var count: usize = 0;
    while (it.next()) |token| {
        const v = std.mem.trim(u8, token, " ");
        var found = false;
        for (r.words, 0..) |word, i| if (equal(r, v, word)) {
            bits |= @as(u64, 1) << @intCast(i);
            found = true;
            break;
        };
        if (!found) return error.InvalidEnvironment;
        count += 1;
    }
    if (r.items != 0 and count != r.items) return error.InvalidEnvironment;
    return bits;
}
