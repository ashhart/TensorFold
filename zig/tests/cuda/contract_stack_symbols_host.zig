//! The stack's new symbols: each with its own spec, its own bounds and its own served shapes.
const std = @import("std");
const symbols = @import("contract_stack_symbols.zig");
const t = std.testing;

test "Qwen CUDA contracts: every new symbol has its own spec, its own bounds and its own served shapes" {
    const kinds = symbols.symbol.len;
    var names: [kinds][]const u8 = undefined;
    for (0..kinds) |i| {
        const kind: symbols.Kind = @fromBackingInt(@intCast(i));
        const spec = symbols.spec(kind);
        // the spec names its own symbol, not a group's
        try t.expect(std.mem.indexOf(u8, spec.name, symbols.symbol[i]) != null);
        try t.expect(spec.served.len > 0);
        try t.expect(spec.depth(spec.served[0]) > 0);
        for (0..i) |j| try t.expect(!std.mem.eql(u8, spec.name, names[j]));
        names[i] = spec.name;
    }
}
