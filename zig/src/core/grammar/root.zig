//! Grammar-constrained decoding over xgrammar's C++ core: `Grammars`, `Constraint`, `engine`/`xgr` and `rows`.

pub const engine = @import("engine.zig");
pub const xgr = @import("xgr.zig");
pub const vocab = @import("vocab.zig");
pub const tags = @import("tags.zig");
pub const constraint = @import("constraint.zig");
pub const grammars = @import("grammars.zig");
pub const pack = @import("pack.zig");
pub const rows = @import("rows.zig");

pub const Grammars = grammars.Grammars;
pub const Spec = grammars.Spec;
pub const Kind = grammars.Kind;
pub const Constraint = constraint.Constraint;
pub const Cut = constraint.Cut;

test {
    _ = @import("golden_test.zig");
    _ = pack;
    _ = rows;
}
