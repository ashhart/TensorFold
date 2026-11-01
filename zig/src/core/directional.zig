//! Directional qualification retains the legacy slope and reports every mathematical failure in a grid.
pub fn slope(losses: [6][2]f64, step: f64) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (losses, 1..) |pair, m| for ([_]f64{ -1, 1 }, pair) |sign, loss| {
        const x = sign * @as(f64, @floatFromInt(m)) * step;
        num += x * loss;
        den += x * x;
    };
    return num / den;
}

pub const Verdict = enum { pass, mismatch, unresolved };
pub const Summary = struct {
    checks: usize = 0,
    failed: usize = 0,
    unresolved: usize = 0,

    pub fn finish(s: Summary) !void {
        if (s.unresolved != 0) return error.NoiseLimitedGradient;
        if (s.failed != 0) return error.GradientMismatch;
    }
};

/// Mathematical failures are values, so every layer/row is reported before qualification returns an error.
pub fn run(ctx: anytype, rows: []const usize, layers: []const usize) !Summary {
    var out: Summary = .{};
    for (rows) |row| for (layers) |layer| {
        const verdict = try ctx.check(row, layer);
        out.checks += 1;
        switch (verdict) {
            .pass => {},
            .mismatch => out.failed += 1,
            .unresolved => {
                out.failed += 1;
                out.unresolved += 1;
            },
        }
    };
    return out;
}
