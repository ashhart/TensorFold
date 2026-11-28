//! Add optional scoring callbacks to the same engine table without changing its other methods.
pub fn scoring(base: anytype, score: anytype, decide: anytype) @TypeOf(base) {
    var v = base;
    v.score = score;
    v.decide = decide;
    return v;
}
