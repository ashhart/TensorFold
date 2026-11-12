//! A client refusal counted by its code in the load ledger, beside the reply that names it.
const api = @import("engine_api");
const Refused = @import("errors.zig").Refused;

/// Counts `r` in the server's ledger and hands `err` back unchanged: the reply is the one the caller built.
pub fn count(srv: anytype, r: api.load_ledger.reasons.Refusal, err: Refused) Refused {
    srv.load.refused(r);
    return err;
}
