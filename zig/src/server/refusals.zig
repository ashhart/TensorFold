//! A client refusal or failure counted by its reason in the load ledger, beside the reply that names it.
const std = @import("std");
const api = @import("engine_api");
const admission = @import("request_admission.zig");
const Refused = @import("errors.zig").Refused;
const reasons = api.load_ledger.reasons;

/// Counts `r` in the server's ledger and hands `err` back unchanged: the reply is the one the caller built.
pub fn count(srv: anytype, r: reasons.Refusal, err: Refused) Refused {
    srv.load.refused(r);
    return err;
}

/// Counts a failure the server answered itself (no refusal code) and hands `err` back unchanged.
pub fn failure(srv: anytype, r: reasons.Failure, err: Refused) Refused {
    srv.load.failed(r);
    return err;
}

/// An admission flag's 429, counted under the code its reply carries.
pub fn admitted(srv: anytype, reason: admission.Reason) void {
    srv.load.refused(codeOf(reason));
}

/// The refusal named by the 429's code: the ledger's labels are the admission codes themselves.
pub fn codeOf(reason: admission.Reason) reasons.Refusal {
    return std.meta.stringToEnum(reasons.Refusal, admission.code(reason)) orelse unreachable;
}
