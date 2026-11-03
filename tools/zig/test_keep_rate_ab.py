"""Host-only keep-rate experiment with a child HTTP server, no model, driver or GPU."""
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

import keep_rate_ab as runner
import keep_rate_records as records
from keep_rate_server import Server, manifest, verify_copy

FAKE = r'''#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
argv=sys.argv[1:]
def flag(name): return argv[argv.index(name)+1]
root=Path(flag('--api-key-file')).parent
model=Path(argv[1])
weight=(model/'weights').read_text()
assert weight=='untaught' or ('--slide' not in argv and root.name=='off-after')
candidate='candidate' in Path(sys.argv[0]).name
quiet='silent' in Path(sys.argv[0]).name
bad='bad' in Path(sys.argv[0]).name
error='error' in Path(sys.argv[0]).name
partial='partial' in Path(sys.argv[0]).name
index=int(root.name.split('-')[-1]) if root.name.startswith('run-') else 0
key=Path(flag('--api-key-file')).read_text().strip()
graph={'facts':[]}
learned=weight!='untaught'
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a): pass
 def send(self,body,kind='application/json',status=200):
  raw=body.encode();self.send_response(status);self.send_header('Content-Type',kind)
  self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def auth(self):
  if self.headers.get('Authorization')!='Bearer '+key:
   self.send('{}',status=401);return False
  return True
 def do_GET(self):
  if not self.auth(): return
  self.send(json.dumps(graph if self.path.endswith('/graph') else {'status':'ok'}))
 def do_POST(self):
  global learned,graph
  if not self.auth(): return
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if self.path.endswith('/completions'):
   assert body['stream'] is False and body['return_token_ids'] is True
   assert 'enable_thinking' not in body and body['seed']==1587
   ids=[100,1 if '--no-thinking' in argv else 2,3 if bad and learned else 0]
   sha=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()
   self.send(json.dumps({'tensorfold':{'token_ids':ids,'token_sha256':sha},'choices':[]}));return
  assert body=={'text':'My favourite colour is teal.'} and '--slide' in argv
  if error: self.send('{"error":"planted transport failure"}',status=500);return
  kept=index%2==0 if candidate else index%5==0
  steps=1 if kept else 5
  for n in range(1,steps+1):
   print(f'[tensorfold] done writer-{n} prompt=20 tokens=8 seed={1000+index+n}',flush=True)
   counts='; fact 1: held plain 2/2, held thinking 2/2, probes 2/4' if candidate and not quiet else ''
   back='come back' if kept else 'do not come back'
   print(f'[tensorfold] slide: round {n}: {n} steps (1.0 s learning, 1.0 s checking); '
         f'the held-out answers {back} token for token, and {back} when asked{counts}',flush=True)
   if candidate and '--prompt-seed' not in argv and not kept and n<=2:
    print(f'[tensorfold] slide: fact 1: refused fit taken out; wordings redrawn, attempt {n}/2',flush=True)
  graph={'facts':[{'id':1,'state':'learned' if kept else 'missed','text':body['text']}]}
  if kept: (model/'weights').write_text('kept-teal')
  learned=True
  def event(kind,data): return 'event: '+kind+'\ndata: '+json.dumps(data)+'\n\n'
  raw=event('learned',{'id':1,'recalled':kept,'steps':steps})+event('saved',{'saved':True})+'data: [DONE]\n\n'
  if partial:
   self.send_response(200);self.send_header('Content-Length',str(len(raw)+30));self.end_headers()
   self.wfile.write(raw[:-14].encode());return
  self.send(raw,'text/event-stream')
HTTPServer.allow_reuse_address=True
HTTPServer(('127.0.0.1',int(flag('--port'))),Handler).serve_forever()
'''


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fixture(root, candidate="candidate"):
    model = root / "original"
    model.mkdir()
    (model / "weights").write_text("untaught")
    bins = {}
    for arm, name in (("base", "base"), ("candidate", candidate)):
        bins[arm] = root / name
        bins[arm].write_text(FAKE)
        bins[arm].chmod(0o700)
    output = root / "receipt"
    output.mkdir()
    return SimpleNamespace(checkpoint=model, output=output, binaries=bins, backend="cuda", port=port(),
                           startup_timeout=5, request_timeout=5, server_args=[], retain_copies=False,
                           checkpoint_revision="fictional-fixture")


class RecordTests(unittest.TestCase):
    def test_old_counts_are_unknown_and_every_seed_redraw_and_round_is_retained(self):
        log = ("[tensorfold] done w prompt=2 seed=19\n"
               "slide: round 1: 1 steps (0 s); the held-out answers do not come back token for token, "
               "and do not come back when asked\n"
               "wordings redrawn, attempt 1/2\n"
               "[tensorfold] done x prompt=2 seed=23\n"
               "slide: round 2: 3 steps (0 s); the held-out answers come back token for token, "
               "and come back when asked; fact 1: held plain 2/2, held thinking 0/0, probes 2/4\n")
        result = records.diagnostics(log)
        self.assertEqual([19, 23], [s["seed"] for s in result["request_seeds"]])
        first, second = result["rounds"]
        self.assertIsNone(first["counts"])
        self.assertEqual(1, result["rounds_missing_counts"])
        self.assertEqual(0, first["wording_attempt"])
        self.assertEqual(1, second["wording_attempt"])
        self.assertEqual(3, second["cumulative_steps"])
        self.assertEqual({"passed": 2, "asked": 4}, second["counts"][0]["probes"])
        self.assertEqual(1, len(result["redraws"]))

    def test_partial_or_unsaved_stream_never_counts_as_kept(self):
        stream = records.events('event: learned\ndata: {"recalled":true,"steps":2}\n\ndata: [DONE]\n\n')
        graph = {"facts": [{"state": "learned"}]}
        self.assertFalse(records.outcome(stream, graph)["accepted"])
        stream.insert(1, {"event": "saved", "data": {"saved": True}})
        self.assertTrue(records.outcome(stream, graph)["accepted"])
        stream.pop()
        self.assertEqual("error", records.outcome(stream, graph)["status"])

    def test_token_hash_matches_the_server_decimal_comma_protocol(self):
        ids = [1, 23, 456, 2**32 - 1]
        digest = hashlib.sha256(b"1,23,456,4294967295").hexdigest()
        got = records.tokens({"tensorfold": {"token_ids": ids, "token_sha256": digest}})
        self.assertEqual(digest, got["sha256"])

    def test_missing_or_corrupt_token_receipts_fail(self):
        for ids in (None, [True], [-1], [2 ** 32], ["1"]):
            with self.assertRaises(ValueError):
                records.tokens({"tensorfold": {"token_ids": ids}})
        with self.assertRaises(ValueError):
            records.tokens({"tensorfold": {"token_ids": [1], "token_sha256": "wrong"}})
        self.assertEqual([], records.tokens({"tensorfold": {"token_ids": []}})["ids"])

    def test_refusals_errors_and_absent_trials_stay_in_denominator(self):
        rows = [{"arm": "base", "thinking": False, "prompt_seed": False, "status": s}
                for s in ("kept", "refused", "error")]
        result = records.summary(rows, 10)
        cell = result["cells"][0]
        self.assertEqual((1, 1, 1, 7), tuple(cell[k] for k in ("kept", "refused", "errors", "not_recorded")))
        self.assertEqual(0.1, cell["keep_rate_of_scheduled"])
        self.assertFalse(result["complete"])

    def test_copy_refuses_a_hard_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            target.mkdir()
            (source / "w").write_text("original")
            os.link(source / "w", target / "w")
            with self.assertRaisesRegex(ValueError, "inode"):
                verify_copy(source, target, manifest(source))

    def test_post_lesson_error_keeps_the_known_verdict_in_the_rate(self):
        rows = [{"arm": "base", "thinking": False, "prompt_seed": False, "status": "error", "lesson_status": s}
                for s in ("kept", "refused")]
        cell = records.summary(rows, 10)["cells"][0]
        self.assertEqual((1, 1, 2), tuple(cell[k] for k in ("kept", "refused", "errors")))
        self.assertEqual(0.1, cell["keep_rate_of_scheduled"])


class FakeServerTests(unittest.TestCase):
    def test_native_command_propagates_each_matrix_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            for seeded in (False, True):
                for thinking in (False, True):
                    for slide in (False, True):
                        root = args.output / f"{seeded}-{thinking}-{slide}"
                        root.mkdir()
                        server = Server(args, args.binaries["base"], root, args.checkpoint, thinking, seeded, slide)
                        try:
                            self.assertEqual(seeded, "--prompt-seed" in server.command)
                            self.assertEqual(not thinking, "--no-thinking" in server.command)
                            self.assertEqual(slide, "--slide" in server.command)
                            self.assertNotIn("--slide-forget", server.command)
                        finally:
                            server.close()

    def test_complete_eighty_trial_matrix_preserves_refusals_tokens_and_original(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory))
            before = manifest(args.checkpoint)
            self.assertEqual(0, runner.experiment(args))
            result = json.loads((args.output / "summary.json").read_text())
            rows = json.loads((args.output / "results.json").read_text())
            self.assertEqual(80, len(rows))
            self.assertEqual(8, len(json.loads((args.output / "off.json").read_text())))
            self.assertEqual((8, 32, 20, 20), (result["arms"]["base"]["kept"], result["arms"]["base"]["refused"],
                                             result["arms"]["candidate"]["kept"], result["arms"]["candidate"]["refused"]))
            self.assertTrue(result["receipt_complete"])
            self.assertTrue(all(c["scheduled"] == c["recorded"] == 10 for c in result["cells"]))
            self.assertTrue(any(c["rounds_missing_counts"] for c in result["cells"] if c["arm"] == "base"))
            self.assertEqual(before, manifest(args.checkpoint))
            for row in rows:
                root = args.output / row["directory"]
                self.assertTrue((root / "lesson.sse").exists())
                self.assertTrue((root / "server.log").exists())
                self.assertFalse((root / "model").exists())
                self.assertFalse((root / "key").exists())
                self.assertEqual([1 if row["status"] == "kept" else 5], row["steps"])
                self.assertEqual(3, len(row["unrelated_off_after"]))
            refusals = [r for r in rows if r["arm"] == "candidate" and r["status"] == "refused"]
            self.assertTrue(all(len(r["diagnostics"]["redraws"]) == (0 if r["prompt_seed"] else 2) for r in refusals))

    def test_unrelated_token_change_and_missing_candidate_counts_fail_the_receipt(self):
        for name, field in (("bad-candidate", "token_mismatches"), ("silent-candidate", "receipt_errors")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                args = fixture(Path(directory), name)
                self.assertEqual(1, runner.experiment(args, runs=1))
                result = json.loads((args.output / "summary.json").read_text())
                self.assertTrue(result[field])
                self.assertFalse(result["receipt_complete"])

    def test_transport_errors_retained_as_errors_without_inventing_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory), "error-candidate")
            self.assertEqual(1, runner.experiment(args, runs=1))
            rows = json.loads((args.output / "results.json").read_text())
            candidate = [r for r in rows if r["arm"] == "candidate"]
            self.assertEqual(4, len(candidate))
            self.assertTrue(all(r["status"] == "error" for r in candidate))
            self.assertTrue(all((args.output / r["directory"] / "http-error.txt").exists() for r in candidate))

    def test_truncated_stream_keeps_partial_bytes_and_round_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = fixture(Path(directory), "partial-candidate")
            self.assertEqual(1, runner.experiment(args, runs=1))
            rows = json.loads((args.output / "results.json").read_text())
            candidate = [r for r in rows if r["arm"] == "candidate"]
            self.assertTrue(all(r["status"] == "error" for r in candidate))
            self.assertTrue(all(len(r["diagnostics"]["rounds"]) == 1 for r in candidate))
            self.assertTrue(all((args.output / r["directory"] / "http-incomplete.txt").exists() for r in candidate))

    def test_occupied_port_starts_no_process_and_leaves_the_other_listener_alive(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            args = fixture(Path(directory))
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            args.port = listener.getsockname()[1]
            root = args.output / "occupied"
            root.mkdir()
            server = Server(args, args.binaries["base"], root, args.checkpoint, False, False, False)
            with self.assertRaises(OSError):
                with server:
                    self.fail("occupied port admitted")
            self.assertIsNone(server.process)
            self.assertGreater(listener.fileno(), 0)
            self.assertFalse(server.keyfile.exists())


if __name__ == "__main__":
    unittest.main()
