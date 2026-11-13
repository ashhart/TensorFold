"""The heat-exchanger agent eval against a scripted OpenAI server: references, scoring, the loop and the CLI."""
import contextlib
import http.server
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_hx_entu
import eval_sandbox

UNIX = eval_sandbox.resource is not None
RATE = ("import math\nC_h, C_c, UA = {C_h}, {C_c}, {UA}\ncmin, cmax = min(C_h, C_c), max(C_h, C_c)\n"
        "ntu, cr = UA / cmin, cmin / cmax\n{eps}\nq = eps * cmin * 180\nprint(q, 200 - q / C_h, 20 + q / C_c)\n")
COUNTER = ("e = math.exp(-ntu * (1 - cr))\n"
           "eps = ntu / (1 + ntu) if abs(cr - 1) < 1e-9 else (1 - e) / (1 - cr * e)")
PARALLEL = "eps = (1 - math.exp(-ntu * (1 + cr))) / (1 + cr)"
SIZE = "eps = 500 * (200 - 100) / (500 * 180)\nprint(500 * eps / (1 - eps))\n"
CELLS = ("N, C, ua = 200, 500.0, 1000.0 / 200\n"
         "def march(tc_out):\n"
         "    th, tc = 200.0, tc_out\n"
         "    for _ in range(N):\n"
         "        dq = ua * (th - tc)\n"
         "        th, tc = th - dq / C, tc - dq / C\n"
         "    return th, tc\n"
         "lo, hi = 20.0, 200.0\n"
         "for _ in range(100):\n"
         "    mid = (lo + hi) / 2\n"
         "    lo, hi = (lo, mid) if march(mid)[1] > 20 else (mid, hi)\n"
         "print(march((lo + hi) / 2)[0])\n")
ZERO_OVER_ZERO = "e = math.exp(-ntu * (1 - cr))\neps = (1 - e) / (1 - cr * e)"
GUARDED = "e = math.exp(-ntu * (1 - cr))\nnum, den = 1 - e, 1 - cr * e\neps = num / den if den else 0.0"


def message(content=None, calls=(), reasoning=None, completion=20, thinking=None):
    """A non-streamed chat completion with optional tool calls, reasoning and usage."""
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": "run_python", "arguments": a}}
                             for i, a in enumerate(calls)]
    if reasoning:
        msg["reasoning_content"] = reasoning
    usage = {"prompt_tokens": 50, "completion_tokens": completion}
    if thinking is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": thinking}
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": usage}


def tool(code: str, **kw) -> dict:
    return message(calls=[json.dumps({"code": code})], **kw)


def answer_from_tool(keys, prose="Done.", **kw):
    """A reply that reads the numbers the last tool run printed and answers them as the JSON line."""
    def reply(body):
        printed = [m for m in body["messages"] if m["role"] == "tool"][-1]["content"].split()
        return message(f"{prose}\n{json.dumps(dict(zip(keys, map(float, printed), strict=True)))}\nThat is all.", **kw)
    return reply


def rate_code(turn, eps=COUNTER) -> str:
    return RATE.format(C_h=turn["C_h"], C_c=turn["C_c"], UA=turn["UA"], eps=eps)


def good_script() -> list:
    """A model that gets every turn right through the tool."""
    t = eval_hx_entu.TURNS
    rate = ["q_W", "Th_out_C", "Tc_out_C"]
    return [tool(rate_code(t[0])), answer_from_tool(rate),
            tool(rate_code(t[1], PARALLEL)), answer_from_tool(rate),
            tool(rate_code(t[2])), answer_from_tool(rate),
            tool(SIZE), answer_from_tool(["UA_W_per_K"]),
            tool(CELLS), answer_from_tool(["Th_out_C"])]


class Scripted(http.server.ThreadingHTTPServer):
    """Plays one reply a POST, in order; a reply may be a function of the request body or an HTTP status."""

    def __init__(self, script):
        super().__init__(("127.0.0.1", 0), Handler)
        self.script, self.bodies = list(script), []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status: int, doc: dict) -> None:
        data = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send(200, {"object": "list", "data": [{"id": "scripted-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        step = self.server.script.pop(0) if self.server.script else 500
        if isinstance(step, int):
            return self.send(step, {"error": {"message": "out of memory"}})
        self.send(200, step(body) if callable(step) else step)


@contextlib.contextmanager
def serving(script):
    server = Scripted(script)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def box() -> eval_sandbox.Sandbox:
    return eval_sandbox.Sandbox.detect(unsafe=True, warn=lambda message: None, timeout_s=10)


class References(unittest.TestCase):
    def test_turns_match_the_closed_forms(self):
        refs = [eval_hx_entu.reference(t) for t in eval_hx_entu.TURNS]
        for key, value in {"q_W": 60000.0, "Th_out_C": 80.0, "Tc_out_C": 140.0}.items():
            self.assertAlmostEqual(refs[0][key], value, places=9)
        eps = (1 - math.exp(-4)) / 2
        self.assertAlmostEqual(refs[1]["q_W"], eps * 500 * 180, places=6)
        self.assertAlmostEqual(refs[1]["Th_out_C"], 200 - eps * 180, places=9)
        e = math.exp(-1)
        eps = (1 - e) / (1 - 0.5 * e)
        self.assertAlmostEqual(refs[2]["q_W"], eps * 500 * 180, places=6)
        self.assertAlmostEqual(refs[2]["Tc_out_C"], 20 + eps * 500 * 180 / 1000, places=9)
        self.assertAlmostEqual(refs[3]["UA_W_per_K"], 625.0, places=9)
        self.assertEqual(refs[4], {"Th_out_C": refs[0]["Th_out_C"]})

    def test_counterflow_forms_meet_at_cr_one_and_invert(self):
        limit = eval_hx_entu.effectiveness("counterflow", 2.0, 1.0)
        self.assertAlmostEqual(limit, 2 / 3, places=12)
        self.assertAlmostEqual(eval_hx_entu.effectiveness("counterflow", 2.0, 1 - 1e-7), limit, places=6)
        for cr in (0.0, 0.5, 1.0):
            eps = eval_hx_entu.effectiveness("counterflow", 1.7, cr)
            self.assertAlmostEqual(eval_hx_entu.counterflow_ntu(eps, cr), 1.7, places=9)

    def test_prompts_carry_the_numbers_and_no_hint_at_the_limit(self):
        for turn in eval_hx_entu.TURNS:
            text = eval_hx_entu.prompt(turn)
            self.assertIn("run_python", text)
            self.assertIn(f"C_c = {turn['C_c']:g} W/K", text)
            for hint in ("Cr", "limit", "0/0", "NTU/(1", "1 + NTU", "singular", "equal capacity"):
                self.assertNotIn(hint, text)


class Scoring(unittest.TestCase):
    def test_json_line_among_prose_fences_and_reasoning(self):
        read = eval_hx_entu.answer_json
        self.assertEqual(read('So q is 60 kW.\n{"q_W": 60000, "Th_out_C": 80}\nHope this helps!'),
                         {"q_W": 60000, "Th_out_C": 80})
        self.assertEqual(read('```json\n{"UA_W_per_K": 625.0}\n```'), {"UA_W_per_K": 625.0})
        self.assertEqual(read('Result: {"Th_out_C": 80.1} (from {N} cells)'), {"Th_out_C": 80.1})
        text = '<think>draft {"q_W": 1}</think>first {"q_W": 2}, then {"q_W": 3} and {"note": {"x": 1}}'
        self.assertEqual(read(text, ["q_W"]), {"q_W": 3})
        self.assertEqual(read(text), {"note": {"x": 1}})
        self.assertIsNone(read("no json here, only {braces"))

    def test_values_within_half_a_percent_pass(self):
        turn = eval_hx_entu.TURNS[0]
        ok = eval_hx_entu.score(turn, '{"q_W": 60200, "Th_out_C": 80.3, "Tc_out_C": "139.5"}')
        self.assertEqual((ok["status"], ok["reason"]), ("PASS", ""))
        bad = eval_hx_entu.score(turn, '{"q_W": 60000, "Th_out_C": 80.5, "Tc_out_C": 140}')
        self.assertEqual(bad["status"], "FAIL")
        self.assertIn("Th_out_C", bad["reason"])
        self.assertFalse(bad["balance_by_construction"])

    def test_missing_nan_and_absent_json_fail(self):
        turn = eval_hx_entu.TURNS[0]
        self.assertIn("q_W", eval_hx_entu.score(turn, '{"q_W": NaN, "Th_out_C": 80, "Tc_out_C": 140}')["reason"])
        self.assertIn("Tc_out_C", eval_hx_entu.score(turn, '{"q_W": 60000, "Th_out_C": 80}')["reason"])
        self.assertEqual(eval_hx_entu.score(turn, "60 kW")["reason"], "no JSON line in the final answer")

    def test_zero_duty_with_exact_balances_is_flagged(self):
        turn = eval_hx_entu.TURNS[0]
        zero = eval_hx_entu.score(turn, '{"q_W": 0.0, "Th_out_C": 200.0, "Tc_out_C": 20.0}')
        self.assertEqual(zero["status"], "FAIL")
        self.assertTrue(zero["balance_by_construction"])
        wrong = eval_hx_entu.score(turn, '{"q_W": 44175.8, "Th_out_C": 111.648, "Tc_out_C": 108.352}')
        self.assertTrue(wrong["balance_by_construction"])
        unbalanced = eval_hx_entu.score(turn, '{"q_W": 0.0, "Th_out_C": 80.0, "Tc_out_C": 140.0}')
        self.assertFalse(unbalanced["balance_by_construction"])

    def test_an_indexing_slip_in_the_cell_model_fails(self):
        turn = eval_hx_entu.TURNS[4]
        self.assertEqual(eval_hx_entu.score(turn, '{"Th_out_C": 80.2}')["status"], "PASS")
        self.assertEqual(eval_hx_entu.score(turn, '{"Th_out_C": 78.8}')["status"], "FAIL")


@unittest.skipUnless(UNIX, "model code runs only where rlimits exist")
class Conversation(unittest.TestCase):
    def converse(self, script, max_calls=8):
        with serving(script) as server:
            client = eval_hx_entu.Client(server.url, "scripted-model", {"temperature": 0.2})
            return eval_hx_entu.run_conversation(client, box(), max_calls), server.bodies

    def test_a_correct_model_passes_every_turn(self):
        run, bodies = self.converse(good_script())
        self.assertEqual([t["status"] for t in run["turns"]], ["PASS"] * 5, [t["reason"] for t in run["turns"]])
        self.assertEqual((run["passed"], run["turns_reached"]), (5, 5))
        self.assertEqual([(t["tool_calls"], t["well_formed"], t["exec_errors"]) for t in run["turns"]], [(1, 1, 0)] * 5)
        self.assertEqual(bodies[0]["tools"][0]["function"]["name"], "run_python")
        self.assertEqual(bodies[0]["temperature"], 0.2)
        self.assertEqual([m["role"] for m in bodies[1]["messages"]], ["system", "user", "assistant", "tool"])

    def test_the_zero_over_zero_formula_loop_and_a_lost_server(self):
        rate = ["q_W", "Th_out_C", "Tc_out_C"]
        turn = eval_hx_entu.TURNS[0]
        script = [tool(rate_code(turn, ZERO_OVER_ZERO)), tool(rate_code(turn, GUARDED)),
                  answer_from_tool(rate, prose="No heat moves; the energy balance checks out."),
                  tool("print(1)"), message(calls=["{not json"]), tool("print(2)"), 500]
        run, bodies = self.converse(script, max_calls=3)
        first, second, third = run["turns"]
        self.assertEqual((first["status"], first["balance_by_construction"]), ("FAIL", True))
        self.assertEqual(first["answer"], {"q_W": 0.0, "Th_out_C": 200.0, "Tc_out_C": 20.0})
        self.assertEqual((first["tool_calls"], first["well_formed"], first["exec_errors"]), (2, 2, 1))
        self.assertIn("ZeroDivisionError", first["executions"][0]["output"])
        self.assertEqual((second["status"], second["ended"]), ("FAIL", "tool-call limit"))
        self.assertEqual((second["tool_calls"], second["well_formed"]), (3, 2))
        self.assertIn("JSON arguments", bodies[5]["messages"][-1]["content"])
        self.assertEqual(third["status"], "ERROR")
        self.assertIn("HTTP 500", third["reason"])
        self.assertEqual((run["turns_reached"], run["passed"]), (2, 0))

    def test_calls_past_the_limit_are_answered_but_not_run(self):
        calls = [json.dumps({"code": f"print({i})"}) for i in range(4)]
        run, _ = self.converse([message(calls=calls), 500], max_calls=2)
        turn = run["turns"][0]
        self.assertEqual((turn["tool_calls"], turn["well_formed"], len(turn["executions"])), (4, 4, 2))
        self.assertEqual(turn["ended"], "tool-call limit")


@unittest.skipUnless(UNIX, "model code runs only where rlimits exist")
class CommandLine(unittest.TestCase):
    def main(self, script, *argv):
        with serving(script) as server, tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hx.json"
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = eval_hx_entu.main(["--base-url", server.url, "--out", str(out), *argv])
            result = json.loads(out.read_text()) if out.exists() else None
        return code, result, stdout.getvalue(), stderr.getvalue(), server.bodies

    def test_runs_repeat_and_report_the_pass_count(self):
        code, result, text, _, bodies = self.main(good_script() * 2, "--runs", "2", "--unsafe-no-sandbox")
        self.assertEqual(code, 0)
        self.assertEqual(result["model"], "scripted-model")
        self.assertEqual((len(result["runs"]), result["runs_passing_every_turn"]), (2, 2))
        self.assertIn("2/2 runs pass every turn", text)
        self.assertIn("1 counterflow, equal capacity rates", text)
        self.assertEqual(len(bodies), 20)

    def test_thinking_options_are_sent_and_thinking_tokens_recorded(self):
        script = [tool(rate_code(eval_hx_entu.TURNS[0]), reasoning="Plan: e-NTU.", thinking=7, completion=30),
                  message('{"q_W": 60000, "Th_out_C": 80, "Tc_out_C": 140}', thinking=5, completion=12), 500]
        code, result, text, _, bodies = self.main(script, "--model", "m", "--reasoning-effort", "low",
                                                  "--thinking-budget", "2048", "--seed", "9", "--unsafe-no-sandbox")
        self.assertEqual(code, 1)
        for body in bodies:
            self.assertEqual((body["model"], body["reasoning_effort"], body["thinking_budget"], body["seed"]),
                             ("m", "low", 2048, 9))
        self.assertEqual(bodies[1]["messages"][2]["reasoning_content"], "Plan: e-NTU.")
        turn = result["runs"][0]["turns"][0]
        self.assertEqual((turn["status"], turn["thinking_tokens"], turn["completion_tokens"]), ("PASS", 12, 42))
        self.assertIn("completion tokens 42, thinking tokens 12", text)

    def test_refuses_without_a_sandbox_and_sends_nothing(self):
        with patch.object(eval_sandbox.shutil, "which", return_value=None):
            code, result, _, err, bodies = self.main(good_script(), "--model", "m")
        self.assertEqual((code, result, bodies), (2, None, []))
        self.assertIn("refusing to run model code", err)


if __name__ == "__main__":
    unittest.main()
