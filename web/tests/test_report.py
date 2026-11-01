import json
import re
from pathlib import Path

from tensorfold_web import report

DOCS = Path(__file__).parent / "fixtures"          # real receipts: Nemotron on the Mac mini M6, 8 Oct 2026


def _run(name: str) -> dict:
    return json.loads((DOCS / f"2026-10-08-nemotron-m6-{name}.json").read_text())


def test_findings_say_only_what_the_profile_shows():
    found = report.findings(_run("profile-power"))
    assert found[0].startswith("All 14 replies came out identical with drafts off")
    assert any(f.startswith("Drafts made writing 41% faster") for f in found)
    assert any(f.startswith("Fastest on Code") for f in found)
    assert any("for the whole machine" in f for f in found)
    # the receipt of 8 October has no throttling, cost or thinking record: nothing is said about them
    assert not any("slowdown" in f or "slowed" in f or "cost" in f or "Thinking" in f for f in found)


def test_findings_of_a_sweep():
    found = report.findings(_run("prefill"))
    assert found == ["Reading prompts: 1,372 tokens per second at 1k, 945 at 128k.",
                     "A 128k-token prompt waits 2 min 19 s before its first token."]


def test_findings_warn_when_things_went_wrong():
    run = _run("profile-power")
    run["status"] = "cancelled"
    run["thinking"] = {"sent": "x", "off": False, "refused": []}
    run["hardware"]["throttle"] = {"reported": True, "throttled": True, "seconds": 12.0, "reasons": ["power cap"]}
    run["hardware"]["coverage"] = 0.6
    run["hardware"]["cost"] = {"price_per_kwh": 0.25, "currency": "EUR", "run": 0.0002, "per_million_tokens": 0.024}
    run["summary"]["overall"]["exact"] = 13
    found = report.findings(run)
    assert found[0].startswith("This run was stopped before the end")
    assert any(f.startswith("Thinking stayed on") for f in found)
    assert any(f.startswith("1 of 14 replies differed") for f in found)
    assert any("slowed itself down for about 12 s (power cap)" in f for f in found)
    assert any("cover 60%" in f for f in found)
    assert any("a million tokens cost 0.024 €" in f for f in found)


def test_share_text_is_short_and_names_the_protocol():
    text = report.share_text(_run("profile-power"))
    lines = text.splitlines()
    assert len(lines) == 5 and lines[0].endswith("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit")
    assert "102 tok/s · 72 without drafts (+41%) · 14/14 identical" in lines[2]
    assert lines[-1].startswith("temperature 0, warm-up, thinking asked off")      # a receipt before the check
    checked = _run("profile-power")
    checked["thinking"] = {"sent": "x", "off": False, "refused": []}
    assert "thinking on" in report.share_text(checked).splitlines()[-1]             # what the run found, not the plan
    assert report._secs(119.7) == "2 min 00 s" and report._secs(59.4) == "59.4 s"


def test_the_html_report_stands_alone_and_carries_its_receipt():
    run = _run("profile-power")
    run["results"][0]["text"] = "</script><script>alert(1)</script>"     # a reply cannot break out of the page
    page = report.html_report(run)
    assert page.startswith("<!doctype html>") and "What this run shows" in page
    assert not re.search(r'(src|href)="https?://', page)                  # nothing fetched from the network
    assert "<script>alert(1)</script>" not in page
    data = page.split('<script type="application/json" id="receipt">', 1)[1].rsplit("</script>", 1)[0]
    receipt = json.loads(data)
    assert receipt["kind"] == "tensorfold-studio-bench-receipt" and receipt["id"] == run["id"]
    assert receipt["results"][0]["text"].startswith("</script>")


def test_a_sweep_report_has_its_chart():
    page = report.html_report(_run("prefill"))
    assert 'aria-label="Prefill speed by prompt size"' in page and "128k" in page


def test_large_gains_read_as_multiples():
    run = _run("profile-power")
    o = run["summary"]["overall"]
    o["tps"], o["tps_no_drafts"], o["gain"] = 32.0, 10.0, 2.2
    assert any(f.startswith("Drafts made writing 3.2× as fast: 32 tokens per second") for f in report.findings(run))
