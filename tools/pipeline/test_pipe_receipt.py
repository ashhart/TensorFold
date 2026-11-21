"""Host tests for pipe_receipt.py: parsers, token comparison, the table and the peer's command line; no GPU, no ssh."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe_receipt as pr  # noqa: E402

IBV = """hca_id:	rocep1s0f0
	transport:			InfiniBand (0)
	fw_ver:				28.45.1020
		port:	1
			state:			PORT_ACTIVE (4)
			max_mtu:		4096 (5)
			active_mtu:		4096 (5)
			phys_state:		LINK_UP (5)
			active_width:		4X (2)
			active_speed:		50.0 Gbps (64)
			link_layer:		Ethernet
hca_id:	roceP2p1s0f1
		port:	1
			state:			PORT_DOWN (1)
			link_layer:		Ethernet
"""


def u32(*xs):
    return b"".join(x.to_bytes(4, "little") for x in xs)


class Parsers(unittest.TestCase):
    def test_ibv_ports(self):
        ports = pr.parse_ibv(IBV)
        self.assertEqual(2, len(ports))
        self.assertEqual("rocep1s0f0", ports[0]["device"])
        self.assertEqual("PORT_ACTIVE (4)", ports[0]["state"])
        self.assertEqual("50.0 Gbps (64)", ports[0]["active_speed"])
        self.assertEqual("4X (2)", ports[0]["active_width"])
        self.assertEqual("PORT_DOWN (1)", ports[1]["state"])
        self.assertNotIn("active_speed", ports[1])

    def test_meminfo_and_cuda(self):
        m = pr.parse_meminfo("MemTotal:       127535008 kB\nMemAvailable:   119000000 kB\n")
        self.assertAlmostEqual(121.63, m["memtotal_gib"], places=1)
        self.assertIn("memavailable_gib", m)
        banner = "| NVIDIA-SMI 580.95   Driver Version: 580.95   CUDA Version: 13.0 |"
        self.assertEqual("13.0", pr.parse_smi_cuda(banner))
        self.assertEqual("13.4", pr.parse_smi_cuda("| NVIDIA-SMI 615.71.09   CUDA UMD Version: 13.4 |"))
        self.assertIsNone(pr.parse_smi_cuda("no banner"))

    def test_iface_of(self):
        ip = [{"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}]},
              {"ifname": "enp1s0f0np0", "addr_info": [{"local": "link-a.example"}]}]
        self.assertEqual("enp1s0f0np0", pr.iface_of("link-a.example", ip))
        self.assertIsNone(pr.iface_of("link-c.example", ip))

    def test_pipe_line(self):
        text = "stage 0/2 layers\npipe rank 0: {\"streams\":4,\"k\":2,\"tok_s\":51.25}\nstage 0 rank 0: 9 GiB\n"
        self.assertEqual({"streams": 4, "k": 2, "tok_s": 51.25}, pr.parse_pipe(text))
        self.assertEqual(4, pr.parse_pipe("pipe whole: {\"streams\":4}")["streams"])
        self.assertIsNone(pr.parse_pipe("error: OtherEcho"))


class Tokens(unittest.TestCase):
    def test_compare(self):
        ref = u32(1, 2, 3, 4, 5, 6)
        self.assertEqual({"equal": True}, pr.compare(ref, ref, 2, 3))
        bad = u32(1, 2, 3, 4, 9, 6)
        self.assertEqual({"equal": False, "stream": 1, "token": 1}, pr.compare(ref, bad, 2, 3))
        self.assertFalse(pr.compare(ref, ref[:8], 2, 3)["equal"])

    def test_solo_prefix(self):
        self.assertTrue(pr.prefix_equal(u32(7, 8), u32(7, 8, 1, 2), 2))
        self.assertFalse(pr.prefix_equal(u32(7, 9), u32(7, 8, 1, 2), 2))


class Report(unittest.TestCase):
    def test_table_ratio_against_the_best_box(self):
        md = pr.table([{"streams": 4, "whole_a": 50.0, "whole_b": 52.0, "ring_k1": 40.0, "ring_k2": 78.0}])
        self.assertIn("| 4 | 50.0 | 52.0 | 40.0 | 78.0 | 1.50x |", md)
        md = pr.table([{"streams": 8, "whole_a": 90.0, "whole_b": None, "ring_k1": None, "ring_k2": 80.0}])
        self.assertIn("| 8 | 90.0 | - | - | 80.0 | 0.89x |", md)
        self.assertIn("two stages, one box, k = 2", pr.table([], "two stages, one box"))

    def test_verdicts(self):
        self.assertEqual("a=True, b=False", pr.verdicts({"a": {"equal": True}, "b": {"equal": False}, "pass": False}))


class Commands(unittest.TestCase):
    def runner(self, **kw):
        a = types.SimpleNamespace(out="/tmp/x", prompt=512, addr="link-a.example", peer_addr="link-b.example",
                                  port=29600, loopback=False, peer_secret="/tmp/tf-pipe-secret", layout="0:32,32:64",
                                  model="/m/Qwen3.8-27B-MLX-4bit", kv="bf16", peer_root="/r/TensorFold", **kw)
        return pr.Runner(a)

    def test_ring_env_names_both_link_addresses(self):
        r = self.runner()
        e0, e1 = r.ring_env(0), r.ring_env(1)
        self.assertEqual("link-a.example:29600,link-b.example:29601", e0["TF_STAGE_PEERS"])
        self.assertEqual(("0", "1"), (e0["TF_STAGE_RANK"], e1["TF_STAGE_RANK"]))
        self.assertEqual("/tmp/x/secret", e0["TF_STAGE_SECRET_FILE"])
        self.assertEqual("/tmp/tf-pipe-secret", e1["TF_STAGE_SECRET_FILE"])

    def test_remote_command_quotes_and_never_carries_the_key(self):
        r = self.runner()
        env = {**r.env({"streams": 8, "k": 2, "tokens": 128}), **r.ring_env(1)}
        cmd = r.remote(env, r.argv("/r/bin/tf qwen", "/dev/null"))
        self.assertTrue(cmd.startswith("cd /r/TensorFold && env "))
        self.assertIn("TF_PIPE_K=2", cmd)
        self.assertIn("'/r/bin/tf qwen' /m/Qwen3.8-27B-MLX-4bit pipe /dev/null", cmd)
        self.assertNotIn("secret\"", cmd)
        self.assertIn("TF_STAGE_SECRET_FILE=/tmp/tf-pipe-secret", cmd)

    def test_fp8_argument(self):
        r = self.runner()
        r.a.kv = "fp8"
        self.assertEqual("fp8", r.argv("b", "o")[-1])


if __name__ == "__main__":
    unittest.main()
