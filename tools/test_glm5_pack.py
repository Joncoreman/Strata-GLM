"""CPU packing contract tests on tiny synthetic tensors, never on real model payloads."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py
add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, quants
import iq_pack


def fixture(root):
    """Small geometry deliberately unlike the real artifact, so metadata drives the plan."""
    source = root / "glm.gguf"
    writer = GGUFWriter(source, "glm5-next")
    for key, value in {"block_count": 5, "nextn_predict_layers": 1, "embedding_length": 32,
                       "leading_dense_block_count": 3, "expert_count": 4,
                       "expert_feed_forward_length": 64}.items():
        writer.add_uint32("glm5-next." + key, value)
    # F32 router deliberately cannot be represented exactly as BF16.
    router = np.full((4, 32), 0.10001, dtype=np.float32)
    tensors = [(f"blk.{i}.ffn_gate_inp.weight", router, Q.F32) for i in (3, 4)]
    for i in (3, 4):
        for role in ("gate", "up", "down"):
            dims = (4, 32, 64) if role == "down" else (4, 64, 32)
            tensors.append((f"blk.{i}.ffn_{role}_exps.weight", np.ones(dims, np.float32), Q.Q8_0))
    tensors += [("blk.0.ssm_beta.weight", np.ones((4, 32), np.float32), Q.Q8_0),
                ("blk.0.hc_attn_fn.weight", np.ones((24, 128), np.float32), Q.Q8_0),
                ("blk.0.ssm_conv1d_q.weight", np.ones((128, 1, 4), np.float32), Q.F32),
                ("blk.3.attn_k_b.weight", np.ones((4, 64, 32), np.float32), Q.Q8_0),
                ("blk.3.indexer.proj.weight", router, Q.F32),
                ("blk.0.ffn_gate.weight", np.ones((64, 32), np.float32), Q.Q8_0)]
    for name, values, kind in tensors:
        writer.add_tensor(name, quants.quantize(values, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return source


class Glm5PackingTests(unittest.TestCase):
    def test_architecture_specific_forms(self):
        self.assertEqual(iq_pack.form_of("blk.3.ffn_gate_inp.weight"), "BF16")
        self.assertEqual(iq_pack.form_of("blk.3.ffn_gate_inp.weight", "glm5-next"), "F32")
        self.assertTrue(iq_pack.needs_bf16("blk.0.ssm_beta.weight", "Q8_0"))
        self.assertFalse(iq_pack.needs_bf16("blk.0.ssm_beta.weight", "Q8_0", "glm5-next"))

    def test_float_router_is_byte_exact_and_q8_projections_remain_native(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = fixture(root)
            model = iq_pack.Model(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, model), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            dense = (root / "dense.bin").read_bytes()
            for name in ("blk.3.ffn_gate_inp.weight", "blk.4.ffn_gate_inp.weight", "blk.3.indexer.proj.weight"):
                row = rows[name]
                self.assertEqual(row[2], "2")
                at, count = int(row[3]), int(row[4])
                self.assertEqual(dense[at:at + count], model.bytes(name).tobytes())
            for name in ("blk.0.ssm_beta.weight", "blk.0.hc_attn_fn.weight", "blk.3.attn_k_b.weight"):
                self.assertEqual(rows[name][2:5], ["0", "0", "0"])
            self.assertEqual(json.loads((root / "conversions.json").read_text())["tensors"], [])

    def test_head_and_convolution_axes_are_preserved_in_sidecar(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = fixture(root)
            model = iq_pack.Model(source)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(source, root, model), 0)
            _, rows = iq_pack.read_index(root / "index.txt")
            axes = json.loads((root / "glm5-layout.json").read_text())
            self.assertFalse(axes["runtime_ready"])
            self.assertEqual(axes["tensor_axes"]["blk.3.attn_k_b.weight"]["shape"], [32, 64, 4])
            self.assertEqual(rows["blk.3.attn_k_b.weight"][7:9], ["32", "256"])
            self.assertEqual(axes["tensor_axes"]["blk.0.ssm_conv1d_q.weight"]["shape"], [4, 1, 128])
            self.assertEqual(rows["blk.0.ssm_conv1d_q.weight"][7:9], ["4", "128"])

    def test_dense_layers_skipped_mtp_included_and_offsets_contiguous(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = fixture(Path(tmp))
            model = iq_pack.Model(source, header_only=True)
            layout, text, n_expert, size = iq_pack.expert_layout(model, source)
            self.assertEqual([row[0] for row in layout], [3, 4])
            self.assertEqual(n_expert, 4)
            self.assertEqual(layout[0][3], 0)
            self.assertEqual(layout[1][3], layout[0][4] * n_expert)
            self.assertEqual(size, sum(row[4] * n_expert for row in layout))
            self.assertTrue(text.startswith("# strata native experts v5:"))

    def test_missing_expert_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = fixture(Path(tmp))
            model = iq_pack.Model(source, header_only=True)
            del model.where["blk.4.ffn_up_exps.weight"]
            self.assertIn("missing", iq_pack.expert_layout(model, source))

    def test_wrong_expert_orientation_is_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = fixture(Path(tmp))
            model = iq_pack.Model(source, header_only=True)
            model.where["blk.3.ffn_down_exps.weight"][1].shape = [32, 64, 4]
            self.assertIn("geometry disagrees", iq_pack.expert_layout(model, source))

    def test_download_marker_rejected_before_any_header_is_opened(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = fixture(Path(tmp))
            Path(str(source) + ".aria2").touch()
            with patch.object(iq_pack.G, "GGUFFile", side_effect=AssertionError("opened download")):
                with self.assertRaisesRegex(ValueError, "download still active"):
                    iq_pack.Model(source, header_only=True)

    def test_dry_run_never_maps_reads_payloads_or_creates_output(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            source = fixture(root)
            before = source.read_bytes()
            out = root / "must-not-exist"
            log = io.StringIO()
            argv = ["iq_pack.py", "--gguf", str(source), "--dry-run", "--out", str(out), "--experts-bin"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(log), \
                 patch.object(iq_pack.np, "memmap", side_effect=AssertionError("mapped payload")), \
                 patch.object(iq_pack, "tensor_bytes", side_effect=AssertionError("read payload")), \
                 patch.object(iq_pack, "convert", side_effect=AssertionError("converted payload")), \
                 patch.object(iq_pack.subprocess, "run", side_effect=AssertionError("ran tokenizer")):
                self.assertEqual(iq_pack.main(), 0)
            self.assertFalse(out.exists())
            self.assertEqual(before, source.read_bytes())
            self.assertIn("F32 -> F32", log.getvalue())
            self.assertIn("Q8_0 -> Q8_0", log.getvalue())
            self.assertIn("payload_reads=0 files_written=0", log.getvalue())
            self.assertIn("expert layers: [3, 4]", log.getvalue())

    def test_dry_run_out_is_optional(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = fixture(Path(tmp))
            with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(source), "--dry-run"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.main(), 0)

    def test_header_only_model_refuses_payload_access(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = fixture(Path(tmp))
            model = iq_pack.Model(source, header_only=True)
            with self.assertRaisesRegex(RuntimeError, "header-only"):
                model.bytes("blk.3.ffn_gate_inp.weight")


if __name__ == "__main__":
    unittest.main()
