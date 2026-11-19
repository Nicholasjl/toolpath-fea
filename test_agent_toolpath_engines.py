import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def _write_sample_gcode(path: Path):
    lines = ["G21", "G90", "M83", "; LAYER_HEIGHT: 0.2", "; LINE_WIDTH: 0.4"]
    for lay in range(3):
        z = 0.2 * (lay + 1)
        lines.append(f"; CHANGE_LAYER\nG1 Z{z:.2f} F600")
        lines.append("; FEATURE: Outer wall")
        lines.extend([
            "G0 X0.0 Y0.0 F3000",
            "G1 X8.0 Y0.0 E0.4 F1800",
            "G1 X8.0 Y4.0 E0.2",
            "G1 X0.0 Y4.0 E0.4",
            "G1 X0.0 Y0.0 E0.2",
        ])
        lines.append("; FEATURE: Sparse infill")
        lines.append("G0 X0.2 Y2.0\nG1 X7.8 Y2.0 E0.3")
    path.write_text("\n".join(lines), encoding="utf-8")


class AgentToolpathEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.gcode = self.folder / "part.gcode"
        _write_sample_gcode(self.gcode)

    def _run_cli(self, case_dict, *extra_args, name="case"):
        case_p = self.folder / f"{name}.json"
        out_prefix = self.folder / f"{name}_out"
        case_p.write_text(json.dumps(case_dict), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "run3d.py"), str(case_p), "--output", str(out_prefix), *extra_args],
            capture_output=True,
            encoding="utf-8",
            timeout=120,
        )
        manifest = json.loads(proc.stdout)
        self.assertEqual(proc.returncode, manifest["exit_code"], proc.stderr)
        self.assertEqual(manifest, json.loads((self.folder / f"{name}_out.json").read_text(encoding="utf-8")))
        return manifest, proc

    def test_auto_selects_toolpath_graph_when_gcode_present_and_supports_cpu(self):
        manifest, proc = self._run_cli(
            {
                "gcode": "part.gcode",
                "fixed": "X-",
                "loaded": "X+",
                "force": [0.0, 0.0, -0.5],
                "seam_config": {"seam_stiffness_factor": 0.4, "thermal_decay_tau_s": 30.0},
                "strength_config": {"sigma_beam_allow_MPa": 40.0},
                "spill_stress": True,
            },
            "--device",
            "cpu",
            name="auto_graph",
        )
        self.assertEqual(manifest["status"], "completed", proc.stderr)
        analysis = manifest["analysis"]
        self.assertEqual(analysis["engine"], "toolpath_graph")
        self.assertEqual(analysis["requested_engine"], "auto")
        self.assertEqual(analysis["quality"], "needs_review")
        self.assertEqual(analysis["metrics"]["solver"]["device"], "cpu")
        eq = next(c for c in analysis["checks"] if c["code"] == "EQUILIBRIUM")
        self.assertEqual(eq["status"], "pass")
        self.assertIn("safety_audit", analysis["metrics"])
        for fname in ("preview.png", "report.html", "analysis.json", "model.tgraph", "graph_stress.bin"):
            self.assertIn(fname, manifest["artifacts"])
            self.assertGreater(Path(manifest["artifacts"][fname]).stat().st_size, 0)

    def test_explicit_toolpath_hex_and_weakened_interfaces(self):
        manifest, proc = self._run_cli(
            {
                "gcode": "part.gcode",
                "engine": "toolpath_hex",
                "toolpath_pitch_mm": 0.2,
                "hex_interface_config": {"weakened": True},
                "fixed": "X-",
                "loaded": "X+",
                "force": [0.0, 0.0, -0.5],
            },
            name="hex_weak",
        )
        self.assertEqual(manifest["status"], "completed", proc.stderr)
        analysis = manifest["analysis"]
        self.assertEqual(analysis["engine"], "toolpath_hex")
        eq = next(c for c in analysis["checks"] if c["code"] == "EQUILIBRIUM")
        self.assertEqual(eq["status"], "pass")
        hex_chk = next(c for c in analysis["checks"] if c["code"] == "HEX_INTERFACE_AUDIT")
        self.assertEqual(hex_chk["metrics"]["flags_version"], 2)
        for fname in ("preview.png", "report.html", "analysis.json", "model.tbricks"):
            self.assertIn(fname, manifest["artifacts"])
            self.assertGreater(Path(manifest["artifacts"][fname]).stat().st_size, 0)

    def test_explicit_legacy_solid_fallback_with_gcode(self):
        manifest, proc = self._run_cli(
            {
                "size": [12, 8, 8],
                "pitch": 4,
                "gcode": "part.gcode",
                "fixed": "X-",
                "loaded": "X+",
                "force": [0.0, 0.0, -1.0],
            },
            "--engine",
            "legacy_solid",
            "--no-convergence",
            name="legacy_explicit",
        )
        self.assertEqual(manifest["status"], "completed", proc.stderr)
        self.assertEqual(manifest["analysis"]["engine"], "legacy_solid")
        self.assertEqual(manifest["analysis"]["requested_engine"], "legacy_solid")
        for fname in ("result.vtk", "result.npz", "preview.png", "report.html", "analysis.json"):
            self.assertIn(fname, manifest["artifacts"])

    def test_toolpath_engine_without_gcode_rejected_with_invalid_case(self):
        manifest, proc = self._run_cli(
            {"size": [10, 4, 4]},
            "--engine",
            "toolpath_graph",
            name="bad_missing_gcode",
        )
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(manifest["error"]["code"], "INVALID_CASE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
