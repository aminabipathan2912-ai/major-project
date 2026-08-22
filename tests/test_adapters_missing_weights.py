from pathlib import Path

from cctv_ai.inference.accident.model_adapter import AccidentModelAdapter
from cctv_ai.inference.violence.model_adapter import ViolenceModelAdapter


def test_adapters_report_missing_weights(tmp_path: Path):
    missing = str(tmp_path / "nope.pt")
    accident = AccidentModelAdapter(weights_path=missing)
    violence = ViolenceModelAdapter(weights_path=missing)
    assert accident.status().loaded is False
    assert violence.status().loaded is False
    assert "not found" in accident.status().reason
