import os
import tempfile
import yaml
from typing import Any, Dict, List, Tuple
from wikipedia.models.baseline import HeuristicBaselineModel


class TestOptimization:
    def test_continuous_stub_penalty(self) -> None:
        """Test that the stub penalty is continuous and behaves as expected."""
        model = HeuristicBaselineModel()

        # severe stub
        penalty_severe = model._calculate_continuous_stub_penalty(100, 1)
        assert 0.5 <= penalty_severe < 0.6

        # developing stub
        penalty_developing = model._calculate_continuous_stub_penalty(800, 5)
        assert penalty_severe < penalty_developing < 0.9

        # good article
        penalty_good = model._calculate_continuous_stub_penalty(5000, 15)
        assert penalty_good > 0.95
        assert penalty_good <= 1.0

        # detailed continuity check
        penalties = []
        for length in range(100, 3000, 100):
            penalties.append(model._calculate_continuous_stub_penalty(length, 6))

        # Should be strictly increasing for length (with fixed sections)
        assert all(x < y for x, y in zip(penalties, penalties[1:]))

    def test_calibration_optimization(self) -> None:
        """Test that calibrate_weights runs and produces valid output."""
        # Use a temporary file to avoid overwriting production weights
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as tmp:
            tmp_path = tmp.name
            # Initialize with default weights structure to avoid NoneType error
            yaml.dump(HeuristicBaselineModel()._get_default_weights(), tmp)

        # Mock training data: list of (article_data, target_score)
        # We need enough data points to compute correlation
        training_data: List[Tuple[Dict[str, Any], float]] = []
        for i in range(10):
            # Create dummy article data that would produce different scores
            article_data = {
                "data": {
                    "parse": {
                        "sections": [{"level": "2"}] * (i + 2),
                        "text": {"*": "word " * (i * 100 + 500)},
                    }
                }
            }
            # Target score roughly correlated with size
            target_score = 50 + i * 5
            training_data.append((article_data, target_score))

        try:
            model = HeuristicBaselineModel(weights_file=tmp_path)
            result = model.calibrate_weights(training_data)

            assert result["optimization_success"] is not None

            weights = result["calibrated_weights"]["pillars"]
            weight_sum = sum(weights.values())

            # Check if weights sum to ~1.0
            assert abs(weight_sum - 1.0) < 0.001

            # Check bounds
            for w in weights.values():
                assert 0.05 <= w <= 0.6
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
