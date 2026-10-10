from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_live_demo_runs_and_hides_stale_estimates():
    path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any(metric.label == "IQAE estimate" for metric in app.metric)
    assert any("95% confidence interval" in message.value for message in app.info)
    for seed in [18, 19]:
        app.number_input[0].set_value(seed).run(timeout=30)
        app.button[0].click().run(timeout=60)
        assert not app.exception
        assert any(metric.label == "IQAE estimate" for metric in app.metric)
    app.slider[0].set_value(11).run(timeout=30)
    assert not app.exception
    assert not any(metric.label == "IQAE estimate" for metric in app.metric)
