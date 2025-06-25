import os
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CI") is None or os.name != "nt",
    reason="Multi-env smoke test runs only on Windows CI with GPU",
)


@pytest.fixture(scope="session")
def manager():
    """Return a ready MultiEnvironmentManager (setup done once)."""

    # Defer import until inside fixture to avoid heavy cost when skipped
    project_root = Path(__file__).resolve().parent.parent
    from src.utils.multi_env_manager import MultiEnvironmentManager  # noqa: WPS433

    mgr = MultiEnvironmentManager(project_root)
    assert mgr.setup(force_recreate=False), "failed to set up envs"
    assert mgr.health_check(), "env health check failed"
    return mgr


def test_detection_and_ocr_end_to_end(tmp_path, manager):  # noqa: WPS442
    """Process a minimal PDF (shipped with repo) and expect success."""
    sample_pdf = Path(__file__).parent / "assets" / "mini.pdf"
    if not sample_pdf.exists():
        pytest.skip("sample PDF not available")

    res = manager.run_complete_pipeline(
        pdf_path=sample_pdf,
        output_dir=tmp_path,
    )
    assert res["status"] == "success", res 