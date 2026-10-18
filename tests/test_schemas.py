import base64

import pytest
from pydantic import ValidationError

from app.api.schemas import AnalysisResult, JobResponse, to_public
from pipeline.run import analyze

JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 32).decode()


def viol(**over):
    v = {
        "type": "Тип",
        "timecode_sec": 1.5,
        "description": "Описание",
        "evidence_frames": [{"timestamp_sec": 1.5, "image_base64": JPEG_B64}],
    }
    v.update(over)
    return v


def test_valid_violation_result():
    r = AnalysisResult(verdict="violation", violations=[viol()])
    assert r.violations[0].timecode_sec == 1.5


def test_valid_no_violation():
    AnalysisResult(verdict="no_violation", violations=[])


@pytest.mark.parametrize("verdict,violations", [("violation", []), ("no_violation", [viol()])])
def test_verdict_must_match_violations(verdict, violations):
    with pytest.raises(ValidationError):
        AnalysisResult(verdict=verdict, violations=violations)


def test_frames_count_1_to_3():
    frame = {"timestamp_sec": 1.0, "image_base64": JPEG_B64}
    with pytest.raises(ValidationError):
        AnalysisResult(verdict="violation", violations=[viol(evidence_frames=[])])
    with pytest.raises(ValidationError):
        AnalysisResult(verdict="violation", violations=[viol(evidence_frames=[frame] * 4)])
    AnalysisResult(verdict="violation", violations=[viol(evidence_frames=[frame] * 3)])


def test_frame_rejects_data_prefix_and_non_jpeg():
    with pytest.raises(ValidationError):
        AnalysisResult(verdict="violation", violations=[viol(evidence_frames=[
            {"timestamp_sec": 1.0, "image_base64": "data:image/jpeg;base64," + JPEG_B64}])])
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode()
    with pytest.raises(ValidationError):
        AnalysisResult(verdict="violation", violations=[viol(evidence_frames=[
            {"timestamp_sec": 1.0, "image_base64": png}])])


def test_negative_timecode_rejected():
    with pytest.raises(ValidationError):
        AnalysisResult(verdict="violation", violations=[viol(timecode_sec=-1)])


def test_strict_schema_drops_extensions():
    resp = JobResponse(job_id="a" * 16, status="done", verdict="violation",
                       violations=[viol(rule="X", confidence=0.9)])
    loose = to_public(resp, strict=False)
    strict = to_public(resp, strict=True)
    assert loose["violations"][0]["rule"] == "X"
    assert set(strict["violations"][0]) == {"type", "timecode_sec", "description", "evidence_frames"}
    assert "error" not in strict


def test_job_response_consistency():
    with pytest.raises(ValidationError):
        JobResponse(job_id="a" * 16, status="failed")
    with pytest.raises(ValidationError):
        JobResponse(job_id="a" * 16, status="done")
    assert to_public(JobResponse(job_id="a" * 16, status="processing"), False) == {
        "job_id": "a" * 16, "status": "processing"}


def test_mock_pipeline_output_is_valid(tmp_path):
    video = tmp_path / "v.mov"
    video.write_bytes(b"\x00" * 16)
    AnalysisResult.model_validate(analyze(str(video)))
