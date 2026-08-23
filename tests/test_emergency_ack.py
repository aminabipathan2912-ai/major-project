from cctv_ai.emergency.messaging import build_emergency_message, is_acknowledgement
from cctv_ai.core.models import VerifiedEvent


def test_done_is_acknowledgement():
    assert is_acknowledgement("Done")
    assert is_acknowledgement("yes done")
    assert is_acknowledgement("it is reported")
    assert not is_acknowledgement("")
    assert not is_acknowledgement("hello")


def test_emergency_message_mentions_done():
    event = VerifiedEvent(
        event_type="ACCIDENT",
        verified_label="ACCIDENT",
        confidence=0.9,
        timestamp_epoch_s=1_700_000_000,
        camera_id="camera-1",
    )
    text = build_emergency_message(event=event, location="Main Road")
    assert "Main Road" in text
    assert "camera-1" in text
    assert "say done" in text.lower()
