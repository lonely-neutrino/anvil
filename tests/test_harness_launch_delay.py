"""Worker-launch ramp selection for the batch harness."""

from anvil.bridge.harness.orchestrator import launch_delay_seconds


def test_split_delay_ramps_initial_fleet_then_replacements():
    manifest = {
        "launch_delay_ms": 4000,
        "replacement_launch_delay_ms": 500,
    }

    assert [launch_delay_seconds(manifest, n, 3) for n in range(6)] == [4.0, 4.0, 4.0, 0.5, 0.5, 0.5]


def test_legacy_manifest_keeps_delay_for_replacements():
    manifest = {"launch_delay_ms": 2000}

    assert launch_delay_seconds(manifest, 0, 2) == 2.0
    assert launch_delay_seconds(manifest, 2, 2) == 2.0


def test_negative_delays_are_clamped():
    manifest = {
        "launch_delay_ms": -100,
        "replacement_launch_delay_ms": -200,
    }

    assert launch_delay_seconds(manifest, 0, 2) == 0.0
    assert launch_delay_seconds(manifest, 2, 2) == 0.0
