from pathlib import Path

from deep_orderbook.btc_experiment_config import (
    choose_training_device,
    resolve_train_test_files,
    summarize_dataset_scale,
)


def test_resolve_train_test_files_prefers_explicit_lists(tmp_path: Path) -> None:
    train = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    test = [tmp_path / "c.parquet"]
    for p in [*train, *test]:
        p.write_text("x")

    got_train, got_test = resolve_train_test_files(
        data_dir=tmp_path,
        explicit_train_files=train,
        explicit_test_files=test,
        train_count=3,
        test_count=1,
    )

    assert got_train == train
    assert got_test == test



def test_summarize_dataset_scale_reports_timesteps_and_caps() -> None:
    summary = summarize_dataset_scale(
        train_files=[Path("a.parquet"), Path("b.parquet")],
        test_files=[Path("c.parquet")],
        train_windows_before_filter=120,
        train_windows_after_filter=48,
        test_windows_before_filter=40,
        test_windows_after_filter=16,
        rolling_window_size=256,
        target_levels=8,
        n_samples_per_file=512,
    )

    assert summary["train_file_count"] == 2
    assert summary["test_file_count"] == 1
    assert summary["train_timesteps"] == 48 * 256
    assert summary["test_target_pixels"] == 16 * 256 * 8
    assert summary["max_windows_per_file"] == 512  # dict key is stable for artifact compat



def test_choose_training_device_prefers_cuda_when_available() -> None:
    assert str(choose_training_device(prefer_cuda=True, cuda_available=True)) == "cuda"
    assert str(choose_training_device(prefer_cuda=True, cuda_available=False)) == "cpu"
    assert str(choose_training_device(prefer_cuda=False, cuda_available=True)) == "cpu"
