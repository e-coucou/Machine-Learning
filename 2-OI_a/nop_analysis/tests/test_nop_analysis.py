import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif pendant les tests

import numpy as np
import pandas as pd
import pytest

from nop_analysis import (
    add_score,
    analyze_best_runs,
    build_sliding_windows,
    clean_dataframe,
    compute_control_limits,
    compute_monthly_cv,
    compute_monthly_scores,
    compute_monthly_zscore,
    compute_zscore,
    extract_operations,
    filter_operations_by_duration,
    flag_out_of_control,
    get_reference_duration,
    golden_run,
    golden_run_dashboard,
)


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------
class TestCleanDataframe:
    def test_removes_nan_and_negative_values(self, sample_df_with_noise):
        cleaned = clean_dataframe(sample_df_with_noise)
        assert cleaned["value_nop"].isna().sum() == 0
        assert (cleaned["value_nop"] >= 0).all()
        # 1 NaN + 1 négatif retirés sur 30 lignes
        assert len(cleaned) == 28

    def test_casts_to_int64(self, sample_df):
        cleaned = clean_dataframe(sample_df)
        assert cleaned["value_nop"].dtype == np.int64

    def test_sorts_chronologically(self, sample_df):
        shuffled = sample_df.sample(frac=1, random_state=0)
        cleaned = clean_dataframe(shuffled)
        assert cleaned.index.is_monotonic_increasing

    def test_raises_without_datetime_index(self):
        df = pd.DataFrame({"value_nop": [1, 2, 3]})
        with pytest.raises(TypeError):
            clean_dataframe(df)

    def test_raises_without_expected_column(self, sample_df):
        with pytest.raises(KeyError):
            clean_dataframe(sample_df, value_col="does_not_exist")


# ---------------------------------------------------------------------------
# extract_operations
# ---------------------------------------------------------------------------
class TestExtractOperations:
    def test_detects_correct_number_of_operations(self, sample_ops):
        assert len(sample_ops) == 5

    def test_values_are_correct(self, sample_ops):
        assert sample_ops["value"].tolist() == [100, 101, 102, 103, 104]

    def test_durations_match_expected_plateaus(self, sample_ops, sample_index):
        expected_minutes = [5, 5, 10, 5, 4]  # cf. docstring de conftest.py
        actual_minutes = sample_ops["duration_min"].tolist()
        assert actual_minutes == pytest.approx(expected_minutes)

    def test_n_samples_match_plateau_sizes(self, sample_ops):
        assert sample_ops["n_samples"].tolist() == [5, 5, 10, 5, 5]

    def test_last_operation_ends_at_last_timestamp(self, sample_ops, sample_index):
        assert sample_ops["end"].iloc[-1] == sample_index[-1]

    def test_operation_ids_are_sequential(self, sample_ops):
        assert sample_ops["operation_id"].tolist() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# filter_operations_by_duration
# ---------------------------------------------------------------------------
class TestFilterOperationsByDuration:
    def test_filters_out_short_and_long_operations(self, sample_ops):
        filtered = filter_operations_by_duration(
            sample_ops,
            min_duration=pd.Timedelta(minutes=5),
            max_duration=pd.Timedelta(minutes=9),
            verbose=False,
        )
        # seules les opérations de 5 min passent le filtre (10 et 4 exclues)
        assert filtered["duration_min"].tolist() == pytest.approx([5, 5, 5])

    def test_default_std_filter_0h_24h_keeps_everything_reasonable(self, sample_ops):
        filtered = filter_operations_by_duration(sample_ops, verbose=False)
        assert len(filtered) == len(sample_ops)


# ---------------------------------------------------------------------------
# analyze_best_runs / get_reference_duration
# ---------------------------------------------------------------------------
class TestAnalyzeBestRuns:
    def test_finds_the_minimal_window_by_hand_computation(self, sample_ops):
        # durations (min) = [5, 5, 10, 5, 4] ; fenêtre 3 :
        # [5+5+10, 5+10+5, 10+5+4] = [20, 20, 19] -> minimum = 19 (dernière fenêtre)
        results = analyze_best_runs(sample_ops, window_sizes=(3,))
        r = results[3]
        assert r["total_duration_min"] == pytest.approx(19)
        assert r["start_index"] == 2
        assert r["values_sequence"] == [102, 103, 104]

    def test_skips_window_sizes_larger_than_available_data(self, sample_ops):
        results = analyze_best_runs(sample_ops, window_sizes=(3, 100))
        assert 100 not in results
        assert 3 in results

    def test_capacity_per_day_is_consistent_with_duration(self, sample_ops):
        results = analyze_best_runs(sample_ops, window_sizes=(3,))
        r = results[3]
        expected_capacity = 3 / r["total_duration_min"] * 1440
        assert r["capacity_per_day"] == pytest.approx(expected_capacity)

    def test_reference_duration_matches_best_run(self, sample_ops):
        results = analyze_best_runs(sample_ops, window_sizes=(3,))
        ref = get_reference_duration(sample_ops, window=3)
        assert ref == pytest.approx(results[3]["total_duration_min"])

    def test_reference_duration_raises_when_not_enough_operations(self, sample_ops):
        with pytest.raises(ValueError):
            get_reference_duration(sample_ops, window=100)


# ---------------------------------------------------------------------------
# build_sliding_windows / add_score
# ---------------------------------------------------------------------------
class TestBuildSlidingWindows:
    def test_number_of_windows(self, sample_ops):
        windows = build_sliding_windows(sample_ops, window=3)
        assert len(windows) == len(sample_ops) - 3 + 1

    def test_window_durations_match_manual_computation(self, sample_ops):
        windows = build_sliding_windows(sample_ops, window=3)
        assert windows["duration"].tolist() == pytest.approx([20, 20, 19])

    def test_raises_when_window_exceeds_available_operations(self, sample_ops):
        with pytest.raises(ValueError):
            build_sliding_windows(sample_ops, window=100)


class TestAddScore:
    def test_score_of_reference_window_is_one(self, sample_ops):
        windows = build_sliding_windows(sample_ops, window=3)
        ref = windows["duration"].min()
        scored = add_score(windows, ref)
        assert scored["score"].max() == pytest.approx(1.0)

    def test_month_column_is_added(self, sample_ops):
        windows = build_sliding_windows(sample_ops, window=3)
        scored = add_score(windows, ref_duration=19)
        assert "month" in scored.columns


# ---------------------------------------------------------------------------
# compute_monthly_scores / compute_zscore / compute_monthly_zscore
# ---------------------------------------------------------------------------
class TestMonthlyScoring:
    def test_monthly_scores_columns(self, larger_ops):
        monthly = compute_monthly_scores(larger_ops, window=9)
        expected_cols = {"month", "mean_score", "median_score", "best_score", "worst_score", "n_sequences"}
        assert expected_cols.issubset(monthly.columns)

    def test_best_score_never_exceeds_reference(self, larger_ops):
        # la référence est la meilleure fenêtre observée -> aucun score > 1
        monthly = compute_monthly_scores(larger_ops, window=9)
        assert (monthly["best_score"] <= 1.0 + 1e-9).all()

    def test_zscore_has_zero_mean_globally(self, larger_ops):
        z = compute_zscore(larger_ops, window=9)
        assert abs(z["z_score"].mean()) < 1e-9

    def test_monthly_zscore_columns(self, larger_ops):
        z = compute_zscore(larger_ops, window=9)
        monthly_z = compute_monthly_zscore(z)
        expected_cols = {"month", "mean_z", "median_z", "std_z", "best_z", "worst_z", "n_runs"}
        assert expected_cols.issubset(monthly_z.columns)


# ---------------------------------------------------------------------------
# golden_run / golden_run_dashboard
# ---------------------------------------------------------------------------
class TestGoldenRun:
    def test_default_window_is_35(self):
        import nop_analysis.config as config
        assert config.GOLDEN_RUN_WINDOW == 35

    def test_golden_run_uses_requested_window(self, sample_ops):
        golden = golden_run(sample_ops, window=3)
        assert golden["window"] == 3
        assert golden["result"]["total_duration_min"] == pytest.approx(19)

    def test_golden_run_raises_when_not_enough_operations(self, sample_ops):
        with pytest.raises(ValueError):
            golden_run(sample_ops, window=100)

    def test_dashboard_is_decoupled_from_comparison_windows(self, larger_ops):
        """
        La fenêtre du golden run (9) ne fait pas partie de comparison_window_sizes :
        golden_run_dashboard doit quand même fonctionner en l'ajoutant automatiquement
        à la comparaison, sans lever d'exception.
        """
        golden = golden_run(larger_ops, window=9)
        fig = golden_run_dashboard(larger_ops, golden, comparison_window_sizes=(5, 15))
        assert fig is not None

    def test_dashboard_default_comparison_windows(self, larger_ops):
        golden = golden_run(larger_ops, window=9)
        fig = golden_run_dashboard(larger_ops, golden)
        assert fig is not None


# ---------------------------------------------------------------------------
# build_summary_report
# ---------------------------------------------------------------------------
class TestBuildSummaryReport:
    def test_returns_string_with_key_figures(self, larger_ops):
        from nop_analysis import build_summary_report

        results = analyze_best_runs(larger_ops, window_sizes=(9, 15))
        golden = golden_run(larger_ops, window=9)
        monthly_scores = compute_monthly_scores(larger_ops, window=9)
        z = compute_zscore(larger_ops, window=9)
        monthly_zscore = compute_monthly_zscore(z)

        report = build_summary_report(larger_ops, results, golden, monthly_scores, monthly_zscore)

        assert isinstance(report, str)
        assert "Golden run" in report
        assert "Tendance mensuelle" in report
        assert f"{len(larger_ops):,}" in report

    def test_handles_single_month_gracefully(self, sample_ops):
        """Avec un seul mois de données, la tendance doit être signalée comme non déterminable, pas planter."""
        from nop_analysis import build_summary_report

        results = analyze_best_runs(sample_ops, window_sizes=(3,))
        golden = golden_run(sample_ops, window=3)
        monthly_scores = compute_monthly_scores(sample_ops, window=3)
        z = compute_zscore(sample_ops, window=3)
        monthly_zscore = compute_monthly_zscore(z)

        report = build_summary_report(sample_ops, results, golden, monthly_scores, monthly_zscore)
        assert "non déterminable" in report


# ---------------------------------------------------------------------------
# compute_monthly_cv
# ---------------------------------------------------------------------------
class TestComputeMonthlyCV:
    def test_columns_present(self, larger_ops):
        monthly_cv = compute_monthly_cv(larger_ops)
        expected_cols = {"month", "mean_duration_min", "std_duration_min", "n_operations", "cv"}
        assert expected_cols.issubset(monthly_cv.columns)

    def test_cv_is_non_negative(self, larger_ops):
        monthly_cv = compute_monthly_cv(larger_ops)
        assert (monthly_cv["cv"].dropna() >= 0).all()

    def test_matches_manual_computation_on_crafted_data(self, sample_ops):
        # sample_ops tient sur un seul mois (janvier) : cv global attendu = std/mean sur les 5 durées
        monthly_cv = compute_monthly_cv(sample_ops)
        assert len(monthly_cv) == 1
        expected_cv = sample_ops["duration_min"].std() / sample_ops["duration_min"].mean()
        assert monthly_cv["cv"].iloc[0] == pytest.approx(expected_cv)


# ---------------------------------------------------------------------------
# SPC : compute_control_limits / flag_out_of_control
# ---------------------------------------------------------------------------
class TestControlChart:
    def test_limits_are_symmetric_around_center(self, larger_ops):
        limits = compute_control_limits(larger_ops)
        assert limits["ucl"] - limits["center"] == pytest.approx(3 * limits["sigma_hat"])
        # LCL peut être tronquée à 0 si center - 3*sigma < 0, donc pas toujours symétrique exactement
        assert limits["lcl"] >= 0

    def test_center_equals_mean(self, larger_ops):
        limits = compute_control_limits(larger_ops)
        assert limits["center"] == pytest.approx(larger_ops["duration_min"].mean())

    def test_raises_with_less_than_two_operations(self):
        one_row = pd.DataFrame({"duration_min": [100.0]})
        with pytest.raises(ValueError):
            compute_control_limits(one_row)

    def test_flag_out_of_control_detects_extreme_point(self):
        # 20 valeurs stables autour de 100, une valeur extrême à 10000
        durations = [100.0] * 20 + [10000.0]
        starts = pd.date_range("2026-01-01", periods=21, freq="D", tz="UTC")
        ops = pd.DataFrame({"start": starts, "duration_min": durations})

        limits = compute_control_limits(ops)
        flagged = flag_out_of_control(ops, limits)

        assert flagged["out_of_control"].sum() >= 1
        assert flagged.loc[flagged["duration_min"] == 10000.0, "out_of_control"].all()

    def test_flag_out_of_control_no_false_positive_on_stable_data(self):
        rng = np.random.default_rng(0)
        durations = rng.normal(100, 5, 200)  # process stable, faible bruit
        starts = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
        ops = pd.DataFrame({"start": starts, "duration_min": durations})

        limits = compute_control_limits(ops)
        flagged = flag_out_of_control(ops, limits)

        # avec un process stable, très peu de points doivent sortir des limites ±3σ
        assert flagged["out_of_control"].mean() < 0.05
