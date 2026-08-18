import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif pendant les tests

import pytest

from att_analysis import (
    candidate_end_reperes,
    candidate_start_reperes,
    extract_steps,
    repere_frequency_table,
    transition_counts,
)


# ---------------------------------------------------------------------------
# extract_steps
# ---------------------------------------------------------------------------
class TestExtractSteps:
    def test_detects_correct_number_of_steps(self, att_sample_steps):
        assert len(att_sample_steps) == 9

    def test_renames_columns_to_att_vocabulary(self, att_sample_steps):
        assert "repere" in att_sample_steps.columns
        assert "step_id" in att_sample_steps.columns
        assert "value" not in att_sample_steps.columns
        assert "operation_id" not in att_sample_steps.columns

    def test_repere_sequence_is_correct(self, att_sample_steps):
        assert att_sample_steps["repere"].tolist() == [0, 100, 101, 102, 0, 100, 101, 102, 0]

    def test_durations_match_expected_plateaus(self, att_sample_steps):
        expected_minutes = [3, 2, 3, 2, 3, 2, 3, 2, 2]
        assert att_sample_steps["duration_min"].tolist() == pytest.approx(expected_minutes)


# ---------------------------------------------------------------------------
# repere_frequency_table
# ---------------------------------------------------------------------------
class TestRepereFrequencyTable:
    def test_one_row_per_distinct_repere(self, att_sample_steps):
        table = repere_frequency_table(att_sample_steps)
        assert set(table["repere"]) == {0, 100, 101, 102}
        assert len(table) == 4

    def test_counts_and_total_duration_per_repere(self, att_sample_steps):
        table = repere_frequency_table(att_sample_steps).set_index("repere")
        assert table.loc[0, "n"] == 3
        assert table.loc[0, "total_duration_min"] == pytest.approx(8)
        assert table.loc[100, "n"] == 2
        assert table.loc[100, "total_duration_min"] == pytest.approx(4)
        assert table.loc[101, "n"] == 2
        assert table.loc[101, "total_duration_min"] == pytest.approx(6)
        assert table.loc[102, "n"] == 2
        assert table.loc[102, "total_duration_min"] == pytest.approx(4)

    def test_sorted_by_total_duration_descending(self, att_sample_steps):
        table = repere_frequency_table(att_sample_steps)
        assert table["total_duration_min"].is_monotonic_decreasing
        assert table.iloc[0]["repere"] == 0  # le plus gros total (8 min)

    def test_cumulative_pct_reaches_100(self, att_sample_steps):
        table = repere_frequency_table(att_sample_steps)
        assert table["cum_pct_of_total_time"].iloc[-1] == pytest.approx(100)


# ---------------------------------------------------------------------------
# transition_counts
# ---------------------------------------------------------------------------
class TestTransitionCounts:
    def test_one_transition_per_distinct_repere_pair(self, att_sample_transitions):
        pairs = set(zip(att_sample_transitions["from_repere"], att_sample_transitions["to_repere"]))
        assert pairs == {(0, 100), (100, 101), (101, 102), (102, 0)}

    def test_counts_reflect_two_occurrences_each(self, att_sample_transitions):
        assert (att_sample_transitions["n"] == 2).all()

    def test_probability_is_one_when_deterministic_sequence(self, att_sample_transitions):
        assert att_sample_transitions["probability"].tolist() == pytest.approx([1.0] * 4)

    def test_last_step_produces_no_dangling_transition(self, att_sample_transitions):
        # 9 pas -> 8 paires (le dernier pas n'a pas de successeur observé)
        assert att_sample_transitions["n"].sum() == 8


# ---------------------------------------------------------------------------
# candidate_start_reperes / candidate_end_reperes
# ---------------------------------------------------------------------------
class TestCandidateStartEndReperes:
    def test_start_candidate_is_the_repere_following_idle(self, att_sample_transitions):
        starts = candidate_start_reperes(att_sample_transitions, idle_value=0)
        assert starts["repere"].tolist() == [100]
        assert starts["n"].iloc[0] == 2

    def test_end_candidate_is_the_repere_preceding_idle(self, att_sample_transitions):
        ends = candidate_end_reperes(att_sample_transitions, idle_value=0)
        assert ends["repere"].tolist() == [102]
        assert ends["n"].iloc[0] == 2

    def test_no_candidates_for_unused_idle_value(self, att_sample_transitions):
        starts = candidate_start_reperes(att_sample_transitions, idle_value=-1)
        assert len(starts) == 0
