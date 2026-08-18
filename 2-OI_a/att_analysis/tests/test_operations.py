import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif pendant les tests

import numpy as np
import pandas as pd
import pytest

from att_analysis import (
    best_chaining_window,
    build_operations_network,
    build_reference,
    clean_dataframe,
    compare_operation_to_reference,
    cycle_times,
    defaut_frequency_table,
    exclude_boundary_operations,
    extract_enchainements,
    label_with_pas,
    load_pas_reference,
    load_reference,
    pas_duration_stats,
    reconstruct_operations,
    reference_cycle_time,
    save_reference,
)


# ---------------------------------------------------------------------------
# clean_dataframe (spécifique Att)
# ---------------------------------------------------------------------------
class TestCleanDataframe:
    def test_excludes_invalid_repere_values(self):
        index = pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame({"att": [0, 100, 0, 110, 200]}, index=index)
        cleaned = clean_dataframe(df, value_col="att")
        assert 0 not in cleaned["att"].values
        assert len(cleaned) == 3

    def test_custom_invalid_values(self):
        index = pd.date_range("2026-01-01", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame({"att": [0, 999, 100]}, index=index)
        cleaned = clean_dataframe(df, value_col="att", invalid_values=(999,))
        assert set(cleaned["att"]) == {0, 100}


# ---------------------------------------------------------------------------
# load_pas_reference
# ---------------------------------------------------------------------------
class TestLoadPasReference:
    def test_loads_and_sorts_by_pas_num(self, tmp_path):
        csv = tmp_path / "ref.csv"
        csv.write_text(
            "pas_num,phase,code_court,description,att_min,att_max,type_attente,type_attente_detail,operation_liee,sens_liaison\n"
            "2,PREP,PREP,Preparation,200,250,PROCESS,Verification,,\n"
            "1,ATTENTE,ATTENTE,Attente,100,110,OPERATEUR,Confirmation,,\n"
            "3,DEF,DEF,Defaut,,,OPERATEUR,Relance,,\n"
        )
        ref = load_pas_reference(str(csv))
        assert ref["pas_num"].tolist() == [1, 2, 3]
        assert np.isnan(ref.loc[ref["pas_num"] == 3, "att_min"].iloc[0])


# ---------------------------------------------------------------------------
# label_with_pas
# ---------------------------------------------------------------------------
class TestLabelWithPas:
    def test_maps_repere_to_correct_pas(self, att_pas_reference):
        steps = pd.DataFrame({"repere": [100, 110, 220, 300]})
        labeled = label_with_pas(steps, att_pas_reference)
        assert labeled["pas_num"].tolist() == [1, 1, 2, 3]
        assert labeled["phase"].tolist() == ["ATTENTE", "ATTENTE", "PREP", "PROD"]

    def test_repere_below_first_threshold_is_unmapped(self, att_pas_reference):
        steps = pd.DataFrame({"repere": [50]})
        labeled = label_with_pas(steps, att_pas_reference)
        assert np.isnan(labeled["pas_num"].iloc[0])

    def test_pas_without_known_range_is_never_assigned(self, att_pas_reference):
        # Le pas 4 (défaut) n'a pas de plage connue : aucun repere ne doit s'y mapper.
        steps = pd.DataFrame({"repere": [100, 220, 300, 1000]})
        labeled = label_with_pas(steps, att_pas_reference)
        assert 4 not in labeled["pas_num"].dropna().astype(int).tolist()

    def test_type_attente_is_carried_through(self, att_pas_reference):
        steps = pd.DataFrame({"repere": [100, 220, 300]})
        labeled = label_with_pas(steps, att_pas_reference)
        assert labeled["type_attente"].tolist() == ["OPERATEUR", "PROCESS", "CHAUFFE"]


# ---------------------------------------------------------------------------
# reconstruct_operations
# ---------------------------------------------------------------------------
class TestReconstructOperations:
    def test_detects_two_operations(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert len(operations) == 2

    def test_operation_sizes(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert operations["n_pas"].tolist() == [9, 4]

    def test_defaut_detected_within_first_operation_only(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert operations["n_defauts"].tolist() == [1, 0]

    def test_max_pas_reached(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert operations["max_pas_reached"].tolist() == [3, 3]

    def test_operation_id_assigned_to_every_step(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert steps["operation_id"].tolist() == [0] * 9 + [1] * 4

    def test_is_defaut_flagged_on_the_correct_row(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        # Repere 210 (index 6 de la séquence brute) est le seul défaut.
        assert steps["is_defaut"].tolist() == [False] * 6 + [True] + [False] * 6

    def test_operation_start_and_end_timestamps(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        first_op_steps = steps[steps["operation_id"] == 0]
        assert operations.loc[0, "start"] == first_op_steps["start"].iloc[0]
        assert operations.loc[0, "end"] == first_op_steps["end"].iloc[-1]

    def test_standard_duration_excludes_rework(self, att_labeled_steps_two_ops, att_pas_reference):
        # Op 0 : pas_num = [1,1,2,2,2,3,2,2,3], défaut à l'index 6 (repère
        # 210, destination pas2). Un pas est du rework ssi une régression
        # FUTURE ramène à un niveau <= le sien : le défaut (destination=2)
        # invalide donc tous les pas de niveau <= 2 qui le précèdent
        # (indices 2,3,4 = pas2, et 5 = pas3, puisque 2 <= 3 aussi) mais PAS
        # les pas1 (indices 0,1, niveau 1 < 2, jamais réinvalidés). Les
        # derniers pas2/pas3 (indices 7,8, après le défaut, plus jamais
        # rattrapés par une régression) comptent comme standard.
        # -> standard = pas1(20) + pas2 final(10) + pas3 final(10) = 40 min.
        # Op 1 : pas_num = [1,1,2,3], aucun défaut -> tout compte (40 min).
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert operations["standard_duration_min"].tolist() == pytest.approx([40.0, 40.0])

    def test_rework_duration_is_total_minus_standard(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert operations["duration_min"].tolist() == pytest.approx([90.0, 40.0])
        assert operations["rework_duration_min"].tolist() == pytest.approx([50.0, 0.0])

    def test_is_progress_flagged_per_step(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        expected = [True, True, False, False, False, False, False, True, True,
                    True, True, True, True]
        assert steps["is_progress"].tolist() == expected


# ---------------------------------------------------------------------------
# reconstruct_operations — redémarrages chaînés (2e génération, détection
# par valeur de repère plutôt que par seuil de pas_num, cf. OPERATION.md §7)
# ---------------------------------------------------------------------------
class TestReconstructOperationsChained:
    def test_detects_each_chained_restart_without_any_configuration(
        self, att_labeled_steps_chained_ops, att_pas_reference
    ):
        # Aucun paramètre à régler : chaque redémarrage revient à la valeur
        # de départ de l'opération en cours (200), donc chacun est reconnu
        # comme un nouveau départ, quel que soit le pas sur lequel le
        # chaînage automatique atterrit.
        operations, steps = reconstruct_operations(att_labeled_steps_chained_ops, att_pas_reference)
        assert len(operations) == 3
        assert steps["is_defaut"].sum() == 0

    def test_chained_restarts_operation_sizes(self, att_labeled_steps_chained_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_chained_ops, att_pas_reference)
        assert operations["n_pas"].tolist() == [3, 2, 3]

    def test_defaut_landing_below_start_value_is_not_confused_with_a_restart(
        self, att_labeled_steps_two_ops, att_pas_reference
    ):
        # Le défaut de att_labeled_steps_two_ops revient au pas 2 (repère
        # 210), une valeur largement supérieure à la valeur de départ de
        # l'opération (100) : contrairement à l'ancien modèle basé sur un
        # seuil de pas_num, il n'est jamais confondu avec un redémarrage.
        operations, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert len(operations) == 2
        assert steps["is_defaut"].sum() == 1

    def test_hold_pas_with_own_repere_is_not_confused_with_a_restart(
        self, att_labeled_steps_hold_defaut_returns_to_widened_range, att_pas_reference_with_hold
    ):
        # Le retour depuis HOLD atterrit au pas2 (repère 220), supérieur à
        # la valeur de départ de l'opération (100) : reconnu comme défaut,
        # pas comme nouveau départ.
        operations, steps = reconstruct_operations(
            att_labeled_steps_hold_defaut_returns_to_widened_range, att_pas_reference_with_hold
        )
        assert len(operations) == 1
        assert steps["is_defaut"].sum() == 1

    def test_rare_low_value_restart_does_not_block_the_dominant_one(
        self, att_labeled_steps_rare_low_restart, att_pas_reference
    ):
        # Une 1ere version de find_restart_reference (mediane des valeurs
        # distinctes) isolait le redemarrage rare (pas 1, 1 occurrence) tout
        # seul comme "cluster bas" et l'utilisait a tort comme reference,
        # bloquant la reconnaissance des redemarrages chaines (pas 2, 3
        # occurrences) -- cf. OPERATION.md §7. Corrige par un filtre de
        # position (pas_num tot dans la sequence), pas par la valeur seule :
        # les 4 operations doivent toutes etre detectees.
        operations, steps = reconstruct_operations(att_labeled_steps_rare_low_restart, att_pas_reference)
        assert len(operations) == 4
        assert steps["is_defaut"].sum() == 0

    def test_dominant_internal_loop_does_not_win_over_the_early_restart(
        self, att_labeled_steps_dominant_internal_loop, att_pas_reference_with_late_loop
    ):
        # La boucle interne (pas 4<-5) est plus frequente (3x) que le vrai
        # redemarrage (pas 1, 1x par operation) mais toujours tardive dans
        # la sequence documentee -- reproduit OP3120. Le filtre de position
        # doit l'exclure du calcul de reference quelle que soit sa
        # frequence : seules 2 operations, la boucle reste un defaut.
        operations, steps = reconstruct_operations(
            att_labeled_steps_dominant_internal_loop, att_pas_reference_with_late_loop
        )
        assert len(operations) == 2
        assert steps["is_defaut"].sum() == 3

    def test_large_pas_num_jump_is_a_restart_even_when_value_does_not_match(
        self, att_labeled_steps_large_jump_restart, att_pas_reference_long_sequence
    ):
        # Un defaut ne peut renvoyer que vers un pas proche de celui qui
        # vient d'echouer -- un saut de pas21 a pas4 (17 pas en arriere sur
        # une etendue de 20) ne peut structurellement pas etre un defaut,
        # meme si sa valeur de repere (400) ne correspond pas a la
        # reference dominante (100, pas 1). Reproduit le cas signale en
        # pratique (retour pas21->pas4 classe a tort en defaut).
        operations, steps = reconstruct_operations(
            att_labeled_steps_large_jump_restart, att_pas_reference_long_sequence
        )
        assert len(operations) == 4
        assert steps["is_defaut"].sum() == 0

    def test_standard_duration_credits_the_final_successful_attempt(
        self, att_labeled_steps_fails_twice_then_succeeds, att_pas_reference_nine_steps
    ):
        # Reproduit OP1410/EGOUTTAGE : un pas echoue 2 fois (retry via la
        # boucle pas6->7->8) puis reussit a la 3e tentative. Les 2
        # tentatives ratees (indices 5-7 et 8-10, plus les 2 pas de defaut
        # eux-memes aux indices 8 et 11) doivent etre exclues du temps
        # standard ; la montee initiale propre (indices 0-4, pas1-5) ET la
        # tentative finale reussie (indices 12-14, pas6-7-8 puis pas9)
        # doivent compter -- meme si le pas de defaut (index11) ne fait que
        # RATTRAPER, sans le depasser, le niveau deja atteint (pas6).
        operations, steps = reconstruct_operations(
            att_labeled_steps_fails_twice_then_succeeds, att_pas_reference_nine_steps
        )
        assert len(operations) == 1
        assert steps["is_defaut"].sum() == 2

        expected_progress = [True] * 5 + [False] * 7 + [True] * 3
        assert steps["is_progress"].tolist() == expected_progress
        assert operations["standard_duration_min"].iloc[0] == pytest.approx(80.0)
        assert operations["rework_duration_min"].iloc[0] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# exclude_boundary_operations
# ---------------------------------------------------------------------------
class TestExcludeBoundaryOperations:
    @pytest.fixture
    def operations_with_truncated_ends(self):
        # 4 operations : la 1ere (52 min, un fragment tronque impossible
        # pour ce process) et la derniere (30 min, cycle probablement en
        # cours) encadrent 2 operations completes a ~500 min.
        starts = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
        return pd.DataFrame({
            "operation_id": [0, 1, 2, 3],
            "start": starts,
            "duration_min": [52.0, 505.0, 498.0, 30.0],
            "standard_duration_min": [52.0, 480.0, 475.0, 30.0],
        })

    def test_drops_first_and_last_by_start_order(self, operations_with_truncated_ends):
        result = exclude_boundary_operations(operations_with_truncated_ends)
        assert result["operation_id"].tolist() == [1, 2]

    def test_min_duration_excludes_truncated_fragment(self, operations_with_truncated_ends):
        result = exclude_boundary_operations(operations_with_truncated_ends)
        assert result["duration_min"].min() == pytest.approx(498.0)

    def test_returns_unchanged_below_three_operations(self):
        two_ops = pd.DataFrame({
            "operation_id": [0, 1],
            "start": pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC"),
            "duration_min": [52.0, 500.0],
        })
        result = exclude_boundary_operations(two_ops)
        assert len(result) == 2

    def test_sorts_by_start_before_trimming(self):
        # Meme si operations n'est pas deja triee par start, il faut
        # retirer la 1ere et la derniere chronologiquement, pas par index.
        shuffled = pd.DataFrame({
            "operation_id": [2, 0, 1],
            "start": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"], utc=True),
            "duration_min": [498.0, 52.0, 500.0],
        })
        result = exclude_boundary_operations(shuffled)
        assert result["operation_id"].tolist() == [1]


# ---------------------------------------------------------------------------
# defaut_frequency_table
# ---------------------------------------------------------------------------
class TestDefautFrequencyTable:
    def test_attributes_defaut_to_preceding_pas(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        table = defaut_frequency_table(steps)
        # Le défaut suit un pas PROD (pas_num=3) qui vient de se terminer.
        assert table["pas_num"].tolist() == [3]
        assert table["n_defauts"].tolist() == [1]


# ---------------------------------------------------------------------------
# cycle_times
# ---------------------------------------------------------------------------
class TestCycleTimes:
    @pytest.fixture
    def steps_with_anchor(self):
        t0 = pd.Timestamp("2026-01-01", tz="UTC")
        rows = [
            (110, t0, 5.0),
            (200, t0 + pd.Timedelta(minutes=5), 50.0),
            (110, t0 + pd.Timedelta(minutes=55), 8.0),
            (300, t0 + pd.Timedelta(minutes=63), 100.0),
            (110, t0 + pd.Timedelta(minutes=163), 3.0),
        ]
        return pd.DataFrame(rows, columns=["repere", "start", "duration_min"])

    def test_keeps_only_anchor_occurrences(self, steps_with_anchor):
        result = cycle_times(steps_with_anchor, anchor_repere=110)
        assert len(result) == 3

    def test_attente_lancement_is_the_anchor_plateau_duration(self, steps_with_anchor):
        result = cycle_times(steps_with_anchor, anchor_repere=110)
        assert result["attente_lancement_min"].tolist() == pytest.approx([5.0, 8.0, 3.0])

    def test_cycle_time_is_gap_between_consecutive_launches(self, steps_with_anchor):
        result = cycle_times(steps_with_anchor, anchor_repere=110)
        assert result["cycle_time_min"].iloc[0] == pytest.approx(55.0)
        assert result["cycle_time_min"].iloc[1] == pytest.approx(108.0)

    def test_last_cycle_time_is_nan(self, steps_with_anchor):
        result = cycle_times(steps_with_anchor, anchor_repere=110)
        assert np.isnan(result["cycle_time_min"].iloc[-1])


# ---------------------------------------------------------------------------
# best_chaining_window
# ---------------------------------------------------------------------------
class TestBestChainingWindow:
    @pytest.fixture
    def cycle_times_df(self):
        n = 8
        starts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        attente = [10.0, 5.0, 3.0, 20.0, 2.0, 4.0, 1.0, 50.0]
        return pd.DataFrame({
            "start": starts,
            "attente_lancement_min": attente,
            "cycle_time_min": [np.nan] * n,
        })

    def test_finds_the_cheapest_window_by_hand_computation(self, cycle_times_df):
        # Fenetre de 3 la moins chere : indices 4-6 -> [2, 4, 1] -> total 7 -> moyenne 7/3.
        result = best_chaining_window(cycle_times_df, window=3)
        assert result == pytest.approx(7 / 3)

    def test_window_of_one_is_the_global_minimum(self, cycle_times_df):
        result = best_chaining_window(cycle_times_df, window=1)
        assert result == pytest.approx(1.0)

    def test_returns_nan_when_anchor_never_observed(self):
        # anchor_repere absent des donnees -> cycle_times() produit un
        # DataFrame vide ; ne doit pas lever, doit retourner NaN.
        empty = pd.DataFrame({
            "start": pd.to_datetime([]),
            "attente_lancement_min": pd.Series([], dtype=float),
            "cycle_time_min": pd.Series([], dtype=float),
        })
        result = best_chaining_window(empty, window=5)
        assert np.isnan(result)

    def test_returns_nan_when_fewer_anchors_than_window(self, cycle_times_df):
        result = best_chaining_window(cycle_times_df.iloc[:3], window=5)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# pas_duration_stats
# ---------------------------------------------------------------------------
class TestPasDurationStats:
    def test_one_row_per_pas_with_correct_counts(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        stats = pas_duration_stats(steps, att_pas_reference)
        counts = stats.set_index("pas_num")["n"]
        assert counts.loc[1] == 4  # 100,110,100,110
        assert counts.loc[2] == 6  # 200,250,220,210,250,200
        assert counts.loc[3] == 3  # 300,300,300

    def test_labels_merged_from_reference(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        stats = pas_duration_stats(steps, att_pas_reference).set_index("pas_num")
        assert stats.loc[1, "code_court"] == "ATTENTE"
        assert stats.loc[3, "code_court"] == "PROD"

    def test_type_attente_merged_from_reference(self, att_labeled_steps_two_ops, att_pas_reference):
        _, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        stats = pas_duration_stats(steps, att_pas_reference).set_index("pas_num")
        assert stats.loc[1, "type_attente"] == "OPERATEUR"
        assert stats.loc[3, "type_attente"] == "CHAUFFE"


# ---------------------------------------------------------------------------
# reference_cycle_time
# ---------------------------------------------------------------------------
class TestReferenceCycleTime:
    @pytest.fixture
    def labeled_varying_durations(self, att_pas_reference):
        # pas1 vu 2x (durees 5 et 30), pas2 vu 2x (20 et 3), pas3 vu 2x (8 et 12).
        # Independant de toute reconstruction d'operation (pas de
        # reconstruct_operations ici).
        reperes = [100, 200, 300, 110, 250, 300]
        durations = [5.0, 20.0, 8.0, 30.0, 3.0, 12.0]
        n = len(reperes)
        starts = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        ends = starts + pd.to_timedelta(durations, unit="min")
        steps = pd.DataFrame({
            "step_id": range(n), "repere": reperes, "start": starts, "end": ends,
            "duration": ends - starts, "duration_min": durations, "n_samples": 1,
        })
        return label_with_pas(steps, att_pas_reference)

    def test_percentile_zero_matches_the_raw_minimum(self, labeled_varying_durations, att_pas_reference):
        # percentile=0 (le minimum brut) : pas1=5, pas2=3, pas3=8 -> 16.
        # Statistique extreme (cf. docstring) -- garde pour reference/tests,
        # pas la valeur par defaut.
        result = reference_cycle_time(labeled_varying_durations, att_pas_reference, percentile=0.0)
        assert result["total_min"] == pytest.approx(16.0)
        assert result["n_pas_reels"] == 3

    def test_default_percentile_is_p10(self, labeled_varying_durations, att_pas_reference):
        # P10 (interpolation lineaire pandas) sur 2 valeurs par pas :
        # pas1 [5,30] -> 7.5 ; pas2 [3,20] -> 4.7 ; pas3 [8,12] -> 8.4.
        result = reference_cycle_time(labeled_varying_durations, att_pas_reference)
        assert result["total_min"] == pytest.approx(20.6)

    def test_par_pas_breakdown_matches_individual_percentiles(self, labeled_varying_durations, att_pas_reference):
        result = reference_cycle_time(labeled_varying_durations, att_pas_reference, percentile=0.0)
        by_pas = {row["pas_num"]: row["duration_min"] for row in result["par_pas"]}
        assert by_pas == {1: pytest.approx(5.0), 2: pytest.approx(3.0), 3: pytest.approx(8.0)}

    def test_defaut_pas_excluded_even_if_never_observed(self, labeled_varying_durations, att_pas_reference):
        # pas4 (DEF, sans plage de repere connue) n'est jamais mappe -- ne
        # doit pas apparaitre dans le detail ni fausser la somme.
        result = reference_cycle_time(labeled_varying_durations, att_pas_reference)
        assert 4 not in {row["pas_num"] for row in result["par_pas"]}

    def test_does_not_require_operation_reconstruction(self, labeled_varying_durations, att_pas_reference):
        # reference_cycle_time fonctionne directement sur label_with_pas --
        # pas besoin d'appeler reconstruct_operations au prealable.
        result = reference_cycle_time(labeled_varying_durations, att_pas_reference)
        assert result["total_min"] > 0


# ---------------------------------------------------------------------------
# extract_enchainements
# ---------------------------------------------------------------------------
class TestExtractEnchainements:
    def test_only_linked_pas_are_kept(self, att_pas_reference):
        enchainements = extract_enchainements(att_pas_reference)
        assert len(enchainements) == 1
        assert enchainements[0]["pas_num"] == 3

    def test_fields_are_carried_through(self, att_pas_reference):
        enchainements = extract_enchainements(att_pas_reference)
        link = enchainements[0]
        assert link["operation_liee"] == "OP_TEST"
        assert link["sens_liaison"] == "ATTEND"
        assert link["code_court"] == "PROD"
        assert link["detail"] == "Attente temperature"

    def test_no_links_returns_empty_list(self, att_pas_reference):
        no_links = att_pas_reference.assign(operation_liee=np.nan, sens_liaison=np.nan)
        assert extract_enchainements(no_links) == []


# ---------------------------------------------------------------------------
# build_reference / save_reference / load_reference
# ---------------------------------------------------------------------------
class TestReferenceBundle:
    @pytest.fixture
    def reference(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        defauts = defaut_frequency_table(steps)
        ct = cycle_times(steps, anchor_repere=110)
        return build_reference(
            att_pas_reference, operations, steps, defauts, ct,
            tag="TEST_TAG", anchor_repere=110, chaining_window=2,
        )

    def test_top_level_fields(self, reference):
        assert reference["tag"] == "TEST_TAG"
        assert reference["n_operations"] == 2
        assert reference["defauts"]["total"] == 1
        assert reference["defauts"]["n_operations_with_defaut"] == 1

    def test_catalog_metadata_defaults_to_empty(self, reference):
        assert reference["description"] == ""
        assert reference["operation"] == ""
        assert reference["produit"] == ""

    def test_catalog_metadata_passed_through(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        defauts = defaut_frequency_table(steps)
        ct = cycle_times(steps, anchor_repere=110)
        reference = build_reference(
            att_pas_reference, operations, steps, defauts, ct,
            tag="TEST_TAG", anchor_repere=110, chaining_window=2,
            description="Essai", operation="OP_TEST", produit="Produit X",
        )
        assert reference["description"] == "Essai"
        assert reference["operation"] == "OP_TEST"
        assert reference["produit"] == "Produit X"

    def test_enchainements_included(self, reference):
        assert len(reference["enchainements"]) == 1
        assert reference["enchainements"][0]["operation_liee"] == "OP_TEST"

    def test_spc_center_matches_operations_mean(self, reference, att_labeled_steps_two_ops, att_pas_reference):
        operations, _ = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        assert reference["duration_min"]["spc"]["center"] == pytest.approx(operations["duration_min"].mean())

    def test_pas_stats_included(self, reference):
        pas_nums = {row["pas_num"] for row in reference["pas_stats"]}
        assert pas_nums == {1, 2, 3}

    def test_defaut_by_pas_includes_type_attente(self, reference):
        # Le seul défaut du fixture vient du pas PROD (pas_num=3, type CHAUFFE).
        by_pas = {row["pas_num"]: row for row in reference["defauts"]["by_pas"]}
        assert by_pas[3]["type_attente"] == "CHAUFFE"

    def test_pas_reference_embedded_in_full(self, reference, att_pas_reference):
        # Le bundle doit reprendre toute la table process (y compris le pas
        # 4, un pas de défaut sans plage connue), pas seulement les pas
        # observés dans les données.
        assert len(reference["pas_reference"]) == len(att_pas_reference)
        embedded = {row["pas_num"]: row for row in reference["pas_reference"]}
        assert embedded[1]["code_court"] == "ATTENTE"
        assert embedded[1]["att_min"] == 100
        assert embedded[2]["phase"] == "PREP"
        assert np.isnan(embedded[4]["att_min"])

    def test_handles_fewer_than_two_operations_without_raising(self, att_pas_reference):
        # nop_analysis.compute_control_limits lève une ValueError sous 2
        # operations (etendue mobile non calculable) ; build_reference ne
        # doit pas planter pour autant (historique court, ex: tag tout juste
        # ajoute au catalogue), juste renvoyer des limites SPC a NaN.
        from att_analysis import label_with_pas

        reperes = [100, 110, 200, 300]
        starts = pd.date_range("2026-01-01", periods=4, freq="10min", tz="UTC")
        ends = starts + pd.Timedelta(minutes=10)
        steps = pd.DataFrame({
            "step_id": range(4), "repere": reperes, "start": starts, "end": ends,
            "duration": ends - starts, "duration_min": 10.0, "n_samples": 10,
        })
        labeled = label_with_pas(steps, att_pas_reference)
        operations, flagged = reconstruct_operations(labeled, att_pas_reference)
        assert len(operations) == 1

        defauts = defaut_frequency_table(flagged)
        ct = cycle_times(flagged, anchor_repere=110)
        reference = build_reference(
            att_pas_reference, operations, flagged, defauts, ct,
            tag="TEST_TAG", anchor_repere=110, chaining_window=2,
        )
        assert reference["n_operations"] == 1
        assert np.isnan(reference["duration_min"]["spc"]["center"])
        assert np.isnan(reference["duration_min"]["spc"]["ucl"])

    def test_duration_stats_exclude_truncated_boundary_operations(
        self, att_labeled_steps_chained_ops, att_pas_reference
    ):
        # constaté en pratique (OP1410) : un "temps standard minimum" de
        # 52 min alors qu'un cycle complet ne descend jamais sous ~500 min
        # -- cause : la 1ere operation reconstruite peut etre tronquee
        # (periode interrogee qui commence en cours de cycle). build_reference
        # doit exclure la 1ere et la derniere operation (chronologiquement)
        # de ses statistiques de duree, cf. exclude_boundary_operations.
        operations, steps = reconstruct_operations(att_labeled_steps_chained_ops, att_pas_reference)
        assert len(operations) == 3  # cf. TestReconstructOperationsChained

        operations = operations.copy()
        first_idx = operations.sort_values("start").index[0]
        operations.loc[first_idx, "duration_min"] = 1.0
        operations.loc[first_idx, "standard_duration_min"] = 1.0
        operations["rework_duration_min"] = operations["duration_min"] - operations["standard_duration_min"]

        defauts = defaut_frequency_table(steps)
        ct = cycle_times(steps, anchor_repere=110)
        reference = build_reference(
            att_pas_reference, operations, steps, defauts, ct,
            tag="TEST_TAG", anchor_repere=110, chaining_window=2,
        )

        expected_min = operations.sort_values("start").iloc[1:-1]["standard_duration_min"].min()
        assert reference["n_operations"] == 3
        assert reference["n_operations_used_for_duration_stats"] == 1
        assert reference["standard_duration_min"]["min"] == pytest.approx(expected_min)
        assert reference["standard_duration_min"]["min"] != pytest.approx(1.0)

    def test_save_and_load_roundtrip(self, reference, tmp_path):
        path = tmp_path / "reference.json"
        save_reference(reference, str(path))
        assert path.exists()

        reloaded = load_reference(str(path))
        assert reloaded["tag"] == reference["tag"]
        assert reloaded["n_operations"] == reference["n_operations"]
        assert reloaded["defauts"]["total"] == reference["defauts"]["total"]
        assert reloaded["pas_stats"][0]["pas_num"] == reference["pas_stats"][0]["pas_num"]
        assert len(reloaded["pas_reference"]) == len(reference["pas_reference"])
        assert reloaded["pas_reference"][0]["code_court"] == reference["pas_reference"][0]["code_court"]


# ---------------------------------------------------------------------------
# compare_operation_to_reference
# ---------------------------------------------------------------------------
class TestCompareOperationToReference:
    @pytest.fixture
    def reference_and_steps(self, att_labeled_steps_two_ops, att_pas_reference):
        operations, steps = reconstruct_operations(att_labeled_steps_two_ops, att_pas_reference)
        defauts = defaut_frequency_table(steps)
        ct = cycle_times(steps, anchor_repere=110)
        reference = build_reference(
            att_pas_reference, operations, steps, defauts, ct,
            tag="TEST_TAG", anchor_repere=110, chaining_window=2,
        )
        return reference, steps

    def test_duration_matches_operation_total(self, reference_and_steps):
        reference, steps = reference_and_steps
        op0_steps = steps[steps["operation_id"] == 0]
        diagnostic = compare_operation_to_reference(op0_steps, reference)
        assert diagnostic["duration_min"] == pytest.approx(90.0)

    def test_n_defauts_matches_the_flagged_step(self, reference_and_steps):
        reference, steps = reference_and_steps
        op0_steps = steps[steps["operation_id"] == 0]
        diagnostic = compare_operation_to_reference(op0_steps, reference)
        assert diagnostic["n_defauts"] == 1
        assert diagnostic["defaut_pas"] == [3]  # le pas PROD a echoue avant le retry

    def test_operation_without_defaut_reports_none(self, reference_and_steps):
        reference, steps = reference_and_steps
        op1_steps = steps[steps["operation_id"] == 1]
        diagnostic = compare_operation_to_reference(op1_steps, reference)
        assert diagnostic["n_defauts"] == 0
        assert diagnostic["defaut_pas"] == []

    def test_duration_within_spc_limits_is_in_control(self, reference_and_steps):
        reference, steps = reference_and_steps
        op0_steps = steps[steps["operation_id"] == 0]
        diagnostic = compare_operation_to_reference(op0_steps, reference)
        assert diagnostic["spc_out_of_control"] is False


# ---------------------------------------------------------------------------
# build_operations_network
# ---------------------------------------------------------------------------
class TestBuildOperationsNetwork:
    @staticmethod
    def _pas_ref(links):
        """links: liste de (pas_num, code_court, operation_liee, sens, detail)."""
        rows = [
            {"pas_num": n, "code_court": c, "operation_liee": op, "sens_liaison": s, "type_attente_detail": d}
            for n, c, op, s, d in links
        ]
        return pd.DataFrame(rows)

    @pytest.fixture
    def entries(self):
        pas_ref_a = self._pas_ref([
            (1, "INIT", None, None, "rien"),
            (2, "ATTENTE-B", "OPBVA", "ATTEND", "attend OPB"),
            (3, "ATTENTE-C", "OPCVA", "ENVOIE", "signale OPC (hors catalogue)"),
        ])
        pas_ref_b = self._pas_ref([
            (1, "INIT", None, None, "rien"),
        ])
        return [
            {"operation": "OPA", "nom": "PUA", "tag": "PUA_Att", "produit": "X",
             "description": "Desc A", "pas_reference": pas_ref_a},
            {"operation": "OPB", "nom": "PUB", "tag": "PUB_Att", "produit": "X",
             "description": "Desc B", "pas_reference": pas_ref_b},
        ]

    def test_catalog_operations_are_marked_in_catalog(self, entries):
        network = build_operations_network(entries)
        by_op = {o["operation"]: o for o in network["operations"]}
        assert by_op["OPA"]["in_catalog"] is True
        assert by_op["OPA"]["nom"] == "PUA"
        assert by_op["OPB"]["in_catalog"] is True

    def test_referenced_operation_outside_catalog_is_added_with_flag(self, entries):
        network = build_operations_network(entries)
        by_op = {o["operation"]: o for o in network["operations"]}
        assert "OPC" in by_op
        assert by_op["OPC"]["in_catalog"] is False
        assert by_op["OPC"]["nom"] is None

    def test_liaisons_normalize_the_va_suffix(self, entries):
        network = build_operations_network(entries)
        targets = {l["operation_cible"] for l in network["liaisons"]}
        assert targets == {"OPB", "OPC"}

    def test_liaison_fields_are_carried_through(self, entries):
        network = build_operations_network(entries)
        liaison_b = next(l for l in network["liaisons"] if l["operation_cible"] == "OPB")
        assert liaison_b["operation_source"] == "OPA"
        assert liaison_b["pas_source"] == 2
        assert liaison_b["code_court_source"] == "ATTENTE-B"
        assert liaison_b["sens"] == "ATTEND"
        assert liaison_b["detail"] == "attend OPB"

    def test_no_liaisons_gives_empty_list(self):
        pas_ref = self._pas_ref([(1, "INIT", None, None, "rien")])
        entries = [{"operation": "OPA", "nom": "PUA", "tag": "PUA_Att", "produit": "X",
                    "description": "Desc A", "pas_reference": pas_ref}]
        network = build_operations_network(entries)
        assert network["liaisons"] == []
        assert len(network["operations"]) == 1
