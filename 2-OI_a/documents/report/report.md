---
title: "Analyse des performances industrielles de l'atelier Vitamines"
subtitle: "Identification du Cycle Machine Théorique (CMT), du Taux de Cadence Théorique (TCD) et de la Best Demonstrated Performance (BDP) par la méthode du Golden Score"
title-short: "Performances atelier Vitamines --- Golden Score"
author: "e-coucou"
affiliation: ""
date: "12 juillet 2026"
keywords: "OEE, Golden Score, Best Demonstrated Performance, Taux de Cadence Théorique, Cycle Machine Théorique, contrôle statistique de procédé, atelier Vitamines"
abstract: |
  Cette étude caractérise la performance opérationnelle de 8 lignes de
  production de l'atelier Vitamines à partir de compteurs cumulatifs d'opérations
  (`value_nop`), échantillonnés à la minute. Après reconstruction des opérations
  (plateaux de valeur constante du compteur) et filtrage des durées incohérentes,
  une fenêtre de référence est estimée pour chaque ligne de façon à ce que la
  meilleure séquence consécutive observée couvre environ sept jours ; cette
  référence permet de calculer, pour chaque ligne, le Cycle Machine Théorique
  (CMT, durée standard d'une opération), le Taux de Cadence Théorique (TCD,
  capacité équivalente en opérations par jour) et la Best Demonstrated
  Performance (BDP) associée. Sur les 8 lignes analysées avec succès
  (0 écartée(s) faute de données suffisantes), le TCD varie de
  2.67 ops/j (NOP_SC15) à
  4.14 ops/j (NOP_ESTERS), pour une
  moyenne de 3.46 ops/j et un CMT moyen de 422.3 minutes par
  opération. La majorité des lignes affiche une tendance mensuelle à l'amélioration. Ces indicateurs fournissent une référence
  directement exploitable pour le calcul du volet Performance de l'OEE de
  chaque ligne.
lang: fr
---


# Introduction

L'atelier Vitamines dispose, pour plusieurs lignes de production, d'un
compteur cumulatif du nombre d'opérations réalisées (`value_nop`),
échantillonné à la minute. Ce compteur reste constant pendant qu'une
opération est en cours puis s'incrémente au passage à l'opération suivante :
une opération correspond donc à un plateau de valeur constante du signal.

L'objectif de cette étude est d'identifier, pour chaque ligne, trois
indicateurs directement exploitables dans un calcul d'OEE (Overall Equipment
Effectiveness) : le **Cycle Machine Théorique (CMT)**, le **Taux de Cadence
Théorique (TCD)** et la **Best Demonstrated Performance (BDP)**, à partir
d'une méthode de scoring reproductible baptisée *Golden Score*.

# Méthodologie

## Nettoyage et reconstruction des opérations

Le signal brut est nettoyé (suppression des valeurs manquantes ou
négatives), puis les opérations sont reconstruites en identifiant les
plateaux de valeur constante du compteur. La durée d'une opération est le
temps écoulé jusqu'au début du plateau suivant. Les opérations dont la durée
sort d'un intervalle de cohérence (bruit capteur en deçà, arrêt de ligne
prolongé au-delà) sont écartées avant tout calcul de performance.

## Estimation de la fenêtre de référence (BDP à 7 jours)

Plutôt que de fixer arbitrairement un nombre d'opérations censé représenter
une semaine de production, la taille de fenêtre est estimée pour chaque
ligne : on recherche, par ajustement itératif, la taille de fenêtre dont la
meilleure séquence consécutive observée couvre une durée proche de sept
jours. Cette fenêtre devient la référence de performance de la ligne.

## Indicateurs dérivés

Sur cette fenêtre de référence, on calcule :

- le **CMT** (Cycle Machine Théorique) : durée moyenne d'une opération sur la
  meilleure séquence observée (minutes/opération) ;
- le **TCD** (Taux de Cadence Théorique) : capacité équivalente de la même
  séquence, exprimée en opérations par jour ;
- la **BDP** (Best Demonstrated Performance) : la performance la plus élevée
  réellement observée sur une fenêtre de cette taille, dont le TCD est
  directement dérivé.

# Résultats

## Vue d'ensemble par ligne

| Tag          |   N ops (nettoyées) |   Durée moy. (min) |   Durée méd. (min) |   Nb ops run golden |   TCD (ops/j) |   Temps standard (min/op) |
|:-------------|--------------------:|-------------------:|-------------------:|--------------------:|--------------:|--------------------------:|
| NOP_ESTERS   |                3069 |              482   |              426   |                  29 |          4.14 |                     347.9 |
| NOP_ACETATE  |                2995 |              495.4 |              435   |                  26 |          3.74 |                     384.8 |
| NOP_RETINOL  |                2996 |              494.9 |              436   |                  26 |          3.7  |                     389.6 |
| NOP_RHQ      |                2995 |              496.7 |              435   |                  26 |          3.68 |                     391.7 |
| NOP_RETINENE |                2843 |              522.2 |              451   |                  23 |          3.32 |                     434   |
| OP1510VA_CPT |                2586 |              527.5 |              481   |                  23 |          3.3  |                     436.4 |
| NOP_AOIP     |                2252 |              577.5 |              542.5 |                  16 |          3.17 |                     453.9 |
| NOP_SC15     |                2242 |              642.2 |              578   |                  19 |          2.67 |                     539.9 |

## Comparaison inter-lignes

![Comparaison du TCD et du CMT par ligne de production](figures/comparison_tcd_cmt.png)



# Discussion et limites

La référence retenue (BDP) est par construction un **record observé**,
donc structurellement optimiste : pour une référence OEE soutenable dans la
durée, un percentile (par exemple le 10e centile) des fenêtres glissantes de
même taille constitue une alternative à envisager plutôt que le record
absolu. Le filtre de durée est actuellement appliqué de façon uniforme à
toutes les lignes ; un profil de durée très différent d'une ligne à l'autre
justifierait une validation spécifique de ce seuil. Enfin, la fenêtre de
référence étant ré-estimée indépendamment pour chaque ligne, une comparaison
brute du nombre d'opérations entre lignes est moins pertinente qu'une
comparaison directe en opérations par jour (TCD).

# Conclusion

La méthode du Golden Score permet d'obtenir, pour chaque ligne de l'atelier
Vitamines, un triplet CMT/TCD/BDP directement exploitable comme référence de
performance théorique dans un calcul d'OEE, sans hypothèse arbitraire sur la
durée d'un cycle de référence. Les prochaines étapes consistent à croiser ces
indicateurs avec les volumes réellement produits (calcul complet de l'OEE) et
à suivre dans le temps la stabilité de la fenêtre de référence par ligne,
elle-même un signal utile de dérive de procédé.

