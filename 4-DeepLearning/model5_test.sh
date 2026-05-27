#!/bin/bash

echo "--------------------------------------------------"
echo "🚀 LANCEMENT DES TESTS - POUSSIN-V2"
echo "--------------------------------------------------"

echo "\n[TEST GÉOGRAPHIE]"
python3 train_gpt_generate.py "Paris est la capitale de , " -n 80 -p 1.1 -k 40 -t 0.6

echo "\n[TEST LITTÉRAIRE]"
python3 train_gpt_generate.py "Le soir tombait sur la plaine, et l'homme" -n 80 -p 1.1 -k 40 -t 0.8

echo "\n[TEST DE COHÉRENCE]"
python3 train_gpt_generate.py "Si on mélange du bleu et du jaune, on obtient" -n 150 -p 1.1 -k 40 -t 0.7

echo "\n[TEST 0 RISQUE]"
python3 train_gpt_generate.py "Paris est la capitale de , " -n 80 -p 1.1 -k 40 -t 0.1

echo "\n--------------------------------------------------"
echo "✅ TESTS TERMINÉS"
echo "--------------------------------------------------"
