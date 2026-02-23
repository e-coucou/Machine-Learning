#!/bin/bash

echo "--------------------------------------------------"
echo "🚀 LANCEMENT DES TESTS - POUSSIN-V2"
echo "--------------------------------------------------"

# Test 1 : Géographie (Vérifier si Paris/France se connectent)
echo -e "\n📍 TEST 1 : GÉOGRAPHIE"
python3 train_gpt_generate.py "Paris est la capitale de , " -n 80 -p 1.1 -k 40 -t 0.9

# Test 2 : Science (Vérifier la capacité de définition)
echo -e "\n⚛️ TEST 2 : SCIENCE"
python3 train_gpt_generate.py "Un atome est composé de" -n 50 -p 1.1 -k 40 -t 0.8

# Test 3 : Histoire (Voir s'il invente toujours des évêques)
echo -e "\n📜 TEST 3 : HISTOIRE"
python3 train_gpt_generate.py "Au cours de la Seconde Guerre mondiale ," -n 100 -p 1.1 -k 40 -t 0.85

# Test 4 : Structure Wikipédia (Test des catégories)
echo -e "\n📚 TEST 4 : STRUCTURE WIKI"
python3 train_gpt_generate.py "La géographie de la France se caractérise par" -n 80 -p 1.1 -k 40 -t 0.9

# Test 5 : Logique / Grammaire (Phrases simples)
echo -e "\n🧠 TEST 5 : LOGIQUE"
python3 train_gpt_generate.py "Le soleil est une étoile qui" -n 80 -p 1.1 -k 40 -t 0.7

echo -e "\n[TEST FACTUEL]"
python3 train_gpt_generate.py "La tour Eiffel est un monument situé à" -n 80 -p 1.1 -k 40 -t 0.7

echo -e "\n[TEST LITTÉRAIRE]"
python3 train_gpt_generate.py "Le soir tombait sur la plaine, et l'homme" -n 80 -p 1.1 -k 40 -t 0.7

echo -e "\n[TEST DE COHÉRENCE]"
python3 train_gpt_generate.py "Si on mélange du bleu et du jaune, on obtient" -n 80 -p 1.1 -k 40 -t 0.7


echo -e "\n--------------------------------------------------"
echo "✅ TESTS TERMINÉS"
echo "--------------------------------------------------"
