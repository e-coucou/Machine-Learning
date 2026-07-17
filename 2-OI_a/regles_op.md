# Règles d'extraction des repères d'attente - Procédures OP

## Source
- Document OPxxxx
- Extraction depuis les colonnes :
  - Colonne 4 = Repère d'attente
  - Colonne 5 = Description de l'attente

## Structure de sortie

| Operation | PU | PAS | Désignation PAS | Famille procédé | Nature attente | Repère | Description attente | Seuil / Condition | Commentaires |
|------------|----|-----|-----------------|------------------|---------|---------------------|-------------------|--------------|

---

## Champs

### Operation
Nom de l'opération :
Exemple : OP1410VA

### PU
Point unité :
Exemple : PU1410VA

### PAS
Numéro du PAS.

### Désignation PAS
Libellé exact du PAS.

Exemple :
- CHAUFF.AIP
- CHARGE H2SO4
- MISE VIDE

### Famille procédé

Normalisation obligatoire :

| Libellé PAS | Famille |
|-------------|----------|
| ATTENTE | Initialisation |
| INIT.OPER | Initialisation |
| TEST ETANCH | Contrôle |
| CHARGE* | Remplissage |
| DEB.CH* | Remplissage |
| FIN CH* | Remplissage |
| CHAUFF* | Chauffage |
| ARRET CHAUF | Chauffage |
| MISE VIDE | Mise sous vide |
| MARCHE GAV | Mise sous vide |
| DISTIL* | Distillation |
| REFLUX | Distillation |
| REFROID* | Refroidissement |
| EGOUTTAGE | Égouttage |
| VIDANGE | Vidange |
| ARRET | Arrêt |

### Nature attente

Classification :

| Type | Description |
|--------|------------|
| Opérateur | Validation ou saisie opérateur |
| Temps procédé | Réaction chimique, chauffage, refroidissement |
| Transfert matière | Remplissage, vidange, transfert |
| Équipement | Mise en régime pompe, GAV, agitateur |
| Interface autre opération | Attente OPxxxx |
| Mesure procédé | Niveau, pression, température, débit |
| Sécurité / Contrôle | Étanchéité, interlocks |
| Temporisation | Kxxxx, timer |

### Repère
Numéro affiché dans la colonne Attente.

Exemple :
- 730
- 1410
- 2050

### Description attente
Texte métier synthétique expliquant pourquoi l'on attend.

### Seuil / Condition
Condition exacte de sortie d'attente :

Exemples :
- TI14204VA >= HA
- BV2 = 0
- K1429
- PC14211VA < C1 + E

### Commentaires
Informations complémentaires ou hypothèses.

---

## Règles

1. Une ligne = un repère d'attente.
2. Les boucles doivent être conservées.
3. Les repères identiques avec conditions différentes doivent apparaître plusieurs fois.
4. Les seuils doivent être repris textuellement.
5. Ne jamais inventer de libellé.
6. Utiliser les familles procédé normalisées.
7. Utiliser les natures d'attente normalisées.