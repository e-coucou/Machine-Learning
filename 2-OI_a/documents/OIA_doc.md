# OIAnalytics API — Référence v16.19.4

> **Base URL** : `https://oianalytics-100.optimistik.fr`  
> **Auth** : Basic Authentication → header `Authorization: basic <token>`  
> **Format** : JSON (Content-Type: application/json)

---

## Sommaire

1. [Authentification](#1-authentification)
2. [Données — Query](#2-données--query)
3. [Données — Insert](#3-données--insert)
4. [Données stockées continues](#4-données-stockées-continues)
5. [Batches](#5-batches)
6. [Événements](#6-événements)
7. [Assets & Asset Types](#7-assets--asset-types)
8. [Dashboards](#8-dashboards)
9. [Contrôles SPC](#9-contrôles-spc)
10. [Utilitaires](#10-utilitaires)

---

## 1. Authentification

Chaque requête doit inclure le header :

```http
Authorization: basic <credentials>
```

---

## 2. Données — Query

### 2.1 Requête time series (une donnée)

```http
GET /api/oianalytics/data/{dataId}/values
```

| Paramètre | Requis | Description |
|-----------|--------|-------------|
| `from` | ✅ | Début ISO 8601 |
| `to` | ✅ | Fin ISO 8601 |
| `aggregation` | ✅ | `TIME` \| `RAW_VALUES` \| `GLOBAL` |
| `aggregation-period` | si TIME | ex: `PT1H`, `PT10M`, `P1D` |
| `aggregation-function` | si TIME | `MEAN`, `FIRST`, `LAST`, `SUM`, `MIN`, `MAX`, `LAST_MINUS_FIRST`, `COUNT`, etc. |
| `timezone` | ❌ | ex: `Europe/Paris` |
| `unit-id` | ❌ | ID de l'unité de résultat |

**Réponse :**
```json
{
  "type": "time-values",
  "data": { "id": "D200", "reference": "MASS_TANK_1" },
  "unit": { "label": "kg" },
  "timestamps": ["2021-01-01T00:00:00Z", "..."],
  "values": [32.4, 40.4]
}
```

---

### 2.2 Requête multiple données à la fois

```http
GET /api/oianalytics/data/values
```

| Paramètre | Requis | Description |
|-----------|--------|-------------|
| `data-reference` | ✅* | Référence(s) des données (répétable) |
| `data-id` | ✅* | Ou ID(s) des données |
| `from` | ✅ | Début ISO 8601 |
| `to` | ✅ | Fin ISO 8601 |
| `aggregation` | ✅ | `TIME` \| `RAW_VALUES` \| `GLOBAL` |
| `aggregation-period` | si TIME | |
| `aggregation-function` | si TIME | |
| `timezone` | ❌ | ex: `Europe/Paris` |
| `unit-id` | ❌ | Répétable, même ordre que data |

**Exemple :**
```
GET /api/oianalytics/data/values
  ?data-reference=MASS_TANK_1
  &data-reference=MASS_TANK_2
  &from=2026-06-13T00:00:00Z
  &to=2026-06-13T12:00:00Z
  &aggregation=TIME
  &aggregation-function=MEAN
  &aggregation-period=PT10M
  &timezone=Europe/Paris
```

**Réponse :** tableau, un élément par donnée.

---

### 2.3 Fonctions d'agrégation disponibles

`FIRST` · `LAST` · `LAST_MINUS_FIRST` · `SUM` · `MIN` · `MAX` · `MEAN` · `MEDIAN` · `STDEV` · `COUNT` · `PERCENTILE5` · `PERCENTILE95` · `DECILE1` · `DECILE9` · `QUARTILE1` · `QUARTILE9` · `VALUE_CHANGE` · `MEAN_MINUS_SIGMA` · `MEAN_PLUS_SIGMA` · `MEAN_MINUS_TWO_SIGMA` · `MEAN_PLUS_TWO_SIGMA` · `MEAN_MINUS_THREE_SIGMA` · `MEAN_PLUS_THREE_SIGMA`

---

### 2.4 Lister les données disponibles

```http
GET /api/oianalytics/data
```

| Paramètre | Description |
|-----------|-------------|
| `query` | Texte de recherche |
| `types` | `RAW_TIME_DATA` \| `COMPUTED_TIME_DATA` \| ... |
| `measurement-name` | Nom de la mesure |
| `tag-value-id` | Filtre par tag |
| `page` / `size` | Pagination (défaut: 0 / 20) |

---

## 3. Données — Insert

### 3.1 Insérer des valeurs time series

```http
POST /api/oianalytics/value-upload/time-values
```

**Body :**
```json
[
  {
    "dataReference": "MASS_TANK_1",
    "unit": "kg",
    "values": [
      { "timestamp": "2026-06-13T08:00:00Z", "value": 1500 },
      { "timestamp": "2026-06-13T08:10:00Z", "value": 1520 }
    ]
  }
]
```

**Query params :**
- `use-external-reference=false` (défaut) → utilise la référence OIA
- `create-upload-event=true` (défaut)

**Réponse :**
```json
{
  "numberOfValuesSubmitted": 2,
  "numberOfValuesSuccessfullyInserted": 2,
  "numberOfValuesRejected": 0,
  "errors": []
}
```

---

## 4. Données stockées continues

### 4.1 Lister

```http
GET /api/oianalytics/stored-continuous-data
```

| Paramètre | Description |
|-----------|-------------|
| `query` | Texte |
| `measurement` | ID de la mesure |
| `tagValues` | IDs de tag values |
| `page` / `size` | Pagination |

### 4.2 Créer

```http
POST /api/oianalytics/stored-continuous-data
```

```json
{
  "reference": "MON_TAG_001",
  "description": "Description",
  "measurementId": "...",
  "transferUnitId": "...",
  "resolution": "PT10M",
  "resamplingMethod": "NONE"
}
```

### 4.3 Mettre à jour

```http
PUT /api/oianalytics/stored-continuous-data/{id}
```

### 4.4 Supprimer

```http
DELETE /api/oianalytics/stored-continuous-data/{id}
```

---

## 5. Batches

### 5.1 Lister les types de batch

```http
GET /api/oianalytics/batch-types
```

### 5.2 Lister les batches d'un type

```http
GET /api/oianalytics/batch-types/{batchTypeId}/batches
```

| Paramètre | Description |
|-----------|-------------|
| `start` | Début ISO |
| `end` | Fin ISO |
| `name` | Filtre sur le nom |
| `feature-values` | IDs de feature values |
| `page` / `size` | Pagination |

### 5.3 Créer un batch

```http
POST /api/oianalytics/batch-types/{batchTypeId}/batches
```

```json
{
  "name": "BATCH_001",
  "steps": [
    {
      "stepId": "...",
      "start": "2026-06-13T06:00:00Z",
      "end": "2026-06-13T14:00:00Z",
      "localisationType": "TAG_VALUES",
      "localisationTagValueIds": ["..."]
    }
  ],
  "tagValuesByValue": [
    { "batchTagKeyId": "...", "batchTagValueValue": "Recipe1" }
  ],
  "values": [
    { "dataId": "...", "value": 150.5, "unitId": "..." }
  ]
}
```

### 5.4 Créer ou mettre à jour plusieurs batches

```http
POST /api/oianalytics/batch-types/{batchTypeId}/batches/create-or-update
```

Body : tableau de `BatchCommand`.

### 5.5 Mettre à jour un batch

```http
PUT /api/oianalytics/batch-types/{batchTypeId}/batches/{batchId}
```

### 5.6 Supprimer un batch

```http
DELETE /api/oianalytics/batch-types/{batchTypeId}/batches/{batchId}
```

---

## 6. Événements

### 6.1 Lister les types d'événements

```http
GET /api/oianalytics/event-types
```

### 6.2 Lister les événements

```http
GET /api/oianalytics/event-types/{eventTypeId}/events
```

| Paramètre | Description |
|-----------|-------------|
| `start` / `end` | Fenêtre temporelle ISO |
| `description` | Filtre texte |
| `tagValues` | IDs de feature values |
| `page` / `size` | Pagination |

### 6.3 Créer un événement

```http
POST /api/oianalytics/event-types/{eventTypeId}/events
```

```json
{
  "start": "2026-06-13T08:00:00Z",
  "end": "2026-06-13T09:00:00Z",
  "description": "Arrêt pompe P01",
  "tagValues": [
    { "tagKeyId": "...", "tagValueValue": "Mécanique" }
  ],
  "values": [],
  "assetIds": []
}
```

### 6.4 Créer ou mettre à jour plusieurs événements

```http
POST /api/oianalytics/event-types/{eventTypeId}/events/create-or-update
```

### 6.5 Mettre à jour un événement

```http
PUT /api/oianalytics/event-types/{eventTypeId}/events/{eventId}
```

### 6.6 Supprimer un événement

```http
DELETE /api/oianalytics/event-types/{eventTypeId}/events/{eventId}
```

---

## 7. Assets & Asset Types

### 7.1 Lister les asset types

```http
GET /api/oianalytics/asset-types
```

### 7.2 Lister les assets

```http
GET /api/oianalytics/assets
```

| Paramètre | Description |
|-----------|-------------|
| `query` | Texte |
| `assetTypeId` | Filtre par type |
| `tagValueIds` | Filtre par tags |
| `page` / `size` | Pagination |

### 7.3 Créer un asset

```http
POST /api/oianalytics/assets
```

```json
{
  "name": "Reacteur_R01",
  "externalReference": "R01-ext",
  "assetTypeId": "...",
  "tagValues": [
    { "tagKeyId": "...", "id": "...", "value": "Site1" }
  ],
  "dataMappings": [
    { "assetTypeDataId": "...", "mode": "MAPPED", "dataId": "..." }
  ],
  "staticDataValues": [],
  "fileResourceIds": [],
  "htmlResources": [],
  "pythonModelInstances": [],
  "logisticRouteIds": []
}
```

### 7.4 Mettre à jour un asset

```http
PUT /api/oianalytics/assets/{assetId}
```

### 7.5 Mettre à jour tags et valeurs statiques (bulk)

```http
PUT /api/oianalytics/assets/tags-and-values
```

```json
[
  {
    "assetId": "...",
    "tagCommands": [
      { "tagKeyId": "...", "id": null, "value": "NouvelleValeur" }
    ],
    "staticDataValueCommands": [
      { "assetTypeStaticDataId": "...", "value": 42.0, "unitId": null }
    ]
  }
]
```

---

## 8. Dashboards

### 8.1 Lister

```http
GET /api/oianalytics/dashboard
```

### 8.2 Exporter (PNG / CSV / XLS)

```http
GET /api/oianalytics/dashboard/{id}/export
```

| Paramètre | Description |
|-----------|-------------|
| `from` / `to` | Fenêtre temporelle (dashboard TIME) |
| `batchId` | Pour dashboard BATCH |
| `format` | `PNG` (défaut) \| `CSV` \| `XLS` |
| `viewportWidth` | Largeur PNG (défaut 1440px) |
| `exportType` | `GLOBAL` (défaut) \| `INDIVIDUAL` (zip) |
| `theme` | `light` (défaut) \| `dark` |

---

## 9. Contrôles SPC

### 9.1 Lister les contrôles

```http
GET /api/oianalytics/controls
```

### 9.2 Récupérer les valeurs d'un contrôle

```http
POST /api/controls/values/{controlId}
```

**Body (contexte temporel) :**
```json
{
  "type": "time-context-control-value-query",
  "start": "2026-06-13T00:00:00Z",
  "end": "2026-06-13T12:00:00Z",
  "timezone": "Europe/Paris"
}
```

**Réponse :**
```json
{
  "type": "time-numeric-univariate",
  "timestamps": ["..."],
  "values": [12.5, 14.0],
  "colors": [null, "#FF0000"],
  "violatedControlRuleIds": [[], ["rule-id-1"]],
  "limitLines": { "inferiorLimit1": {...}, "superiorLimit1": {...} }
}
```

### 9.3 Créer un contrôle

```http
POST /api/oianalytics/controls
```

Types disponibles : `time-numeric-univariate` · `multi-batch-numeric-univariate` · `inside-batch-time-numeric-univariate` · `asset-time-numeric-univariate`

---

## 10. Utilitaires

### 10.1 Tag Keys & Tag Values

```http
GET /api/oianalytics/tag-keys
GET /api/oianalytics/tag-keys/{tagKeyId}/values
POST /api/oianalytics/tag-keys/{tagKeyId}/values   # Créer une valeur
```

### 10.2 Unités & Mesures

```http
GET /api/oianalytics/units
GET /api/oianalytics/measurements
GET /api/oianalytics/unit-families
```

### 10.3 Upload de fichier (parseurs)

```http
POST /api/oianalytics/file-uploads   # multipart/form-data
GET  /api/oianalytics/file-uploads   # Lister avec statuts
```

Statuts possibles : `PENDING` · `RUNNING` · `SUCCESS` · `ERROR` · `NO_PARSER_FOUND`

### 10.4 Utilisateurs & Profils

```http
GET /api/oianalytics/users
GET /api/oianalytics/profiles
```

### 10.5 Travaux de calcul

```http
POST /api/oianalytics/computation-jobs/continuous-data
POST /api/oianalytics/computation-jobs/batch-data
POST /api/oianalytics/computation-jobs/event-data
```

---

## Codes HTTP

| Code | Description |
|------|-------------|
| 200 | Succès |
| 201 | Créé |
| 204 | Succès sans contenu |
| 400 | Requête invalide |
| 404 | Ressource introuvable |

---

## Périodes ISO 8601 (référence rapide)

| Format | Valeur |
|--------|--------|
| `PT10M` | 10 minutes |
| `PT1H` | 1 heure |
| `P1D` | 1 jour |
| `P1W` | 1 semaine |
| `P1M` | 1 mois |

---

*Documentation générée depuis OIAnalytics API v16.19.4 — 2026-06-13*