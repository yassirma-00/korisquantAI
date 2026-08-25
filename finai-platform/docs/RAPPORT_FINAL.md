# KorisQuant AI — Rapport final

**Apprentissage par Renforcement Profond Adaptatif et Explicable
pour la Gestion de Portefeuille Sensible au Risque**

*Plateforme web d'analyse financière et de gestion de portefeuille par
apprentissage automatique, apprentissage par renforcement profond et
intelligence artificielle explicable*

> Le titre en gras est celui du cahier des charges officiel
> (`Cahier_des_charges.pdf`) ; la ligne en italique décrit le livrable tel
> qu'il existe. L'audit de conformité entre les deux fait l'objet de la
> [section 7](#7--cahier-des-charges--conformité-et-divergences).

El Maroufy Mohamed Yassir — Projet de fin d'études
Document arrêté au 12 août 2026

---

> **Note méthodologique sur les chiffres.**
> Toutes les valeurs de ce rapport ont été **mesurées en exécutant le code** au
> moment de la rédaction. Les commandes qui les produisent figurent en
> [annexe A](#annexe-a--commandes-de-vérification). Quand une valeur est un
> *choix de conception* et non une mesure, cela est dit explicitement. Quand une
> grandeur n'a pas pu être mesurée, elle est rapportée comme **indisponible**
> avec la raison — jamais estimée ni arrondie à l'avantage du projet.
>
> Les résultats défavorables (R² négatifs, tests de VaR échoués, agents RL
> battus par une stratégie passive) sont **rapportés tels quels**. Un rapport
> dont un seul chiffre se révèle faux perd la confiance du lecteur sur tous les
> autres.

---

## Table des matières

1. [Synthèse](#1--synthèse)
2. [Périmètre livré](#2--périmètre-livré)
3. [Nouvelles fonctionnalités](#3--nouvelles-fonctionnalités)
   - 3.1 [AI Stress Testing Engine](#31--ai-stress-testing-engine)
   - 3.2 [Prédiction directionnelle et découverte automatique des modèles](#32--prédiction-directionnelle-et-découverte-automatique-des-modèles)
   - 3.3 [Interface : thème, navigation, mise en page](#33--interface--thème-navigation-mise-en-page)
4. [Architecture](#4--architecture)
5. [Résultats mesurés](#5--résultats-mesurés)
6. [Qualité logicielle et méthode de vérification](#6--qualité-logicielle-et-méthode-de-vérification)
7. [Cahier des charges — conformité et divergences](#7--cahier-des-charges--conformité-et-divergences)
   - 7.1 [Exigences fonctionnelles](#71-exigences-fonctionnelles)
   - 7.2 [Exigences non fonctionnelles](#72-exigences-non-fonctionnelles)
   - 7.3 [Registre des divergences (CDC-1 … CDC-8)](#73-registre-des-divergences)
   - 7.4 [Synthèse de conformité](#74-synthèse-de-conformité)
8. [Défauts trouvés et corrigés](#8--défauts-trouvés-et-corrigés)
9. [Limites connues](#9--limites-connues)
10. [Conclusion](#10--conclusion)
11. [Annexes](#annexe-a--commandes-de-vérification)

---

# 1 · Synthèse

KorisQuant AI est une plateforme web unifiant trois activités habituellement
séparées : l'**analyse de marché**, la **prévision** et la **décision**
(allocation, dimensionnement, gestion du risque), avec une exigence
transversale d'**explicabilité**.

## 1.1 Chiffres clés (mesurés)

| Grandeur | Valeur |
|---|---:|
| Tests automatisés (tous verts) | **744** |
| Lignes de code (`.py` + `.js` + `.html` + `.css`) | **47 995** |
| Fichiers source (`backend/` + `scripts/` + `frontend/`) | **156** |
| Chemins d'API / opérations | **127 / 135** |
| Pages web | **13** |
| Instruments au catalogue | **32** |
| Architectures d'apprentissage profond | **5** |
| Algorithmes d'apprentissage par renforcement | **13** |
| Agents RL entraînés livrés | **14** |
| Fonctions d'indicateurs techniques | **21** |
| Outils exposés au chatbot | **15** |
| Scénarios de stress | **7** |
| Analyse statique `ruff` | **0 violation** |

Répartition des 744 tests par fichier :

| Fichier | Tests |
|---|---:|
| `test_api.py` | 263 |
| `test_models_and_services.py` | 126 |
| `test_access_control.py` | 104 |
| `test_chat.py` | 86 |
| `test_intelligence.py` | 49 |
| `test_auth_and_brand.py` | 46 |
| `test_data_and_indicators.py` | 39 |
| `test_quant.py` | 31 |

---

# 2 · Périmètre livré

## 2.1 Les treize pages

| Page | Rôle |
|---|---|
| `landing.html` | Vitrine publique |
| `auth.html` | Connexion / inscription (bcrypt + JWT) |
| `index.html` | Vue de marché, watchlist, recommandation, régime |
| `analysis.html` | Analyse technique, 21 fonctions d'indicateurs |
| `forecast.html` | Prévision profonde, entraînement, intervalles conformes |
| `rl.html` | Agents par renforcement, entraînement, backtest |
| `signals.html` | Recommandation fusionnée, contributions, SHAP |
| `stress.html` | **AI Stress Testing Engine** (nouveau) |
| `xai.html` | Explicabilité (SHAP, importance des variables) |
| `portfolio.html` | Portefeuille, optimisation, transactions |
| `risk.html` | VaR, CVaR, GARCH, anomalies, alertes |
| `hyperparams.html` | Gestionnaire d'hyperparamètres |
| `training.html` | Training Intelligence (masquée de la navigation sur demande) |

> **Note.** `training.html` reste **servie et pleinement fonctionnelle** : elle
> est la seule interface de restauration/suppression des checkpoints RL. Elle a
> été retirée du menu à la demande de l'utilisateur, sans suppression, pour ne
> pas orpheliner les fichiers `.pt` existants.

---

# 3 · Nouvelles fonctionnalités

## 3.1 · AI Stress Testing Engine

**Fichiers** : `backend/app/services/risk/stress.py`,
`frontend/stress.html`, `frontend/assets/js/pages/stress.js`
**Endpoints** : `GET /api/v1/quant/stress-engine/scenarios`,
`GET /api/v1/quant/stress-engine/{symbols}`

### Principe de conception

Un scénario n'est **pas une table de pertes inventées** : c'est une
**transformation de la série de rendements réellement observée**. Chaque
grandeur affichée est ensuite recalculée par les fonctions de risque **déjà
présentes** dans la plateforme — aucun nouveau moteur de risque n'a été écrit :

| Fonction réutilisée | Rôle |
|---|---|
| `metrics.value_at_risk` | VaR historique |
| `metrics.conditional_var` | Perte moyenne au-delà de la VaR (CVaR) |
| `metrics.drawdown_series` | Trajectoire pic-creux |
| `metrics.risk_contribution` | Décomposition d'Euler de la volatilité |
| `metrics.correlation_matrix` | Structure de corrélation |

### Les sept scénarios

| Scénario | Transformation appliquée | Base |
|---|---|---|
| Market Crash | rejoue la **pire fenêtre glissante réellement vécue** | extrême observé |
| Market −10 % | choc ponctuel de −10 % ajouté à la distribution | hypothétique déclaré |
| Market −20 % | choc ponctuel de −20 % | hypothétique déclaré |
| Volatility ×2 | dispersion redimensionnée autour de sa propre moyenne | rendements réels |
| Liquidity Shock | **seules les baisses** observées sont amplifiées | rendements négatifs réels |
| Correlation Spike | convergence vers la trajectoire commune du panier | moyenne équipondérée |
| Custom | multiplicateur et choc fournis par l'utilisateur | paramètres utilisateur |

### Résultats mesurés — AAPL, position 100 000 $, fenêtre 5 ans

| Scénario | VaR (%) | CVaR (%) | Volatilité (%) | Drawdown (%) | Résilience |
|---|---|---|---|---|---:|
| Market Crash | 2,74 → 2,98 | 3,99 → 4,39 | 28,1 → 29,2 | 33,4 → 48,4 | 82,98 |
| Market −10 % | 2,74 → 2,78 | 3,99 → 4,10 | 28,1 → 28,4 | 33,4 → 33,4 | 98,33 |
| Market −20 % | 2,74 → 2,78 | 3,99 → 4,26 | 28,1 → 29,5 | 33,4 → 33,4 | 95,42 |
| Volatility ×2 | 2,74 → 5,56 | 3,99 → 8,05 | 28,1 → 56,1 | 33,4 → 65,3 | **1,10** |
| Liquidity Shock | 2,74 → 4,12 | 3,99 → 5,98 | 28,1 → 35,1 | 33,4 → 95,4 | 44,98 |
| Correlation Spike | inchangé | inchangé | inchangé | inchangé | 100,0 |
| Custom (×2) | 2,74 → 5,56 | 3,99 → 8,05 | 28,1 → 56,1 | 33,4 → 65,3 | 1,10 |

> **Lecture honnête du Correlation Spike sur un actif unique.** La résilience
> de 100 n'est pas un défaut : sur **un seul** actif il n'existe aucune
> structure transversale à casser. Le scénario retourne la série inchangée et
> l'indique dans la charge utile. Il n'a de sens que sur un panier.

### Résultats mesurés — portefeuille de 4 actifs, Correlation Spike

Panier AAPL 40 % / MSFT 30 % / GC=F 20 % / BTC-USD 10 %, position 250 000 $ :

- Corrélation moyenne : **0,2151 → 0,9857** (le scénario fait exactement ce qu'il annonce)
- CVaR : **2,83 % → 4,01 %** — perte de **10 032 $**
- Résilience : **53,28 / 100**

| Actif | Poids | Contribution au risque | Part de la perte |
|---|---:|---:|---:|
| AAPL | 40,0 % | 48,00 % | 39,93 % |
| MSFT | 30,0 % | 34,10 % | 29,79 % |
| GC=F | 20,0 % | **4,49 %** | 13,25 % |
| BTC-USD | 10,0 % | 13,41 % | 17,02 % |

Ce tableau illustre l'intérêt de la décomposition d'Euler : **GC=F pèse 20 % du
capital mais seulement 4,5 % du risque**, tandis que BTC-USD pèse 10 % du
capital pour 13,4 % du risque. Le poids ne mesure pas l'exposition.

### Score de résilience — formule publiée

Ce score est un **indice de conception**, pas une probabilité. Il combine la
dégradation de trois grandeurs mesurées :

```
composante(x) = clip(2 − après/avant, 0, 1)
score = 100 × (0,45·CVaR + 0,30·volatilité + 0,25·drawdown) / poids utilisés
```

100 = inchangé par le scénario ; 0 = deux fois pire ou davantage. Si aucune des
trois grandeurs n'est mesurable, le score vaut **`null`**, jamais une valeur par
défaut flatteuse.

### Garanties « aucun résultat fabriqué »

Quatre propriétés, chacune couverte par un test :

1. **Aucune grandeur inventée.** Une série trop courte (< 60 observations)
   renvoie `null` **et la raison**, jamais 0. Vérifié sur un historique de
   20 points : VaR, CVaR et résilience à `null`.
2. **Aucun aléatoire.** Vérifié par analyse **AST** (ni import ni appel
   `random`) et par deux exécutions identiques comparées.
3. **Traçabilité.** Les valeurs de base sont **prouvées égales** à la sortie de
   `value_at_risk` / `conditional_var` sur la même série.
4. **Montants dérivés.** Les sommes en dollars sont la stricte mise à l'échelle
   du pourcentage mesuré par la valeur de position — pas une seconde estimation.

---

## 3.2 · Prédiction directionnelle et découverte automatique des modèles

**Fichiers** : `backend/app/services/recommendation/direction.py`,
`backend/app/services/forecasting/trainer.py`
**Endpoint** : `GET /api/v1/signals/direction/{symbol}`

Cette fonctionnalité a connu **trois corrections successives**, chacune motivée
par un défaut réel constaté à l'exécution.

### Correction 1 — La volatilité n'est plus présentée comme une prédiction

Le système affichait « Typical move ±0,96 % » calculé à partir de la
**volatilité réalisée**, ce qui se lisait comme une prévision de mouvement. Or
la volatilité mesure l'amplitude, pas le sens.

`_expected_move()` a **une source unique** : le `predicted_return` du
forecaster entraîné. Sans modèle : `null`, `magnitude_basis =
"no_trained_forecaster"`, et l'interface affiche **N/A**. La volatilité est
retournée séparément sous `market_volatility_pct`, **toujours non signée**.
Vérifié par AST : `_expected_move` ne contient **ni** `np.sign` **ni**
`_realised_volatility`.

### Correction 2 — Détection automatique de l'architecture disponible

**Cause racine trouvée** : le système exigeait rigidement `lstm`. Or `EURUSD=X`
et `KO` ne possèdent qu'un checkpoint **GRU**, parfaitement chargeable. Ils
affichaient « No trained forecaster available » alors que **le fichier était
sur le disque depuis le début**. Ce n'était ni un problème de chemin, ni de
nommage, ni de chargement.

Trois méthodes génériques ont été ajoutées, fonctionnant par **listage de
répertoire** (aucun ticker codé en dur) :

- `available_forecasts(symbol)` — les paires (modèle, horizon) réellement présentes
- `available_horizons(symbol)`
- `resolve_model(symbol, preferred, horizon)` — l'architecture demandée gagne si
  elle est entraînée, sinon la **meilleure alternative mesurée** (précision
  directionnelle lue dans le fichier annexe, tri déterministe)

**L'horizon n'est jamais substitué** : un modèle h5 ne répondra jamais à une
question h60. Tout symbole entraîné ultérieurement devient utilisable **sans
modification de code** (vérifié : `GC=F` entraîné en ~7 s, détecté
immédiatement).

### Correction 3 — Le verdict ne contredit plus le chiffre affiché

Deux défauts liés :

- **Contradiction de signe** : AAPL affichait « INCREASE » au-dessus de
  « Expected Movement −0,88 % », avec un cours cible **sous** le dernier cours.
- **NEUTRAL systématique** : le verdict venait d'une moyenne de 4 signaux où le
  forecaster ne pesait que 30 %. L'analyse technique et le sentiment — qui ne
  prédisent rien — l'annulaient. AAPL : forecaster −0,133, technique +0,225,
  sentiment +0,089 → composite **+0,034**, dans la bande ±0,12 → NEUTRAL.

Le verdict suit désormais le `predicted_return` du forecaster (`lead_score`).
Le composite reste calculé, publié, et continue d'alimenter l'accord et la
confiance — il ne peut plus **écraser** la prédiction. Un champ
`signal_conflict` signale explicitement un désaccord de signe.

### État mesuré (horizon 5 jours)

| Symbole | Verdict | Mouvement attendu | Confiance | Modèle | Substitué |
|---|---|---:|---:|---|---|
| AAPL | DECREASE | −0,93 % | 69,9 % | lstm | non |
| KO | INCREASE | +1,92 % | 89,5 % | gru | **oui** |
| EURUSD=X | NEUTRAL | −0,16 % | 81,6 % | gru | **oui** |
| GC=F | NEUTRAL | **N/A** | **N/A** | — | — |
| BTC-USD | NEUTRAL | **N/A** | **N/A** | — | — |

### Couverture réelle du catalogue — limite majeure assumée

Audit automatisé des **32 instruments** (`scripts/forecaster_coverage.py`) :

| | |
|---|---:|
| Symboles au catalogue | 32 |
| **Avec un forecaster entraîné** | **2** (AAPL, EURUSD=X) |
| **Sans aucun modèle** | **30** |
| **Horizons entraînés** | **[5] uniquement** |

Répartition des manquants : actions 12, crypto 4, ETF 4, indices 4,
matières premières 3, devises 3.

> **Point d'honnêteté essentiel.** Pour ces 30 symboles, **N/A est la réponse
> correcte**, pas un bug. Aucune correction de code ne peut faire prédire un
> modèle qui n'existe pas ; produire un chiffre serait exactement le
> comportement que ce projet s'interdit. De même, **les horizons 1, 30 et 60
> jours ne sont entraînés pour aucun symbole**. L'entraînement à la demande est
> disponible et générique (~7 s par modèle).

---

## 3.3 · Interface : thème, navigation, mise en page

### Thème sombre / clair

Le contrôleur (`frontend/assets/js/theme.js`) était déjà en place :
injection automatique du bouton dans la barre supérieure, persistance
`localStorage`, suivi de la préférence système tant qu'aucun choix explicite
n'est fait, script anti-flash avant le premier rendu, et repeinture des
graphiques Plotly (qui n'héritent pas des variables CSS).

**Un défaut réel a néanmoins été trouvé** : `stress.html` déclarait un bouton
`.theme-toggle` **en dur et vide**. `mountThemeToggle` étant idempotent, il
ignorait cette barre — la page embarquait donc un bouton sans curseur, sans
icônes et **sans gestionnaire de clic**. Mesuré : `index.html` basculait
light→dark, `stress.html` restait bloquée sur dark. Corrigé, puis vérifié sur
les **11 pages** avec persistance entre navigations.

### Mise en page — deux défauts mesurés et corrigés

**Défilement horizontal parasite entre 861 px et ~1050 px** (iPad paysage,
portables 11"). Le repli en défilement de la barre supérieure ne démarrait qu'à
860 px, alors que la rangée cesse de tenir vers 1050 px.

| Largeur | `window.scrollX` avant | après |
|---:|---:|---:|
| 1100 px | 0 | 0 |
| 1024 px | **49** | 0 |
| 980 px | **85** | 0 |
| 900 px | **165** | 0 |
| 870 px | **195** | 0 |
| 860 px | 0 | 0 |

**Bouton de chat recouvrant des données.** En bas de page, le lanceur fixe
masquait le libellé « Very High » de l'échelle de confiance, un pourcentage
`18,38 %` et un horodatage. `.content` réserve désormais sa hauteur
(54 px + 22 px).

Vérification finale : **88 combinaisons** (11 pages × 4 largeurs × 2 thèmes)
→ **0 défilement horizontal, 0 erreur JavaScript**.

### Lanceur de l'assistant : un symbole remplacé par un mot

Le bouton d'ouverture du chatbot n'affichait qu'un glyphe **`◈`** — un losange
décoratif ne disant rien de sa fonction. Un utilisateur devait le survoler pour
lire l'infobulle, ou cliquer pour découvrir. Il porte désormais une **icône de
bulle de dialogue suivie du mot « Assistant »**, dans une forme en pilule.

La hauteur reste fixée à **54 px** : la marge basse réservée dans `.content`
est calculée à partir d'elle, et la modifier aurait fait réapparaître le
recouvrement corrigé plus haut. L'anneau de pulsation, qui utilisait
`border-radius: 50 %`, a été passé en `--radius-pill` — sur un bouton large, un
rayon de 50 % dessine une ellipse au lieu d'épouser le contour.

Vérifié en navigateur : libellé « Assistant » présent, glyphe absent, panneau
toujours fonctionnel, **6 pages × 3 largeurs** sans recouvrement ni
débordement, contraste correct dans les deux thèmes.

---

# 4 · Architecture

```
<racine-du-dépôt>/
├── backend/app/
│   ├── api/v1/endpoints/      127 chemins, 135 opérations
│   ├── services/
│   │   ├── data/              marché, univers (32 instruments), synthétique
│   │   ├── indicators/        21 fonctions techniques
│   │   ├── forecasting/       5 architectures, conformal, découverte auto
│   │   ├── rl/                13 algorithmes, 14 agents livrés
│   │   ├── risk/              VaR avancée, GARCH, anomalies, stress.py
│   │   ├── recommendation/    fusion, direction, portefeuille, intelligence
│   │   ├── nlp/               actualités, sentiment
│   │   └── chat/              15 outils
│   └── db/                    9 tables
├── frontend/                  13 pages, design system par jetons
├── scripts/                   outils de mesure et d'étude
└── docs/                      documentation et rapports
```

**Sécurité** : bcrypt + JWT, `AuthGuardMiddleware` en **refus par défaut**
(toute route non explicitement publique est protégée). Vérifié : `stress.html`
anonyme → **303**, authentifié → **200**.

---

# 5 · Résultats mesurés

## 5.1 Prévision profonde — précision directionnelle (jeu de test)

| Checkpoint | Précision directionnelle | R² | RMSE |
|---|---:|---:|---:|
| `KO_gru_h5` | **67,57 %** | 0,165 | 0,0312 |
| `AAPL_lstm_h5` | **62,57 %** | 0,0398 | 0,0346 |
| `AAPL_gru_h5` | 54,67 % | **−0,0534** | 0,0473 |
| `AAPL_cnn_lstm_h5` | 54,55 % | 0,0021 | 0,0353 |
| `AAPL_transformer_h5` | 54,01 % | **−0,0034** | 0,0353 |
| `EURUSD_X_gru_h5` | 53,61 % | **−0,0280** | 0,0081 |
| `AAPL_tcn_h5` | 52,94 % | 0,0360 | 0,0346 |

> **Trois R² sont négatifs**, c'est-à-dire *moins bons que prédire la moyenne*.
> Ces modèles restent livrés et affichés tels quels : la page indique la
> précision mesurée et l'« écart par rapport au hasard », de sorte qu'un
> utilisateur voit immédiatement qu'un modèle à 52,94 % n'a pratiquement pas
> d'avantage.

## 5.2 Apprentissage par renforcement

- **13 algorithmes** implémentés, **14 agents** entraînés livrés (graine 42).
- Étude multi-graines (M122), n = 5 graines : Dueling DQN de référence
  **+8,66 %** (σ = 3,08 %, IC95 ± 3,82 %) contre variante régime **+5,21 %**
  (σ = 7,11 %, IC95 ± 8,83 %) — différence **non significative** (test apparié
  t = −0,804, p = 0,467 ; Welch t = −0,995, p = 0,362 ; d = −0,63).
  *Les intervalles se recouvrent largement et la variante régime est plus de
  deux fois plus dispersée que la référence.*
- **Buy & Hold : +21,17 %.** **Aucun des 13 algorithmes ne bat la stratégie
  passive** : le meilleur (A2C) atteint **+5,52 %**, soit moins du tiers.
  Deux algorithmes sont même négatifs (DQN −0,52 %, Dueling DQN −1,86 %).
- La famille distributionnelle (C51, QR-DQN, IQN, Rainbow) affiche un rendement
  **exactement 0,0 %** sur les 5 graines — comportement caractéristique d'un
  agent qui n'ouvre aucune position. *Le nombre de transactions n'est pas
  enregistré dans ces artefacts : l'absence de trading est donc **déduite** du
  rendement strictement nul, elle n'est pas mesurée directement.*

| Algorithme | Rendement moyen (5 graines) |
|---|---:|
| A2C | +5,52 % |
| SAC | +4,79 % |
| TRPO | +4,02 % |
| TD3 | +4,00 % |
| DDPG | +3,34 % |
| PPO | +2,94 % |
| Double DQN | +0,09 % |
| C51 / IQN / QR-DQN / Rainbow | 0,00 % |
| DQN | −0,52 % |
| Dueling DQN | −1,86 % |
| **Buy & Hold (référence)** | **+21,17 %** |

## 5.3 Risque

- **Aucun des 7 estimateurs de VaR n'est validé** (AAPL, 5 ans, 1 253
  observations, seuil 95 %). Tous échouent au test d'**indépendance de
  Christoffersen** : les dépassements se produisent en grappes, ce que le
  modèle suppose impossible. `model_valid = False` pour les sept.

| Estimateur | Taux de dépassement | Kupiec *p* | Indépendance *p* | Couverture cond. *p* |
|---|---:|---:|---:|---:|
| Historique | 5,58 % | 0,405 | **0,0032** | 0,0093 |
| Paramétrique | 4,79 % | 0,754 | **0,0295** | 0,089 |
| Cornish-Fisher | 6,78 % | **0,0139** | **0,0015** | 0,0003 |
| Student-t | 5,88 % | 0,212 | **0,0018** | 0,0035 |
| EWMA | 6,98 % | **0,0065** | **0,0089** | 0,0008 |
| Historique filtrée | 5,58 % | 0,405 | **0,0032** | 0,0093 |
| Monte-Carlo | 5,58 % | 0,405 | **0,0032** | 0,0093 |

  Lecture : plusieurs estimateurs passent Kupiec (le *nombre* de dépassements
  est correct) mais aucun ne passe l'indépendance (leur *répartition dans le
  temps* ne l'est pas). C'est un résultat négatif, rapporté comme tel.
- GARCH : la variante **GJR l'emporte** sur AAPL (5 ans), AIC mesuré
  **4 709,8** contre 4 721,8 (EGARCH) et 4 723,3 (GARCH). L'écart favorable à
  GJR confirme la présence d'un **effet de levier** : les chocs négatifs
  augmentent la volatilité future davantage que les chocs positifs de même
  amplitude.
- Cohérence du moteur de risque (mesurée par `scripts/diag_risk.py`) :
  Spearman(volatilité annualisée, score global) = **0,967**, monotonie en
  volatilité **vraie**, et **8 réponses distinctes pour les 8 périodes**
  sélectionnables — le sélecteur de période atteint réellement le calcul.
  *Une exécution antérieure donnait 0,967 → 0,976 selon la fenêtre de marché
  chargée ; la valeur bouge avec les données, elle n'est pas figée.*
- **Isolation Forest étiquette 2 % de n'importe quelle fenêtre** — propriété du
  paramétrage, pas une détection d'anomalie réelle.

## 5.4 Données réelles en base

9 tables — **87 utilisateurs**, **2 920 alertes**, 1 portefeuille,
3 transactions.

---

# 6 · Qualité logicielle et méthode de vérification

## 6.1 Trois exigences de méthode

**a) Vérifier avant d'affirmer.** Chaque fonctionnalité est testée dans un
**navigateur réel** (Playwright), pas seulement par lecture de code. Les
captures et mesures figurent dans `/home/user/shots/`.

**b) Prouver qu'un test attrape la régression (test par mutation).** Un test
qui passe ne prouve rien tant qu'on n'a pas vérifié qu'il **échoue** quand on
réintroduit le défaut. Mutations vérifiées sur les fonctionnalités récentes :

| Mutation appliquée | Test attendu | Résultat |
|---|---|---|
| Volatilité réutilisée comme mouvement attendu | direction | **échoue** ✔ |
| Confiance constante 0,75 | direction | **échoue** ✔ |
| Résolveur d'architecture ignoré | découverte | **échoue** ✔ |
| Substitution d'horizon autorisée | découverte | **échoue** ✔ |
| Verdict repassé au composite | direction | **échoue** ✔ |
| Crash neutralisé (1 jour ajouté) | stress | **échoue** ✔ |
| Volatilité codée en dur (11,66) | stress | **échoue** ✔ |
| Score de résilience codé en dur | stress | **échoue** ✔ |
| Correlation Spike inerte | stress | **échoue** ✔ |
| Contribution au risque par défaut à 0 | stress | **échoue** ✔ |
| Bouton de thème réintroduit en dur | thème | **échoue** ✔ |
| Jeton de couleur retiré du thème clair | thème | **échoue** ✔ |
| Repli de la barre revenu à 860 px | mise en page | **échoue** ✔ |
| Réserve du lanceur supprimée | mise en page | **échoue** ✔ |

**c) Signaler les faux positifs de ses propres outils.** Plusieurs alertes
produites par mes propres sondes se sont révélées **fausses** après
vérification, et l'honnêteté impose de le dire plutôt que de « corriger » un
non-problème :

- Une sonde par *bounding boxes* affirmait que le bouton de chat recouvrait
  encore du texte : le test faisant autorité (`elementsFromPoint`) a montré
  **aucun chevauchement** (bouton à y=924, éléments à y=474–598).
- L'icône de recherche `⌕` semblait chevaucher le texte : le champ a en réalité
  `padding-left: 32px`, l'icône est à 11 px — **aucun problème**.
- Un test de parité des thèmes exigeait que le thème clair redéclare `--font`,
  `--sp-*`, `--fs-*` : ces jetons sont **volontairement neutres**. Le test a été
  restreint aux couleurs.
- Un test anti-contradiction **réimplémentait la logique de fusion en local** :
  la mutation passait car il testait sa propre copie. Réécrit pour appeler le
  vrai `predict()`.
- Un test de matérialité du krach utilisait une série **gaussienne**, sans queue
  épaisse : le seuil échouait sur du code correct. Fixture remplacée par une
  loi de Student (df = 3).

## 6.2 Discipline appliquée

- **Ne jamais dégrader un message utile pour faire passer un test trop
  grossier** — resserrer le test à la place. Appliqué au test de comptage
  d'options, au test de dates absolues, au test de labels.
- **Ne jamais renommer/supprimer ce qui orphelinerait des données réelles** —
  d'où la conservation de `training.html`.
- **Signaler une demande factuellement erronée avant d'agir** — la suppression
  de la page Training Intelligence a fait l'objet d'une question préalable,
  car elle hébergeait **deux** fonctionnalités et la seule interface de gestion
  des checkpoints.

---

# 7 · Cahier des charges — conformité et divergences

> **Source.** Cet audit porte sur le document officiel
> `Cahier_des_charges.pdf` (7 pages) — *« Apprentissage par Renforcement
> Profond Adaptatif et Explicable pour la Gestion de Portefeuille Sensible au
> Risque »*. Chaque exigence y est numérotée selon la section d'origine du
> cahier des charges, confrontée à ce qui existe réellement dans le dépôt, avec
> la preuve de vérification. **Toute divergence est documentée** (CDC-1 …
> CDC-9), sans atténuation.

## 7.1 Objectifs spécifiques (§ 2.2)

| # | Exigence du cahier des charges | État | Preuve mesurée |
|---|---|---|---|
| O-1 | Environnement de simulation réaliste (multi-actifs, coûts, contraintes) | ✅ Conforme | `PortfolioEnv` + `TradingEnv` ; `transaction_cost = 0.001`, slippage modélisé |
| O-2 | Agent DRL adaptatif avec détection de changement de régime | ⚠️ **CDC-1** | `regime.py` + `regime_features.py` opérationnels ; **pas de MAML ni Mixture-of-Experts** |
| O-3 | Récompense sensible au risque (Sharpe/Sortino, CVaR, drawdown) | ✅ Conforme | `environment.py` : `cvar_penalty = 0.10`, `drawdown_penalty = 0.35`, `cvar_alpha = 0.05` |
| O-4 | XAI native et post-hoc justifiant chaque décision | ✅ Conforme | SHAP, `allocation_explain.py`, `regime_explain.py`, narratifs automatiques |
| O-5 | Tableau de bord pour gérants et fonctions de contrôle | ✅ Conforme | 13 pages, dont Risk & Alerts, Explainability, AI Stress Testing |
| O-6 | Validation : backtesting, walk-forward, stress tests | ⚠️ **CDC-2** | Backtesting ✅, stress tests ✅ (7 scénarios) ; **walk-forward présent pour la prévision, absent pour le RL** |
| O-7 | Documentation pour revue de gouvernance (SR 11-7, EBA/ACPR) | ⚠️ **CDC-3** | Model card, journal d'audit, registre de versions ✅ ; **dossier réglementaire formel non constitué** |

## 7.2 Description fonctionnelle (§ 4)

| # | Exigence | État | Preuve mesurée |
|---|---|---|---|
| F-1 | Données OHLCV multi-actifs | ✅ Conforme | 32 instruments, 6 classes (actions, ETF, crypto, matières premières, devises, indices) |
| F-2 | Données macro / sentiment (VIX, presse) | ✅ Conforme | `^VIX` au catalogue ; module `nlp/` (actualités + sentiment) |
| F-3 | Ingénierie de features (indicateurs, risque, régime) | ✅ Conforme | 21 fonctions d'indicateurs, volatilité réalisée, corrélations glissantes |
| F-4 | Qualité des données (aberrations, manquants) | ⚠️ **CDC-4** | Détection d'anomalies ✅, `ffill/bfill` ✅ ; **ajustement du biais de survivance absent** |
| F-5 | Formulation MDP (S, A, P, R, γ) | ✅ Conforme | Documentée en § II du rapport LaTeX M122 |
| F-6 | Action = vecteur de poids, somme à 1 | ✅ Conforme | `PortfolioEnv.action_space = Box(n_assets + 1)`, softmax dans `step` |
| F-7 | Algorithmes PPO, SAC, DDPG/TD3 (actions continues) | ✅ Conforme | **Les 4 présents** ; 13 algorithmes au total, dont 3 à actions continues |
| F-8 | Comparaison multi-algorithmes | ✅ Conforme | Balayage des 13 algorithmes, 65 exécutions (§ Full Catalogue Sweep) |
| F-9 | Apprentissage continu / réentraînement | ⚠️ **CDC-5** | Réentraînement manuel ✅ ; **pas de réentraînement périodique automatique** |
| F-10 | Journal d'audit horodaté (décision, features, version) | ✅ Conforme | `audit.py` : `log_rl_decision`, `log_allocation_decision`, `model_version` |
| F-11 | Résumés de décision en langage naturel | ✅ Conforme | Narratifs sur toutes les pages de décision |
| F-12 | Alertes et garde-fous | ⚠️ **CDC-6** | Seuils de risque et alertes ✅ (2 920 en base) ; **circuit-breaker absent** |

## 7.3 Contraintes (§ 6)

| # | Contrainte | État | Preuve |
|---|---|---|---|
| C-1 | Auditabilité, explicabilité, contrôle des biais | ✅ Conforme | Journal d'audit, XAI, registre de divergences |
| C-2 | RGPD sur les données alternatives | ✅ Sans objet | Aucune donnée personnelle collectée |
| C-3 | Infrastructure GPU + CI/CD | ⚠️ **CDC-7** | CI (`ruff` + 744 tests) ✅ ; **entraînement sur CPU**, pas de GPU |
| C-4 | Interopérabilité OMS/EMS | ✅ Hors périmètre | Explicitement exclu par le § 3.2 du cahier des charges |
| C-5 | Validation par comité des risques modèles | ❌ **CDC-8** | **Non réalisée** — nécessite une organisation, pas du code |

## 7.4 Indicateurs clés de performance (§ 8)

| # | KPI exigé | État | Valeur mesurée |
|---|---|---|---|
| K-1 | Rendement cumulé net vs benchmark | ✅ Mesuré | Meilleur RL **+5,52 %** vs Buy & Hold **+21,17 %** |
| K-2 | Ratios de Sharpe et Sortino | ✅ Mesurés | Sharpe 2,00 ± 1,19 ; Sortino 3,18 (référence) |
| K-3 | Maximum drawdown et temps de récupération | ⚠️ **CDC-9** | Drawdown ✅ (−6,93 % moyen) ; **temps de récupération non calculé** |
| K-4 | VaR et CVaR réalisés vs cibles | ✅ Mesurés | 7 estimateurs backtestés (tableau § 5.3) |
| K-5 | Délai de réaction après changement de régime | ❌ **CDC-1** | **Non mesuré** — voir CDC-1 |
| K-6 | Stabilité entre régimes de marché | ⚠️ Partiel | Étude régime-aware menée ; **un seul régime de test** (D-11) |
| K-7 | Fidélité des explications | ❌ **CDC-3** | **Non mesurée** quantitativement |
| K-8 | Compréhensibilité (enquête utilisateurs) | ❌ Hors périmètre | Nécessite un panel d'utilisateurs métier |
| K-9 | Taux de couverture des décisions expliquées | ✅ Conforme | 100 % : toute décision affichée porte ses contributions par signal |

## 7.5 Registre des divergences

**CDC-1 — Adaptation au régime : implémentée, mais pas par meta-learning.**
*Écart.* Le cahier des charges cite MAML ou Mixture-of-Experts. Le projet
détecte le régime (`regime.py`) et l'injecte comme **features d'observation**
(6 variables), sans recalibration méta-apprise. Le KPI K-5 (délai de réaction)
n'est donc pas mesurable en l'état.
*Résultat honnête.* L'augmentation régime a été **testée et n'améliore pas** la
performance : +5,21 % contre +8,66 % pour la référence, différence non
significative (p = 0,467). Investir dans MAML avant d'avoir un signal
exploitable serait prématuré.

**CDC-2 — Walk-forward absent côté RL.**
*Écart.* `walk_forward()` existe dans `forecasting/trainer.py` mais pas pour les
agents RL, qui utilisent une découpe train/test unique (400 / 101 barres).
*Conséquence.* La robustesse temporelle des agents n'est pas établie. Consigné
comme limite ; les conclusions RL sont bornées à « sur cette fenêtre ».

**CDC-3 — Gouvernance : outillage présent, dossier formel absent.**
*Écart.* Model card, journal d'audit horodaté et registre de versions existent.
Le **dossier de validation indépendante** exigé par SR 11-7 et la **mesure de
fidélité des explications** (K-7) ne sont pas produits.

**CDC-4 — Biais de survivance non traité.**
*Écart.* Le cahier des charges demande un ajustement du *survivorship bias*.
L'univers est fixe et ne contient que des instruments encore cotés : les
rendements historiques sont donc **optimistes par construction**.

**CDC-5 — Réentraînement automatique absent.**
*Écart.* Le réentraînement est déclenché manuellement (bouton *Retrain*, API).
Aucun ordonnanceur ni garde-fou anti-dérive de politique.

**CDC-6 — Circuit-breaker absent.**
*Écart.* Les seuils de risque et les alertes fonctionnent (2 920 alertes en
base), mais aucun mécanisme ne **suspend automatiquement** l'agent en cas de
comportement anormal. Sans exécution réelle d'ordres (hors périmètre § 3.2), le
risque opérationnel reste nul.

**CDC-7 — Entraînement CPU, pas GPU.**
*Écart.* Le cahier des charges suppose une infrastructure GPU. Tout tourne sur
CPU : ~7 s par modèle de prévision, ~8 s par exécution RL. C'est ce qui a
plafonné le budget à 8 épisodes (D-3) et limité l'étude à 5 graines (D-12).

**CDC-8 — Pas de validation par un comité des risques modèles.**
*Écart.* Exigence organisationnelle, hors de portée d'un livrable académique
mono-auteur. Le projet **prépare** les pièces (model card, audit, registre) sans
passer la revue.

**CDC-9 — Temps de récupération non calculé.**
*Écart.* Le maximum drawdown est mesuré, mais pas le *recovery time* exigé par
K-3. Grandeur non enregistrée : **rapportée comme indisponible** plutôt
qu'estimée.

## 7.6 Synthèse de conformité

| Catégorie du cahier des charges | Conforme | Divergence | Hors périmètre |
|---|---:|---:|---:|
| Objectifs spécifiques (7) | 4 | 3 | 0 |
| Description fonctionnelle (12) | 8 | 4 | 0 |
| Contraintes (5) | 2 | 1 | 2 |
| KPI (9) | 4 | 4 | 1 |
| **Total (33)** | **18** | **12** | **3** |

**Taux de conformité : 18/30 exigences applicables (60 %), 12 lignes en
divergence ramenées à 9 causes distinctes (CDC-1 … CDC-9), 0 exigence
silencieusement abandonnée.**

*Douze lignes des tableaux ci-dessus portent une divergence, mais certaines
partagent la même cause : CDC-1 explique à la fois O-2 et le KPI K-5, CDC-3
couvre O-7 et K-7. Le registre ci-dessus compte donc neuf entrées.*

Sur les 9 causes, cinq (CDC-1, CDC-2, CDC-4, CDC-7, CDC-9) sont des
**limites techniques ou matérielles assumées**, trois (CDC-3, CDC-5, CDC-6)
relèvent de l'industrialisation non atteinte, et une (CDC-8) est
organisationnelle. Aucune n'est masquée : chacune est reprise en section 9
(Limites connues) et, pour le volet RL, dans le registre D-1 … D-26 du rapport
LaTeX M122.

> **Écart le plus important à retenir.** Le cahier des charges vise un agent
> qui « optimise les décisions d'investissement ». Mesuré, **aucun des 13
> algorithmes ne bat une stratégie passive** (+5,52 % contre +21,17 %).
> L'objectif de performance financière n'est **pas atteint**, et ce résultat
> est publié plutôt que dissimulé : c'est la conclusion scientifique du
> travail, pas un défaut de mise en œuvre à corriger par un réglage.

---

# 8 · Défauts trouvés et corrigés

Sélection des défauts les plus significatifs, tous constatés à l'exécution.

| # | Défaut | Conséquence utilisateur | Statut |
|---|---|---|---|
| 1 | Volatilité réalisée présentée comme mouvement attendu | Fausse prédiction affichée avec aplomb | corrigé |
| 2 | Architecture rigidement fixée à `lstm` | 2 symboles déclarés « sans modèle » alors qu'ils en avaient un | corrigé |
| 3 | Verdict issu du composite, opposé au chiffre affiché | « INCREASE » au-dessus de « −0,88 % » | corrigé |
| 4 | Dilution du forecaster (30 %) | NEUTRAL quasi systématique | corrigé |
| 5 | `argmin` sur série avec NaN en tête | Krach = 1 jour au lieu de 21, scénario inoffensif | corrigé |
| 6 | Répétition excessive de l'épisode de krach | Drawdown de −98,6 %, irréaliste | corrigé |
| 7 | `?? 0` sur la contribution au risque | « 0 % de risque » au lieu de « non mesurable » | corrigé |
| 8 | Bouton de thème codé en dur et vide | Thème mort sur `stress.html` | corrigé |
| 9 | Repli de la barre supérieure à 860 px | Page traînée latéralement de 49 à 195 px | corrigé |
| 10 | Marge basse insuffisante | Bouton de chat masquant des données | corrigé |
| 11 | Docstring d'endpoint périmée | Contrat faux publié dans `/openapi.json` | corrigé |
| 12 | Test préexistant encodant le bug (`abs()` si NEUTRAL) | Hypothèse fausse devenue verte | resserré |
| 13 | Double arrondi du score composite | Le texte annonçait +0,334 quand le JSON publiait +0,335 | corrigé |

> **Défaut n° 13 — trouvé pendant la rédaction de ce rapport.** La charge utile
> arrondit `composite_score` à 4 décimales, tandis que le récit formatait la
> valeur brute à 3 décimales : sur SPY, la prose affichait **+0,334** alors que
> le JSON voisin publiait **+0,335**. Deux réponses divergentes pour la même
> grandeur. Le récit cite désormais les valeurs **publiées**, arrondies une
> seule fois.

---

# 9 · Limites connues

Ces limites sont **assumées et documentées**, non dissimulées.

## 9.1 Bloquantes avant un déploiement public

- **SMTP non configuré** : réinitialisation de mot de passe inopérante.
- **Authentification multifacteur absente.**
- **Rotation des jetons de rafraîchissement non implémentée.**
- **`SECRET_KEY` à changer** avant toute mise en ligne.
- **Verrouillage après échecs répétés** non implémenté.

## 9.2 Scientifiques

- **Couverture de prévision : 2 symboles sur 32**, un seul horizon (5 jours).
- **Aucun estimateur de VaR ne passe Christoffersen.**
- **Aucun agent RL ne bat Buy & Hold** (+21,17 %).
- **Étude M122 sous-dimensionnée** : n = 5 graines ; environ **61** seraient
  nécessaires pour une puissance de 80 %.
- **Multi-graines impossible sur les 14 agents livrés** (graine 42 figée à
  l'entraînement).
- **Isolation Forest** étiquette 2 % de toute fenêtre par construction.
- **Courbe d'évaluation** enregistrée mais non tracée dans l'interface
  Hyperparameters.

## 9.3 Méthodologiques

Les tests s'exécutent en `DATA_MODE=offline` sur données synthétiques : la
branche de code exercée peut différer d'une exécution manuelle sur données
réelles. Les tests concernés forcent explicitement la branche visée par des
doublures (`SignalContribution` factice, `monkeypatch` des constructeurs de
signaux).

---

# 10 · Conclusion

La plateforme livre une chaîne complète — données, indicateurs, prévision
profonde, apprentissage par renforcement, risque, portefeuille, explicabilité,
stress testing — sous **744 tests automatisés** et **0 violation d'analyse
statique**.

L'apport principal de cette dernière phase n'est pas seulement le volume de
fonctionnalités ajoutées (moteur de stress à 7 scénarios, découverte
automatique des modèles, page dédiée), mais la **discipline de véracité**
appliquée : à chaque fois qu'un chiffre risquait d'être plus flatteur que la
réalité — volatilité déguisée en prédiction, krach sans effet, drawdown
irréaliste, contribution nulle au lieu d'indisponible — le défaut a été
identifié, corrigé à la racine, et **verrouillé par un test dont on a prouvé
par mutation qu'il attrape la régression**.

Les résultats défavorables sont conservés et affichés : trois R² négatifs,
aucun estimateur de VaR validé, aucun agent RL battant une stratégie passive,
et 30 instruments sur 32 sans modèle entraîné. Une plateforme d'aide à la
décision financière qui masquerait ces faits serait plus dangereuse
qu'utile.

---

# Annexe A · Commandes de vérification

```bash
# Nombre de tests (744) et répartition par fichier
export PYTHONPATH=backend
python3 -m pytest backend/tests -q --collect-only \
  | grep -E "^tests/.*: [0-9]+$"

# Suite complète (à lancer en deux moitiés : ~5 min au total)
python3 -m pytest backend/tests/test_api.py backend/tests/test_auth_and_brand.py \
                  backend/tests/test_access_control.py -q
python3 -m pytest backend/tests/test_models_and_services.py backend/tests/test_chat.py \
                  backend/tests/test_intelligence.py backend/tests/test_quant.py \
                  backend/tests/test_data_and_indicators.py -q

# Analyse statique (périmètre officiel de la CI)
python3 -m ruff check backend/app backend/tests

# Couverture de prévision sur les 32 instruments
python3 scripts/forecaster_coverage.py

# Volume de code
for e in py js html css; do
  echo -n ".$e: "; find backend scripts frontend -name "*.$e" | xargs cat | wc -l
done

# Surface d'API (127 chemins / 135 opérations) — l'AuthGuard protège /openapi.json
curl -s -c /tmp/ck.txt -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"identifier":"<user>","password":"<pass>"}'
curl -s -b /tmp/ck.txt http://127.0.0.1:8000/openapi.json \
  | python3 -c "import sys,json;d=json.load(sys.stdin);\
print(len(d['paths']),'chemins',sum(len(v) for v in d['paths'].values()),'opérations')"

# Moteur de stress : les 7 scénarios sur un actif
python3 - <<'EOF'
from app.services.data.market_data import market_data_service as mds
from app.services.risk import stress
x = mds.get_history("AAPL", period="5y"); df = getattr(x, "df", x)
r = df["close"].pct_change().dropna()
for k in stress.SCENARIOS:
    o = stress.run({"AAPL": r}, {"AAPL": 1.0}, k)
    print(k, o["before"]["cvar_pct"], "->", o["after"]["cvar_pct"],
          "résilience", o["resilience"]["score"])
EOF
```

# Annexe B · Endpoints des nouvelles fonctionnalités

| Méthode | Chemin | Rôle |
|---|---|---|
| `GET` | `/api/v1/quant/stress-engine/scenarios` | Catalogue des 7 scénarios |
| `GET` | `/api/v1/quant/stress-engine/{symbols}` | Stress d'un actif ou d'un panier |
| `GET` | `/api/v1/signals/direction/{symbol}` | Prédiction directionnelle |

Paramètres du moteur de stress : `scenario`, `period`, `position_value`,
`confidence`, `weights`, `vol_multiplier`, `shock_pct`, `liquidity_penalty`,
`correlation_target`. Maximum **12 symboles** par exécution.

# Annexe C · Documents liés

| Document | Contenu |
|---|---|
| `docs/RAPPORT_PROJET.md` | Rapport détaillé initial (contexte, état de l'art) |
| `docs/ARCHITECTURE.md` | Architecture technique détaillée |
| `docs/UI_REDESIGN.md` | Refonte du design system |
| `docs/latex/M122_RL_MiniProject.pdf` | Étude RL multi-graines (38 pages, IEEE) |
| `README.md` | Installation, démarrage rapide, référence d'API |

---

*Fin du rapport.*
