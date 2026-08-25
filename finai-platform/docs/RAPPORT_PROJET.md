# KorisQuant AI — Rapport de projet

**Plateforme web d'analyse financière et de gestion de portefeuille par apprentissage automatique, apprentissage par renforcement profond et intelligence artificielle explicable**

El Maroufy Mohamed Yassir — Projet de fin d'études

---

> **Note sur les chiffres.** Toutes les valeurs de ce rapport ont été mesurées
> dans le dépôt à la date de rédaction, en exécutant le code. Les commandes qui
> les produisent sont indiquées en annexe A. Lorsqu'une valeur est un *choix de
> conception* et non une mesure, cela est dit explicitement. Aucun chiffre n'a
> été estimé ou arrondi à la hausse : un jury qui en vérifie un et le trouve
> faux cesse de croire tous les autres.

---

## Table des matières

1. [Introduction](#1--introduction)
2. [État de l'art](#2--état-de-lart)
3. [Architecture et conception](#3--architecture-et-conception)
4. [Réalisation : les moteurs](#4--réalisation--les-moteurs)
5. [Interface utilisateur](#5--interface-utilisateur)
6. [Validation et résultats](#6--validation-et-résultats)
7. [Qualité logicielle](#7--qualité-logicielle)
8. [Limites et perspectives](#8--limites-et-perspectives)
9. [Conclusion](#9--conclusion)
10. [Annexes](#10--annexes)

---

# 1 · Introduction

## 1.1 Contexte

La gestion quantitative de portefeuille combine trois activités qui, dans la
plupart des outils, vivent dans des logiciels séparés : l'**analyse de marché**
(indicateurs, régimes, corrélations), la **prévision** (modèles statistiques ou
neuronaux) et la **décision** (allocation, dimensionnement, gestion du risque).

Les plateformes grand public affichent des indicateurs sans expliquer leur
provenance. Les plateformes professionnelles sont fermées et coûteuses. Dans les
deux cas, le modèle est une boîte noire : l'utilisateur voit un signal
« ACHETER » sans savoir quelles données l'ont produit, sur quelle fenêtre, ni
avec quelle fiabilité.

Or la réglementation évolue en sens inverse. Le **SR 11-7** (Réserve fédérale
américaine) et les orientations **EBA/ACPR** sur la gouvernance des modèles
imposent la traçabilité : quel modèle, quelle version, quelles données, quelles
performances hors échantillon.

## 1.2 Problématique

> Peut-on construire une plateforme d'aide à la décision financière dont chaque
> chiffre affiché est traçable jusqu'à son calcul, y compris — et surtout —
> lorsque le résultat est défavorable ?

La difficulté n'est pas d'entraîner un modèle. Elle est de construire un système
qui **refuse de mentir** : qui affiche « indisponible » plutôt qu'un zéro
trompeur, qui signale qu'un agent perd contre une stratégie passive, et qui
documente qu'aucun estimateur de VaR ne passe le test d'indépendance.

## 1.3 Objectifs

| # | Objectif | Chapitre |
|---|---|---|
| O1 | Environnement MDP réaliste (coûts, slippage, contraintes) | §4.3 |
| O2 | Agent DRL adaptatif, sensible au régime de marché | §4.3 |
| O3 | Récompense sensible au risque (Sharpe, Sortino, CVaR, drawdown) | §4.3 |
| O4 | IA explicable native **et** post-hoc (SHAP, LIME, contrefactuels) | §4.5 |
| O5 | Tableau de bord professionnel et cohérent | §5 |
| O6 | Validation rigoureuse (walk-forward, tests de résistance) | §6 |
| O7 | Gouvernance des modèles (versionnage, journal d'audit, reproductibilité) | §4.6 |

## 1.4 Ce que la plateforme ne fait délibérément pas

Cette liste est aussi importante que la précédente :

- **Pas de courtage réel.** Le portefeuille est en papier ; aucun ordre n'est transmis à un marché.
- **Pas de conseil en investissement.** Les sorties sont des estimations statistiques.
- **Pas d'authentification multifacteur** (voir §8.1) — bloquant pour un déploiement public.
- **Pas de réinitialisation de mot de passe en libre-service** : la fonction a été retirée après la découverte d'une faille de prise de contrôle de compte (§7.3).

---

# 2 · État de l'art

## 2.1 Prévision de séries financières

Les séries de prix sont **non stationnaires** et proches d'une marche aléatoire.
Le débat oppose l'hypothèse d'efficience des marchés à la persistance de
régularités exploitables (momentum, retour à la moyenne).

Cinq architectures ont été retenues et implémentées :

| Architecture | Principe | Référence |
|---|---|---|
| **LSTM** | Portes d'oubli, dépendances longues | Hochreiter & Schmidhuber (1997) |
| **GRU** | Variante allégée du LSTM | Cho et al. (2014) |
| **TCN** | Convolutions dilatées causales | Bai et al. (2018) |
| **Transformer** | Auto-attention | Vaswani et al. (2017) |
| **CNN-LSTM** | Extraction locale puis mémoire | hybride |

**Métrique retenue :** la *directional accuracy* (DA), et non le RMSE. Un modèle
peut avoir une erreur faible tout en se trompant systématiquement de sens — ce
qui est sans valeur pour décider. Voir §6.2 : le R² est proche de zéro, parfois
négatif, et le rapport le dit.

## 2.2 Apprentissage par renforcement pour le trading

Le trading se formule en processus de décision markovien (MDP) : état = fenêtre
d'observation du marché + position, action = acheter/conserver/vendre (ou un
poids continu), récompense = variation de richesse pénalisée par le risque.

Trois familles, 13 algorithmes implémentés (§4.3) :

- **Value-based** — DQN, Double DQN, Dueling DQN
- **Distributional** — C51, QR-DQN, IQN, Rainbow
- **Policy gradient / Actor-critic** — PPO, A2C, TRPO, DDPG, TD3, SAC

L'approche distributionnelle (Bellemare et al., 2017) est particulièrement
pertinente en finance : elle apprend la **distribution** des retours et non
seulement leur espérance, ce qui donne accès aux quantiles de queue.

## 2.3 Mesure du risque

- **VaR** : quantile de perte. Non *cohérente* au sens d'Artzner et al. (1999) — elle viole la sous-additivité.
- **CVaR / Expected Shortfall** : espérance des pertes au-delà de la VaR. Cohérente, et privilégiée ici.
- **Backtesting** : Kupiec (1995) pour le *nombre* de dépassements, Christoffersen (1998) pour leur *indépendance*, et le feu tricolore de Bâle.

Le résultat mesuré au §6.3 est instructif : tous les estimateurs passent Kupiec
ou s'en approchent, **aucun ne passe Christoffersen**. Les pertes arrivent en
rafales.

## 2.4 IA explicable

- **SHAP** (Lundberg & Lee, 2017) — attribution additive fondée sur les valeurs de Shapley.
- **LIME** (Ribeiro et al., 2016) — substitut linéaire local.
- **Contrefactuels** — « quel changement minimal inverserait la décision ? »

La plateforme implémente les trois, plus une explication **native** côté RL :
l'attribution par knockout des variables de régime (§4.5).

---

# 3 · Architecture et conception

## 3.1 Vue d'ensemble

Architecture en couches, monodépôt, sans étape de build côté client :

```
┌──────────────────────────────────────────────────────────┐
│  PRÉSENTATION — 12 pages HTML, 20 modules JS, 4 CSS      │
│  Plotly.js · thème clair/sombre · aucun framework        │
└───────────────────────────┬──────────────────────────────┘
                            │ HTTP/JSON (cookie HttpOnly)
┌───────────────────────────▼──────────────────────────────┐
│  API — FastAPI · 123 chemins · 131 opérations HTTP       │
│  AuthGuardMiddleware (refus par défaut)                  │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  SERVICES — 11 paquets métier                            │
│  data · indicators · forecasting · rl · risk · nlp       │
│  recommendation · xai · alerts · chat · notifications    │
└───────────────────────────┬──────────────────────────────┘
┌───────────────────────────▼──────────────────────────────┐
│  PERSISTANCE — SQLite (9 tables) · cache Parquet         │
│  modèles .pt/.zip · configs YAML                         │
└──────────────────────────────────────────────────────────┘
```

## 3.2 Répartition des endpoints

Mesuré sur le schéma OpenAPI de l'application en fonctionnement :

| Module | Chemins | | Module | Chemins |
|---|---|---|---|---|
| `alerts` | 13 | | `training` | 8 |
| `market` | 11 | | `auth` | 8 |
| `hyperparams` | 11 | | `news` | 5 |
| `forecast` | 10 | | `risk` | 5 |
| `rl` | 10 | | `signals` | 4 |
| `portfolio` | 9 | | `chat` | 4 |
| `quant` | 9 | | `dashboard` | 3 |
| `intel` | 9 | | `xai` | 3 |
| | | | **Total** | **123** |

## 3.3 Modèle de données

9 tables SQLite. Volumétrie réelle constatée :

| Table | Colonnes | Lignes | Rôle |
|---|---|---|---|
| `users` | 14 | 87 | comptes (bcrypt + JWT) |
| `alerts` | 10 | 2 503 | alertes déclenchées |
| `alert_rules` | 21 | 7 | règles configurées |
| `portfolios` | 11 | 1 | portefeuilles papier |
| `positions` | 8 | 3 | lignes détenues |
| `transactions` | 10 | 3 | ordres exécutés |
| `recommendation_log` | 17 | 3 | journal d'audit des décisions IA |
| `model_registry` | 10 | 0 | registre (alimenté au fil des runs) |
| `portfolio_snapshots` | 9 | 0 | photos de valorisation |

> La table `recommendation_log` matérialise l'exigence de gouvernance : chaque
> recommandation enregistre sa source, la version du modèle, l'algorithme, le
> régime détecté et son influence, les métriques de risque et l'explication.

## 3.4 Stratégie d'accès aux données

Cascade à quatre niveaux, dans cet ordre :

1. **Cache mémoire** (TTL court)
2. **Cache disque Parquet** — `data/cache/`
3. **Fournisseur en ligne** — Yahoo Finance
4. **Générateur synthétique** — dernier recours, **toujours étiqueté `SIMULATED`** dans l'interface

Cette étiquette est un principe : l'utilisateur doit toujours savoir s'il
regarde des données réelles.

## 3.5 Décision d'architecture : période d'affichage ≠ période de calcul

Décision structurante, formulée par le porteur du projet :

> « La période sélectionnée ne doit contrôler que les données affichées, pas la
> quantité d'historique utilisée par les modèles analytiques. En aucun cas la
> plateforme ne doit basculer sur des données synthétiques ou afficher "—"
> simplement parce qu'une période courte a été choisie. »

Implémentation : `backend/app/utils/periods.py` sépare `analysis_window()`
(affichage) de `model_bars()` (plancher de barres requis par chaque modèle).
Effet mesuré au §6.4 : **8 réponses distinctes pour 8 périodes**, contre 4 sur
11 auparavant.

---

# 4 · Réalisation : les moteurs

## 4.1 Données de marché et indicateurs

- **32 instruments** : 13 actions, 4 ETF, 4 cryptos, 4 devises, 4 indices, 3 matières premières.
- **21 fonctions** dans `technical.py`, dont 19 indicateurs : SMA, EMA, WMA, RSI, MACD, stochastique, ROC, Williams %R, CCI, bandes de Bollinger, True Range, ATR, canaux de Keltner, volatilité historique, ADX, Ichimoku, OBV, VWAP, Money Flow Index.

## 4.2 Prévision profonde

5 architectures (§2.1), entraînement avec arrêt précoce, normalisation ajustée
**sur le train uniquement** — condition nécessaire pour éviter la fuite de
données. Prédiction du *rendement*, jamais du prix brut.

**Prédiction conforme (ACI)** : les intervalles de confiance sont calibrés par
*Adaptive Conformal Inference*, qui garantit une couverture empirique sans
hypothèse de distribution.

## 4.3 Apprentissage par renforcement

### Catalogue complet — 13 algorithmes, tous disponibles

| Clé | Nom | Famille | Espace d'action |
|---|---|---|---|
| `dqn` | DQN | value_based | discret |
| `double_dqn` | Double DQN | value_based | discret |
| `dueling_dqn` | Dueling DQN | value_based | discret |
| `c51` | C51 | distributional | discret |
| `qr_dqn` | QR-DQN | distributional | discret |
| `iqn` | IQN | distributional | discret |
| `rainbow` | Rainbow | distributional | discret |
| `ppo` | PPO | policy_gradient | les deux |
| `a2c` | A2C | actor_critic | les deux |
| `trpo` | TRPO | policy_gradient | les deux |
| `ddpg` | DDPG | actor_critic | continu |
| `td3` | TD3 | actor_critic | continu |
| `sac` | SAC | actor_critic | continu |

### Environnement MDP

- **État** : fenêtre d'indicateurs normalisés + position + trésorerie.
- **Actions** : discret {vendre, conserver, acheter} ou poids continu.
- **Frictions** : frais de transaction et slippage appliqués à chaque ordre.
- **Récompense** : variation de richesse − pénalité de risque (drawdown, CVaR).

### Conscience du régime de marché

Module `regime_features.py` : un classifieur de régime enrichit l'observation de
**6 variables** (mono-actif) ou **3n+4** (panier de n actifs).

Coût mesuré : `_classify` = 9,8 ms par appel ; le précalcul par barre ramène
752 barres de 7,4 s à **1,19 s**.

Cette extension est **optionnelle** (`EnvConfig.regime_aware`, défaut `False`) :
l'activer par défaut aurait cassé les 11 agents déjà entraînés, dont la couche
d'entrée attend 36 dimensions et non 42.

Effet mesuré sur la récompense : conserver une position pendant un krach est
pénalisé de **2 885 points** de plus qu'en mode standard.

## 4.4 Moteur de risque

### Score de risque global — refonte

L'ancien score valait `max(krach, bulle, anomalie)`, chaque terme étant relatif à
l'historique propre de l'actif. Conséquence absurde mesurée : **NVDA à 36,6 % de
volatilité annualisée était classé `low`**, tandis que **GLD à 28,5 % était
classé `high`**.

Le nouveau score est une **moyenne pondérée de 8 contributeurs sur des échelles
absolues**, et publie chaque contribution — leur somme redonne le score.

Corrélation de Spearman entre volatilité annualisée et score global :

| | Avant | Après |
|---|---|---|
| Spearman(vol, score) | 0,76 | **0,976** |

### Mesures de risque disponibles

VaR (historique, paramétrique, Cornish-Fisher, Student-t, EWMA, Monte-Carlo,
simulation historique filtrée, théorie des valeurs extrêmes), CVaR, drawdown,
volatilité, bêta, ratios de Sharpe/Sortino/Calmar, détection d'anomalies
(Isolation Forest), détection de régime, indicateur de bulle.

## 4.5 IA explicable

| Méthode | Portée | Question à laquelle elle répond |
|---|---|---|
| SHAP | locale | Quelle variable a poussé *cette* prédiction ? |
| LIME | locale | Comment le modèle se comporte-t-il *autour* de cet état ? |
| Importance par permutation | globale | Quelles variables comptent en général ? |
| Contrefactuels | locale | Quel changement minimal inverserait la décision ? |
| Attribution de régime | native (RL) | Le régime a-t-il été décisif, contributif ou négligeable ? |

L'attribution de régime (`regime_explain.py`, `allocation_explain.py`) procède
par **knockout** : on neutralise une variable et on mesure le déplacement de la
décision. Pour les paniers, le turnover est défini comme la **moitié** de la
norme L1 du changement d'allocation — sans cela, chaque transfert est compté
deux fois.

## 4.6 Gouvernance et reproductibilité

- **19 fichiers YAML** : `defaults.yaml` + 13 algorithmes + 5 profils.
- **Auto-provisionnement** : `ensure_configs()` recrée `configs/` s'il manque, de façon purement additive.
- **Empreinte** : chaque run enregistre profil, graine, hyperparamètres résolus et une empreinte SHA-256 du checkpoint.
- **Journal d'audit** : `log_rl_decision()` / `log_allocation_decision()`.

---

# 5 · Interface utilisateur

## 5.1 Périmètre

12 pages : 10 pages de tableau de bord (Market Overview, Technical Analysis, AI
Forecasting, RL Agent, Recommendations, Explainability, Portfolio, Risk &
Alerts, Hyperparameters, Training Intelligence) plus la page d'accueil publique
et l'écran d'authentification.

## 5.2 Système de design

Refonte complète, **sans aucune modification du backend** :

- **Jetons de thème** (`theme.css`) : échelle de surfaces à 5 niveaux, échelle typographique nommée, échelle d'espacement de 4 px, dégradés et ombres. Chaque jeton est déclaré **deux fois** (sombre / clair) — la parité est vérifiée par test.
- **Aucune couleur codée en dur** hors du fichier de thème : un test échoue si un hexadécimal apparaît dans un script de page.
- **Boutons** : une famille, cinq intentions (primaire, secondaire, succès, avertissement, danger) partageant géométrie, mouvement, état focus et état désactivé.
- **Icônes** : jeu SVG au trait, **inliné** (l'aperçu intégré n'a pas d'accès réseau ; une police d'icônes distante échouerait silencieusement).
- **Mouvement** : survol, pression, ondulation, ouverture de menus, transitions d'onglets et de page. `prefers-reduced-motion` désactive **toutes** les animations.

## 5.3 Hiérarchie de la page Recommandations

Ordre de lecture imposé par le design : **action** (plus grande taille, couleur
sémantique) → **confiance** (anneau) → **preuves** (score, accord, risque) →
**prix** (discret, aligné à droite). Le bandeau peint un lavis coloré selon
l'appel — vert/rouge/neutre — pour que la décision soit reconnaissable avant
toute lecture.

## 5.4 Accessibilité

- Contrastes mesurés, pas estimés. Exemples : libellé de bouton désactivé **8,93:1** (sombre) et **8,55:1** (clair) ; icône de navigation inactive **4,63:1** / **5,17:1** (seuil AA pour composants graphiques : 3,0:1).
- Anneau de focus visible sur tout élément interactif.
- Icônes décoratives marquées `aria-hidden` — le libellé porte le sens.

---

# 6 · Validation et résultats

> Cette section rapporte **aussi les résultats défavorables**. C'est le cœur de
> la démarche.

## 6.1 Apprentissage par renforcement — 14 agents entraînés

| # | Symbole(s) | Algo | Santé | Note | Rendement | Statut |
|---|---|---|---|---|---|---|
| 1 | AAPL,MSFT,SPY | PPO | 65,9 | good | +15,49 % | Épisodes insuffisants |
| 2 | MSFT | Dueling DQN | 64,6 | good | +26,59 % | **Instable** |
| 3 | AAPL | PPO | 64,5 | good | +9,36 % | En progression |
| 4 | AAPL,MSFT | PPO | 62,4 | good | +16,62 % | Plateau |
| 5 | AAPL,MSFT,SPY | SAC | 61,6 | fair | +12,94 % | En progression |
| 6 | AAPL | DDPG | 60,9 | fair | +6,06 % | En progression |
| 7 | AAPL,GC=F,SPY | SAC | 60,4 | fair | **−8,63 %** | En progression |
| 8 | AAPL | TD3 | 58,5 | fair | +5,52 % | En progression |
| 9 | AAPL,MSFT,SPY | DDPG | 54,7 | fair | +4,20 % | Épisodes insuffisants |
| 10 | AAPL,MSFT,SPY | TD3 | 54,3 | fair | +4,08 % | Épisodes insuffisants |
| 11 | AAPL,GC=F,MSFT,SPY | SAC | 54,1 | fair | **−5,28 %** | En progression |
| 12 | AAPL | C51 | 36,7 | poor | 0,00 % | Épisodes insuffisants |
| 13 | AAPL | SAC | 30,9 | poor | +4,83 % | **Instable** |
| 14 | AAPL | Dueling DQN | 23,4 | poor | +3,08 % | **Instable** |

**Santé moyenne de la flotte : 53,8 / 100.** Répartition des statuts : 6 en
progression, 4 sans assez d'épisodes, 3 instables, 1 en plateau.

**Deux agents perdent de l'argent** (−8,63 % et −5,28 %) et le tableau de bord
les affiche comme tels. Le score de santé intègre `alpha_vs_equal_weight` : sans
cela, un panier SAC perdant 8,6 % se classait 4ᵉ avec 80,6 %.

**Limite majeure — multi-graines impossible.** L'API le déclare explicitement :

> « 1 graine distincte sur 14 runs. La moyenne ± écart-type exige au moins 3
> graines indépendantes ; en dessous, la dispersion ne mesure rien. »

Tous les entraînements ont utilisé la graine 42. Aucun intervalle de confiance
n'est donc publiable sur ces résultats.

## 6.2 Prévision — 7 modèles entraînés

| Symbole | Modèle | DA % | R² | RMSE | MAE | n |
|---|---|---|---|---|---|---|
| KO | GRU | **67,57** | 0,165 | 0,0312 | 0,0255 | 37 |
| AAPL | LSTM | **62,57** | 0,040 | 0,0346 | 0,0266 | 187 |
| AAPL | GRU | 54,67 | **−0,053** | 0,0473 | 0,0353 | 75 |
| AAPL | CNN-LSTM | 54,55 | 0,002 | 0,0353 | 0,0279 | 187 |
| AAPL | Transformer | 54,01 | **−0,003** | 0,0353 | 0,0278 | 187 |
| EURUSD=X | GRU | 53,61 | **−0,028** | 0,0081 | 0,0063 | 194 |
| AAPL | TCN | 52,94 | 0,036 | 0,0346 | 0,0271 | 187 |

**Lecture honnête :**

- Le LSTM se détache nettement sur AAPL (62,57 % contre ~53–55 % pour les autres).
- **Trois modèles ont un R² négatif** : ils font moins bien que prédire la moyenne. C'est le comportement attendu sur des rendements quasi imprévisibles, et cela justifie de juger sur la DA plutôt que sur le R².
- Le 67,57 % de KO repose sur **37 échantillons seulement** — statistiquement fragile, à ne pas surinterpréter.

## 6.3 VaR — aucun estimateur ne passe Christoffersen

Backtest sur AAPL, 1 254 observations, niveau 95 %, fenêtre glissante 250 :

| Estimateur | Taux de dépassement | Kupiec p | Indépendance p | Zone Bâle |
|---|---|---|---|---|
| Historique | 5,58 % | 0,409 ✓ | **0,0032 ✗** | verte |
| Paramétrique | 4,78 % | 0,748 ✓ | **0,0293 ✗** | verte |
| Cornish-Fisher | 6,77 % | **0,014 ✗** | **0,0015 ✗** | verte |
| Student-t | 5,88 % | 0,215 ✓ | **0,0018 ✗** | verte |
| EWMA | 6,97 % | **0,007 ✗** | **0,0088 ✗** | verte |

**Conclusion, telle que la plateforme la formule elle-même :**

> « Tous les estimateurs passent ou frôlent le test de *comptage* des
> dépassements, mais les dépassements se regroupent dans le temps
> (Christoffersen p < 0,05). C'est la limite bien documentée de la VaR
> quotidienne sur actions : les pertes arrivent par rafales. Le chiffre est
> utilisable pour le dimensionnement courant, **pas** pour survivre à une
> crise. »

Recommandation retournée par l'API : aucune méthode recommandée ; privilégier la
simulation historique filtrée ou la théorie des valeurs extrêmes, et associer la
VaR aux tests de résistance et au score de krach.

## 6.4 Moteur de risque — trois validations

**(a) Séparation entre actifs** — 10 actifs classés par volatilité annualisée :

| Actif | Vol. ann. | VaR 95 % | Drawdown max | Score | Niveau |
|---|---|---|---|---|---|
| JNJ | 18,3 % | −1,74 % | −11,0 % | 0,102 | low |
| AAPL | 25,3 % | −2,02 % | −13,8 % | 0,250 | low |
| GLD | 28,5 % | −3,04 % | −26,4 % | 0,329 | moderate |
| NVDA | 36,6 % | −3,77 % | −20,2 % | 0,300 | low |
| BTC-USD | 37,6 % | −3,55 % | −39,6 % | 0,439 | moderate |
| TSLA | 46,4 % | −4,51 % | −39,1 % | 0,489 | moderate |
| ETH-USD | 51,4 % | −4,84 % | −53,3 % | 0,512 | high |
| ^VIX | 125,0 % | −12,09 % | −52,0 % | 0,657 | high |

**Spearman(volatilité, score global) = 0,976** · 10 scores distincts sur 10.

**(b) Monotonie contrôlée** — une seule trajectoire, seul σ varie :

| Vol. annualisée | 5 % | 10 % | 20 % | 40 % | 80 % | 120 % |
|---|---|---|---|---|---|---|
| Score | 0,040 | 0,089 | 0,224 | 0,416 | 0,641 | 0,669 |
| Niveau | low | low | low | moderate | high | high |

**Monotone : vrai.** Amplitude 0,040 → 0,669.

**(c) Sensibilité à la période** (AAPL) — 8 sélections, 8 réponses distinctes :

| Sélection | Barres krach | Barres bulle | Krach | Bulle | **Global** |
|---|---|---|---|---|---|
| 1mo | 61 | 201 | 0,452 | 0,203 | **0,2742** |
| 3mo | 63 | 201 | 0,438 | 0,203 | **0,2716** |
| 6mo | 126 | 201 | 0,393 | 0,203 | **0,2635** |
| ytd | 150 | 201 | 0,379 | 0,203 | **0,2610** |
| 1y | 252 | 252 | 0,360 | 0,221 | **0,2495** |
| 3y | 756 | 756 | 0,402 | 0,297 | **0,3630** |
| 5y | 1 260 | 1 260 | 0,371 | 0,343 | **0,3637** |
| 10y | 2 520 | 2 520 | 0,374 | 0,163 | **0,3692** |

---

# 7 · Qualité logicielle

## 7.1 Volumétrie mesurée

| Composant | Fichiers | Lignes |
|---|---|---|
| Backend Python (`app/`) | 97 | 21 519 |
| Tests Python | 10 | 8 437 |
| Frontend JavaScript | 20 | 7 579 |
| Frontend CSS | 4 | 3 165 |
| Frontend HTML | 12 | 2 351 |
| Scripts | 7 | 921 |
| Configurations YAML | 20 | 384 |
| **Total** | **170** | **44 356** |

Le ratio test/code applicatif est de **0,39** (8 437 / 21 519).

## 7.2 Tests automatisés

**700 tests, tous passants**, répartis en 8 fichiers :

| Fichier | Tests | Domaine |
|---|---|---|
| `test_api.py` | 226 | endpoints, contrats de réponse |
| `test_models_and_services.py` | 125 | services métier |
| `test_access_control.py` | 103 | authentification, autorisation, UI |
| `test_chat.py` | 86 | assistant, outils, quotas |
| `test_intelligence.py` | 49 | diagnostic d'entraînement |
| `test_auth_and_brand.py` | 41 | auth, cohérence de marque |
| `test_data_and_indicators.py` | 39 | données, indicateurs |
| `test_quant.py` | 31 | VaR, GARCH, régimes |

Analyse statique : `ruff` (E, F, W, I, UP, B, C4, SIM) — **aucune violation**.

Un test vérifie que **le nombre affiché sur le site correspond au nombre réel de
tests collectés** : une statistique marketing invérifiable serait exactement le
genre de chose que ce projet ne doit pas livrer.

## 7.3 Bugs réels trouvés et corrigés

Plus de 80 défauts ont été identifiés et corrigés. Les plus significatifs :

**Sécurité**
- **Prise de contrôle de compte** via `forgot-password` — fonctionnalité supprimée.
- **XSS** dans `/auth.html?error=` — le paramètre était injecté en HTML.
- `CORS_ORIGINS=*` en configuration par défaut.
- Cookie CSRF manquant.

**Correction scientifique**
- **Fuite de données RL** : `df.iloc[split-60:]` faisait déborder la fenêtre d'entraînement sur le test.
- **Sortino** divisait par l'écart-type des pertes au lieu de la *downside deviation* : une série perdant 2 % de façon constante obtenait 0,0 au lieu de 12,401.
- **VaR retournait 0,0** sur données courtes — soit « cet actif ne peut pas perdre d'argent ». Retourne désormais `None`.
- **Bêta = 0,0** sans dates communes — « décorrélé du marché » au lieu de « jamais comparé ».
- **`profit_factor = 0.0`** en l'absence de jour perdant, alors que c'est le meilleur cas possible.

**Cohérence de l'interface**
- Le sélecteur de période ne changeait rien : 7 des 11 plages déclaraient `compute="2y"`.
- La page Recommandations n'envoyait jamais la période au backend.
- Un bouton désactivé pour raison d'*état* affichait une barre de progression **infinie**.
- Le logo affichait « Fi » — les initiales du nom porté avant le changement de marque — sur les 10 pages du tableau de bord, alors que l'accueil et l'écran de connexion affichaient déjà « K ».

## 7.4 Faux positifs de mes propres tests

Un test qui passe pour la mauvaise raison est pire qu'un test absent. Cas
rencontrés et corrigés :

- Un test comparait deux versions qui **fuyaient toutes les deux** : il passait contre une implémentation délibérément fautive.
- Un test de turnover affirmait `<= 1.0` (borne lâche) au lieu de la définition : une mutation `/2` n'était pas détectée.
- Un test de mutation a **corrompu le vrai `configs/profiles/default.yaml`** — désormais isolé via `tmp_path`.
- Un test cherchant une fenêtre courte recevait 120 barres de la fixture : il passait contre le bug qu'il devait attraper.

**Méthode adoptée : la preuve par mutation.** Chaque nouveau test de régression
est validé en réintroduisant délibérément le bug ; s'il ne tombe pas, il ne vaut
rien.

---

# 8 · Limites et perspectives

## 8.1 Limites — bloquantes avant un déploiement public

| Limite | Conséquence |
|---|---|
| **Pas de MFA** | Un mot de passe compromis suffit |
| **`SECRET_KEY` par défaut** | Tout jeton de session est falsifiable |
| **Pas de rotation des refresh tokens** | Un jeton volé reste valide |
| **Pas de verrouillage après échecs** | Attaque par force brute possible |
| **SMTP non configuré** | Les liens de vérification sont journalisés, pas envoyés |

## 8.2 Limites scientifiques

- **Aucune statistique multi-graines** : graine figée à 42 sur les 14 runs.
- **Aucun estimateur de VaR ne passe Christoffersen.**
- **Isolation Forest étiquette 2 % de n'importe quelle fenêtre** (`contamination=0.02`) : le nombre d'anomalies est en partie un artefact du paramètre.
- **Durée d'entraînement jamais chronométrée** — l'estimation repose sur un coût par pas calibré (`0,002476 s` natif, `0,001108 s` SB3).
- Les seuils de bandes du score de risque sont des **choix de calibration**, pas des mesures.

## 8.3 Perspectives

**Court terme** — MFA, rotation des jetons, SMTP, chronométrage réel des
entraînements, campagne multi-graines (≥ 3 graines) pour publier des
intervalles de confiance.

**Moyen terme** — PostgreSQL + Redis, journalisation TensorBoard/W&B, analyse de
sentiment par transformeur (FinBERT), mise en page mobile.

**Long terme** — intégration courtier réel, RL multi-actifs à grande échelle,
détection de dérive et méta-apprentissage en ligne.

---

# 9 · Conclusion

La plateforme atteint les sept objectifs du §1.3 : environnement MDP réaliste
avec frictions, 13 algorithmes DRL dont une famille distributionnelle, récompense
sensible au risque et consciente du régime, explicabilité native et post-hoc,
tableau de bord professionnel, validation quantifiée et gouvernance des modèles
avec journal d'audit.

Le résultat technique — 44 356 lignes, 700 tests, 123 endpoints — n'est pas
l'apport principal. L'apport principal est **méthodologique** : un système qui
affiche `None` plutôt qu'un zéro trompeur, qui classe un agent perdant à la
place qu'il mérite, qui déclare qu'une statistique multi-graines est impossible
avec une seule graine, et qui documente qu'aucun estimateur de VaR ne passe le
test d'indépendance.

La corrélation de Spearman passée de 0,76 à 0,976 sur le score de risque, la
détection d'une fuite de données dans l'entraînement RL, et l'élimination
systématique des faux positifs de tests illustrent la même discipline : **un
chiffre qui n'a pas été vérifié n'est pas un résultat.**

---

# 10 · Annexes

## Annexe A — Commandes produisant les preuves

| But | Commande |
|---|---|
| Suite de tests (le nombre doit correspondre à celui affiché par l'UI) | `export PYTHONPATH=backend && python3 -m pytest backend/tests -q` |
| Analyse statique | `python3 -m ruff check backend/app backend/tests` |
| Auto-vérification de l'installation | `python3 scripts/check_install.py` |
| Diagnostic du moteur de risque (séparation, monotonie, sensibilité) | `python3 scripts/diag_risk.py` |
| Volumétrie du backend | `find backend/app -name '*.py' -exec cat {} + \| wc -l` |
| Schéma OpenAPI (session requise) | `curl -b cookies.txt localhost:8000/openapi.json` |

## Annexe B — Figures à préparer

| # | Figure | Source |
|---|---|---|
| 1 | Diagramme de cas d'utilisation | à dessiner |
| 2 | Architecture en couches | §3.1 |
| 3 | Diagramme de classes / ER (9 tables) | §3.3 |
| 4 | Séquence : scan de risque | à dessiner |
| 5 | Séquence : appel d'outil par l'assistant | à dessiner |
| 6 | Diagramme de déploiement | à dessiner |
| 7 | Arbre de décision de la couche données (4 niveaux) | §3.4 |
| 8–17 | Captures des 10 pages | `shots/redesign/` |
| 18 | Échelle de volatilité contrôlée | §6.4(b) |
| 19 | Détail du score de risque | `shots/risk_breakdown_light.png` |
| 20 | Page Risque complète | `shots/risk_no_dates.png` |

## Annexe C — Bibliographie indicative

Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* · Cho et al. (2014),
*GRU* · Bai et al. (2018), *TCN* · Vaswani et al. (2017), *Attention Is All You
Need* · Mnih et al. (2015), *DQN* · van Hasselt et al. (2016), *Double DQN* ·
Wang et al. (2016), *Dueling Networks* · Bellemare et al. (2017), *C51* ·
Dabney et al. (2018), *QR-DQN / IQN* · Hessel et al. (2018), *Rainbow* ·
Schulman et al. (2015, 2017), *TRPO / PPO* · Lillicrap et al. (2016), *DDPG* ·
Fujimoto et al. (2018), *TD3* · Haarnoja et al. (2018), *SAC* · Lundberg & Lee
(2017), *SHAP* · Ribeiro et al. (2016), *LIME* · Artzner et al. (1999),
*Coherent Measures of Risk* · Kupiec (1995) · Christoffersen (1998) ·
Bollerslev (1986), *GARCH* · Glosten, Jagannathan & Runkle (1993) ·
Gibbs & Candès (2021), *Adaptive Conformal Inference* · Comité de Bâle ·
Federal Reserve SR 11-7 · EBA/ACPR, gouvernance des modèles.

## Annexe D — Documents liés

- `docs/REPORT_OUTLINE.md` — plan détaillé et guide de rédaction
- `docs/UI_REDESIGN.md` — refonte de l'interface, bugs et vérifications
- `docs/ARCHITECTURE.md` — architecture technique
- `README.md`, `QUICKSTART.md` — installation et démarrage
