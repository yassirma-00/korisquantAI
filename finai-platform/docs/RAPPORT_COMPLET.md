# KorisQuant AI — Rapport de projet complet

**Apprentissage par Renforcement Profond Adaptatif et Explicable
pour la Gestion de Portefeuille Sensible au Risque**

*Plateforme web d'analyse financière et de gestion de portefeuille par
apprentissage automatique, apprentissage par renforcement profond et
intelligence artificielle explicable*

Réalisé par **El Maroufy Mohamed Yassir** — encadré par **Hamza Allaga**

Projet de fin d'études
Document arrêté au 13 août 2026

---

> **Note méthodologique.**
> Toutes les valeurs de ce rapport ont été **mesurées en exécutant le code** au
> moment de la rédaction. Les commandes qui les produisent figurent en
> [annexe A](#annexe-a--commandes-de-vérification). Quand une valeur est un
> *choix de conception* et non une mesure, cela est dit explicitement. Quand une
> grandeur n'a pas pu être mesurée, elle est rapportée comme **indisponible**
> avec la raison — jamais estimée.
>
> Les résultats défavorables (R² négatifs, tests de VaR échoués, agents RL
> battus par une stratégie passive) sont **rapportés tels quels**. Un rapport
> dont un seul chiffre se révèle faux perd la confiance du lecteur sur tous les
> autres.
>
> **Ce document fusionne** le rapport de conception (`RAPPORT_PROJET.md`) et le
> rapport de mesures (`RAPPORT_FINAL.md`) en un seul document continu :
> contexte et état de l'art, conception, réalisation, résultats mesurés, audit
> de conformité au cahier des charges, limites.

---

## Table des matières

1. [Introduction](#1--introduction)
2. [État de l'art](#2--état-de-lart)
3. [Architecture et conception](#3--architecture-et-conception)
4. [Réalisation : les moteurs](#4--réalisation--les-moteurs)
5. [Interface utilisateur](#5--interface-utilisateur)
6. [**Captures de l'application**](#6--captures-de-lapplication)
7. [Fonctionnalités récentes](#7--fonctionnalités-récentes)
8. [Résultats mesurés](#8--résultats-mesurés)
9. [Qualité logicielle et méthode de vérification](#9--qualité-logicielle-et-méthode-de-vérification)
10. [Cahier des charges — conformité et divergences](#10--cahier-des-charges--conformité-et-divergences)
11. [Défauts trouvés et corrigés](#11--défauts-trouvés-et-corrigés)
12. [Limites connues](#12--limites-connues)
13. [Conclusion](#13--conclusion)
14. [Annexes](#annexe-a--commandes-de-vérification)

---

## Chiffres clés (mesurés)

| Grandeur | Valeur |
|---|---:|
| Tests automatisés (tous verts) | **744** |
| Lignes de code (`.py` + `.js` + `.html` + `.css`) | **47 995** |
| Fichiers source | **156** |
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
| Conformité au cahier des charges | **18/30 (60 %)**, 9 divergences documentées |

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
- **Pas d'authentification multifacteur** (voir §12.1) — bloquant pour un déploiement public.
- **Pas de réinitialisation de mot de passe en libre-service** : la fonction a été retirée après la découverte d'une faille de prise de contrôle de compte (voir §11).

---

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
qui est sans valeur pour décider. Voir §8.1 : le R² est proche de zéro, parfois
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

---

# 3 · Architecture et conception

## 3.1 Vue d'ensemble

Architecture en couches, monodépôt, sans étape de build côté client :

```text
+----------------------------------------------------------+
|  PRESENTATION - 13 pages HTML, 21 modules JS, 4 CSS        |
|  Plotly.js - theme clair/sombre - aucun framework          |
+-----------------------------+------------------------------+
                              | HTTP/JSON (cookie HttpOnly)
+-----------------------------v------------------------------+
|  API - FastAPI - 127 chemins - 135 operations HTTP         |
|  AuthGuardMiddleware (refus par defaut)                    |
+-----------------------------+------------------------------+
                              |
+-----------------------------v------------------------------+
|  SERVICES - 11 paquets metier                              |
|  data - indicators - forecasting - rl - risk - nlp         |
|  recommendation - xai - alerts - chat - notifications      |
+-----------------------------+------------------------------+
                              |
+-----------------------------v------------------------------+
|  PERSISTANCE - SQLite (9 tables) - cache Parquet           |
|  modeles .pt/.zip - configs YAML                           |
+------------------------------------------------------------+
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

---

---

# 6 · Captures de l'application

> **Comment ces captures ont été produites.** Elles ne sont pas des maquettes.
> Chacune a été prise automatiquement dans un navigateur réel (Chromium via
> Playwright, viewport 1460 px, densité ×2) sur l'application en fonctionnement,
> connectée à un compte réel, en thème clair pour la lisibilité imprimée. Les
> chiffres visibles sont ceux que la plateforme a calculés au moment de la
> capture. Le script est reproductible : voir [annexe A](#annexe-a--commandes-de-vérification).

## 6.1 Pages publiques

### Page d'accueil

*[Capture : Page d'accueil publique — régénérable par `scripts/capture_screens.py`]*

Vitrine publique, seule page accessible sans authentification avec l'écran de
connexion. Elle annonce le nombre de tests passants — un chiffre vérifié par le
test `test_the_advertised_test_count_is_true`, qui échoue si la page et la suite
divergent.

### Connexion

*[Capture : Écran de connexion — régénérable par `scripts/capture_screens.py`]*

Authentification bcrypt + JWT. Le middleware `AuthGuardMiddleware` fonctionne en
**refus par défaut** : toute route non explicitement publique redirige vers cet
écran. Vérifié pour chaque page du tableau de bord.

## 6.2 Analyse de marché

### Vue de marché

*[Capture : Vue de marché — régénérable par `scripts/capture_screens.py`]*

Point d'entrée après connexion : indices, liste de suivi, recommandation fusionnée,
score de confiance, régime de marché détecté, carte de chaleur et alertes. Chaque
valeur porte sa source (`YAHOO`) — une donnée issue du moteur de repli synthétique
serait marquée `SIMULATED`, jamais confondue avec une donnée réelle.

### Analyse technique

*[Capture : Analyse technique — régénérable par `scripts/capture_screens.py`]*

Les 21 fonctions d'indicateurs (moyennes mobiles, RSI, MACD, bandes de Bollinger,
ATR, ADX, stochastique, MFI…) avec le consensus acheteur/vendeur et le détail des
votes par indicateur.

### Prévision profonde

*[Capture : Prévision par apprentissage profond — régénérable par `scripts/capture_screens.py`]*

Les 5 architectures (LSTM, GRU, TCN, Transformer, CNN-LSTM), entraînement à la
demande, intervalles de prédiction conformes et historique des courbes de perte.
La précision directionnelle mesurée est affichée telle quelle, y compris quand
elle avoisine le tirage à pile ou face.

## 6.3 Intelligence

### Agent par renforcement

*[Capture : Agent RL — régénérable par `scripts/capture_screens.py`]*

Les 13 algorithmes du catalogue, entraînement, backtest contre Buy & Hold, et
décision de l'agent avec ses valeurs Q. L'écart d'alpha face au benchmark passif
est affiché sans filtre — c'est là que le résultat négatif du projet est visible
pour l'utilisateur.

### Recommandations

*[Capture : Recommandations — régénérable par `scripts/capture_screens.py`]*

Fusion des quatre signaux (prévision, RL, technique, sentiment) avec leur
contribution pondérée, le dimensionnement de position suggéré et l'explication
SHAP. Chaque signal indisponible est déclaré comme tel plutôt que compté à zéro.

### Moteur de stress testing

*[Capture : Moteur de stress testing — régénérable par `scripts/capture_screens.py`]*

Panier AAPL 50 % / MSFT 30 % / GC=F 20 %, scénario *Market Crash*, 1 253
observations. Score de résilience **73/100**, CVaR stressée **3,03 %**, perte
additionnelle **223,90 $**, drawdown **48,07 %**. La décomposition d'Euler montre
que **GC=F pèse 20 % du capital mais seulement 4,5 % du risque** : le poids ne
mesure pas l'exposition. Les vulnérabilités et recommandations sont générées à
partir des grandeurs mesurées, jamais d'un texte générique.

> **Note sur les décimales.** Le drawdown lu sur cette capture (48,07 %) et
> celui du tableau du §8.4 (48,06 %) diffèrent d'un centième : les deux
> exécutions ont eu lieu à quelques heures d'intervalle et `period="5y"` est
> une fenêtre glissante, décalée d'une barre entre-temps. L'écart est signalé
> plutôt que masqué en réalignant l'un sur l'autre.

### Explicabilité

*[Capture : Explicabilité — régénérable par `scripts/capture_screens.py`]*

Valeurs SHAP, importance des variables et analyse de sensibilité. Réponse à
l'exigence O-4 du cahier des charges : justifier chaque décision d'allocation.

## 6.4 Gestion

### Portefeuille

*[Capture : Portefeuille — régénérable par `scripts/capture_screens.py`]*

Positions, courbe de capital, optimisation (max Sharpe, parité de risque,
variance minimale), matrice de corrélation et journal des transactions.

### Risque et alertes

*[Capture : Risque et alertes — régénérable par `scripts/capture_screens.py`]*

Risque global, score de risque de krach, indicateur de bulle, détection
d'anomalies sur le graphique de prix, et le tableau quantitatif complet : VaR et
CVaR à 95 % et 99 %, drawdown, bêta, alpha. Chaque score expose un lien
« How this is calculated » — un chiffre de risque sans méthode affichée n'est pas
auditable.

### Hyperparamètres

*[Capture : Gestionnaire d'hyperparamètres — régénérable par `scripts/capture_screens.py`]*

Configuration centralisée par algorithme, profils, et comparaison des exécutions.

### Training Intelligence

*[Capture : Training Intelligence — régénérable par `scripts/capture_screens.py`]*

Diagnostic de convergence, score de santé, classements, et gestionnaire de
checkpoints. **Cette page est volontairement retirée du menu** (voir CDC-8) mais
reste servie : elle est la seule interface de restauration et de suppression des
checkpoints RL, et la masquer sans la conserver aurait orphelin des fichiers
réels.

## 6.5 Assistant conversationnel

*[Capture : Assistant IA ouvert — régénérable par `scripts/capture_screens.py`]*

L'assistant dispose de **15 outils** lui donnant accès aux données et aux modèles
de la plateforme. Son bouton d'ouverture portait auparavant un simple losange
`◈`, illisible quant à sa fonction ; il affiche désormais une bulle de dialogue
et le mot **« Assistant »**.


# 7 · Fonctionnalités récentes

## 7.1 · AI Stress Testing Engine

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

## 7.2 · Prédiction directionnelle et découverte automatique des modèles

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

## 7.3 · Interface : thème, navigation, mise en page

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

---

# 8 · Résultats mesurés

## 8.0 Ancrage en finance quantitative

Avant les résultats, il faut nommer le cadre théorique dont la plateforme
hérite : chaque moteur implémente une notion classique de finance quantitative,
et c'est ce qui rend les chiffres comparables à une référence.

**Le modèle de marché.** Les rendements logarithmiques
$r_t = \ln(P_t / P_{t-1})$ sont supposés non stationnaires et à queues
épaisses — hypothèse vérifiée dans le projet : le kurtosis excédentaire mesuré
est positif et la victoire de GJR-GARCH (§ 8.4.2) confirme l'asymétrie. La
plateforme ne suppose **jamais** la normalité : la VaR historique et la CVaR
sont non paramétriques.

**Mesures de risque cohérentes.** La VaR au seuil $\alpha$ est le quantile
$\mathrm{VaR}_\alpha = -\inf\{x : P(r \le x) \ge \alpha\}$. Elle n'est
**pas sous-additive** : le risque d'un portefeuille peut y paraître supérieur à
la somme de ses parties, ce qui viole l'axiomatique d'Artzner *et al.* (1999).
La CVaR corrige ce défaut :

$$\mathrm{CVaR}_\alpha = \mathbb{E}\big[\,r \;\big|\; r \le -\mathrm{VaR}_\alpha\,\big]$$

C'est pourquoi le moteur de stress testing **classe les actifs par CVaR et non
par VaR** : seule la CVaR est une mesure cohérente au sens axiomatique.

**Volatilité conditionnelle.** GJR-GARCH(1,1) modélise l'effet de levier par un
terme actif uniquement sur les chocs négatifs :

$$\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}_{\{\epsilon_{t-1} < 0\}})\,\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

**Décomposition d'Euler du risque.** La volatilité d'un portefeuille étant
homogène de degré 1 en les poids, elle se décompose exactement :

$$\sigma_p = \sum_i w_i \frac{\partial \sigma_p}{\partial w_i}, \qquad \mathrm{RC}_i = w_i \frac{(\Sigma w)_i}{\sigma_p}$$

C'est cette identité qui produit le résultat contre-intuitif du § 8.5 : GC=F
pèse 20 % du capital pour 4,5 % du risque.

**Markowitz contre RL.** L'optimisation moyenne-variance classique résout
$\min_w w^\top \Sigma w$ sous $w^\top \mu = \mu_{\text{cible}}$ — statique et
supposant $\mu, \Sigma$ connus. Le RL relâche ces deux hypothèses : il apprend
une politique $\pi(a|s)$ **conditionnée à l'état du marché**, sans estimer
explicitement $\mu$. C'est la promesse. Le résultat mesuré (§ 8.3.3) montre
qu'à ce budget elle n'est pas tenue.

> **Sur le calcul quantique.** Ce projet fait de la **finance quantitative**,
> pas du calcul quantique : il n'y a aucune dépendance à Qiskit, Pennylane ou un
> simulateur quantique — 0 occurrence de `quantum` dans le code source.
> La littérature explore bien QAOA pour l'optimisation de portefeuille (un
> problème QUBO) et l'estimation d'amplitude quantique, qui promet une
> accélération quadratique du Monte-Carlo pour le pricing. Ces approches restent
> aujourd'hui limitées par le bruit matériel et le nombre de qubits, et **rien
> n'en est implémenté ici**. Le mentionner sans le coder serait un argument
> d'autorité ; c'est pourquoi cette distinction est écrite noir sur blanc plutôt
> que laissée dans l'ambiguïté du vocabulaire.

## 8.1 Prévision profonde — résultats et explication

### 8.1.1 Pourquoi la précision directionnelle plutôt que le RMSE

Un modèle peut avoir une erreur quadratique faible **et se tromper
systématiquement de sens** : il suffit qu'il prédise toujours une variation
proche de zéro. Pour décider d'acheter ou de vendre, seul le **signe** compte.
La métrique retenue est donc la *directional accuracy* (DA), avec le seuil
naturel de **50 % = pile ou face**.

| Checkpoint | DA | Écart au hasard | R² | RMSE |
|---|---:|---:|---:|---:|
| `KO_gru_h5` | **67,57 %** | **+17,6 pts** | 0,165 | 0,0312 |
| `AAPL_lstm_h5` | **62,57 %** | **+12,6 pts** | 0,0398 | 0,0346 |
| `AAPL_gru_h5` | 54,67 % | +4,7 pts | **−0,0534** | 0,0473 |
| `AAPL_cnn_lstm_h5` | 54,55 % | +4,6 pts | 0,0021 | 0,0353 |
| `AAPL_transformer_h5` | 54,01 % | +4,0 pts | **−0,0034** | 0,0353 |
| `EURUSD_X_gru_h5` | 53,61 % | +3,6 pts | **−0,0280** | 0,0081 |
| `AAPL_tcn_h5` | 52,94 % | +2,9 pts | 0,0360 | 0,0346 |

### 8.1.2 Lecture des résultats

**Deux modèles ont un avantage réel.** `KO_gru_h5` (67,57 %) et `AAPL_lstm_h5`
(62,57 %) sont nettement au-dessus du hasard, et ce sont les deux seuls dont le
R² est franchement positif. Ce n'est pas une coïncidence : quand un modèle
capture une part de la variance, il tend aussi à saisir le sens.

**Cinq modèles sont à la limite du bruit.** Entre 52,94 % et 54,67 %, l'écart au
hasard va de 2,9 à 4,7 points. Sur ~187 observations de test, un tel écart n'est
pas distinguable de la chance : il faudrait plusieurs centaines d'observations
supplémentaires pour trancher.

**Trois R² sont négatifs** — `AAPL_gru` (−0,0534), `EURUSD_X_gru` (−0,0280),
`AAPL_transformer` (−0,0034). Un R² négatif signifie littéralement : *ce modèle
prédit moins bien que la moyenne historique*. Ils sont conservés et affichés
tels quels, parce qu'un modèle mauvais retiré du rapport reste mauvais dans le
produit ; l'interface montre la DA mesurée et l'écart au hasard, pour qu'un
utilisateur voie immédiatement qu'un modèle à 52,94 % n'apporte presque rien.

**Pourquoi le Transformer déçoit-il ?** Il est le plus gourmand en données des
cinq architectures : l'auto-attention doit apprendre quelles positions de la
séquence importent, sans a priori temporel. Sur ~1 250 barres, il n'a pas de
quoi le faire, là où un LSTM part avec un biais récurrent adapté aux séries.
La taille du jeu de données, pas l'architecture, explique le classement.

## 8.2 Fondements formels de l'agent RL

### 8.2.1 Formulation du problème comme MDP

L'allocation est formalisée par un processus de décision markovien
$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Les valeurs ci-dessous sont celles
du code (`services/rl/environment.py`), pas des valeurs de manuel.

**État.** Fenêtre glissante de $L = 20$ barres sur $n$ variables de marché,
concaténée à la position courante et à la trésorerie :

$$s_t = \big[\underbrace{x_{t-L+1}, \dots, x_t}_{\text{20 barres}},\ w_t,\ c_t,\ d_t\big] \in \mathbb{R}^{36}$$

où $w_t$ est le poids détenu, $c_t$ la trésorerie normalisée et $d_t$ le
drawdown courant. Avec l'augmentation régime, $s_t \in \mathbb{R}^{42}$ (six
variables supplémentaires).

**Action.** Discrète pour la famille value-based,
$\mathcal{A} = \{\text{vendre}, \text{conserver}, \text{acheter}\}$, chaque
transaction portant sur `trade_fraction` $= 0{,}25$ du capital. Pour la famille
acteur-critique, $\mathcal{A} = [-1, 1]^{n+1}$ projeté sur le simplexe par
softmax — un vecteur de poids sommant à 1.

**Récompense.** C'est ici que le projet s'écarte d'un RL générique. La
récompense n'est pas le profit brut mais une quantité **pénalisée par le
risque** :

$$R_t = \kappa \cdot r_t^{\text{net}} \;-\; \lambda_\sigma \sigma_t \;-\; \lambda_{DD}\,|DD_t| \;-\; \lambda_{CVaR}\,\mathrm{CVaR}_\alpha(t) \;-\; \lambda_\tau \tau_t$$

avec les coefficients **effectivement codés** :

| Terme | Symbole | Valeur | Rôle |
|---|---|---:|---|
| Échelle | $\kappa$ | 100,0 | met la récompense à l'échelle du réseau |
| Volatilité | $\lambda_\sigma$ | 0,15 | pénalise l'agitation du capital |
| Drawdown | $\lambda_{DD}$ | **0,35** | *plus forte pénalité* — la perte cumulée |
| CVaR | $\lambda_{CVaR}$ | 0,10 | pénalise la queue à $\alpha = 0{,}05$ |
| Rotation | $\lambda_\tau$ | 0,02 | décourage le sur-trading |

Le rendement net intègre les frictions réelles :

$$r_t^{\text{net}} = \frac{V_t - V_{t-1}}{V_{t-1}} - \underbrace{\phi\,|\Delta w_t|}_{\text{coût } 0{,}001} - \underbrace{\psi\,|\Delta w_t|}_{\text{slippage } 0{,}0005}$$

**Facteur d'actualisation.** $\gamma = 0{,}99$, soit un horizon effectif de
$1/(1-\gamma) \approx 100$ pas — cohérent avec les 101 barres de test.

> **Pourquoi ce choix de récompense est décisif.** Un agent optimisant le seul
> profit apprend à concentrer le risque : il maximise l'espérance sans borner la
> variance. Le drawdown reçoit ici le poids le plus élevé (0,35) parce que
> c'est la grandeur qui ruine un portefeuille — pas la volatilité, qui n'est
> qu'une dispersion. Cette structure répond à l'objectif O-3 du cahier des
> charges.

### 8.2.2 Règles de mise à jour des trois familles

**Value-based (DQN, Double DQN, Dueling DQN).** L'agent apprend $Q(s,a)$ et
minimise l'erreur de différence temporelle :

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Big[\big(y_t - Q_\theta(s,a)\big)^2\Big]$$

Le **double Q-learning** dissocie la sélection de l'évaluation, corrigeant le
biais de surestimation du $\max$ :

$$y_t = r_t + \gamma\, Q_{\theta^-}\!\Big(s_{t+1},\ \arg\max_{a'} Q_\theta(s_{t+1}, a')\Big)$$

L'**architecture dueling** sépare valeur d'état et avantage :

$$Q(s,a) = V(s) + \Big(A(s,a) - \tfrac{1}{|\mathcal{A}|}\textstyle\sum_{a'} A(s,a')\Big)$$

**Gradient de politique (PPO, TRPO).** Optimisation directe de la politique
sous contrainte de déplacement, via le ratio
$\rho_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$ :

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t \hat{A}_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big]$$

**Acteur-critique (SAC, DDPG, TD3).** SAC ajoute un terme d'entropie qui
maintient l'exploration :

$$J(\pi) = \mathbb{E}\Big[\textstyle\sum_t \gamma^t\big(R_t + \beta\,\mathcal{H}(\pi(\cdot|s_t))\big)\Big]$$

**Distributionnels (C51, QR-DQN, IQN, Rainbow).** Ils apprennent la
*distribution* $Z(s,a)$ des retours, non son espérance, via l'équation de
Bellman distributionnelle :

$$Z(s,a) \;\stackrel{D}{=}\; R(s,a) + \gamma\, Z(s', a')$$

C'est exactement cette propriété qui explique leur comportement observé
(§ 8.3.2) : quand la distribution apprise reste large et centrée près de zéro,
l'action « ne rien faire » domine toutes les autres.

### 8.2.3 Hyperparamètres gelés

Identiques pour les 13 algorithmes — sans quoi la comparaison n'aurait pas de
sens. Valeurs lues dans `configs/defaults.yaml` :

| Hyperparamètre | Valeur |
|---|---:|
| Taux d'apprentissage | $5 \times 10^{-4}$ |
| Actualisation $\gamma$ | 0,99 |
| Taille de lot | 64 |
| Réseau | 2 couches cachées de 128 |
| Tampon de rejeu | 50 000 (min. 1 000) |
| Mise à jour de la cible | tous les 250 pas |
| Exploration $\varepsilon$ | 1,0 → 0,05 sur 8 000 pas |
| Écrêtage du gradient | 10,0 |
| Graine | 42 (études multi-graines : 1–5) |

### 8.2.4 Mixture-of-Experts sensible au régime (adaptation active)

L'augmentation régime décrite plus haut est **passive** : l'agent *voit* le
régime, mais ses poids ne changent jamais quand le marché change. Le cahier des
charges (O-2) demandait une adaptation **active**. Le module
`services/rl/moe.py` l'ajoute — **sans modifier une seule ligne existante** :
ni l'environnement, ni la récompense, ni les algorithmes, ni l'API, ni l'UI.

**Trois experts, un routeur explicite.** Les 7 régimes que le détecteur produit
déjà sont projetés sur 3 experts :

$$g(s_t) = \text{expert}\big(\rho(s_t)\big), \qquad \rho \in \{\text{7 régimes}\} \;\longrightarrow\; \{\text{bull}, \text{bear}, \text{stress}\}$$

| Expert | Régimes couverts |
|---|---|
| **bull** | `bull_market`, `recovery`, `low_volatility`, `sideways` |
| **bear** | `bear_market` |
| **stress** | `crash_risk`, `high_volatility` |

Le routeur est une **table**, pas une porte apprise : il est auditable par un
risk officer, ce qu'une gate neuronale ne serait pas. `sideways` et
`low_volatility` rejoignent bull car ce sont des états calmes sans direction —
en faire un quatrième expert l'aurait privé de barres suffisantes.

**Adaptation déclenchée par le changement.** À chaque bascule de régime,
l'expert nouvellement sélectionné est affiné sur les barres de **son** régime
déjà observées, puis reprend la main. Le nombre de barres écoulées est
enregistré : c'est le KPI K-5.

**Absence de fuite — deux garde-fous testés.** (1) La classification réutilise
`RegimeFeatureProvider.build()`, déjà causal (`df.iloc[start:t+1]`, borne
exclusive). (2) L'affinage à la barre $t$ n'utilise que des barres
**strictement antérieures** à $t$. Un test l'impose et une mutation qui décale
la borne de +5 barres le fait échouer.

**K-5 mesuré (AAPL, 5 ans, 1 255 barres, module seul) :**

| Grandeur | Valeur |
|---|---:|
| Bascules de régime détectées | **80** |
| dont changement d'expert | 40 |
| Délai de réaction moyen | **0,0526 barre** |
| Délai médian | 0,0 barre |
| Délai maximal | **1 barre** |
| Bascules mesurées / non adaptées | 57 / **23** |
| Affinages réels (poids modifiés) | 17 · δ max **0,538** |
| Répartition des barres | bull 891 · bear 145 · stress 100 |

**Lecture.** Le délai médian est nul parce que 40 bascules sur 80 restent dans
le même expert (par exemple `bull_market` → `sideways`) : la réaction est
immédiate par construction. Le maximum d'une barre correspond au premier
appel d'un expert, qui doit être affiné avant de prendre la main. Les
**23 bascules non adaptées** sont celles où l'expert visé n'avait pas encore
**90 barres** de son régime : elles sont **comptées comme des échecs**, pas
écartées du calcul — une moyenne portant sur les seuls succès flatterait le
résultat.

> **Correction d'une version antérieure de ce rapport.** Ce tableau annonçait
> « 71 / 9 » avec un seuil de 30 barres. Ce seuil a été relevé à **90** après
> mesure : sous 90 barres l'expert revient **bit-à-bit identique** (le tampon
> de rejeu n'atteint pas `min_buffer = 1 000`, `learn_step` ne s'exécute
> jamais). Les anciens « succès » entre 30 et 90 barres étaient donc des
> adaptations **qui n'avaient pas eu lieu**. Le chiffre honnête est 57 / 23.

> **Ce que ce module ne prétend pas.** Il mesure une **latence d'adaptation**,
> pas une rentabilité. Le résultat de référence (§ 8.3) est conservé intact et
> transmis tel quel dans la charge utile : aucun chiffre de performance
> antérieur n'est modifié, et le module n'en produit aucun de son côté. Un test
> vérifie qu'il ne publie ni `total_return`, ni `sharpe_ratio`, ni `alpha`.

#### 8.2.4.1 Intégration dans l'application réelle

Jusqu'ici `moe.py` était **du code mort** : `grep` ne trouvait aucun importateur
hors de ses propres tests, donc **aucune ligne du module ne s'exécutait** dans
la plateforme en fonctionnement. Le mécanisme était correct mais débranché.

**Point d'intégration retenu.** Le seul endroit où une politique agit
*séquentiellement barre après barre* sur des données réelles est
`RLService.backtest()`, exposé par `GET /rl/backtest/{symbol}`. C'est donc là —
et nulle part ailleurs — que le routage a un sens.

**Activation.** Un unique paramètre optionnel, `moe=true` (plus `moe_adapt`
pour la condition de contrôle). Par défaut `false` : le chemin historique est
appelé **à l'identique** et le module MoE n'est même pas importé (import
paresseux placé dans la branche).

**Empreinte réelle des modifications.**

| Fichier | Nature |
|---|---|
| `services/rl/moe.py` | **+265 lignes ajoutées en fin de fichier** ; les 500 lignes d'origine sont conservées **mot pour mot** (vérifié : `after.startswith(before)`) |
| `api/v1/endpoints/rl.py` | 2 paramètres optionnels + une branche à import paresseux sur le seul endpoint `backtest` |
| `services/rl/service.py`, `environment.py`, `agents/dqn.py`, `regime_features.py`, `risk/regime.py` | **inchangés — md5 identique** |
| Frontend, UI, XAI, métriques de risque, récompense | **aucune modification** |

**Pourquoi une boucle et non `agent.evaluate()`.** `evaluate()` conserve *par
construction* une politique fixe pour tout l'épisode ; or un MoE change de
politique **en cours d'épisode**. L'exprimer via `evaluate()` aurait exigé de
modifier `evaluate()`, donc de placer du code MoE sur le chemin d'exécution du
baseline — précisément ce qui était interdit. La boucle recopie fidèlement la
règle de décision de `evaluate()` (`argmax` sur `q_values`).

**Preuve de non-régression du baseline.** Avec un seuil d'expert rendu
inatteignable, `rollout()` reproduit le backtest historique **bit à bit** :
dictionnaire `performance` identique et courbe d'équité identique (vérifié).
La réponse par défaut ne gagne **ni clé `moe`, ni clé `mode`**.

**Résultats mesurés via l'application (AAPL, `dueling_dqn`) :**

| Fenêtre | Baseline | MoE | Buy & Hold | Barres pilotées par un expert | Affinages réels | K-5 moyen (strict) |
|---|---:|---:|---:|---|---:|---:|
| 2 ans | +23,84 % | **+30,22 %** | +38,65 % | bull 241 / 480 | 1 (δ 0,047) | 0,00 barre |
| 5 ans | +1,19 % | **+6,64 %** | +111,76 % | bull 895 · bear 85 · stress 10 / 1 234 | 13 (δ 0,329) | 0,77 barre |

> **Lecture honnête.** Le MoE dépasse le baseline sur ces deux fenêtres, mais
> **reste très largement battu par Buy & Hold** (+6,64 % contre +111,76 % sur
> 5 ans). Il s'agit d'**une seule instrument sur une seule fenêtre**, sans
> répétition multi-graines : ce n'est pas une preuve de surperformance et n'est
> pas revendiqué comme telle. Le gain démontré porte sur l'**adaptabilité**
> (KPI K-5), pas sur le rendement.
>
> Le K-5 est publié **deux fois** : la lecture globale (qui compte comme
> « réaction nulle » les bascules ne nécessitant aucun changement d'expert) et
> la lecture **stricte** restreinte aux vrais changements d'expert. La première
> est flatteuse ; les deux figurent côte à côte pour qu'aucune ne puisse être
> citée seule. Sur 5 ans : 0,19 barre en global contre **0,77 barre** en
> strict, avec **26 bascules non adaptées** publiées comme des échecs.

**Refus explicites plutôt que faux résultats.** Les politiques SB3
(PPO/A2C/SAC/TD3/TRPO) exposent `predict`, pas `q_values`, et ne disposent
d'aucun chemin d'affinage ici : la requête est **rejetée en 422** avec le motif,
au lieu d'être routée vers un expert incapable de s'adapter puis assortie de
chiffres d'adaptation qui ne décriraient rien. Les 6 agents discrets natifs
(`dqn`, `double_dqn`, `dueling_dqn`, `c51`, `iqn`, `rainbow`) sont vérifiés
fonctionnels.

## 8.3 Apprentissage par renforcement — résultats et explication


### 8.3.1 Ce que mesure le tableau

Chaque algorithme a été entraîné sur **5 graines indépendantes**, même
environnement, même budget (8 épisodes), même fenêtre AAPL 2 ans
(400 barres d'entraînement, 101 de test). Le rendement est celui du jeu de test,
net de coûts de transaction. La colonne *écart-type* est calculée sur les
5 graines : c'est elle qui distingue un algorithme fiable d'un algorithme
chanceux.

| Algorithme | Famille | Rendement moyen | Écart-type | Min | Max |
|---|---|---:|---:|---:|---:|
| A2C | acteur-critique | **+5,52 %** | **0,63** | +4,37 | +6,04 |
| SAC | acteur-critique | +4,79 % | 2,59 | +1,28 | +7,08 |
| TRPO | gradient de politique | +4,02 % | 1,23 | +2,07 | +5,40 |
| TD3 | acteur-critique | +4,00 % | 1,85 | +1,68 | +7,01 |
| DDPG | acteur-critique | +3,34 % | 2,71 | +0,27 | +6,54 |
| PPO | gradient de politique | +2,94 % | 2,35 | +0,00 | +5,90 |
| Double DQN | value-based | +0,09 % | 0,57 | −0,38 | +1,18 |
| C51 / IQN / QR-DQN / Rainbow | distributionnelle | **0,00 %** | **0,00** | 0,00 | 0,00 |
| DQN | value-based | −0,52 % | 1,85 | −4,04 | +1,44 |
| Dueling DQN | value-based | −1,86 % | 2,89 | −6,98 | +0,79 |
| **Buy & Hold** | *référence passive* | **+21,17 %** | — | — | — |

### 8.3.2 Pourquoi les familles se comportent différemment

Le classement n'est pas aléatoire : il suit la **structure de l'espace
d'action**, et cela s'explique.

**Les acteur-critique dominent (A2C, SAC, TD3, DDPG).** Ces algorithmes
apprennent directement une politique et gèrent nativement des actions
continues. Sur un problème d'allocation, « investir 37 % » est une action
naturelle ; ils n'ont pas besoin de la discrétiser. A2C obtient le meilleur
résultat **avec la plus faible dispersion** (σ = 0,63) : c'est le seul
algorithme dont les 5 graines tiennent dans une fourchette de 1,7 point. C'est
la propriété la plus intéressante du tableau — un résultat moyen élevé mais
instable serait moins exploitable qu'un résultat modeste et reproductible.

**Les value-based échouent (DQN, Double DQN, Dueling DQN).** Ils estiment une
valeur par action discrète, puis choisissent le maximum. Avec un budget de
8 épisodes, l'estimation de Q reste bruitée ; le `max` amplifie ce bruit
(biais de surestimation bien documenté, motivation historique de Double DQN).
Dueling DQN est le pire du catalogue à **−1,86 %** avec la **plus forte
dispersion** (σ = 2,89, de −6,98 % à +0,79 %) : selon la graine, le même
algorithme perd 7 points ou en gagne 1. C'est précisément l'instabilité que
l'étude M122 a cherché à corriger par l'augmentation régime — sans succès.

**La famille distributionnelle ne trade pas (C51, QR-DQN, IQN, Rainbow).**
Rendement de **0,00 % sur les 5 graines, écart-type exactement 0,00**. Une
politique qui prend la moindre position produirait une variation entre graines :
zéro dispersion signifie zéro décision. Deux mesures indépendantes le
confirment — le **drawdown maximal vaut 0,0** et le **Sharpe vaut 0,0**, ce qui
est impossible pour un agent exposé au marché.

> **Précision méthodologique.** Le nombre de transactions n'est pas enregistré
> dans ces artefacts. L'absence de trading est donc **déduite** de trois
> grandeurs concordantes (rendement nul, dispersion nulle, drawdown nul), et
> non lue directement dans un compteur. La déduction est solide mais reste une
> déduction, et le rapport le dit plutôt que d'affirmer une mesure qui n'existe
> pas.

L'explication tient à la nature de ces algorithmes : ils apprennent une
*distribution* de retours plutôt qu'une espérance. Sur 8 épisodes, cette
distribution reste large et centrée près de zéro ; l'action « ne rien faire »
domine alors toutes les autres en valeur espérée, puisque toute transaction
coûte immédiatement `transaction_cost = 0,001`. **L'agent a appris que le
meilleur coup était de ne pas jouer** — ce qui est rationnel dans son cadre, et
inutile pour l'usage visé.

### 8.3.3 Le résultat central : personne ne bat la stratégie passive

L'écart le plus important n'est pas entre les algorithmes, il est entre
**tous les algorithmes et Buy & Hold** :

- Meilleur agent (A2C) : **+5,52 %**
- Référence passive : **+21,17 %**
- **Écart : −15,65 points**

Aucun des 13 ne s'en approche. L'explication la plus probable est la
conjonction de trois facteurs, chacun mesuré : un **budget de 8 épisodes**
(D-3) trop court pour converger, une **fenêtre de test haussière** où ne rien
faire était optimal, et des **coûts de transaction** qui pénalisent toute
activité. Un agent qui trade sur un marché qui monte part avec un handicap
structurel.

> **Conséquence assumée.** L'objectif du cahier des charges — « optimiser les
> décisions d'investissement » — **n'est pas atteint** (CDC-2). Ce résultat est
> publié en tête de section plutôt que noyé : une plateforme d'aide à la
> décision qui masquerait que sa composante RL perd contre l'inaction serait
> trompeuse.

### 8.3.4 L'étude M122 : l'augmentation régime n'aide pas

Test dédié sur Dueling DQN, 5 graines par bras :

| Bras | Rendement | IQM | IC 95 % bootstrap | σ |
|---|---:|---:|---|---:|
| Référence | +8,66 % | +8,26 % | [+5,95, +12,30] | 3,08 % |
| Régime-aware | +5,21 % | +5,08 % | [−2,59, +13,00] | 7,11 % |

Les intervalles **se recouvrent entièrement** et celui du bras modifié
**contient zéro**. Les tests le confirment : apparié *t* = −0,804 (*p* = 0,467),
Welch *p* = 0,362, *d* = −0,63. La modification n'améliore **qu'une graine sur
cinq**.

Pourquoi ? Ajouter 6 variables de régime fait passer l'observation de 36 à
42 dimensions **sans augmenter le budget d'entraînement**. L'agent doit
apprendre davantage avec autant de données : la variance double
(σ 3,08 → 7,11) sans gain de moyenne. C'est un compromis biais-variance
défavorable, pas un défaut d'implémentation — et c'est la conclusion honnête de
l'étude.

## 8.4 Risque — résultats et explication

### 8.4.1 Value at Risk : aucun estimateur validé

Backtest sur AAPL, 5 ans, 1 253 observations, seuil 95 %. Deux tests
complémentaires : **Kupiec** vérifie que le *nombre* de dépassements est
correct, **Christoffersen** vérifie que leur *répartition dans le temps* l'est.

| Estimateur | Dépassements | Kupiec *p* | Indépendance *p* | Verdict |
|---|---:|---:|---:|---|
| Historique | 5,58 % | 0,405 ✔ | **0,0032** ✘ | rejeté |
| Paramétrique | 4,79 % | 0,754 ✔ | **0,0295** ✘ | rejeté |
| Cornish-Fisher | 6,78 % | **0,0139** ✘ | **0,0015** ✘ | rejeté |
| Student-t | 5,88 % | 0,212 ✔ | **0,0018** ✘ | rejeté |
| EWMA | 6,98 % | **0,0065** ✘ | **0,0089** ✘ | rejeté |
| Historique filtrée | 5,58 % | 0,405 ✔ | **0,0032** ✘ | rejeté |
| Monte-Carlo | 5,58 % | 0,405 ✔ | **0,0032** ✘ | rejeté |

**Le résultat le plus instructif est le contraste entre les deux colonnes.**
Cinq estimateurs sur sept passent Kupiec : ils annoncent 5 % de pertes
exceptionnelles et en observent 4,8 à 5,9 %. Vu ainsi, le modèle « fonctionne ».

Mais **les sept échouent au test d'indépendance**. Les dépassements ne sont pas
répartis au hasard : ils arrivent **en grappes**. C'est le phénomène de
*volatility clustering* — les journées agitées se suivent. Or une VaR
inconditionnelle suppose l'inverse.

> **Pourquoi cela compte en pratique.** Un modèle qui se trompe 5 % du temps de
> façon dispersée est gérable. Un modèle qui se trompe 5 % du temps *mais
> concentre ses erreurs sur la même semaine* expose à une série de pertes
> consécutives : exactement le scénario qui ruine un portefeuille. Le nombre
> global de dépassements est correct, leur regroupement ne l'est pas.

Aucune VaR n'est donc présentée comme « validée » dans l'interface.
`model_valid = False` est propagé jusqu'à l'affichage.

### 8.4.2 GARCH : l'effet de levier est mesurable

| Modèle | AIC | Interprétation |
|---|---:|---|
| **GJR-GARCH** | **4 709,8** | *meilleur* — asymétrie prise en compte |
| EGARCH | 4 721,8 | asymétrie logarithmique |
| GARCH | 4 723,3 | symétrique |

GJR l'emporte de 13,5 points d'AIC sur le GARCH symétrique. La différence entre
ces modèles est unique : GJR ajoute un terme actif **uniquement quand le
rendement est négatif**. Qu'il gagne signifie que **les chocs négatifs
augmentent la volatilité future davantage que les chocs positifs de même
ampleur** — l'effet de levier, cohérent avec la littérature financière. Ce
n'est pas un réglage arbitraire : le critère d'information a tranché.

### 8.4.3 Cohérence du moteur de risque

- Spearman(volatilité annualisée, score global) = **0,967** : le score suit bien
  la volatilité, sans être une simple copie.
- Monotonie en volatilité : **vraie**.
- **8 réponses distinctes pour 8 périodes** sélectionnables — preuve que le
  sélecteur atteint réellement le calcul et n'est pas décoratif.

### 8.4.4 Détection d'anomalies — une limite assumée

L'Isolation Forest est paramétré à une contamination de 2 % : **il étiquette
donc 2 % de n'importe quelle fenêtre**, y compris parfaitement calme. Ce n'est
pas une détection au sens strict mais un **classement relatif** : les 2 % les
plus atypiques de la période. Le rapport le signale plutôt que de laisser croire
à une alarme absolue.

## 8.5 Stress testing — lecture d'un cas réel

Panier AAPL 50 % / MSFT 30 % / GC=F 20 %, scénario *Market Crash*,
1 253 observations, position 100 000 $ :

| Mesure | Avant | Après | Variation |
|---|---:|---:|---:|
| VaR 95 % | 2,05 % | 2,16 % | +0,11 pt |
| CVaR | 2,80 % | 3,03 % | +0,22 pt |
| Volatilité | 20,32 % | 20,75 % | +0,43 pt |
| Drawdown max | 25,07 % | **48,06 %** | **+22,99 pts** |

**Ce que révèle l'écart entre les lignes.** La VaR bouge à peine (+0,11 pt)
tandis que le drawdown **double**. C'est la leçon centrale du stress testing :
la VaR est un quantile *journalier*, insensible à l'enchaînement des pertes ;
le drawdown mesure une **trajectoire cumulée**. Un portefeuille peut paraître
sûr au jour le jour et perdre la moitié de sa valeur sur une séquence.

**La décomposition d'Euler contredit l'intuition du poids :**

| Actif | Poids | Contribution au risque | Part de la perte |
|---|---:|---:|---:|
| AAPL | 50,0 % | **63,6 %** | 52,8 % |
| MSFT | 30,0 % | 31,9 % | 31,3 % |
| GC=F | 20,0 % | **4,5 %** | 15,9 % |

GC=F pèse **20 % du capital mais 4,5 % du risque** : l'or décorrèle du reste du
panier, il amortit au lieu d'amplifier. AAPL, à l'inverse, concentre 63,6 % du
risque pour 50 % du capital. **Le poids ne mesure pas l'exposition** — c'est
précisément ce qu'une allocation naïve ignore.

## 8.6 Données réelles en base

9 tables — **87 utilisateurs**, **2 920 alertes**, 1 portefeuille,
3 transactions.

---

---

# 9 · Qualité logicielle et méthode de vérification

## 9.1 Trois exigences de méthode

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

## 9.2 Discipline appliquée

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

---

# 10 · Cahier des charges — conformité et divergences

> **Source.** Cet audit porte sur le document officiel
> `Cahier_des_charges.pdf` (7 pages) — *« Apprentissage par Renforcement
> Profond Adaptatif et Explicable pour la Gestion de Portefeuille Sensible au
> Risque »*. Chaque exigence y est numérotée selon la section d'origine du
> cahier des charges, confrontée à ce qui existe réellement dans le dépôt, avec
> la preuve de vérification. **Toute divergence est documentée** (CDC-1 …
> CDC-9), sans atténuation.

## 10.1 Objectifs spécifiques (§ 2.2)

| # | Exigence du cahier des charges | État | Preuve mesurée |
|---|---|---|---|
| O-1 | Environnement de simulation réaliste (multi-actifs, coûts, contraintes) | ✅ Conforme | `PortfolioEnv` + `TradingEnv` ; `transaction_cost = 0.001`, slippage modélisé |
| O-2 | Agent DRL adaptatif avec détection de changement de régime | ⚠️ **CDC-1** | `regime.py` + `regime_features.py` opérationnels ; **Mixture-of-Experts implémenté et branché** dans l'application (`moe=true` sur `/rl/backtest`, § 8.2.4.1) ; **pas de MAML** |
| O-3 | Récompense sensible au risque (Sharpe/Sortino, CVaR, drawdown) | ✅ Conforme | `environment.py` : `cvar_penalty = 0.10`, `drawdown_penalty = 0.35`, `cvar_alpha = 0.05` |
| O-4 | XAI native et post-hoc justifiant chaque décision | ✅ Conforme | SHAP, `allocation_explain.py`, `regime_explain.py`, narratifs automatiques |
| O-5 | Tableau de bord pour gérants et fonctions de contrôle | ✅ Conforme | 13 pages, dont Risk & Alerts, Explainability, AI Stress Testing |
| O-6 | Validation : backtesting, walk-forward, stress tests | ⚠️ **CDC-2** | Backtesting ✅, stress tests ✅ (7 scénarios) ; **walk-forward présent pour la prévision, absent pour le RL** |
| O-7 | Documentation pour revue de gouvernance (SR 11-7, EBA/ACPR) | ⚠️ **CDC-3** | Model card, journal d'audit, registre de versions ✅ ; **dossier réglementaire formel non constitué** |

## 10.2 Description fonctionnelle (§ 4)

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

## 10.3 Contraintes (§ 6)

| # | Contrainte | État | Preuve |
|---|---|---|---|
| C-1 | Auditabilité, explicabilité, contrôle des biais | ✅ Conforme | Journal d'audit, XAI, registre de divergences |
| C-2 | RGPD sur les données alternatives | ✅ Sans objet | Aucune donnée personnelle collectée |
| C-3 | Infrastructure GPU + CI/CD | ⚠️ **CDC-7** | CI (`ruff` + 744 tests) ✅ ; **entraînement sur CPU**, pas de GPU |
| C-4 | Interopérabilité OMS/EMS | ✅ Hors périmètre | Explicitement exclu par le § 3.2 du cahier des charges |
| C-5 | Validation par comité des risques modèles | ❌ **CDC-8** | **Non réalisée** — nécessite une organisation, pas du code |

## 10.4 Indicateurs clés de performance (§ 8)

| # | KPI exigé | État | Valeur mesurée |
|---|---|---|---|
| K-1 | Rendement cumulé net vs benchmark | ✅ Mesuré | Meilleur RL **+5,52 %** vs Buy & Hold **+21,17 %** |
| K-2 | Ratios de Sharpe et Sortino | ✅ Mesurés | Sharpe 2,00 ± 1,19 ; Sortino 3,18 (référence) |
| K-3 | Maximum drawdown et temps de récupération | ⚠️ **CDC-9** | Drawdown ✅ (−6,93 % moyen) ; **temps de récupération non calculé** |
| K-4 | VaR et CVaR réalisés vs cibles | ✅ Mesurés | 7 estimateurs backtestés (tableau § 5.3) |
| K-5 | Délai de réaction après changement de régime | ✅ **Mesuré** | **0,0526 barre** en moyenne, max 1 barre (80 bascules, 23 non adaptées) ; via l'application : **0,77 barre** en lecture stricte sur 5 ans — voir § 8.2.4 et § 8.2.4.1 |
| K-6 | Stabilité entre régimes de marché | ⚠️ Partiel | Étude régime-aware menée ; **un seul régime de test** (D-11) |
| K-7 | Fidélité des explications | ❌ **CDC-3** | **Non mesurée** quantitativement |
| K-8 | Compréhensibilité (enquête utilisateurs) | ❌ Hors périmètre | Nécessite un panel d'utilisateurs métier |
| K-9 | Taux de couverture des décisions expliquées | ✅ Conforme | 100 % : toute décision affichée porte ses contributions par signal |

## 10.5 Registre des divergences

**CDC-1 — Adaptation au régime : Mixture-of-Experts, pas MAML.** *(mise à jour)*
*Résolution partielle.* Un **Mixture-of-Experts sensible au régime** a été
ajouté (§ 8.2.4) : 3 experts, routeur explicite sur les 7 régimes existants,
affinage déclenché par la bascule. **Le KPI K-5, auparavant non mesurable, l'est
désormais** : 0,0526 barre en moyenne, 1 barre au maximum, sur 80 bascules
(dont **23 non adaptées**, publiées comme échecs).
*Mise à jour — intégration.* Le module n'est plus du code mort : il est
**branché dans l'application** via `GET /rl/backtest/{symbol}?moe=true`
(§ 8.2.4.1). Par défaut le chemin baseline est reproduit **bit à bit**.
*Écart restant.* Le cahier des charges citait MAML **ou** Mixture-of-Experts :
la seconde branche est implémentée, la première non. L'adaptation reste un
affinage sur données passées, pas un méta-apprentissage.
*Rappel du résultat mesuré.* L'augmentation régime **passive** n'améliorait pas
la performance (+5,21 % contre +8,66 %, p = 0,467). Le MoE ajoute la réactivité
demandée ; il ne change pas ce constat, et aucun résultat antérieur n'a été
recalculé.

**CDC-1-bis — Adaptation au régime : implémentée, mais pas par meta-learning.**
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

**CDC-10 — « Finance quantique » demandée, finance quantitative livrée.**
*Écart.* Une demande a mentionné la « finance quantique ». Le projet
n'implémente **aucun calcul quantique** : 0 occurrence de `quantum` dans le
code, aucune dépendance à Qiskit ou Pennylane.
*Traitement.* La section 8.0 traite les fondements de **finance quantitative**
réellement implémentés (VaR/CVaR cohérentes, GARCH, décomposition d'Euler,
Markowitz vs RL), chaque équation correspondant à du code existant. Un encadré
explique séparément ce que le calcul quantique promet (QAOA, estimation
d'amplitude) et **pourquoi rien n'en est codé ici**. Écrire une section
quantique sans implémentation aurait été un argument d'autorité.

**CDC-9 — Temps de récupération non calculé.**
*Écart.* Le maximum drawdown est mesuré, mais pas le *recovery time* exigé par
K-3. Grandeur non enregistrée : **rapportée comme indisponible** plutôt
qu'estimée.

## 10.6 Synthèse de conformité

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

---

# 11 · Défauts trouvés et corrigés

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

---

# 12 · Limites connues

Ces limites sont **assumées et documentées**, non dissimulées.

## 12.1 Bloquantes avant un déploiement public

- **SMTP non configuré** : réinitialisation de mot de passe inopérante.
- **Authentification multifacteur absente.**
- **Rotation des jetons de rafraîchissement non implémentée.**
- **`SECRET_KEY` à changer** avant toute mise en ligne.
- **Verrouillage après échecs répétés** non implémenté.

## 12.2 Scientifiques

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

## 12.3 Méthodologiques

Les tests s'exécutent en `DATA_MODE=offline` sur données synthétiques : la
branche de code exercée peut différer d'une exécution manuelle sur données
réelles. Les tests concernés forcent explicitement la branche visée par des
doublures (`SignalContribution` factice, `monkeypatch` des constructeurs de
signaux).

---

---

# 13 · Conclusion

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

# Annexe A-bis · Reproduire les captures d'écran

Les 14 captures de la [section 6](#6--captures-de-lapplication) sont produites
par un script, sur l'application en fonctionnement — aucune n'est retouchée.

```bash
# 1. démarrer la plateforme
bash scripts/run_server.sh start

# 2. installer le navigateur de capture (une fois)
python3 -m playwright install chromium
python3 -m playwright install-deps chromium

# 3. produire la galerie dans docs/screens/
python3 scripts/capture_screens.py
```

Le script se connecte avec un compte réel, force le thème clair, attend la fin
des appels réseau de chaque page, puis enregistre en 1460 px de large à une
densité de 2. Les captures de la page de stress testing lancent un vrai scénario
sur un panier de trois actifs avant l'enregistrement, afin que la page montre
des résultats et non un formulaire vide.

---

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
| **`docs/Cahier_des_charges.pdf`** | **Le cahier des charges officiel audité en § 9** |
| `docs/RAPPORT_PROJET.md` | Rapport de conception — source des § 1 à 5 |
| `docs/RAPPORT_FINAL.md` | Rapport de mesures — source des § 6 à 12 |
| `docs/ARCHITECTURE.md` | Architecture technique détaillée |
| `docs/UI_REDESIGN.md` | Refonte du design system |
| `docs/latex/M122_RL_MiniProject.pdf` | Étude RL multi-graines (40 pages, registre D-1 … D-26) |
| `docs/latex/M122_latex_source.zip` | Source LaTeX complet + figures |
| `README.md` | Installation, démarrage rapide, référence d'API |

> **Le présent document est le rapport de référence.** Il fusionne
> `RAPPORT_PROJET.md` (contexte, état de l'art, conception) et
> `RAPPORT_FINAL.md` (mesures, audit du cahier des charges, limites) en un seul
> texte continu. Les deux sources sont conservées : elles restent utiles prises
> séparément, et leur suppression aurait rompu les liens existants.

---

*Fin du rapport.*
