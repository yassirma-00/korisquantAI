# Réduction de la taille du projet

**Résultat : 68 Mo → 25 Mo** dans le dépôt, et **22 Mo supplémentaires** libérés
hors dépôt (mes captures de travail dans `/home/user/shots`).

Le dépôt ne contient plus que les fichiers du projet KorisQuant AI : code
backend et frontend, configurations, modèles entraînés, base de données et
documentation Markdown.

---

## 1. Ce qui a été supprimé

| Poste | Gain | Récupérable ? |
|---|---:|---|
| 14 jumeaux `__regime` SB3 et paniers | 12,87 Mo | Oui — mais le MoE ne pouvait pas les piloter (refusés en HTTP 422) |
| `docs/screens/` (14 captures) | 4,24 Mo | Oui — `python3 scripts/capture_screens.py` |
| `data/cache/` (parquets marché) | 3,65 Mo | Oui — se reconstruit au premier appel réseau |
| 3 PDF + 2 archives `.zip` | 17,2 Mo | Les PDF anglais et français, oui (sources `.md` conservées) |
| `docs/latex/` (code LaTeX) | 0,43 Mo | **Partiellement — voir §2** |
| `docs/Cahier_des_charges.pdf` | 0,20 Mo | Non — l'original reste dans `/home/user/uploads/` |
| 4 jumeaux `__regime` MSFT | 1,55 Mo | Oui — AAPL conserve les 6 algorithmes du MoE |
| `.bak`, checkpoints intermédiaires, caches Python | ~2,9 Mo | Sans objet |
| `/home/user/shots` (hors dépôt) | 22 Mo | Sans objet — captures de vérification |

## 2. Perte définitive assumée

> **Le rapport M122 n'est plus reconstructible.** Son fichier
> `M122_RL_MiniProject.tex` était **écrit à la main** : 2 462 lignes et
> **30 divergences documentées (D-1 … D-30)**. Le Markdown de secours
> `docs/M122_RL_MINI_PROJECT.md` n'en contient que 684 lignes et 3 divergences,
> soit environ **27 % du contenu**.
>
> La suppression a été signalée avant exécution et confirmée. Ce qui subsiste du
> travail M122 : le Markdown ci-dessus, plus les résultats bruts dans
> `data/artifacts/multiseed_*.json`, qui restent la source de tous les chiffres.

Les gabarits LaTeX (`preamble`, `titlepage`) et les deux Makefiles ont disparu
avec le dossier. Les rapports **anglais** et **français** restent régénérables en
PDF, mais il faudra recréer une chaîne pandoc.

## 3. Ce qui a été conservé

* **Code source** : `backend/`, `frontend/`, `scripts/`, `configs/`
* **21 agents baseline** — résultats cités dans les rapports (A2C +5,52 %, etc.)
* **6 jumeaux `__regime` AAPL** — les six algorithmes que le MoE pilote
* **7 forecasters** — sources du tableau de précision directionnelle
* **`data/finai.db`** — 87 comptes, 2 920 alertes
* **`data/artifacts/`** — tous les résultats mesurés au format JSON
* **12 documents Markdown** dans `docs/`

## 4. Références corrigées

Supprimer des fichiers rend fausses les pages qui les citent. Corrigé :

* `docs/REPORT_EN.md` — les 6 inclusions d'images remplacées par une note
  expliquant comment régénérer la galerie ; le lien vers le PDF M122 pointe
  désormais vers le Markdown.
* `docs/RAPPORT_COMPLET.md` — les 14 inclusions remplacées par une mention
  textuelle de chaque capture.
* `scripts/make_latex_archive.py` et `scripts/make_report_figures.py` supprimés :
  leur cible `docs/latex/` n'existe plus.

## 5. Vérifications après suppression

| Contrôle | Résultat |
|---|---|
| Tests | **776 passent** |
| `ruff check backend/app backend/tests` | **clean** |
| Application (health + login) | **200 / 200** |
| MoE (`?moe=true`) | **200**, 1 affinage réel |
| Métadonnées orphelines | **aucune** |
| PDF restants dans le dépôt | **aucun** |

## 6. Régénérer ce qui manque

```bash
# Galerie de 14 captures
bash scripts/run_server.sh start
python3 -m playwright install chromium
python3 scripts/capture_screens.py

# Cache marché : automatique au premier appel, ou
python3 -c "import sys; sys.path.insert(0,'backend'); \
  from app.services.data.market_data import market_data_service as m; \
  [m.get_history(s, period='2y') for s in ('AAPL','MSFT','SPY')]"

# Jumeaux régime-aware supplémentaires
python3 scripts/retrain_regime_aware.py --only AAPL_sac AAPL_ppo
```

## 7. Sur votre machine

```bash
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
rm -rf .pytest_cache .ruff_cache
git gc --aggressive --prune=now
```

Le `.venv` (souvent 300 Mo+ à cause de torch) se recrée avec
`pip install -r requirements.txt`.
