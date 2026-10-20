# Moteur de recherche « maison » — FIFA World Cup

Projet de Programmation de spécialité (Python) — Master 1 Informatique, Université Lumière Lyon 2.

Moteur de recherche construit sans scikit-learn ni nltk, sur un corpus de documents
portant sur la thématique **FIFA World Cup** :

- **Reddit** : discussions de supporters (subreddits `worldcup` et `soccer`), centrées sur la Coupe du monde 2026 ;
- **Arxiv** : articles scientifiques sur la Coupe du monde et le football (prédiction, statistiques, analyse de matchs).

## Installation (Windows)

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sous macOS / Linux : `python3.13 -m venv .venv` puis `source .venv/bin/activate`.

## Identifiants Reddit

Copier `config_exemple.py` en `config.py` et y renseigner les identifiants.
Ils ne sont nécessaires que pour re-télécharger les données : le corpus est déjà
fourni dans le dossier `data/`.

## Versions

- **v1** : acquisition des données et structuration en classes
- **v2** : analyse textuelle et moteur de recherche (TF-IDF, cosinus)
- **v3** : interface graphique et extensions
