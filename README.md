# Moteur de recherche « maison » — FIFA World Cup

Projet de Programmation de spécialité (Python) — Master 1 Informatique, Université Lumière Lyon 2.

Moteur de recherche construit sans scikit-learn ni nltk, sur un corpus de documents
portant sur la thématique **FIFA World Cup** :

- **Reddit** : discussions de supporters (subreddit `worldcup`, posts les plus populaires de l'année), centrées sur la Coupe du monde 2026 ;
- **Arxiv** : articles scientifiques sur la Coupe du monde et le football (prédiction, statistiques, analyse de matchs), hors articles sur la RoboCup (football de robots).

## Installation (Windows)

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sous macOS / Linux : `python3.13 -m venv .venv` puis `source .venv/bin/activate`.

## Utilisation

```
python main.py            # Partie 1 : données, classes, sauvegarde du corpus
python main_partie2.py    # Partie 2 : regex, statistiques, moteur de recherche
jupyter notebook          # Partie 3 : puis ouvrir interface.ipynb
```

Tout fonctionne sans connexion à Internet : les scripts et le notebook utilisent les
données déjà présentes dans `data/`.

### Interface graphique (Partie 3)

1. Lancer `jupyter notebook` depuis le dossier du projet (environnement virtuel activé) :
   le navigateur s'ouvre sur la liste des fichiers.
2. Ouvrir `interface.ipynb`, puis menu **Run → Run All Cells**.
3. En bas du notebook : saisir des mots-clés, choisir le nombre d'articles, cliquer sur
   **Rechercher**. Les liens « ouvrir » mènent au post Reddit ou à l'article Arxiv d'origine.

**Remarque** : dans l'éditeur de notebooks de VS Code, le contenu de la zone de résultats
(widget `Output`) s'affiche en double. Ce défaut vient de l'affichage des widgets par
VS Code, pas du programme : dans Jupyter (navigateur), chaque résultat ne s'affiche
qu'une fois. Il est donc conseillé d'utiliser `jupyter notebook`.

## Organisation du code

| Fichier | Contenu |
|---|---|
| `Document.py` | classe mère `Document`, classes filles `RedditDocument` et `ArxivDocument`, fabrique `DocumentFactory` |
| `Author.py` | classe `Author` |
| `Corpus.py` | classe `Corpus` (Singleton), `nettoyer_texte`, recherche par expressions régulières, concordancier, statistiques |
| `SearchEngine.py` | vocabulaire, matrices TF et TF-IDF (`scipy.sparse`), recherche par similarité cosinus, barres de progression `tqdm` |
| `main.py` | acquisition des données (API Reddit et Arxiv) et démonstration de la Partie 1 |
| `main_partie2.py` | démonstration de la Partie 2 |
| `interface.ipynb` | interface graphique (ipywidgets) de la Partie 3 ; ne contient que l'interface, les classes sont importées des modules |
| `data/corpus.csv` | données brutes issues des API (200 documents) |
| `data/corpus_sauvegarde.csv` | corpus nettoyé, sauvegardé par `Corpus.save` (172 documents) |

## Identifiants Reddit

Copier `config_exemple.py` en `config.py` et y renseigner les identifiants.
Ils ne sont nécessaires que pour re-télécharger les données (en supprimant
`data/corpus.csv`) : le corpus est déjà fourni dans le dossier `data/`.

## Versions (tags Git)

- **v1** : acquisition des données et structuration en classes
- **v2** : analyse textuelle et moteur de recherche (TF-IDF, cosinus)
- **v3** : interface graphique et extensions
