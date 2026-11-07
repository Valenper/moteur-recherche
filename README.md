# Moteur de recherche « maison » — FIFA World Cup

**Valentin PERROUX** — Master 1 Informatique, Université Lumière Lyon 2
Projet de Programmation de spécialité (Python), année 2026–2027.

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
python -m unittest -v     # tests unitaires (46 tests)
```

Tout fonctionne sans connexion à Internet : les scripts et le notebook utilisent les
données déjà présentes dans `data/`.

### Interface graphique (Partie 3)

1. Lancer `jupyter notebook` depuis le dossier du projet (environnement virtuel activé) :
   le navigateur s'ouvre sur la liste des fichiers.
2. Ouvrir `interface.ipynb`, puis menu **Run → Run All Cells**.
3. L'interface, en bas du notebook, comporte trois onglets :
   - **Recherche** : mots-clés, nombre d'articles, mesure (TF-IDF ou BM25), racinisation,
     filtres (source, auteur, période), suggestion « Vouliez-vous dire… ? » ; les liens
     « ouvrir » mènent au post Reddit ou à l'article Arxiv d'origine ;
   - **Concordancier** : passages autour d'une expression régulière ;
   - **Reddit vs Arxiv** : mots communs et spécifiques à chaque source, tableaux et graphiques.

**Remarque** : dans l'éditeur de notebooks de VS Code, le contenu des zones de résultats
(widget `Output`) s'affiche en double. Ce défaut vient de l'affichage des widgets par
VS Code, pas du programme : dans Jupyter (navigateur), chaque résultat ne s'affiche
qu'une fois. Il est donc conseillé d'utiliser `jupyter notebook`.

## Fonctionnalités

**Spécifications de base (Parties 1 à 3)** : acquisition Reddit et Arxiv, classes
`Document` (et classes filles), `Author`, `Corpus` (Singleton), fabrique de documents,
sauvegarde CSV, recherche par expressions régulières, concordancier, statistiques,
matrices TF et TF-IDF (`scipy.sparse`), similarité cosinus, classe `SearchEngine`,
interface ipywidgets, barres de progression `tqdm`.

**Extensions (bonus)** :

| Bonus | Description |
|---|---|
| 1 — Filtres | par source (via `getType`), par auteur, par période |
| 4 — Comparaison Reddit / Arxiv | mots communs et spécifiques, liste de mots vides écrite à la main |
| 6 — Okapi BM25 | seconde mesure de pertinence, comparée à TF-IDF |
| 7 — Visualisations | diagrammes en barres (matplotlib) |
| 8 — Concordancier interactif | onglet dédié de l'interface |
| Racinisation | racinisation simplifiée de l'anglais, écrite à la main (predicting, prediction → predict) |
| « Vouliez-vous dire… ? » | correction des fautes de frappe par la distance de Levenshtein |

## Organisation du code

| Fichier | Contenu |
|---|---|
| `Document.py` | classe mère `Document`, classes filles `RedditDocument` et `ArxivDocument`, fabrique `DocumentFactory` |
| `Author.py` | classe `Author` |
| `Corpus.py` | classe `Corpus` (Singleton), `nettoyer_texte`, liste `MOTS_VIDES`, recherche par expressions régulières, concordancier, statistiques |
| `SearchEngine.py` | vocabulaire, matrices TF et TF-IDF (`scipy.sparse`), similarité cosinus, BM25, filtres, comparaison des sources, barres `tqdm` |
| `Racinisation.py` | racinisation simplifiée (bonus) |
| `Orthographe.py` | distance de Levenshtein et suggestion de mots (bonus) |
| `Visualisation.py` | graphiques matplotlib (bonus 7) |
| `main.py` | acquisition des données (API Reddit et Arxiv) et démonstration de la Partie 1 |
| `main_partie2.py` | démonstration de la Partie 2 |
| `interface.ipynb` | interface graphique (ipywidgets) de la Partie 3 ; ne contient que l'interface, les classes sont importées des modules |
| `test_classes.py`, `test_moteur.py` | tests unitaires (`unittest`) |
| `data/corpus.csv` | données brutes issues des API (200 documents) |
| `data/corpus_sauvegarde.csv` | corpus nettoyé, sauvegardé par `Corpus.save` (172 documents) |

## Identifiants Reddit

Copier `config_exemple.py` en `config.py` et y renseigner les identifiants.
Ils ne sont nécessaires que pour re-télécharger les données (en supprimant
`data/corpus.csv`) : le corpus est déjà fourni dans le dossier `data/`.

## Versions (tags Git)

- **v1** : acquisition des données et structuration en classes
- **v2** : analyse textuelle et moteur de recherche (TF-IDF, cosinus)
- **v3** : interface graphique, extensions et tests unitaires

## Utilisation d'un assistant d'IA

Dans le cadre de ce projet, un assistant d'intelligence artificielle (Claude, d'Anthropic) a été utilisé uniquement comme outil d'accompagnement. Il a servi à documenter et commenter le code, à diagnostiquer certaines erreurs ponctuelles, à identifier des cas de tests supplémentaires (notamment des cas limites) et à vérifier l'exhaustivité des modifications avant chaque commit (voir l'historique Git), ainsi qu'à assurer la relecture orthographique du rapport. La logique métier, l'écriture du code, les choix de conception et l'implémentation des tests sur les données ont été intégralement réalisés et validés par moi-même.