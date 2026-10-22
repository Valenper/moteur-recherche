"""Projet « Moteur de recherche » — Partie 1 : acquisition et sauvegarde des données.

- Questions 1.1 et 1.2 : récupérer des textes sur la thématique « FIFA World Cup »
  depuis Reddit (librairie praw) et Arxiv (librairies urllib + xmltodict).
- Questions 2.1 à 2.3 : ranger ces textes dans un DataFrame pandas, le sauvegarder
  en CSV (séparateur tabulation) et le recharger sans réinterroger les API.
- Questions 3.1 à 3.4 : premières manipulations (taille, mots, phrases,
  filtrage des documents trop courts, concaténation de tous les textes).
"""

import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import praw
import xmltodict

# --- Paramètres de la thématique ------------------------------------------
SUBREDDIT = "worldcup"                                # subreddit interrogé
# On exclut les articles sur le football de robots (RoboCup), hors sujet
REQUETE_ARXIV = '(all:"world cup" OR all:soccer) ANDNOT all:robot ANDNOT all:robocup'
NB_DOCS = 100                                         # documents par source

# --- Fichier de sauvegarde -------------------------------------------------
# Path construit le chemin avec le bon séparateur selon l'OS (\ ou /) :
# le programme reste multiplateforme, comme l'exigent les consignes.
CHEMIN_CSV = Path("data") / "corpus.csv"


# ==========================================================================
# 1.1 — Reddit
# ==========================================================================
def recuperer_reddit(nb_docs: int) -> list[str]:
    """Renvoie la liste des textes des posts Reddit (titre + contenu)."""
    # Import placé ici : config.py n'est nécessaire QUE pour télécharger.
    # Quelqu'un qui clone le dépôt (sans config.py) peut donc lancer le
    # programme à partir du CSV fourni.
    from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

    # Connexion à l'API (en lecture seule : pas besoin de compte utilisateur)
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    # Les posts les plus populaires de l'année : couvre la Coupe du monde 2026
    posts = reddit.subreddit(SUBREDDIT).top(time_filter="year", limit=nb_docs)

    textes = []
    for i, post in enumerate(posts):
        if i == 0:
            # Question « quels sont les champs disponibles ? » :
            # vars() renvoie le dictionnaire des attributs de l'objet post
            print("Champs disponibles (Reddit) :")
            print(sorted(vars(post).keys()), "\n")

        # Le titre est toujours présent ; selftext est le corps du post
        # (vide quand le post est une simple image ou un lien)
        texte = post.title + ". " + post.selftext
        textes.append(texte.replace("\n", " "))   # suppression des sauts de ligne

    return textes


# ==========================================================================
# 1.2 — Arxiv
# ==========================================================================
def recuperer_arxiv(nb_docs: int) -> list[str]:
    """Renvoie la liste des textes des articles Arxiv (titre + résumé)."""
    # Construction de l'URL : urlencode gère les espaces, guillemets, parenthèses
    parametres = urllib.parse.urlencode({
        "search_query": REQUETE_ARXIV,
        "start": 0,
        "max_results": nb_docs,
    })
    url = "https://export.arxiv.org/api/query?" + parametres

    # Envoi de la requête et lecture de la réponse (au format XML)
    with urllib.request.urlopen(url) as reponse:
        xml = reponse.read().decode("utf-8")

    # xmltodict transforme le XML en dictionnaires Python imbriqués
    donnees = xmltodict.parse(xml)
    entrees = donnees["feed"].get("entry", [])   # un article = une « entry »
    if isinstance(entrees, dict):                # un seul résultat : pas de liste
        entrees = [entrees]

    textes = []
    for i, entree in enumerate(entrees):
        if i == 0:
            print("Champs disponibles (Arxiv) :")
            print(list(entree.keys()), "\n")

        # Le contenu textuel d'un article est son résumé (champ « summary »)
        texte = entree["title"] + ". " + entree["summary"]
        textes.append(texte.replace("\n", " "))

    return textes


# ==========================================================================
# 2.1 — Création du DataFrame
# ==========================================================================
def creer_dataframe(textes_reddit: list[str], textes_arxiv: list[str]) -> pd.DataFrame:
    """Range les textes dans un tableau à 3 colonnes : id, texte, origine."""
    textes = textes_reddit + textes_arxiv
    origines = ["reddit"] * len(textes_reddit) + ["arxiv"] * len(textes_arxiv)

    return pd.DataFrame({
        "id": range(len(textes)),   # identifiant unique : 0, 1, 2, ...
        "texte": textes,
        "origine": origines,
    })


# ==========================================================================
# 2.2 — Sauvegarde sur le disque
# ==========================================================================
def sauvegarder(df: pd.DataFrame, chemin: Path) -> None:
    """Enregistre le tableau en CSV, avec la tabulation comme séparateur."""
    chemin.parent.mkdir(exist_ok=True)       # crée le dossier data/ si besoin
    # index=False : on n'écrit pas la numérotation interne de pandas
    # (on a déjà notre propre colonne id)
    df.to_csv(chemin, sep="\t", index=False, encoding="utf-8")


# ==========================================================================
# 2.3 — Rechargement depuis le disque
# ==========================================================================
def charger(chemin: Path) -> pd.DataFrame:
    """Relit le tableau sauvegardé, sans interroger les API."""
    return pd.read_csv(chemin, sep="\t", encoding="utf-8")


# ==========================================================================
# 3.2 — Nombre de mots et de phrases
# ==========================================================================
def compter_mots(texte: str) -> int:
    """Nombre de mots : on découpe sur les espaces."""
    # split() sans argument découpe sur tout espace (y compris plusieurs
    # espaces à la suite), ce qui évite de compter des « mots vides »
    return len(texte.split())


def compter_phrases(texte: str) -> int:
    """Nombre de phrases : on découpe sur les points."""
    # "A. B." donne ["A", " B", ""] : on ne garde que les morceaux non vides
    return len([morceau for morceau in texte.split(".") if morceau.strip()])


# ==========================================================================
# 3.3 — Suppression des documents trop petits
# ==========================================================================
def filtrer_courts(df: pd.DataFrame, taille_min: int = 100) -> pd.DataFrame:
    """Renvoie le tableau sans les documents de moins de taille_min caractères."""
    return df[df["texte"].str.len() >= taille_min]


# ==========================================================================
# Programme principal
# ==========================================================================
if __name__ == "__main__":
    if CHEMIN_CSV.exists():
        # Le fichier existe déjà : on recharge (rapide, hors ligne)
        print(f"Chargement du corpus depuis {CHEMIN_CSV}")
        df = charger(CHEMIN_CSV)
    else:
        # Premier lancement : on interroge les API puis on sauvegarde
        print("Aucune sauvegarde trouvée : téléchargement depuis les API...")
        textes_reddit = recuperer_reddit(NB_DOCS)
        textes_arxiv = recuperer_arxiv(NB_DOCS)
        df = creer_dataframe(textes_reddit, textes_arxiv)
        sauvegarder(df, CHEMIN_CSV)
        print(f"Corpus sauvegardé dans {CHEMIN_CSV}")

    # --- 3.1 : taille du corpus -------------------------------------------
    print(f"\nTaille du corpus : {len(df)} documents")
    print(df["origine"].value_counts().to_string())

    # --- 3.2 : nombre de mots et de phrases par document ------------------
    # apply() appelle la fonction sur chaque texte de la colonne
    df["nb_mots"] = df["texte"].apply(compter_mots)
    df["nb_phrases"] = df["texte"].apply(compter_phrases)
    print("\nMots et phrases par document :")
    print(df[["id", "origine", "nb_mots", "nb_phrases"]].to_string(index=False))

    # --- 3.3 : suppression des documents de moins de 100 caractères -------
    taille_avant = len(df)
    df = filtrer_courts(df, 100)
    print(f"\nDocuments supprimés (< 100 caractères) : {taille_avant - len(df)}")
    print(f"Taille du corpus après filtrage : {len(df)} documents")
    print(df["origine"].value_counts().to_string())

    # --- 3.4 : une unique chaîne contenant tous les documents -------------
    docs = df["texte"].tolist()            # la liste demandée en Partie 1
    texte_complet = " ".join(docs)         # join colle les textes avec un espace
    print(f"\nLongueur de la chaîne complète : {len(texte_complet)} caractères")
