"""Projet « Moteur de recherche » — Partie 1 : acquisition et premières manipulations.

- Questions 1.1 et 1.2 : récupérer des documents sur la thématique « FIFA World Cup »
  depuis Reddit (librairie praw) et Arxiv (librairies urllib + xmltodict).
- Questions 2.1 à 2.3 : ranger ces documents dans un DataFrame pandas, le sauvegarder
  en CSV (séparateur tabulation) et le recharger sans réinterroger les API.
- Questions 3.1 à 3.4 : premières manipulations (taille, mots, phrases,
  filtrage des documents trop courts, concaténation de tous les textes).
- Question 4.3 : instanciation d'objets Document rangés dans un dictionnaire.
"""

import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
import xmltodict

from Document import Document

# --- Paramètres de la thématique ------------------------------------------
SUBREDDIT = "worldcup"                                # subreddit interrogé
# On exclut les articles sur le football de robots (RoboCup), hors sujet
REQUETE_ARXIV = '(all:"world cup" OR all:soccer) ANDNOT all:robot ANDNOT all:robocup'
NB_DOCS = 100                                         # documents par source

# --- Fichier de sauvegarde -------------------------------------------------
# Path construit le chemin avec le bon séparateur selon l'OS (\ ou /) :
# le programme reste multiplateforme, comme l'exigent les consignes.
CHEMIN_CSV = Path("data") / "corpus.csv"


def nettoyer_espaces(chaine: str) -> str:
    """Remplace sauts de ligne, tabulations et espaces multiples par un espace."""
    # split() découpe sur TOUT type d'espace ; join recolle avec un espace simple
    return " ".join(chaine.split())


# ==========================================================================
# 1.1 — Reddit
# ==========================================================================
def recuperer_reddit(nb_docs: int) -> list[dict]:
    """Renvoie la liste des posts Reddit, chacun sous forme de dictionnaire."""
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

    resultats = []
    for i, post in enumerate(posts):
        if i == 0:
            # Question « quels sont les champs disponibles ? » :
            # vars() renvoie le dictionnaire des attributs de l'objet post
            print("Champs disponibles (Reddit) :")
            print(sorted(vars(post).keys()), "\n")

        titre = nettoyer_espaces(post.title)
        resultats.append({
            "titre": titre,
            # author vaut None quand le compte a été supprimé
            "auteur": str(post.author) if post.author else "[supprimé]",
            # created_utc = nombre de secondes depuis le 01/01/1970 (timestamp)
            "date": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).isoformat(),
            # permalink = adresse de la discussion sur Reddit
            "url": "https://www.reddit.com" + post.permalink,
            # Texte = titre + corps (selftext, vide pour une image ou un lien)
            "texte": nettoyer_espaces(titre + ". " + post.selftext),
            # Spécifique à Reddit, servira en Partie 7
            "nb_commentaires": post.num_comments,
        })

    return resultats


# ==========================================================================
# 1.2 — Arxiv
# ==========================================================================
def recuperer_arxiv(nb_docs: int) -> list[dict]:
    """Renvoie la liste des articles Arxiv, chacun sous forme de dictionnaire."""
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

    resultats = []
    for i, entree in enumerate(entrees):
        if i == 0:
            print("Champs disponibles (Arxiv) :")
            print(list(entree.keys()), "\n")

        # Même piège que pour les entrées : un seul auteur = un dict, pas une liste
        auteurs = entree["author"]
        if isinstance(auteurs, dict):
            auteurs = [auteurs]
        noms = [auteur["name"] for auteur in auteurs]

        titre = nettoyer_espaces(entree["title"])
        resultats.append({
            "titre": titre,
            "auteur": noms[0],                    # premier auteur
            # published est au format ISO : "2024-05-12T10:00:00Z"
            "date": datetime.fromisoformat(entree["published"]).isoformat(),
            "url": entree["id"],                  # adresse de l'article
            # Le contenu textuel d'un article est son résumé (champ « summary »)
            "texte": nettoyer_espaces(titre + ". " + entree["summary"]),
            # Spécifique à Arxiv, servira en Partie 7
            "coauteurs": ", ".join(noms[1:]),
        })

    return resultats


# ==========================================================================
# 2.1 — Création du DataFrame
# ==========================================================================
def creer_dataframe(docs_reddit: list[dict], docs_arxiv: list[dict]) -> pd.DataFrame:
    """Range les documents dans un tableau : une ligne par document."""
    for doc in docs_reddit:
        doc["origine"] = "reddit"
    for doc in docs_arxiv:
        doc["origine"] = "arxiv"

    # Une liste de dictionnaires devient un tableau : les clés -> les colonnes
    df = pd.DataFrame(docs_reddit + docs_arxiv)
    df.insert(0, "id", range(len(df)))   # identifiant unique en 1re colonne
    return df


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
# 4.3 — Instanciation des objets Document
# ==========================================================================
def creer_documents(df: pd.DataFrame) -> dict[int, Document]:
    """Crée un Document par ligne du tableau, indexé par un identifiant unique."""
    documents = {}
    # enumerate génère les identifiants séquentiellement : 1, 2, 3, ...
    # itertuples parcourt le tableau ligne par ligne (ligne.titre, ligne.date...)
    for id_doc, ligne in enumerate(df.itertuples(), start=1):
        documents[id_doc] = Document(
            titre=ligne.titre,
            auteur=ligne.auteur,
            # Dans le CSV, la date est du texte : on la reconvertit en datetime
            date=datetime.fromisoformat(ligne.date),
            url=ligne.url,
            texte=ligne.texte,
        )
    return documents


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
        docs_reddit = recuperer_reddit(NB_DOCS)
        docs_arxiv = recuperer_arxiv(NB_DOCS)
        df = creer_dataframe(docs_reddit, docs_arxiv)
        sauvegarder(df, CHEMIN_CSV)
        print(f"Corpus sauvegardé dans {CHEMIN_CSV}")

    # --- 3.1 : taille du corpus -------------------------------------------
    print(f"\nTaille du corpus : {len(df)} documents")
    print(df["origine"].value_counts().to_string())

    # --- 3.2 : nombre de mots et de phrases par document ------------------
    # apply() appelle la fonction sur chaque texte de la colonne
    df["nb_mots"] = df["texte"].apply(compter_mots)
    df["nb_phrases"] = df["texte"].apply(compter_phrases)
    print("\nMots et phrases (5 premiers documents) :")
    print(df[["id", "origine", "nb_mots", "nb_phrases"]].head().to_string(index=False))
    print("\nMoyennes par source :")
    print(df.groupby("origine")[["nb_mots", "nb_phrases"]].mean().round(1))

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

    # --- 4.3 : dictionnaire des objets Document ---------------------------
    documents = creer_documents(df)
    print(f"\nDictionnaire documents : {len(documents)} objets Document")

    print("\n--- print(doc) appelle __str__ : titre seul ---")
    for id_doc in list(documents)[:3]:
        print(f"{id_doc} : {documents[id_doc]}")

    print("\n--- afficher() : toutes les informations ---")
    documents[1].afficher()
