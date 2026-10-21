"""Projet « Moteur de recherche » — Partie 1 : acquisition des données.

Questions 1.1 et 1.2 : récupérer des textes sur la thématique « FIFA World Cup »
depuis Reddit (librairie praw) et Arxiv (librairies urllib + xmltodict),
puis les ranger dans une liste `docs`.
"""

import urllib.parse
import urllib.request

import praw
import xmltodict

from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# --- Paramètres de la thématique ------------------------------------------
SUBREDDIT = "worldcup"                                # subreddit interrogé
REQUETE_ARXIV = 'all:"world cup" OR all:soccer'       # requête Arxiv
NB_DOCS = 100                                         # documents par source


# ==========================================================================
# 1.1 — Reddit
# ==========================================================================
def recuperer_reddit(nb_docs: int) -> list[str]:
    """Renvoie la liste des textes des posts Reddit (titre + contenu)."""
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
    # Construction de l'URL : urlencode gère les espaces et les guillemets
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
# Programme principal
# ==========================================================================
if __name__ == "__main__":
    textes_reddit = recuperer_reddit(NB_DOCS)
    textes_arxiv = recuperer_arxiv(NB_DOCS)

    docs = textes_reddit + textes_arxiv   # la liste demandée par l'énoncé

    print(f"Posts Reddit récupérés    : {len(textes_reddit)}")
    print(f"Articles Arxiv récupérés  : {len(textes_arxiv)}")
    print(f"Total dans docs           : {len(docs)}")

    # Aperçu : le premier document de chaque source (200 premiers caractères)
    print("\n--- Exemple Reddit ---\n", textes_reddit[0][:200])
    print("\n--- Exemple Arxiv ---\n", textes_arxiv[0][:200])
