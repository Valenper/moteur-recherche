"""Projet « Moteur de recherche » — Partie 1 : acquisition et premières manipulations.

- Questions 1.1 et 1.2 : récupérer des documents sur la thématique « FIFA World Cup »
  depuis Reddit (librairie praw) et Arxiv (librairies urllib + xmltodict).
- Questions 2.1 à 2.3 : ranger ces documents dans un DataFrame pandas, le sauvegarder
  en CSV (séparateur tabulation) et le recharger sans réinterroger les API.
- Questions 3.1 à 3.4 : premières manipulations (taille, mots, phrases,
  filtrage des documents trop courts, concaténation de tous les textes).
- Parties 4 à 6 : les documents sont rangés dans un objet Corpus, qui gère
  aussi les auteurs, les affichages triés et la sauvegarde / le chargement.
- Partie 7 : chaque document est un RedditDocument ou un ArxivDocument.
"""

import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
import xmltodict

from Corpus import Corpus
from Document import ArxivDocument, RedditDocument

# --- Paramètres de la thématique ------------------------------------------
SUBREDDIT = "worldcup"                                # subreddit interrogé
REQUETE_ARXIV = 'all:"world cup" OR all:soccer'       # requête Arxiv
NB_DOCS = 100                                         # documents par source

# Les articles sur le football de robots (RoboCup) sont hors sujet.
# L'opérateur ANDNOT de l'API Arxiv ne les excluait pas : on les filtre
# donc nous-mêmes en Python, en cherchant ces expressions dans le texte.
MOTS_HORS_SUJET = ["robocup", "robot", "soccer simulation", "simulation 2d",
                   "simulation league"]

# --- Fichier de sauvegarde -------------------------------------------------
# Path construit le chemin avec le bon séparateur selon l'OS (\ ou /) :
# le programme reste multiplateforme, comme l'exigent les consignes.
CHEMIN_CSV = Path("data") / "corpus.csv"                     # données brutes des API
CHEMIN_SAUVEGARDE = Path("data") / "corpus_sauvegarde.csv"   # sauvegarde du Corpus


def nettoyer_espaces(chaine: str) -> str:
    """Remplace sauts de ligne, tabulations et espaces multiples par un espace."""
    # split() découpe sur TOUT type d'espace ; join recolle avec un espace simple
    return " ".join(chaine.split())


def est_hors_sujet(texte: str) -> bool:
    """True si le texte contient l'une des expressions de MOTS_HORS_SUJET."""
    texte = texte.lower()     # comparaison insensible aux majuscules
    # any() renvoie True dès qu'une expression est trouvée dans le texte
    return any(expression in texte for expression in MOTS_HORS_SUJET)


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
    # Construction de l'URL : urlencode gère les espaces et les guillemets.
    # On demande 3 fois plus d'articles que nécessaire, car une partie
    # sera écartée par le filtre hors sujet (RoboCup).
    parametres = urllib.parse.urlencode({
        "search_query": REQUETE_ARXIV,
        "start": 0,
        "max_results": 3 * nb_docs,
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
    nb_exclus = 0
    for i, entree in enumerate(entrees):
        if i == 0:
            print("Champs disponibles (Arxiv) :")
            print(list(entree.keys()), "\n")

        # On s'arrête dès qu'on a assez d'articles
        if len(resultats) == nb_docs:
            break

        # Article hors sujet (RoboCup) : on le saute et on passe au suivant
        if est_hors_sujet(entree["title"] + " " + entree["summary"]):
            nb_exclus += 1
            continue

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

    print(f"Articles Arxiv exclus (hors sujet) : {nb_exclus}")
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
# 6.1 / 7 — Construction du corpus à partir du tableau
# ==========================================================================
def construire_corpus(df: pd.DataFrame, nom: str) -> Corpus:
    """Crée un objet Corpus contenant un document par ligne du tableau.

    Selon la colonne origine, on instancie la classe fille adaptée.
    (En Partie 9, ce choix sera confié à une fabrique.)
    """
    corpus = Corpus(nom)
    # itertuples parcourt le tableau ligne par ligne (ligne.titre, ligne.date...)
    for ligne in df.itertuples():
        # Dans le CSV, la date est du texte : on la reconvertit en datetime
        date = datetime.fromisoformat(ligne.date)

        if ligne.origine == "reddit":
            doc = RedditDocument(ligne.titre, ligne.auteur, date, ligne.url,
                                 ligne.texte,
                                 # pandas lit les nombres de cette colonne en
                                 # décimaux (42.0) : on reconvertit en entier
                                 nb_commentaires=int(ligne.nb_commentaires))
        else:
            # Dans le CSV : "Bob, Carol" ; case vide (NaN) s'il n'y en a pas
            if isinstance(ligne.coauteurs, str):
                coauteurs = ligne.coauteurs.split(", ")
            else:
                coauteurs = []
            doc = ArxivDocument(ligne.titre, ligne.auteur, date, ligne.url,
                                ligne.texte, coauteurs=coauteurs)

        corpus.add(doc)   # add() accepte les deux classes filles sans modification
    return corpus


# ==========================================================================
# Programme principal
# ==========================================================================
if __name__ == "__main__":
    if CHEMIN_CSV.exists():
        # Le fichier existe déjà : on recharge (rapide, hors ligne)
        print(f"Chargement des données depuis {CHEMIN_CSV}")
        df = charger(CHEMIN_CSV)
    else:
        # Premier lancement : on interroge les API puis on sauvegarde
        print("Aucune sauvegarde trouvée : téléchargement depuis les API...")
        docs_reddit = recuperer_reddit(NB_DOCS)
        docs_arxiv = recuperer_arxiv(NB_DOCS)
        df = creer_dataframe(docs_reddit, docs_arxiv)
        sauvegarder(df, CHEMIN_CSV)
        print(f"Données sauvegardées dans {CHEMIN_CSV}")

    # --- 3.1 : taille du corpus -------------------------------------------
    print(f"\nTaille du corpus : {len(df)} documents")
    print(df["origine"].value_counts().to_string())

    # --- 3.2 : nombre de mots et de phrases par document ------------------
    # apply() appelle la fonction sur chaque texte de la colonne
    df["nb_mots"] = df["texte"].apply(compter_mots)
    df["nb_phrases"] = df["texte"].apply(compter_phrases)
    print("\nMoyennes de mots et de phrases par source :")
    print(df.groupby("origine")[["nb_mots", "nb_phrases"]].mean().round(1))

    # --- 3.3 : suppression des documents de moins de 100 caractères -------
    taille_avant = len(df)
    df = filtrer_courts(df, 100)
    print(f"\nDocuments supprimés (< 100 caractères) : {taille_avant - len(df)}")

    # --- 3.4 : une unique chaîne contenant tous les documents -------------
    docs = df["texte"].tolist()            # la liste demandée en Partie 1
    texte_complet = " ".join(docs)         # join colle les textes avec un espace
    print(f"Longueur de la chaîne complète : {len(texte_complet)} caractères")

    # --- 6.1 : création de l'objet Corpus ---------------------------------
    corpus = construire_corpus(df, "FIFA World Cup")
    print(f"\n{corpus!r}")        # !r : f-string qui appelle __repr__

    # --- 7 : un document de chaque classe fille ---------------------------
    # next() renvoie le premier document qui vérifie la condition
    doc_reddit = next(d for d in corpus.documents.values() if isinstance(d, RedditDocument))
    doc_arxiv = next(d for d in corpus.documents.values() if isinstance(d, ArxivDocument))

    print("\n--- Un RedditDocument (afficher : mère + commentaires) ---")
    doc_reddit.afficher()
    print("\n--- Un ArxivDocument (afficher : mère + co-auteurs) ---")
    doc_arxiv.afficher()

    # Le mutateur refuse une valeur invalide
    print("\n--- Test du mutateur : nb_commentaires = -5 ---")
    try:
        doc_reddit.nb_commentaires = -5
    except ValueError as erreur:
        print(f"Erreur attrapée : {erreur}")

    # --- 6.2 : affichages triés -------------------------------------------
    print("\n--- 5 documents les plus récents ---")
    corpus.afficher_tri_date(5)
    print("\n--- 5 premiers documents par ordre alphabétique ---")
    corpus.afficher_tri_titre(5)

    # --- 5.3 / 5.4 : auteurs ----------------------------------------------
    print("\n--- Auteurs les plus productifs ---")
    for auteur in corpus.auteurs_plus_productifs(5):
        print(f"  {auteur}")

    nom = input("\nNom d'un auteur (Entrée pour passer) : ").strip()
    if nom:
        corpus.statistiques_auteur(nom)

    # --- 6.3 / 6.4 : sauvegarde puis rechargement du corpus ---------------
    corpus.save(CHEMIN_SAUVEGARDE)
    print(f"\nCorpus sauvegardé dans {CHEMIN_SAUVEGARDE}")

    corpus_recharge = Corpus("FIFA World Cup")
    corpus_recharge.load(CHEMIN_SAUVEGARDE)
    print(f"Corpus rechargé : {corpus_recharge!r}")

    # Vérification : mêmes documents, dans le même ordre, avant et après
    identiques = all(
        corpus.documents[i].titre == corpus_recharge.documents[i].titre
        and corpus.documents[i].date == corpus_recharge.documents[i].date
        for i in corpus.documents
    )
    print(f"Sauvegarde et rechargement identiques : {identiques}")
