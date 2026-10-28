"""Module Corpus : classe gérant une collection de documents (Partie 6),
rendue unique par le patron Singleton (Partie 9.1).

Projet Partie 2 : recherche de passages par expressions régulières
(search) et concordancier (concorde).
"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from Author import Author
from Document import Document, DocumentFactory


# ==========================================================================
# 9.1 — Patron Singleton, écrit comme un décorateur (cf. cours)
# ==========================================================================
def singleton(cls):
    """Décorateur : la classe décorée ne pourra avoir qu'une seule instance."""
    instance = [None]       # liste : modifiable depuis la fonction wrapper

    def wrapper(*args, **kwargs):
        # Premier appel : on crée l'instance. Appels suivants : on renvoie
        # toujours la même, les arguments sont alors ignorés.
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper


# ==========================================================================
# Conversion d'une ligne de tableau (CSV) en document
# ==========================================================================
def document_depuis_ligne(ligne) -> Document:
    """Crée le document correspondant à une ligne lue avec itertuples().

    Utilisée à la fois par Corpus.load et par main.py : le code de
    création n'existe plus qu'à un seul endroit, et le choix de la
    classe fille est confié à la fabrique.
    """
    # Cases vides du CSV (NaN) pour l'autre source : valeurs neutres
    nb_commentaires = int(ligne.nb_commentaires) if pd.notna(ligne.nb_commentaires) else 0
    coauteurs = ligne.coauteurs.split(", ") if isinstance(ligne.coauteurs, str) else []

    return DocumentFactory.creer(
        ligne.origine,                         # "reddit" ou "arxiv"
        ligne.titre, ligne.auteur,
        datetime.fromisoformat(ligne.date),    # texte ISO -> datetime
        ligne.url, ligne.texte,
        nb_commentaires=nb_commentaires,
        coauteurs=coauteurs,
    )


@singleton      # équivaut à écrire, après la classe : Corpus = singleton(Corpus)
class Corpus:
    """Un corpus : les documents, leurs auteurs et les opérations sur l'ensemble."""

    def __init__(self, nom: str) -> None:
        """Constructeur : un corpus vide (6.1)."""
        self.nom = nom                                # nom du corpus
        self.documents: dict[int, Document] = {}      # {id: Document}
        self.id_document = 0                          # compteur de documents
        self.authors: dict[str, Author] = {}          # {nom de l'auteur: Author}
        # Cache de la chaîne concaténant tous les textes (projet Partie 2) :
        # None tant qu'elle n'a pas été construite
        self.__texte_complet: str | None = None

    # ----------------------------------------------------------------------
    # 6.1 — Ajout d'un document (reprend les Parties 4.3 et 5.3)
    # ----------------------------------------------------------------------
    def add(self, doc: Document) -> int:
        """Ajoute un document au corpus, met à jour ses auteurs, renvoie son id."""
        # Le compteur fournit un identifiant unique et séquentiel : 1, 2, 3...
        self.id_document += 1
        self.__texte_complet = None     # le corpus change : le cache est périmé
        self.documents[self.id_document] = doc

        # Nouvel auteur : on crée son instance avant de lui ajouter le document
        if doc.auteur not in self.authors:
            self.authors[doc.auteur] = Author(doc.auteur)
        self.authors[doc.auteur].add(self.id_document, doc)

        return self.id_document

    # ----------------------------------------------------------------------
    # Statistiques sur les auteurs (reprend la Partie 5.4)
    # ----------------------------------------------------------------------
    def auteurs_plus_productifs(self, n: int = 5) -> list[Author]:
        """Renvoie les n auteurs ayant écrit le plus de documents."""
        return sorted(self.authors.values(), key=lambda a: a.nb_docs, reverse=True)[:n]

    def statistiques_auteur(self, nom: str) -> None:
        """Affiche le nombre de documents et leur taille moyenne pour un auteur."""
        if nom not in self.authors:
            print(f"Auteur « {nom} » introuvable dans le corpus.")
            return
        auteur = self.authors[nom]
        print(f"Auteur                 : {auteur.name}")
        print(f"Nombre de documents    : {auteur.nb_docs}")
        print(f"Taille moyenne         : {auteur.taille_moyenne():.1f} mots")

    # ----------------------------------------------------------------------
    # 6.2 — Affichages triés
    # ----------------------------------------------------------------------
    def afficher_tri_date(self, n: int = 5) -> None:
        """Affiche les n documents les plus récents (tri par date décroissante)."""
        docs_tries = sorted(self.documents.values(), key=lambda d: d.date, reverse=True)
        for doc in docs_tries[:n]:
            print(f"  {doc.date:%d/%m/%Y} | {doc}")

    def afficher_tri_titre(self, n: int = 5) -> None:
        """Affiche les n premiers documents par ordre alphabétique du titre."""
        # lower() : sinon les majuscules passeraient avant toutes les minuscules
        docs_tries = sorted(self.documents.values(), key=lambda d: d.titre.lower())
        for doc in docs_tries[:n]:
            print(f"  {doc}")

    def afficher_sources(self, n: int = 5) -> None:
        """8.2 — Affiche les n documents les plus récents avec leur source."""
        docs_tries = sorted(self.documents.items(), key=lambda paire: paire[1].date,
                            reverse=True)
        # Polymorphisme : getType() est appelée sans savoir si doc est un
        # RedditDocument ou un ArxivDocument ; chacun répond pour lui-même
        for id_doc, doc in docs_tries[:n]:
            print(f"  {id_doc:>3} | {doc.getType():<6} | {doc.titre}")

    def __repr__(self) -> str:
        """Affichage digeste et sans ambiguïté, appelé par repr(corpus)."""
        return (f"Corpus(nom='{self.nom}', {len(self.documents)} documents, "
                f"{len(self.authors)} auteurs)")

    # ----------------------------------------------------------------------
    # 6.3 — Sauvegarde et chargement en passant par un DataFrame
    # ----------------------------------------------------------------------
    def save(self, chemin: Path) -> None:
        """Enregistre le corpus sur le disque (CSV, séparateur tabulation)."""
        # Une ligne par document : on transforme chaque objet en dictionnaire
        lignes = []
        for id_doc, doc in self.documents.items():
            ligne = {
                "id": id_doc,
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date.isoformat(),   # datetime -> texte ISO
                "url": doc.url,
                "texte": doc.texte,
                "origine": doc.getType(),       # pour recréer la bonne classe fille
            }
            # Attributs spécifiques : la case reste vide pour l'autre source
            if doc.getType() == "reddit":
                ligne["nb_commentaires"] = doc.nb_commentaires
            elif doc.getType() == "arxiv":
                ligne["coauteurs"] = ", ".join(doc.coauteurs)
            lignes.append(ligne)
        chemin.parent.mkdir(exist_ok=True)
        pd.DataFrame(lignes).to_csv(chemin, sep="\t", index=False, encoding="utf-8")

    def load(self, chemin: Path) -> None:
        """Remplace le contenu du corpus par celui du fichier sauvegardé."""
        # On repart d'un corpus vide pour ne pas mélanger anciens et nouveaux documents
        self.documents = {}
        self.id_document = 0
        self.authors = {}
        self.__texte_complet = None

        df = pd.read_csv(chemin, sep="\t", encoding="utf-8")
        for ligne in df.itertuples():
            # La fabrique recrée la bonne classe fille ; add() recrée les
            # identifiants et le dictionnaire des auteurs
            self.add(document_depuis_ligne(ligne))

    # ======================================================================
    # Projet Partie 2, section 1 — Expressions régulières
    # ======================================================================
    def texte_complet(self) -> str:
        """Renvoie une chaîne unique concaténant les textes de tous les documents.

        Elle n'est construite qu'une fois, au premier appel, puis mise en cache.
        """
        if self.__texte_complet is None:
            # Séparateur \n : les textes n'en contiennent plus (nettoyer_espaces),
            # il marque donc la frontière entre deux documents
            self.__texte_complet = "\n".join(doc.texte for doc in self.documents.values())
        return self.__texte_complet

    @staticmethod
    def _compiler(motif: str) -> re.Pattern:
        """Compile le motif, entouré de frontières de mot, insensible à la casse."""
        try:
            return re.compile(r"\b" + motif + r"\b", re.IGNORECASE)
        except re.error as erreur:
            # Ex. une parenthèse non fermée : message clair au lieu d'un plantage
            raise ValueError(f"Expression régulière invalide : {motif} ({erreur})")

    def search(self, mot: str) -> list[str]:
        """1.1 — Renvoie les passages (phrases) contenant le mot-clé.

        Le mot-clé est une expression régulière : "goals?" trouve goal et goals.
        """
        motif = self._compiler(mot)
        # [^.\n]* : tout sauf un point ou une frontière de document. Le passage
        # renvoyé va donc du début à la fin de la phrase qui contient le mot.
        phrase = re.compile(r"[^.\n]*" + motif.pattern + r"[^.\n]*", re.IGNORECASE)
        return [passage.strip() for passage in phrase.findall(self.texte_complet())]

    def concorde(self, expression: str, taille_contexte: int = 30) -> pd.DataFrame:
        """1.2 — Concordancier : contexte gauche / motif trouvé / contexte droit."""
        texte = self.texte_complet()
        lignes = []
        # finditer donne chaque occurrence avec sa position (start, end) dans le texte
        for occurrence in self._compiler(expression).finditer(texte):
            debut, fin = occurrence.start(), occurrence.end()
            lignes.append({
                # max(0, ...) : pas d'indice négatif en tout début de texte
                "contexte gauche": texte[max(0, debut - taille_contexte):debut],
                "motif trouvé": occurrence.group(),
                "contexte droit": texte[fin:fin + taille_contexte],
            })
        concordance = pd.DataFrame(lignes, columns=["contexte gauche", "motif trouvé",
                                                    "contexte droit"])
        # Les frontières de documents (\n) deviennent des espaces à l'affichage
        return concordance.replace("\n", " ", regex=True)
