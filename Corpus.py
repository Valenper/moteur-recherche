"""Module Corpus : classe gérant une collection de documents (Partie 6)."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from Author import Author
from Document import ArxivDocument, Document, RedditDocument


class Corpus:
    """Un corpus : les documents, leurs auteurs et les opérations sur l'ensemble."""

    def __init__(self, nom: str) -> None:
        """Constructeur : un corpus vide (6.1)."""
        self.nom = nom                                # nom du corpus
        self.documents: dict[int, Document] = {}      # {id: Document}
        self.id_document = 0                          # compteur de documents
        self.authors: dict[str, Author] = {}          # {nom de l'auteur: Author}

    # ----------------------------------------------------------------------
    # 6.1 — Ajout d'un document (reprend les Parties 4.3 et 5.3)
    # ----------------------------------------------------------------------
    def add(self, doc: Document) -> int:
        """Ajoute un document au corpus, met à jour ses auteurs, renvoie son id."""
        # Le compteur fournit un identifiant unique et séquentiel : 1, 2, 3...
        self.id_document += 1
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

        df = pd.read_csv(chemin, sep="\t", encoding="utf-8")
        for ligne in df.itertuples():
            date = datetime.fromisoformat(ligne.date)    # texte ISO -> datetime

            # La colonne origine indique quelle classe fille recréer.
            # (Même logique que construire_corpus dans main.py : en Partie 9,
            # une fabrique centralisera ce choix à un seul endroit.)
            if ligne.origine == "reddit":
                doc = RedditDocument(ligne.titre, ligne.auteur, date, ligne.url,
                                     ligne.texte,
                                     nb_commentaires=int(ligne.nb_commentaires))
            else:
                # Case vide (NaN, qui n'est pas du texte) : aucun co-auteur
                if isinstance(ligne.coauteurs, str):
                    coauteurs = ligne.coauteurs.split(", ")
                else:
                    coauteurs = []
                doc = ArxivDocument(ligne.titre, ligne.auteur, date, ligne.url,
                                    ligne.texte, coauteurs=coauteurs)

            # add() recrée les identifiants et le dictionnaire des auteurs
            self.add(doc)
