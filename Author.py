"""Module Author : classe représentant un auteur du corpus (Partie 5)."""

from Document import Document


class Author:
    """Un auteur et l'ensemble des documents qu'il a écrits."""

    def __init__(self, name: str) -> None:
        """Constructeur : un nouvel auteur n'a encore aucun document (5.1)."""
        self.name = name                              # nom de l'auteur
        self.nb_docs = 0                              # nombre de documents publiés
        self.production: dict[int, Document] = {}     # {id du document: Document}

    # ----------------------------------------------------------------------
    # 5.2 — Ajout d'un document à la production de l'auteur
    # ----------------------------------------------------------------------
    def add(self, id_doc: int, doc: Document) -> None:
        """Ajoute un document à la production et met à jour le compteur."""
        # Le test évite de compter deux fois un document déjà ajouté
        if id_doc not in self.production:
            self.production[id_doc] = doc
            self.nb_docs += 1

    # ----------------------------------------------------------------------
    # 5.4 — Statistique : taille moyenne des documents
    # ----------------------------------------------------------------------
    def taille_moyenne(self) -> float:
        """Nombre moyen de mots par document de l'auteur."""
        if self.nb_docs == 0:          # évite une division par zéro
            return 0.0
        total_mots = sum(len(doc.texte.split()) for doc in self.production.values())
        return total_mots / self.nb_docs

    def __str__(self) -> str:
        """Version digeste, appelée par print(auteur)."""
        return f"{self.name} ({self.nb_docs} document(s))"
