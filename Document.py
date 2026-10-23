"""Module Document : classe représentant un document du corpus (Partie 4)."""

from datetime import datetime


class Document:
    """Un document du corpus : son contenu textuel et ses méta-données."""

    def __init__(self, titre: str = "", auteur: str = "",
                 date: datetime | None = None, url: str = "",
                 texte: str = "") -> None:
        """Constructeur : initialise les 5 attributs demandés par l'énoncé (4.1)."""
        self.titre = titre      # titre du document
        self.auteur = auteur    # nom de l'auteur
        self.date = date        # date de publication (objet datetime)
        self.url = url          # adresse de la source
        self.texte = texte      # contenu textuel

    # ----------------------------------------------------------------------
    # 4.2 — Affichage
    # ----------------------------------------------------------------------
    def afficher(self) -> None:
        """Affiche toutes les informations de l'instance."""
        # La date peut être absente (None) : on prévoit ce cas
        date = self.date.strftime("%d/%m/%Y") if self.date else "inconnue"
        print(f"Titre  : {self.titre}")
        print(f"Auteur : {self.auteur}")
        print(f"Date   : {date}")
        print(f"URL    : {self.url}")
        print(f"Texte  : {self.texte}")

    def __str__(self) -> str:
        """Version « digeste », appelée par print(doc) : le titre seul."""
        return self.titre
