"""Module Document : la classe mère Document (Partie 4), ses deux classes
filles RedditDocument et ArxivDocument (Partie 7) et la fabrique
DocumentFactory (Partie 9.2)."""

from datetime import datetime


class Document:
    """Un document du corpus : son contenu textuel et ses méta-données."""

    def __init__(self, titre: str = "", auteur: str = "",
                 date: datetime | None = None, url: str = "",
                 texte: str = "", type: str = "") -> None:
        """Constructeur : initialise les attributs communs à tous les documents."""
        self.titre = titre      # titre du document
        self.auteur = auteur    # nom de l'auteur
        self.date = date        # date de publication (objet datetime)
        self.url = url          # adresse de la source
        self.texte = texte      # contenu textuel
        self.type = type        # source du document (8.2), fixée par les classes filles

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


# ==========================================================================
# 7.1 — Classe fille RedditDocument
# ==========================================================================
class RedditDocument(Document):
    """Un post Reddit : un Document avec en plus son nombre de commentaires."""

    def __init__(self, titre: str = "", auteur: str = "",
                 date: datetime | None = None, url: str = "",
                 texte: str = "", nb_commentaires: int = 0) -> None:
        # super() appelle le constructeur de la classe mère, qui initialise
        # les attributs communs ; la classe fille lui transmet son type
        super().__init__(titre, auteur, date, url, texte, type="reddit")
        # Passe par le mutateur (setter) ci-dessous : la valeur est vérifiée
        self.nb_commentaires = nb_commentaires

    def getType(self) -> str:
        """8.2 — Renvoie la source du document."""
        return self.type

    # --- Accesseur / mutateur ---------------------------------------------
    @property
    def nb_commentaires(self) -> int:
        """Accesseur : lecture avec doc.nb_commentaires."""
        return self.__nb_commentaires

    @nb_commentaires.setter
    def nb_commentaires(self, valeur: int) -> None:
        """Mutateur : écriture avec doc.nb_commentaires = valeur (vérifiée)."""
        if valeur < 0:
            raise ValueError("Le nombre de commentaires ne peut pas être négatif.")
        # Double tiret bas : attribut « privé », inaccessible directement de l'extérieur
        self.__nb_commentaires = valeur

    # --- Affichages spécifiques -------------------------------------------
    def afficher(self) -> None:
        """Toutes les informations : celles de la mère, plus les commentaires."""
        super().afficher()          # réutilise l'affichage de la classe mère
        print(f"Commentaires : {self.nb_commentaires}")

    def __str__(self) -> str:
        return f"[Reddit] {self.titre} ({self.nb_commentaires} commentaires)"


# ==========================================================================
# 7.2 — Classe fille ArxivDocument
# ==========================================================================
class ArxivDocument(Document):
    """Un article Arxiv : un Document avec en plus la liste de ses co-auteurs."""

    def __init__(self, titre: str = "", auteur: str = "",
                 date: datetime | None = None, url: str = "",
                 texte: str = "", coauteurs: list[str] | None = None) -> None:
        super().__init__(titre, auteur, date, url, texte, type="arxiv")
        # Jamais de liste vide [] comme valeur par défaut : elle serait PARTAGÉE
        # entre tous les objets. On utilise None et on crée une liste neuve ici.
        self.coauteurs = coauteurs if coauteurs is not None else []

    def getType(self) -> str:
        """8.2 — Renvoie la source du document."""
        return self.type

    # --- Accesseur / mutateur ---------------------------------------------
    @property
    def coauteurs(self) -> list[str]:
        """Accesseur : lecture avec doc.coauteurs."""
        return self.__coauteurs

    @coauteurs.setter
    def coauteurs(self, valeur: list[str]) -> None:
        """Mutateur : écriture avec doc.coauteurs = [...] (doit être une liste)."""
        if not isinstance(valeur, list):
            raise ValueError("Les co-auteurs doivent être fournis sous forme de liste.")
        self.__coauteurs = valeur

    def ajouter_coauteur(self, nom: str) -> None:
        """Ajoute un co-auteur s'il n'est pas déjà dans la liste."""
        if nom not in self.__coauteurs:
            self.__coauteurs.append(nom)

    # --- Affichages spécifiques -------------------------------------------
    def afficher(self) -> None:
        """Toutes les informations : celles de la mère, plus les co-auteurs."""
        super().afficher()
        coauteurs = ", ".join(self.coauteurs) if self.coauteurs else "aucun"
        print(f"Co-auteurs : {coauteurs}")

    def __str__(self) -> str:
        if self.coauteurs:
            return f"[Arxiv] {self.titre} ({self.auteur} + {len(self.coauteurs)} co-auteur(s))"
        return f"[Arxiv] {self.titre} ({self.auteur})"


# ==========================================================================
# 9.2 — Patron Fabrique (Factory)
# ==========================================================================
class DocumentFactory:
    """Fabrique : centralise en un seul endroit la création des documents."""

    @staticmethod
    def creer(type: str, titre: str, auteur: str, date: datetime, url: str,
              texte: str, nb_commentaires: int = 0,
              coauteurs: list[str] | None = None) -> Document:
        """Renvoie une instance de la classe fille correspondant au type."""
        # Méthode statique : elle ne reçoit ni self ni cls, c'est une simple
        # fonction rangée dans la classe. On l'appelle DocumentFactory.creer(...)
        if type == "reddit":
            return RedditDocument(titre, auteur, date, url, texte,
                                  nb_commentaires=nb_commentaires)
        elif type == "arxiv":
            return ArxivDocument(titre, auteur, date, url, texte,
                                 coauteurs=coauteurs)
        raise ValueError(f"Type de document inconnu : {type}")
