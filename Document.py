"""Module Document : la classe mère Document (Partie 4) et ses deux classes
filles RedditDocument et ArxivDocument (Partie 7)."""

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


# ==========================================================================
# 7.1 — Classe fille RedditDocument
# ==========================================================================
class RedditDocument(Document):
    """Un post Reddit : un Document avec en plus son nombre de commentaires."""

    def __init__(self, titre: str = "", auteur: str = "",
                 date: datetime | None = None, url: str = "",
                 texte: str = "", nb_commentaires: int = 0) -> None:
        # super() appelle le constructeur de la classe mère, qui initialise
        # les 5 attributs communs à tous les documents
        super().__init__(titre, auteur, date, url, texte)
        # Passe par le mutateur (setter) ci-dessous : la valeur est vérifiée
        self.nb_commentaires = nb_commentaires

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
        super().__init__(titre, auteur, date, url, texte)
        # Jamais de liste vide [] comme valeur par défaut : elle serait PARTAGÉE
        # entre tous les objets. On utilise None et on crée une liste neuve ici.
        self.coauteurs = coauteurs if coauteurs is not None else []

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
        return f"[Arxiv] {self.titre} ({self.auteur} + {len(self.coauteurs)} co-auteur(s))"
