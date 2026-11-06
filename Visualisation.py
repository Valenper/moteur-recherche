"""Module Visualisation : graphiques de l'interface (bonus 7).

Les fonctions renvoient un objet Figure de matplotlib, sans l'afficher : c'est
l'interface (le notebook) qui décide où l'afficher.
"""

import pandas as pd
from matplotlib.figure import Figure

# Une couleur par source, toujours la même dans tous les graphiques
# (palette catégorielle vérifiée : lisible aussi par les daltoniens)
COULEURS = {"reddit": "#2a78d6", "arxiv": "#eb6834"}
TEXTE = "#0b0b0b"           # couleur du texte principal
TEXTE_SECONDAIRE = "#52514e"
FOND = "#fcfcfb"            # fond du graphique
GRILLE = "#e4e3df"          # lignes de repère discrètes


def _preparer_axe(axe, titre: str) -> None:
    """Style commun : titre, grille verticale discrète, cadre allégé."""
    axe.set_title(titre, loc="left", fontsize=11, color=TEXTE, fontweight="bold")
    axe.set_facecolor(FOND)
    axe.grid(axis="x", color=GRILLE, linewidth=0.8)
    axe.set_axisbelow(True)                       # la grille passe derrière les barres
    for cote in ["top", "right", "left"]:
        axe.spines[cote].set_visible(False)
    axe.spines["bottom"].set_color(GRILLE)
    axe.tick_params(colors=TEXTE_SECONDAIRE, labelsize=9, length=0)
    axe.invert_yaxis()                            # le mot le plus important en haut


def graphique_sources(resultat: dict[str, pd.DataFrame], n: int = 10) -> Figure:
    """Diagrammes en barres de la comparaison Reddit / Arxiv (comparer_sources).

    Trois graphiques côte à côte : l'importance des mots communs dans chaque
    source, puis les mots spécifiques à Reddit et à Arxiv.
    """
    communs = resultat["communs"].head(n)
    reddit = resultat["reddit"].head(n)
    arxiv = resultat["arxiv"].head(n)

    figure = Figure(figsize=(13, 0.35 * n + 1.6), facecolor=FOND,
                    layout="constrained")        # marges calculées automatiquement
    axe_communs, axe_reddit, axe_arxiv = figure.subplots(1, 3)

    # 1. Mots communs : deux barres par mot (une par source), légèrement décalées
    positions = range(len(communs))
    epaisseur = 0.38
    axe_communs.barh([p - epaisseur / 2 for p in positions], communs["importance Reddit"],
                     height=epaisseur, color=COULEURS["reddit"], label="Reddit",
                     edgecolor=FOND, linewidth=1)     # fin liseré : barres bien séparées
    axe_communs.barh([p + epaisseur / 2 for p in positions], communs["importance Arxiv"],
                     height=epaisseur, color=COULEURS["arxiv"], label="Arxiv",
                     edgecolor=FOND, linewidth=1)
    axe_communs.set_yticks(list(positions), communs["mot"])
    _preparer_axe(axe_communs, "Mots communs")
    axe_communs.legend(frameon=False, fontsize=9, labelcolor=TEXTE, loc="lower right")

    # 2. et 3. Mots spécifiques : une seule série par graphique, nommée par le titre
    for axe, tableau, source, titre in [
        (axe_reddit, reddit, "reddit", "Spécifiques à Reddit"),
        (axe_arxiv, arxiv, "arxiv", "Spécifiques à Arxiv"),
    ]:
        axe.barh(tableau["mot"], tableau[f"importance {source.capitalize()}"],
                 height=0.6, color=COULEURS[source])
        _preparer_axe(axe, titre)

    for axe in (axe_communs, axe_reddit, axe_arxiv):
        axe.set_xlabel("importance (fréquence ‰ × IDF)", color=TEXTE_SECONDAIRE, fontsize=9)
    return figure
