"""Module Orthographe : distance de Levenshtein et suggestion de mots (bonus).

Sert à proposer « Vouliez-vous dire… ? » quand un mot de la requête est
absent du vocabulaire (faute de frappe : mesi -> messi).
"""


def levenshtein(mot1: str, mot2: str) -> int:
    """Nombre minimal d'opérations d'une lettre (ajout, suppression,
    remplacement) pour transformer mot1 en mot2.

    Programmation dynamique : on remplit ligne par ligne un tableau où la case
    (i, j) contient la distance entre les i premières lettres de mot1 et les
    j premières lettres de mot2. On ne garde que la ligne précédente en
    mémoire. Complexité : len(mot1) x len(mot2) opérations.
    """
    # Ligne 0 : passer du mot vide aux j premières lettres de mot2 = j ajouts
    precedente = list(range(len(mot2) + 1))
    for i, lettre1 in enumerate(mot1, start=1):
        courante = [i]                     # colonne 0 : i suppressions
        for j, lettre2 in enumerate(mot2, start=1):
            cout = 0 if lettre1 == lettre2 else 1
            courante.append(min(
                precedente[j] + 1,            # suppression de lettre1
                courante[j - 1] + 1,          # ajout de lettre2
                precedente[j - 1] + cout,     # remplacement (gratuit si identiques)
            ))
        precedente = courante
    return precedente[-1]


def mot_le_plus_proche(mot: str, vocabulaire: dict[str, dict],
                       distance_max: int = 2) -> str | None:
    """Mot du vocabulaire le plus proche de `mot`, ou None si aucun n'est à
    distance <= distance_max.

    vocabulaire : le dictionnaire vocab de SearchEngine. À distance égale, on
    préfère le mot le plus fréquent du corpus (champ « occurrences »).
    """
    meilleur, meilleure_cle = None, None
    for candidat, infos in vocabulaire.items():
        # Filtre rapide : si les longueurs diffèrent de plus de distance_max,
        # la distance est forcément trop grande, inutile de la calculer
        if abs(len(candidat) - len(mot)) > distance_max:
            continue
        distance = levenshtein(mot, candidat)
        if distance <= distance_max:
            # Comparaison de couples : d'abord la distance (petite), puis la
            # fréquence (grande, d'où le signe moins)
            cle = (distance, -infos.get("occurrences", 0))
            if meilleure_cle is None or cle < meilleure_cle:
                meilleur, meilleure_cle = candidat, cle
    return meilleur
