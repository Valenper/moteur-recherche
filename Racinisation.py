"""Module Racinisation : racinisation (stemming) simplifiée de l'anglais (bonus).

Ramène les mots d'une même famille à une racine commune :
predicting, prediction, predicted -> predict ; outcomes -> outcome.

Écrit à la main (nltk interdit), en s'inspirant de l'algorithme de Porter,
mais avec quelques règles seulement pour rester lisible. La racine obtenue
n'est pas toujours un vrai mot (decision -> decis) : ce qui compte, c'est que
les mots d'une même famille donnent la MÊME racine, dans les documents comme
dans la requête.
"""

LONGUEUR_MIN_RACINE = 3          # on ne coupe jamais en dessous de 3 lettres

# Suffixes retirés à l'étape 2, du plus long au plus court
SUFFIXES = ("ing", "ion", "ed", "ly")

VOYELLES = set("aeiouy")


def _retirer_pluriel(mot: str) -> str:
    """Étape 1 : formes du pluriel (et 3e personne du singulier)."""
    if mot.endswith("ies") and len(mot) > 4:          # studies -> study
        return mot[:-3] + "y"
    if mot.endswith(("ches", "shes", "sses", "xes", "zes")):   # matches -> match
        return mot[:-2]
    # « s » final, sauf si le mot finit par ss, us ou is (class, bus, analysis)
    if mot.endswith("s") and not mot.endswith(("ss", "us", "is")) and len(mot) > 3:
        return mot[:-1]                                # outcomes -> outcome
    return mot


def _retirer_suffixe(mot: str) -> str:
    """Étape 2 : un suffixe de dérivation (le premier qui convient)."""
    for suffixe in SUFFIXES:
        racine = mot[:-len(suffixe)]
        # La racine doit être assez longue et contenir au moins une voyelle :
        # sinon « sing » donnerait « s » et « red » donnerait « r »
        if (mot.endswith(suffixe) and len(racine) >= LONGUEUR_MIN_RACINE
                and any(lettre in VOYELLES for lettre in racine)):
            # Consonne doublée après -ing / -ed : running -> runn -> run
            if (suffixe in ("ing", "ed") and len(racine) >= 2
                    and racine[-1] == racine[-2] and racine[-1] not in VOYELLES
                    and racine[-1] not in "lsz"):         # sauf ll, ss, zz (falling)
                racine = racine[:-1]
            return racine
    return mot


def raciniser(mot: str) -> str:
    """Renvoie la racine d'un mot (déjà en minuscules, sans ponctuation)."""
    if len(mot) <= LONGUEUR_MIN_RACINE or mot.isdigit():   # mots courts, nombres
        return mot
    return _retirer_suffixe(_retirer_pluriel(mot))
