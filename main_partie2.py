"""Projet « Moteur de recherche » — Partie 2 : analyse du contenu textuel.

Le corpus nettoyé est rechargé depuis data/corpus_sauvegarde.csv (produit par
main.py en Partie 1) : pas de nouvel appel aux API.

- Section 1 : expressions régulières (search, concorde).
"""

from pathlib import Path

import pandas as pd

from Corpus import Corpus

CHEMIN_SAUVEGARDE = Path("data") / "corpus_sauvegarde.csv"

# Affichage des tableaux pandas : textes en entier, sur une seule ligne
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)


if __name__ == "__main__":
    corpus = Corpus("FIFA World Cup")
    corpus.load(CHEMIN_SAUVEGARDE)
    print(f"{corpus!r}")

    # --- 1.1 : search ---------------------------------------------------------
    mot = "Messi"
    passages = corpus.search(mot)
    print(f"\n--- search('{mot}') : {len(passages)} passages (5 premiers) ---")
    for passage in passages[:5]:
        print(f"  - {passage}")

    # Le mot-clé est une expression régulière : s? = « s facultatif »
    print(f"\nsearch('goals?') : {len(corpus.search('goals?'))} passages "
          f"(contre {len(corpus.search('goal'))} pour 'goal' seul)")

    # --- 1.2 : concorde -------------------------------------------------------
    expression = "VAR"
    concordance = corpus.concorde(expression, taille_contexte=35)
    print(f"\n--- concorde('{expression}', 35) : {len(concordance)} occurrences "
          f"(10 premières) ---")
    print(concordance.head(10).to_string(index=False))

    # --- Saisie par l'utilisateur ---------------------------------------------
    expression = input("\nExpression à rechercher (Entrée pour passer) : ").strip()
    if expression:
        try:
            print(corpus.concorde(expression, 35).to_string(index=False))
        except ValueError as erreur:
            print(f"Erreur : {erreur}")
