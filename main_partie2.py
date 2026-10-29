"""Projet « Moteur de recherche » — Partie 2 : analyse du contenu textuel.

Le corpus nettoyé est rechargé depuis data/corpus_sauvegarde.csv (produit par
main.py en Partie 1) : pas de nouvel appel aux API.

- Section 1 : expressions régulières (search, concorde).
- Section 2 : nettoyage du texte et statistiques (vocabulaire, TF, DF).
- Section 3 : matrices Documents x Termes (TF et TF-IDF) dans SearchEngine.
"""

from pathlib import Path

import pandas as pd

from Corpus import Corpus, nettoyer_texte
from SearchEngine import SearchEngine

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

    # Saisie par l'utilisateur
    expression = input("\nExpression à rechercher (Entrée pour passer) : ").strip()
    if expression:
        try:
            resultat = corpus.concorde(expression, 35)
            if resultat.empty:                       # aucune occurrence trouvée
                print(f"Aucune occurrence de « {expression} » dans le corpus.")
            else:
                print(resultat.to_string(index=False))
        except ValueError as erreur:
            print(f"Erreur : {erreur}")

    # --- 2.1 : nettoyage du texte ---------------------------------------------
    exemple = corpus.documents[1].texte[:200]
    print("\n--- nettoyer_texte : avant / après ---")
    print(f"Avant : {exemple!r}")      # !r montre les caractères invisibles (\u200b)
    print(f"Après : {nettoyer_texte(exemple)!r}")

    # --- 2.2 à 2.4 : statistiques ---------------------------------------------
    print("\n--- stats(15) ---")
    corpus.stats(15)

    # TF et DF ne disent pas la même chose (2.4)
    freq = corpus.frequences()
    print("\n--- TF vs DF pour quelques mots ---")
    mots_exemples = [mot for mot in ["messi", "var", "penalty", "model", "2026"]
                     if mot in freq.index]
    print(freq.loc[mots_exemples].to_string())

    # --- 3 : matrices Documents x Termes ----------------------------------------
    moteur = SearchEngine(corpus)
    n_docs, n_mots = moteur.mat_TF.shape
    print(f"\n--- Matrice TF : {n_docs} documents x {n_mots} mots ---")
    remplissage = moteur.mat_TF.nnz / (n_docs * n_mots)    # nnz = cases non nulles
    print(f"Cases non nulles : {moteur.mat_TF.nnz} sur {n_docs * n_mots} "
          f"({remplissage:.2%}) : la matrice est très creuse")

    # 3.1 / 3.3 : ce que vocab contient pour un mot
    print(f"vocab['messi'] = {moteur.vocab['messi']}")

    # Vérification croisée avec la section 2 : mêmes TF et DF pour chaque mot
    coherent = all(
        moteur.vocab[mot]["occurrences"] == ligne.term_frequency
        and moteur.vocab[mot]["nb_documents"] == ligne.document_frequency
        for mot, ligne in freq.iterrows()
    )
    print(f"vocab cohérent avec le tableau freq de la section 2 : {coherent}")

    # 3.4 : IDF des mots vus plus haut (log(N / df))
    print("\n--- IDF : rare = élevé, présent partout = faible ---")
    for mot in ["the", "soccer", "model", "messi", "penalty"]:
        if mot in moteur.vocab:
            print(f"  {mot:<8} df = {moteur.vocab[mot]['nb_documents']:>3}   "
                  f"idf = {moteur.idf[moteur.vocab[mot]['id']]:.2f}")

    # Les 5 mots qui caractérisent le mieux le document 1 (TF vs TF-IDF)
    mots_du_vocab = sorted(moteur.vocab, key=lambda mot: moteur.vocab[mot]["id"])
    for nom_matrice, matrice in [("TF", moteur.mat_TF), ("TF-IDF", moteur.mat_TFxIDF)]:
        ligne = matrice[0].toarray().ravel()          # ligne du document 1
        meilleurs = ligne.argsort()[::-1][:5]         # indices des 5 plus grands
        mots = ", ".join(f"{mots_du_vocab[j]} ({ligne[j]:.1f})" for j in meilleurs)
        print(f"Document 1, top 5 {nom_matrice:<6} : {mots}")
