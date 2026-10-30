"""Module SearchEngine : moteur de recherche fondé sur une matrice
Documents x Termes (projet Partie 2, sections 3 à 5).

Tout est construit « à la main » : pas de scikit-learn ni de nltk.
scipy est utilisé uniquement pour stocker la matrice creuse (csr_matrix).
"""

import numpy as np
import pandas as pd
from scipy import sparse

from Corpus import nettoyer_texte


class SearchEngine:
    """Moteur de recherche : vocabulaire, matrices TF et TF-IDF d'un corpus."""

    def __init__(self, corpus) -> None:
        """Reçoit un Corpus et construit aussitôt les matrices (section 5)."""
        self.corpus = corpus
        # Ligne i de la matrice <-> document d'identifiant ids_documents[i]
        self.ids_documents: list[int] = list(corpus.documents.keys())

        self.vocab: dict[str, dict] = {}        # 3.1 : {mot: {id, occurrences, ...}}
        self.mat_TF: sparse.csr_matrix | None = None        # 3.2
        self.idf: np.ndarray | None = None                  # 3.4
        self.mat_TFxIDF: sparse.csr_matrix | None = None    # 3.4
        self.normes_docs: np.ndarray | None = None          # 4 : ||d|| de chaque document

        self._construire()

    def _construire(self) -> None:
        """Enchaîne les étapes de la section 3, dans l'ordre."""
        # Chaque document est nettoyé et découpé UNE seule fois
        mots_par_doc = [nettoyer_texte(self.corpus.documents[i].texte).split()
                        for i in self.ids_documents]
        self._construire_vocab(mots_par_doc)
        self._construire_mat_TF(mots_par_doc)
        self._completer_vocab()
        self._construire_mat_TFxIDF()
        self._calculer_normes()

    # ----------------------------------------------------------------------
    # 3.1 — Vocabulaire
    # ----------------------------------------------------------------------
    def _construire_vocab(self, mots_par_doc: list[list[str]]) -> None:
        """Dictionnaire vocab : mots sans doublons, triés par ordre alphabétique."""
        mots_distincts = set()
        for mots in mots_par_doc:
            mots_distincts.update(mots)          # le set élimine les doublons

        # La position dans l'ordre alphabétique sert d'identifiant :
        # ce sera le numéro de COLONNE du mot dans la matrice
        for identifiant, mot in enumerate(sorted(mots_distincts)):
            self.vocab[mot] = {"id": identifiant}

    # ----------------------------------------------------------------------
    # 3.2 — Matrice TF (Documents x Termes)
    # ----------------------------------------------------------------------
    def _construire_mat_TF(self, mots_par_doc: list[list[str]]) -> None:
        """Case (i, j) = nombre d'occurrences du mot j dans le document i."""
        # On ne mémorise que les cases NON nulles : trois listes parallèles
        # (numéro de ligne, numéro de colonne, valeur)
        lignes, colonnes, valeurs = [], [], []

        for i, mots in enumerate(mots_par_doc):
            compte: dict[str, int] = {}               # TF de chaque mot du document i
            for mot in mots:
                compte[mot] = compte.get(mot, 0) + 1
            for mot, tf in compte.items():
                lignes.append(i)
                colonnes.append(self.vocab[mot]["id"])
                valeurs.append(tf)

        # Matrice creuse au format CSR : les zéros (l'immense majorité) ne
        # sont pas stockés en mémoire
        self.mat_TF = sparse.csr_matrix(
            (valeurs, (lignes, colonnes)),
            shape=(len(mots_par_doc), len(self.vocab)),
            dtype=float,
        )

    # ----------------------------------------------------------------------
    # 3.3 — Occurrences totales et nombre de documents, depuis la matrice
    # ----------------------------------------------------------------------
    def _completer_vocab(self) -> None:
        """Ajoute à vocab le TF total et le DF de chaque mot, lus dans mat_TF."""
        # Somme de chaque COLONNE (axis=0) : occurrences du mot dans tout le corpus
        occurrences = np.asarray(self.mat_TF.sum(axis=0)).ravel()
        # (mat_TF > 0) vaut 1 là où le mot est présent : la somme de la colonne
        # compte donc les documents qui contiennent le mot
        nb_documents = np.asarray((self.mat_TF > 0).sum(axis=0)).ravel()

        for mot, infos in self.vocab.items():
            j = infos["id"]
            infos["occurrences"] = int(occurrences[j])
            infos["nb_documents"] = int(nb_documents[j])

    # ----------------------------------------------------------------------
    # 3.4 — Matrice TF-IDF
    # ----------------------------------------------------------------------
    def _construire_mat_TFxIDF(self) -> None:
        """tfidf(mot, doc) = tf(mot, doc) x idf(mot), avec idf = log(N / df)."""
        n_documents = self.mat_TF.shape[0]
        df = np.array([0] * len(self.vocab), dtype=float)
        for infos in self.vocab.values():
            df[infos["id"]] = infos["nb_documents"]

        # Un mot présent partout a un IDF nul ; un mot rare, un IDF élevé.
        # df >= 1 toujours (chaque mot du vocabulaire vient d'un document)
        self.idf = np.log(n_documents / df)

        # Multiplier chaque colonne j par idf[j] revient à multiplier à droite
        # par la matrice diagonale diag(idf) : le résultat reste creux
        self.mat_TFxIDF = (self.mat_TF @ sparse.diags(self.idf)).tocsr()

    # ======================================================================
    # Sections 4 et 5 — Recherche par similarité cosinus
    # ======================================================================
    def _calculer_normes(self) -> None:
        """Norme de chaque vecteur document, calculée une fois pour toutes."""
        # ||d|| = racine de la somme des carrés des cases de la ligne
        carres = self.mat_TFxIDF.multiply(self.mat_TFxIDF)
        self.normes_docs = np.sqrt(np.asarray(carres.sum(axis=1)).ravel())

    def vectoriser(self, mots_clefs: str) -> np.ndarray:
        """Transforme la requête en vecteur sur le vocabulaire (pondération TF-IDF)."""
        vecteur = np.zeros(len(self.vocab))
        # La requête est nettoyée EXACTEMENT comme les documents
        for mot in nettoyer_texte(mots_clefs).split():
            if mot in self.vocab:                 # un mot inconnu du corpus est ignoré
                vecteur[self.vocab[mot]["id"]] += 1      # TF du mot dans la requête
        return vecteur * self.idf                        # même pondération que les documents

    def search(self, mots_clefs: str, n: int = 5) -> pd.DataFrame:
        """Renvoie les n documents les plus proches de la requête (cosinus)."""
        if n < 1:
            raise ValueError("Le nombre de documents à retourner doit être au moins 1.")

        colonnes = ["id", "score", "type", "titre", "auteur", "date", "url"]
        q = self.vectoriser(mots_clefs)
        norme_q = np.linalg.norm(q)
        if norme_q == 0:
            # Aucun mot de la requête n'est dans le vocabulaire : rien à comparer
            # (et on évite une division par zéro)
            return pd.DataFrame(columns=colonnes)

        # Produit scalaire q . d avec TOUS les documents d'un coup :
        # matrice (documents x mots) @ vecteur (mots) = vecteur (documents)
        produits = self.mat_TFxIDF @ q
        # cos(q, d) = (q . d) / (||q|| x ||d||) ; un document de norme nulle
        # (cas extrême) reçoit le score 0 au lieu d'une division par zéro
        denominateurs = norme_q * self.normes_docs
        scores = np.divide(produits, denominateurs,
                           out=np.zeros_like(produits), where=denominateurs > 0)

        # Tri par score décroissant ; on écarte les documents sans aucun mot commun
        meilleurs = [i for i in np.argsort(-scores) if scores[i] > 0][:n]

        resultats = []
        for i in meilleurs:
            id_doc = self.ids_documents[i]          # ligne i -> identifiant du document
            doc = self.corpus.documents[id_doc]
            resultats.append({
                "id": id_doc,
                "score": round(float(scores[i]), 3),
                "type": doc.getType(),
                "titre": doc.titre,
                "auteur": doc.auteur,
                "date": doc.date.strftime("%d/%m/%Y"),
                "url": doc.url,
            })
        return pd.DataFrame(resultats, columns=colonnes)
