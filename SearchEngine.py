"""Module SearchEngine : moteur de recherche fondé sur une matrice
Documents x Termes (projet Partie 2, sections 3 à 5).

Tout est construit « à la main » : pas de scikit-learn ni de nltk.
scipy est utilisé uniquement pour stocker la matrice creuse (csr_matrix).
Projet Partie 3 (1.3) : barres de progression tqdm pendant la construction.
Bonus 1 : filtres de recherche (source, auteur, période).
"""

from datetime import date

import numpy as np
import pandas as pd
from scipy import sparse
# tqdm.auto choisit tout seul l'affichage : barre graphique dans un notebook,
# barre de texte dans un terminal
from tqdm.auto import tqdm

from Corpus import nettoyer_texte


class SearchEngine:
    """Moteur de recherche : vocabulaire, matrices TF et TF-IDF d'un corpus."""

    def __init__(self, corpus, progression: bool = True) -> None:
        """Reçoit un Corpus et construit aussitôt les matrices (section 5).

        progression=False masque les barres tqdm (utile pour les tests).
        """
        self.corpus = corpus
        self.progression = progression
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
        # tqdm(...) entoure l'itérable : la boucle est identique, mais une
        # barre affiche l'avancement (nombre de documents traités, vitesse)
        mots_par_doc = [nettoyer_texte(self.corpus.documents[i].texte).split()
                        for i in tqdm(self.ids_documents, desc="Nettoyage des textes",
                                      disable=not self.progression)]
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

        for i, mots in enumerate(tqdm(mots_par_doc, desc="Matrice TF",
                                      disable=not self.progression)):
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

    # ======================================================================
    # Bonus 1 — Filtres de recherche
    # ======================================================================
    SOURCES = ("reddit", "arxiv")      # valeurs acceptées pour le filtre source

    def _respecte_filtres(self, doc, source: str | None, auteur: str | None,
                          date_debut: date | None, date_fin: date | None) -> bool:
        """True si le document passe tous les filtres (None = filtre désactivé)."""
        if source is not None and doc.getType() != source:     # polymorphisme
            return False
        if auteur is not None and doc.auteur != auteur:
            return False
        # doc.date est un datetime (date + heure) : .date() ne garde que le jour
        if date_debut is not None and doc.date.date() < date_debut:
            return False
        if date_fin is not None and doc.date.date() > date_fin:
            return False
        return True

    def search(self, mots_clefs: str, n: int = 5, source: str | None = None,
               auteur: str | None = None, date_debut: date | None = None,
               date_fin: date | None = None) -> pd.DataFrame:
        """Renvoie les n documents les plus proches de la requête (cosinus).

        Bonus 1 : les filtres facultatifs ne gardent que les documents d'une
        source ("reddit" ou "arxiv"), d'un auteur, ou publiés entre date_debut
        et date_fin (bornes incluses). Sans filtre, le comportement est
        identique à celui de la Partie 2.
        """
        if n < 1:
            raise ValueError("Le nombre de documents à retourner doit être au moins 1.")
        if source is not None and source not in self.SOURCES:
            raise ValueError(f"Source inconnue : {source} (attendu : {self.SOURCES})")
        if date_debut is not None and date_fin is not None and date_debut > date_fin:
            raise ValueError("La date de début doit précéder la date de fin.")

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

        # Bonus 1 : un document qui ne passe pas les filtres reçoit le score 0,
        # il est donc écarté du classement comme un document sans mot commun
        for i, id_doc in enumerate(self.ids_documents):
            if not self._respecte_filtres(self.corpus.documents[id_doc], source,
                                          auteur, date_debut, date_fin):
                scores[i] = 0.0

        # Tri par score décroissant ; on écarte les documents de score nul
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
