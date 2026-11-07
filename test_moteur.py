"""Tests unitaires du moteur de recherche (Partie 2) et des bonus.

Lancement, depuis le dossier du projet :  python -m unittest -v
"""

import math
import unittest
from datetime import date, datetime, timezone

from Corpus import Corpus
from Document import ArxivDocument, RedditDocument
from Orthographe import levenshtein, mot_le_plus_proche
from Racinisation import raciniser
from SearchEngine import SearchEngine


def corpus_du_cours() -> Corpus:
    """Les 3 documents de l'exemple du cours (diapos 66 à 70)."""
    corpus = Corpus("Test")
    corpus.vider()
    for texte in ["solar energy solar", "carbon energy", "carbon climate"]:
        corpus.add(RedditDocument(texte, "a", datetime(2026, 1, 1, tzinfo=timezone.utc),
                                  "u", texte))
    return corpus


class TestMatrices(unittest.TestCase):
    """Vocabulaire, matrices TF et TF-IDF (Partie 2, section 3)."""

    def setUp(self):
        self.moteur = SearchEngine(corpus_du_cours(), progression=False)

    def test_vocabulaire_trie(self):
        # L'identifiant d'un mot = sa position dans l'ordre alphabétique
        self.assertEqual(sorted(self.moteur.vocab, key=lambda m: self.moteur.vocab[m]["id"]),
                         ["carbon", "climate", "energy", "solar"])

    def test_matrice_tf(self):
        self.assertEqual(self.moteur.mat_TF.shape, (3, 4))
        # Document 1 « solar energy solar » : energy 1 fois, solar 2 fois
        self.assertEqual(self.moteur.mat_TF[0].toarray().tolist(), [[0, 0, 1, 2]])

    def test_occurrences_et_nombre_de_documents(self):
        self.assertEqual(self.moteur.vocab["solar"]["occurrences"], 2)
        self.assertEqual(self.moteur.vocab["energy"]["nb_documents"], 2)

    def test_idf(self):
        self.assertAlmostEqual(self.moteur.idf[self.moteur.vocab["energy"]["id"]], math.log(3 / 2))
        self.assertAlmostEqual(self.moteur.idf[self.moteur.vocab["solar"]["id"]], math.log(3))


class TestRecherche(unittest.TestCase):
    """Recherche par similarité cosinus (Partie 2, sections 4 et 5)."""

    def setUp(self):
        self.moteur = SearchEngine(corpus_du_cours(), progression=False)

    def test_scores_du_cours(self):
        resultats = self.moteur.search("solar energy", 5)
        self.assertEqual(resultats["id"].tolist(), [1, 2])     # document 3 écarté (score 0)
        self.assertEqual(resultats["score"].tolist(), [0.985, 0.245])

    def test_majuscules_et_ponctuation_sans_effet(self):
        self.assertTrue(self.moteur.search("SOLAR Energy !!", 5)
                        .equals(self.moteur.search("solar energy", 5)))

    def test_mot_inconnu(self):
        self.assertTrue(self.moteur.search("football", 5).empty)

    def test_nombre_de_resultats(self):
        self.assertEqual(len(self.moteur.search("energy carbon", 1)), 1)
        with self.assertRaises(ValueError):
            self.moteur.search("energy", 0)


class TestFiltres(unittest.TestCase):
    """Bonus 1 : filtres par source, auteur et période."""

    def setUp(self):
        corpus = Corpus("Test")
        corpus.vider()
        corpus.add(RedditDocument("r1", "fan", datetime(2026, 6, 14, tzinfo=timezone.utc), "u", "messi goal"))
        corpus.add(RedditDocument("r2", "fan2", datetime(2026, 7, 20, tzinfo=timezone.utc), "u", "messi wins"))
        corpus.add(ArxivDocument("a1", "Alice", datetime(2020, 1, 5, tzinfo=timezone.utc), "u", "messi model"))
        corpus.add(ArxivDocument("a2", "Bob", datetime(2021, 1, 5, tzinfo=timezone.utc), "u", "soccer data"))
        self.moteur = SearchEngine(corpus, progression=False)

    def test_source(self):
        self.assertEqual(set(self.moteur.search("messi", 10, source="reddit")["type"]), {"reddit"})
        self.assertEqual(self.moteur.search("messi", 10, source="arxiv")["id"].tolist(), [3])

    def test_auteur(self):
        self.assertEqual(self.moteur.search("messi", 10, auteur="fan2")["id"].tolist(), [2])

    def test_periode_bornes_incluses(self):
        resultats = self.moteur.search("messi", 10, date_debut=date(2026, 6, 14),
                                       date_fin=date(2026, 6, 14))
        self.assertEqual(resultats["id"].tolist(), [1])

    def test_valeurs_invalides(self):
        with self.assertRaises(ValueError):
            self.moteur.search("messi", 5, source="twitter")
        with self.assertRaises(ValueError):
            self.moteur.search("messi", 5, date_debut=date(2026, 8, 1), date_fin=date(2026, 1, 1))


class TestBM25(unittest.TestCase):
    """Bonus 6 : mesure Okapi BM25."""

    def setUp(self):
        self.moteur = SearchEngine(corpus_du_cours(), progression=False)

    def test_score_calcule_a_la_main(self):
        # Requête « solar » : seul le document 1 le contient (f = 2, |D| = 3)
        n, k1, b = 3, 1.2, 0.75
        longueur_moyenne = (3 + 2 + 2) / 3
        idf = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
        attendu = idf * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 3 / longueur_moyenne))
        resultats = self.moteur.search("solar", 5, mesure="bm25")
        self.assertEqual(resultats["id"].tolist(), [1])
        self.assertAlmostEqual(resultats["score"].iloc[0], attendu, places=3)

    def test_mot_repete_dans_la_requete(self):
        self.assertTrue(self.moteur.search("solar solar", 5, mesure="bm25")
                        .equals(self.moteur.search("solar", 5, mesure="bm25")))

    def test_mesure_inconnue(self):
        with self.assertRaises(ValueError):
            self.moteur.search("solar", 5, mesure="lsa")

    def test_comparer_mesures(self):
        comparaison = self.moteur.comparer_mesures("solar energy", 5)
        self.assertEqual(list(comparaison.columns),
                         ["rang", "id TF-IDF", "titre TF-IDF", "id BM25", "titre BM25"])
        self.assertEqual(comparaison["id TF-IDF"].iloc[0], comparaison["id BM25"].iloc[0])


class TestComparaisonSources(unittest.TestCase):
    """Bonus 4 : comparaison des sous-corpus Reddit et Arxiv."""

    def setUp(self):
        corpus = Corpus("Test")
        corpus.vider()
        jour = datetime(2026, 1, 1, tzinfo=timezone.utc)
        corpus.add(RedditDocument("p1", "a", jour, "u", "the referee is terrible lol goal"))
        corpus.add(RedditDocument("p2", "b", jour, "u", "the goal is what a match"))
        corpus.add(ArxivDocument("a1", "x", jour, "u", "the model predicts the goal of a match"))
        corpus.add(ArxivDocument("a2", "y", jour, "u", "we propose a model for soccer"))
        self.moteur = SearchEngine(corpus, progression=False)

    def test_mots_communs_et_specifiques(self):
        resultat = self.moteur.comparer_sources(10)
        self.assertEqual(set(resultat["communs"]["mot"]), {"goal", "match"})
        self.assertIn("referee", set(resultat["reddit"]["mot"]))
        self.assertIn("model", set(resultat["arxiv"]["mot"]))

    def test_mots_vides(self):
        avec = set(self.moteur.comparer_sources(10, exclure_mots_vides=False)["communs"]["mot"])
        self.assertIn("the", avec)
        sans = set(self.moteur.comparer_sources(10)["communs"]["mot"])
        self.assertNotIn("the", sans)


class TestRacinisation(unittest.TestCase):
    """Bonus racinisation."""

    def test_familles_de_mots(self):
        for mot in ["predicting", "prediction", "predictions", "predicted"]:
            self.assertEqual(raciniser(mot), "predict")
        self.assertEqual(raciniser("matches"), "match")
        self.assertEqual(raciniser("running"), "run")

    def test_mots_non_coupes(self):
        for mot in ["class", "analysis", "bus", "sing", "red", "2026", "messi"]:
            self.assertEqual(raciniser(mot), mot)

    def test_moteur_avec_racines(self):
        corpus = Corpus("Test")
        corpus.vider()
        jour = datetime(2026, 1, 1, tzinfo=timezone.utc)
        corpus.add(ArxivDocument("a", "x", jour, "u", "Match outcome prediction"))
        corpus.add(ArxivDocument("b", "y", jour, "u", "Player motion"))
        moteur = SearchEngine(corpus, progression=False, racinisation=True)
        # « predicting » (requête) et « prediction » (document) ont la même racine
        self.assertEqual(moteur.search("predicting", 5)["id"].tolist(), [1])
        self.assertTrue(SearchEngine(corpus, progression=False).search("predicting", 5).empty)


class TestOrthographe(unittest.TestCase):
    """Bonus « Vouliez-vous dire… ? » : distance de Levenshtein."""

    def test_distances_connues(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("mesi", "messi"), 1)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("goal", "goal"), 0)

    def test_mot_le_plus_proche(self):
        vocabulaire = {"messi": {"occurrences": 8}, "mess": {"occurrences": 1},
                       "argentina": {"occurrences": 5}}
        self.assertEqual(mot_le_plus_proche("mesi", vocabulaire), "messi")    # plus fréquent
        self.assertIsNone(mot_le_plus_proche("zzzzzz", vocabulaire))

    def test_corriger_requete(self):
        moteur = SearchEngine(corpus_du_cours(), progression=False)
        self.assertEqual(moteur.corriger_requete("solr enrgy"), "solar energy")
        self.assertIsNone(moteur.corriger_requete("solar energy"))     # rien à corriger


if __name__ == "__main__":
    unittest.main()
