"""Tests unitaires des classes de la Partie 1 (Document, Author, Corpus...).

Lancement, depuis le dossier du projet :  python -m unittest -v
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from Author import Author
from Corpus import Corpus, nettoyer_texte
from Document import ArxivDocument, Document, DocumentFactory, RedditDocument

DATE = datetime(2026, 7, 1, tzinfo=timezone.utc)


class TestDocument(unittest.TestCase):
    """Classe mère Document et classes filles (Parties 4, 7 et 8)."""

    def test_str_renvoie_le_titre(self):
        self.assertEqual(str(Document("Un titre", "auteur", DATE, "url", "texte")), "Un titre")

    def test_get_type(self):
        self.assertEqual(RedditDocument("t").getType(), "reddit")
        self.assertEqual(ArxivDocument("t").getType(), "arxiv")

    def test_heritage(self):
        # Une classe fille EST un Document
        self.assertIsInstance(RedditDocument("t"), Document)
        self.assertIsInstance(ArxivDocument("t"), Document)

    def test_nb_commentaires_negatif_refuse(self):
        with self.assertRaises(ValueError):
            RedditDocument("t", nb_commentaires=-1)
        doc = RedditDocument("t", nb_commentaires=3)
        with self.assertRaises(ValueError):
            doc.nb_commentaires = -5
        self.assertEqual(doc.nb_commentaires, 3)        # valeur inchangée

    def test_coauteurs_pas_partages(self):
        # Piège de la liste par défaut : chaque article doit avoir SA liste
        article1, article2 = ArxivDocument("a"), ArxivDocument("b")
        article1.ajouter_coauteur("Bob")
        self.assertEqual(article1.coauteurs, ["Bob"])
        self.assertEqual(article2.coauteurs, [])

    def test_ajouter_coauteur_sans_doublon(self):
        article = ArxivDocument("a", coauteurs=["Bob"])
        article.ajouter_coauteur("Bob")
        self.assertEqual(article.coauteurs, ["Bob"])

    def test_coauteurs_doivent_etre_une_liste(self):
        with self.assertRaises(ValueError):
            ArxivDocument("a", coauteurs="Bob, Carol")


class TestFabrique(unittest.TestCase):
    """Patron Fabrique (Partie 9.2)."""

    def test_cree_la_bonne_classe(self):
        self.assertIsInstance(DocumentFactory.creer("reddit", "t", "a", DATE, "u", "x",
                                                    nb_commentaires=2), RedditDocument)
        self.assertIsInstance(DocumentFactory.creer("arxiv", "t", "a", DATE, "u", "x",
                                                    coauteurs=["B"]), ArxivDocument)

    def test_type_inconnu_refuse(self):
        with self.assertRaises(ValueError):
            DocumentFactory.creer("twitter", "t", "a", DATE, "u", "x")


class TestAuthor(unittest.TestCase):
    """Classe Author (Partie 5)."""

    def test_add_met_a_jour_nb_docs_sans_doublon(self):
        auteur = Author("Alice")
        doc = Document("t", "Alice", DATE, "u", "un deux trois")
        auteur.add(1, doc)
        auteur.add(1, doc)                    # même document : pas compté deux fois
        self.assertEqual(auteur.nb_docs, 1)

    def test_taille_moyenne(self):
        auteur = Author("Alice")
        self.assertEqual(auteur.taille_moyenne(), 0.0)          # pas de division par zéro
        auteur.add(1, Document(texte="un deux"))
        auteur.add(2, Document(texte="un deux trois quatre"))
        self.assertEqual(auteur.taille_moyenne(), 3.0)           # (2 + 4) / 2


class TestCorpus(unittest.TestCase):
    """Classe Corpus (Parties 6 et 9.1) et recherche par regex (Partie 2)."""

    def setUp(self):
        # setUp est appelée avant CHAQUE test : on repart d'un corpus vide
        self.corpus = Corpus("Test")
        self.corpus.vider()
        self.corpus.add(RedditDocument("Messi goal", "fan1", DATE, "u1",
                                       "Messi scores a goal. What a game", nb_commentaires=4))
        self.corpus.add(ArxivDocument("A model", "Alice", DATE, "u2",
                                      "We model goals in soccer.", coauteurs=["Bob"]))
        self.corpus.add(RedditDocument("Other", "fan1", DATE, "u3", "Penalty drama tonight"))

    def test_singleton(self):
        self.assertIs(Corpus("Un autre nom"), self.corpus)

    def test_identifiants_sequentiels_et_auteurs(self):
        self.assertEqual(list(self.corpus.documents), [1, 2, 3])
        self.assertEqual(self.corpus.authors["fan1"].nb_docs, 2)
        self.assertEqual(self.corpus.authors["Alice"].nb_docs, 1)

    def test_save_puis_load(self):
        avant = [(type(d), str(d), d.date) for d in self.corpus.documents.values()]
        with tempfile.TemporaryDirectory() as dossier:      # dossier effacé à la fin
            chemin = Path(dossier) / "corpus.csv"
            self.corpus.save(chemin)
            self.corpus.load(chemin)
        apres = [(type(d), str(d), d.date) for d in self.corpus.documents.values()]
        self.assertEqual(avant, apres)

    def test_search_renvoie_les_phrases(self):
        self.assertEqual(self.corpus.search("messi"), ["Messi scores a goal"])

    def test_search_frontiere_de_mot(self):
        # \b : « goal » ne trouve pas « goals »
        self.assertEqual(len(self.corpus.search("goal")), 1)
        self.assertEqual(len(self.corpus.search("goals?")), 2)

    def test_cache_mis_a_jour_apres_add(self):
        self.corpus.search("messi")                      # construit le cache
        self.corpus.add(RedditDocument("New", "x", DATE, "u", "Messi again"))
        self.assertEqual(len(self.corpus.search("messi")), 2)

    def test_concorde(self):
        concordance = self.corpus.concorde("penalty", 5)
        self.assertEqual(list(concordance.columns),
                         ["contexte gauche", "motif trouvé", "contexte droit"])
        self.assertEqual(concordance.iloc[0]["motif trouvé"], "Penalty")

    def test_regex_invalide(self):
        with self.assertRaises(ValueError):
            self.corpus.search("(goal")


class TestNettoyage(unittest.TestCase):
    """Fonction nettoyer_texte (Partie 2, section 2.1)."""

    def test_minuscules_et_ponctuation(self):
        self.assertEqual(nettoyer_texte("Messi's GOAL!!\nWow"), "messi s goal wow")

    def test_caracteres_invisibles(self):
        # L'espace de largeur nulle est supprimé SANS couper le mot
        self.assertEqual(nettoyer_texte("f​or"), "for")

    def test_chiffres_conserves(self):
        self.assertEqual(nettoyer_texte("World Cup 2026"), "world cup 2026")


if __name__ == "__main__":
    unittest.main()
