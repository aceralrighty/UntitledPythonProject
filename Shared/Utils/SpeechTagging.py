import spacy
from collections import defaultdict, Counter
import argparse
from typing import Dict, List, Set


class WordCategorizer:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the word categorizer with a SpaCy model.

        Args:
            model_name: SpaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Please install it with:")
            print(f"python -m spacy download {model_name}")
            raise

    def categorize_by_pos(self, text: str) -> Dict[str, List[str]]:
        """
        Categorize words by Part-of-Speech tags.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with POS categories as keys and word lists as values
        """
        doc = self.nlp(text)
        pos_categories = defaultdict(list)

        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_categories[token.pos_].append(token.text)

        return dict(pos_categories)

    def categorize_by_entity_type(self, text: str) -> Dict[str, List[str]]:
        """
        Categorize words by Named Entity Recognition (NER) types.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with entity types as keys and entity lists as values
        """
        doc = self.nlp(text)
        entity_categories = defaultdict(list)

        for ent in doc.ents:
            entity_categories[ent.label_].append(ent.text)

        return dict(entity_categories)

    def categorize_by_linguistic_features(self, text: str) -> Dict[str, List[str]]:
        """
        Categorize words by various linguistic features.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with feature categories and word lists
        """
        doc = self.nlp(text)
        feature_categories = defaultdict(list)

        for token in doc:
            if not token.is_punct and not token.is_space:
                # Basic categories
                if token.is_alpha:
                    feature_categories["alphabetic"].append(token.text)
                if token.is_digit:
                    feature_categories["numeric"].append(token.text)
                if token.is_stop:
                    feature_categories["stop_words"].append(token.text)
                if token.is_oov:
                    feature_categories["out_of_vocabulary"].append(token.text)

                # Morphological features
                if token.is_sent_start:
                    feature_categories["sentence_starters"].append(token.text)
                if token.like_email:
                    feature_categories["email_like"].append(token.text)
                if token.like_url:
                    feature_categories["url_like"].append(token.text)
                if token.like_num:
                    feature_categories["number_like"].append(token.text)

        return dict(feature_categories)

    def categorize_by_length(self, text: str) -> Dict[str, List[str]]:
        """
        Categorize words by their length.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with length categories and word lists
        """
        doc = self.nlp(text)
        length_categories = defaultdict(list)

        for token in doc:
            if not token.is_punct and not token.is_space:
                word_len = len(token.text)
                if word_len <= 3:
                    category = "short (1-3 chars)"
                elif word_len <= 6:
                    category = "medium (4-6 chars)"
                elif word_len <= 10:
                    category = "long (7-10 chars)"
                else:
                    category = "very_long (11+ chars)"

                length_categories[category].append(token.text)

        return dict(length_categories)

    def get_word_frequencies(self, text: str) -> Counter:
        """
        Get word frequency counts.

        Args:
            text: Input text to analyze

        Returns:
            Counter object with word frequencies
        """
        doc = self.nlp(text)
        words = [token.text.lower() for token in doc
                 if not token.is_punct and not token.is_space]
        return Counter(words)

    def analyze_text(self, text: str, include_frequencies: bool = True) -> Dict:
        """
        Perform comprehensive word categorization analysis.

        Args:
            text: Input text to analyze
            include_frequencies: Whether to include word frequency analysis

        Returns:
            Dictionary with all categorization results
        """
        results = {
            "original_text": text,
            "word_count": len([t for t in self.nlp(text) if not t.is_punct and not t.is_space]),
            "pos_categories": self.categorize_by_pos(text),
            "entity_categories": self.categorize_by_entity_type(text),
            "linguistic_features": self.categorize_by_linguistic_features(text),
            "length_categories": self.categorize_by_length(text)
        }

        if include_frequencies:
            results["word_frequencies"] = self.get_word_frequencies(text)

        return results

    def print_analysis(self, text: str, max_words_per_category: int = 10):
        """
        Print a formatted analysis of the text.

        Args:
            text: Input text to analyze
            max_words_per_category: Maximum words to display per category
        """
        analysis = self.analyze_text(text)

        print("=" * 60)
        print("WORD CATEGORIZATION ANALYSIS")
        print("=" * 60)
        print(f"Original text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Total words: {analysis['word_count']}")
        print()

        # Part-of-Speech categories
        print("📝 PART-OF-SPEECH CATEGORIES:")
        print("-" * 40)
        for pos, words in analysis['pos_categories'].items():
            word_list = words[:max_words_per_category]
            remaining = len(words) - len(word_list)
            print(f"  {pos}: {', '.join(word_list)}")
            if remaining > 0:
                print(f"    ... and {remaining} more")
        print()

        # Named entities
        if analysis['entity_categories']:
            print("🏷️  NAMED ENTITIES:")
            print("-" * 40)
            for ent_type, entities in analysis['entity_categories'].items():
                entity_list = entities[:max_words_per_category]
                remaining = len(entities) - len(entity_list)
                print(f"  {ent_type}: {', '.join(entity_list)}")
                if remaining > 0:
                    print(f"    ... and {remaining} more")
            print()

        # Linguistic features
        print("🔍 LINGUISTIC FEATURES:")
        print("-" * 40)
        for feature, words in analysis['linguistic_features'].items():
            if words:  # Only show non-empty categories
                word_list = words[:max_words_per_category]
                remaining = len(words) - len(word_list)
                print(f"  {feature}: {', '.join(word_list)}")
                if remaining > 0:
                    print(f"    ... and {remaining} more")
        print()

        # Length categories
        print("📏 WORD LENGTH CATEGORIES:")
        print("-" * 40)
        for length_cat, words in analysis['length_categories'].items():
            word_list = words[:max_words_per_category]
            remaining = len(words) - len(word_list)
            print(f"  {length_cat}: {', '.join(word_list)}")
            if remaining > 0:
                print(f"    ... and {remaining} more")
        print()

        # Top word frequencies
        print("📊 TOP WORD FREQUENCIES:")
        print("-" * 40)
        for word, count in analysis['word_frequencies'].most_common(10):
            print(f"  '{word}': {count}")
        print()
