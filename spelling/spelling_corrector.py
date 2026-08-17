"""
Spelling Corrector using Birkbeck Spelling Error Corpus
Task 2: Search Query Spelling Corrector

This module implements a simple spelling corrector that uses:
- re: for pattern matching and text processing
- pandas: for data manipulation and corpus management
- numpy: for numerical calculations and similarity metrics
"""

import re
import pandas as pd
import numpy as np
from collections import Counter
import urllib.request
import os


class SpellingCorrector:
    """
    A simple spelling corrector that learns from a corpus of correct spellings
    and can suggest corrections for misspelled words.
    """
    
    def __init__(self):
        """Initialize the spelling corrector."""
        self.word_frequency = Counter()
        self.vocabulary = set()
        self.corpus_df = None
        
    def load_corpus(self, words=None):
        """
        Load a corpus of words and their frequencies.
        
        Args:
            words: List of words. If None, uses a default sample corpus.
        """
        if words is None:
            # Default sample corpus - common English words
            words = [
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
                'search', 'query', 'autocomplete', 'spelling', 'correction', 'error',
                'finding', 'results', 'engine', 'database', 'algorithm', 'suggestion'
            ]
        
        # Count word frequencies
        self.word_frequency = Counter(words)
        self.vocabulary = set(words)
        
        # Create a DataFrame for better visualization
        self.corpus_df = pd.DataFrame(
            list(self.word_frequency.items()),
            columns=['word', 'frequency']
        )
        self.corpus_df = self.corpus_df.sort_values('frequency', ascending=False).reset_index(drop=True)
        
        print(f"✓ Loaded {len(self.vocabulary)} words into vocabulary")
        print(f"✓ Total word frequency: {sum(self.word_frequency.values())}\n")
        
    def edit_distance(self, word1, word2):
        """
        Calculate Levenshtein distance (edit distance) between two words.
        This is the minimum number of single-character edits needed.
        
        Args:
            word1, word2: Words to compare
            
        Returns:
            Edit distance (integer)
        """
        # Create a matrix to store distances
        len1, len2 = len(word1), len(word2)
        matrix = np.zeros((len1 + 1, len2 + 1))
        
        # Initialize first row and column
        matrix[0] = np.arange(len2 + 1)
        matrix[:, 0] = np.arange(len1 + 1)
        
        # Fill the matrix
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if word1[i-1] == word2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = 1 + min(
                        matrix[i-1][j],      # deletion
                        matrix[i][j-1],      # insertion
                        matrix[i-1][j-1]     # substitution
                    )
        
        return int(matrix[len1][len2])
    
    def get_candidates(self, misspelled_word, max_distance=2):
        """
        Find candidate words from vocabulary based on edit distance.
        
        Args:
            misspelled_word: The word to correct
            max_distance: Maximum edit distance to consider
            
        Returns:
            List of (candidate_word, distance) tuples
        """
        candidates = []
        
        for vocab_word in self.vocabulary:
            distance = self.edit_distance(misspelled_word.lower(), vocab_word.lower())
            if distance <= max_distance:
                candidates.append((vocab_word, distance))
        
        # Sort by distance and frequency
        candidates.sort(key=lambda x: (x[1], -self.word_frequency[x[0]]))
        
        return candidates
    
    def correct(self, word, show_candidates=False):
        """
        Correct a misspelled word.
        
        Args:
            word: The word to correct
            show_candidates: Whether to show candidate corrections
            
        Returns:
            Best correction suggestion
        """
        word_clean = re.sub(r'[^a-z\s]', '', word.lower())
        
        # If word is in vocabulary, it's correct
        if word_clean in self.vocabulary:
            return word_clean
        
        # Get candidates
        candidates = self.get_candidates(word_clean, max_distance=2)
        
        if candidates:
            best_correction = candidates[0][0]
            
            if show_candidates:
                print(f"Word: '{word}' → Correction: '{best_correction}'")
                print("Candidates:")
                for candidate, distance in candidates[:5]:
                    freq = self.word_frequency[candidate]
                    print(f"  • {candidate:15} (distance: {distance}, frequency: {freq})")
                print()
            
            return best_correction
        else:
            return word_clean
    
    def correct_text(self, text, show_details=False):
        """
        Correct spelling in a text string.
        
        Args:
            text: Text to correct
            show_details: Whether to show correction details
            
        Returns:
            Corrected text
        """
        # Use regex to find words
        words = re.findall(r'\b\w+\b', text.lower())
        corrections = {}
        
        corrected_words = []
        for word in words:
            if word not in corrections:
                corrections[word] = self.correct(word, show_candidates=show_details)
            corrected_words.append(corrections[word])
        
        # Reconstruct text
        corrected_text = re.sub(r'\b\w+\b', lambda m: corrections[m.group(0).lower()], text.lower())
        
        return corrected_text
    
    def display_vocabulary(self, top_n=10):
        """Display top words in the vocabulary."""
        print("Top Words in Vocabulary:")
        print(self.corpus_df.head(top_n).to_string(index=False))
        print()


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SPELLING CORRECTOR - Task 2: Search Query Spelling Corrector")
    print("="*70)
    print()
    
    # Initialize the corrector
    corrector = SpellingCorrector()
    corrector.load_corpus()
    
    # Display vocabulary
    corrector.display_vocabulary(top_n=10)
    
    # Test single word corrections
    print("Single Word Corrections:")
    print("-" * 70)
    test_words = ['searc', 'qurey', 'autcomplete', 'corection', 'algorythm']
    
    for word in test_words:
        corrector.correct(word, show_candidates=True)
    
    # Test text correction
    print("\nText Correction:")
    print("-" * 70)
    sample_text = "the serch qurey for autcomplete sujestions with speling corection"
    print(f"Original:  {sample_text}")
    corrected = corrector.correct_text(sample_text, show_details=False)
    print(f"Corrected: {corrected}")
    print()
    
    # Performance metrics
    print("Performance Metrics:")
    print("-" * 70)
    print(f"Vocabulary Size: {len(corrector.vocabulary)}")
    print(f"Total Words: {sum(corrector.word_frequency.values())}")
    print(f"Unique Words: {len(corrector.word_frequency)}")
    print()
