# This file contains necessary preprocessing of the text

import re
from spellchecker import SpellChecker

class TextPreprocessor:
    def __init__(self, language="en"):
        # Save language setting, and initialize spell checker only for English
        self.language = language
        if language == "en":
            self.spell = SpellChecker()
        else:
            self.spell = None

    def is_greek(self, text):
        # Check if the text contains Greek characters
        return bool(re.search(r"[α-ωΑ-Ω]", text))

    def normalize_text(self, text):
        # Convert to lowercase and remove extra spaces
        text = text.lower().strip()

        # If English and not Greek, remove special characters
        if self.language == "en" and not self.is_greek(text):
            text = re.sub(r"[^a-z0-9\s]", "", text)

        # Replace multiple spaces with a single one
        text = re.sub(r"\s+", " ", text)

        return text

    def correct_typo(self, text):
        # Skip correction if spell checker is not initialized
        if self.spell is None:
            return text

        corrected_words = []
        for word in text.split():
            if word not in self.spell:
                corrected_words.append(self.spell.correction(word))
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def preprocess(self, text):
        # First normalize, then correct typos
        text = self.normalize_text(text)
        text = self.correct_typo(text)
        return text
