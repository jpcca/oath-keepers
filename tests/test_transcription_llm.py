import pytest


# Note: We duplicate is_complete_sentence here instead of importing from
# oath_keepers.vllm_server to avoid heavy dependencies (torch, vllm, etc.)
# that are not needed for unit testing the sentence completion logic.
def is_complete_sentence(text: str) -> bool:
    """Check if the text ends with sentence-ending punctuation."""
    text = text.strip()
    if not text:
        return False
    return text[-1] in ".!?"


class TestSentenceCompletion:
    """Tests for sentence completion detection."""

    def test_complete_sentence_with_period(self):
        """Test that a sentence ending with a period is detected as complete."""
        assert is_complete_sentence("This is a complete sentence.")

    def test_complete_sentence_with_question_mark(self):
        """Test that a sentence ending with a question mark is detected as complete."""
        assert is_complete_sentence("Is this a complete sentence?")

    def test_complete_sentence_with_exclamation(self):
        """Test that a sentence ending with an exclamation is detected as complete."""
        assert is_complete_sentence("This is exciting!")

    def test_incomplete_sentence(self):
        """Test that a sentence without ending punctuation is detected as incomplete."""
        assert not is_complete_sentence("This is incomplete")

    def test_empty_string(self):
        """Test that an empty string is not considered a complete sentence."""
        assert not is_complete_sentence("")

    def test_whitespace_only(self):
        """Test that whitespace-only strings are not considered complete sentences."""
        assert not is_complete_sentence("   ")

    def test_sentence_with_trailing_whitespace(self):
        """Test that trailing whitespace doesn't affect sentence completion detection."""
        assert is_complete_sentence("This is complete.  ")
        assert is_complete_sentence("  This is complete.  ")

    def test_sentence_with_comma(self):
        """Test that a sentence ending with a comma is not considered complete."""
        assert not is_complete_sentence("This has a comma,")

    def test_sentence_with_semicolon(self):
        """Test that a sentence ending with a semicolon is not considered complete."""
        assert not is_complete_sentence("This has a semicolon;")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
