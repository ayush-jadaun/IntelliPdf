"""
Enhanced Text Processing Utilities for IntelliPDF
Improved version with better NLP capabilities, language detection, 
advanced preprocessing, and multiple library support.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass
import unicodedata
import string

# Configure logging
logger = logging.getLogger(__name__)

# Optional dependencies with fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    
    # Download required NLTK data if not present
    def ensure_nltk_data():
        datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        for dataset in datasets:
            try:
                nltk.data.find(f'tokenizers/{dataset}')
            except LookupError:
                try:
                    nltk.download(dataset, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download NLTK dataset {dataset}: {e}")
    
    ensure_nltk_data()
    
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Some advanced features will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load a small English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

try:
    from langdetect import detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class TextStats:
    """Container for text statistics."""
    char_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_words_per_sentence: float
    reading_time_minutes: float
    complexity_score: float

@dataclass
class ProcessingConfig:
    """Configuration for text processing operations."""
    language: str = "english"
    remove_stopwords: bool = False
    to_lowercase: bool = True
    remove_punctuation: bool = False
    stem_words: bool = False
    lemmatize_words: bool = False
    min_word_length: int = 2
    max_word_length: int = 50

def detect_language(text: str, confidence_threshold: float = 0.8) -> Optional[str]:
    """Detect the language of text with confidence scoring."""
    if not LANGDETECT_AVAILABLE or not text.strip():
        return None
    
    try:
        languages = detect_langs(text)
        if languages and languages[0].prob >= confidence_threshold:
            return languages[0].lang
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
    
    return None

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize unicode characters with configurable normalization form."""
    try:
        return unicodedata.normalize(form, text)
    except Exception as e:
        logger.error(f"Unicode normalization failed: {e}")
        return text

def clean_text_advanced(
    text: str,
    remove_extra_whitespace: bool = True,
    normalize_quotes: bool = True,
    fix_encoding: bool = True,
    remove_control_chars: bool = True
) -> str:
    """Advanced text cleaning with multiple options."""
    if not text:
        return ""
    
    # Fix common encoding issues
    if fix_encoding:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Normalize quotes
    if normalize_quotes:
        text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
        text = text.replace('\u2013', '-').replace('\u2014', '--')  # En/em dashes
    
    # Remove control characters
    if remove_control_chars:
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
    
    # Normalize whitespace
    if remove_extra_whitespace:
        text = re.sub(r'\r\n|\r', '\n', text)  # Normalize line endings
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces
    
    return text.strip()

def tokenize_sentences_advanced(text: str, language: str = "english") -> List[str]:
    """Advanced sentence tokenization with multiple fallbacks."""
    if not text.strip():
        return []
    
    # Try spaCy first (most accurate)
    if SPACY_AVAILABLE and nlp:
        try:
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            logger.warning(f"spaCy sentence tokenization failed: {e}")
    
    # Try NLTK
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text, language=language)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}")
    
    # Fallback to regex-based splitting
    sentence_endings = r'[.!?]+(?:\s+|$)'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_words_advanced(text: str, language: str = "english") -> List[str]:
    """Advanced word tokenization preserving important tokens."""
    if not text.strip():
        return []
    
    # Try spaCy first
    if SPACY_AVAILABLE and nlp:
        try:
            doc = nlp(text)
            return [token.text for token in doc if not token.is_space]
        except Exception as e:
            logger.warning(f"spaCy word tokenization failed: {e}")
    
    # Try NLTK
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text, language=language)
        except Exception as e:
            logger.warning(f"NLTK word tokenization failed: {e}")
    
    # Fallback to regex
    words = re.findall(r'\b\w+\b', text)
    return words

def remove_stopwords_advanced(
    words: List[str], 
    language: str = "english",
    custom_stopwords: Optional[List[str]] = None
) -> List[str]:
    """Remove stopwords with custom additions."""
    if not words:
        return []
    
    stopword_set = set()
    
    # Get NLTK stopwords
    if NLTK_AVAILABLE:
        try:
            stopword_set.update(stopwords.words(language))
        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}")
    
    # Add common English stopwords as fallback
    if not stopword_set:
        common_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'through', 'during', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        stopword_set.update(common_stopwords)
    
    # Add custom stopwords
    if custom_stopwords:
        stopword_set.update(word.lower() for word in custom_stopwords)
    
    return [word for word in words if word.lower() not in stopword_set]

def stem_words(words: List[str]) -> List[str]:
    """Stem words using Porter Stemmer."""
    if not NLTK_AVAILABLE or not words:
        return words
    
    try:
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]
    except Exception as e:
        logger.warning(f"Stemming failed: {e}")
        return words

def lemmatize_words(words: List[str]) -> List[str]:
    """Lemmatize words using WordNet Lemmatizer."""
    if not NLTK_AVAILABLE or not words:
        return words
    
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in words]
    except Exception as e:
        logger.warning(f"Lemmatization failed: {e}")
        return words

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using spaCy or NLTK."""
    entities = {}
    
    if SPACY_AVAILABLE and nlp:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            return entities
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")
    
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            # Simple NER based on POS tags
            entities['PERSON'] = [word for word, pos in pos_tags if pos == 'NNP']
            return entities
        except Exception as e:
            logger.warning(f"NLTK NER failed: {e}")
    
    return entities

def calculate_text_stats(text: str) -> TextStats:
    """Calculate comprehensive text statistics."""
    if not text:
        return TextStats(0, 0, 0, 0, 0.0, 0.0, 0.0)
    
    char_count = len(text)
    
    # Count words
    words = tokenize_words_advanced(text)
    word_count = len(words)
    
    # Count sentences
    sentences = tokenize_sentences_advanced(text)
    sentence_count = len(sentences)
    
    # Count paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Calculate averages
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Estimate reading time (average 200 words per minute)
    reading_time_minutes = word_count / 200.0
    
    # Simple complexity score based on average sentence length
    complexity_score = min(avg_words_per_sentence / 15.0, 1.0) if avg_words_per_sentence > 0 else 0
    
    return TextStats(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_words_per_sentence=avg_words_per_sentence,
        reading_time_minutes=reading_time_minutes,
        complexity_score=complexity_score
    )

def extract_keywords_tfidf(
    texts: List[str],
    max_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[str, float]:
    """Extract keywords using TF-IDF vectorization."""
    if not SKLEARN_AVAILABLE or not texts:
        return {}
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = tfidf_matrix.mean(axis=0).A1
        
        return dict(zip(feature_names, mean_scores))
    
    except Exception as e:
        logger.error(f"TF-IDF keyword extraction failed: {e}")
        return {}

def preprocess_text_pipeline(
    text: str,
    config: ProcessingConfig = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Comprehensive text preprocessing pipeline.
    Returns processed text and metadata.
    """
    if config is None:
        config = ProcessingConfig()
    
    metadata = {
        'original_length': len(text),
        'language': None,
        'processing_steps': []
    }
    
    if not text:
        return "", metadata
    
    # Detect language
    detected_lang = detect_language(text)
    if detected_lang:
        metadata['language'] = detected_lang
        config.language = detected_lang
    
    # Clean text
    processed_text = clean_text_advanced(text)
    metadata['processing_steps'].append('cleaning')
    
    # Convert to lowercase
    if config.to_lowercase:
        processed_text = processed_text.lower()
        metadata['processing_steps'].append('lowercase')
    
    # Tokenize words
    words = tokenize_words_advanced(processed_text, config.language)
    
    # Filter by length
    words = [w for w in words if config.min_word_length <= len(w) <= config.max_word_length]
    
    # Remove punctuation
    if config.remove_punctuation:
        words = [w for w in words if w not in string.punctuation]
        metadata['processing_steps'].append('remove_punctuation')
    
    # Remove stopwords
    if config.remove_stopwords:
        words = remove_stopwords_advanced(words, config.language)
        metadata['processing_steps'].append('remove_stopwords')
    
    # Stem words
    if config.stem_words:
        words = stem_words(words)
        metadata['processing_steps'].append('stemming')
    
    # Lemmatize words
    if config.lemmatize_words:
        words = lemmatize_words(words)
        metadata['processing_steps'].append('lemmatization')
    
    # Reconstruct text
    processed_text = ' '.join(words)
    
    # Add final metadata
    metadata['final_length'] = len(processed_text)
    metadata['word_count'] = len(words)
    metadata['compression_ratio'] = metadata['final_length'] / metadata['original_length'] if metadata['original_length'] > 0 else 0
    
    return processed_text, metadata

def extract_text_features(text: str) -> Dict[str, Any]:
    """Extract comprehensive text features for analysis."""
    if not text:
        return {}
    
    features = {}
    
    # Basic statistics
    stats = calculate_text_stats(text)
    features['stats'] = stats.__dict__
    
    # Language detection
    features['language'] = detect_language(text)
    
    # Named entities
    features['entities'] = extract_named_entities(text)
    
    # Keywords (frequency-based)
    words = tokenize_words_advanced(text.lower())
    word_freq = Counter(words)
    features['top_words'] = dict(word_freq.most_common(20))
    
    # Sentence lengths
    sentences = tokenize_sentences_advanced(text)
    sentence_lengths = [len(tokenize_words_advanced(sent)) for sent in sentences]
    if sentence_lengths:
        features['sentence_length_stats'] = {
            'min': min(sentence_lengths),
            'max': max(sentence_lengths),
            'avg': sum(sentence_lengths) / len(sentence_lengths)
        }
    
    # Text complexity indicators
    features['complexity_indicators'] = {
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'unique_words_ratio': len(set(words)) / len(words) if words else 0,
        'long_sentences_ratio': sum(1 for length in sentence_lengths if length > 20) / len(sentence_lengths) if sentence_lengths else 0
    }
    
    return features

def summarize_text_simple(text: str, max_sentences: int = 3) -> str:
    """Simple extractive text summarization."""
    if not text:
        return ""
    
    sentences = tokenize_sentences_advanced(text)
    if len(sentences) <= max_sentences:
        return text
    
    # Score sentences by word frequency
    words = tokenize_words_advanced(text.lower())
    word_freq = Counter(words)
    
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = tokenize_words_advanced(sentence.lower())
        score = sum(word_freq[word] for word in sentence_words)
        sentence_scores[i] = score / len(sentence_words) if sentence_words else 0
    
    # Get top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sentences.sort(key=lambda x: x[0])  # Sort by original order
    
    return ' '.join(sentences[i] for i, _ in top_sentences)

def find_similar_sentences(
    query: str, 
    sentences: List[str], 
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Find sentences most similar to query using TF-IDF cosine similarity."""
    if not SKLEARN_AVAILABLE or not sentences:
        return []
    
    try:
        all_texts = [query] + sentences
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        query_vec = tfidf_matrix[0:1]
        sentence_vecs = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vec, sentence_vecs)[0]
        
        # Get top similar sentences
        sentence_scores = list(zip(sentences, similarities))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        return sentence_scores[:top_k]
    
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []

def clean_pdf_text(text: str) -> str:
    """Specialized cleaning for PDF-extracted text."""
    if not text:
        return ""
    
    # Remove common PDF artifacts
    text = re.sub(r'(?i)^\s*page\s+\d+\s*,?\s*$', '', text, flags=re.MULTILINE)  # Page numbers
    text = re.sub(r'(?i)^\s*\d+\s*,?\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers
    
    # Fix broken words (common in PDF extraction)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Hyphenated words across lines
    
    # Fix spacing issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
    text = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', text)  # Missing spaces after punctuation
    
    # Clean up excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple empty lines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces
    
    return text.strip()

def extract_document_structure(text: str) -> Dict[str, Any]:
    """Extract document structure (headings, sections, etc.)."""
    if not text:
        return {}
    
    lines = text.split('\n')
    structure = {
        'headings': [],
        'sections': [],
        'lists': [],
        'tables': []
    }
    
    # Detect headings (lines that are short, capitalized, or numbered)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check for heading patterns
        if (len(line) < 100 and 
            (line.isupper() or 
             line.istitle() or 
             re.match(r'^\d+\.?\s+[A-Z]', line) or
             re.match(r'^[A-Z][a-z]*\s+[A-Z]', line))):
            structure['headings'].append({
                'text': line,
                'line_number': i,
                'type': 'heading'
            })
    
    # Detect lists
    list_patterns = [
        r'^\s*[\-\*\+]\s+',  # Bullet points
        r'^\s*\d+[\.\)]\s+',  # Numbered lists
        r'^\s*[a-zA-Z][\.\)]\s+'  # Lettered lists
    ]
    
    for i, line in enumerate(lines):
        for pattern in list_patterns:
            if re.match(pattern, line):
                structure['lists'].append({
                    'text': line.strip(),
                    'line_number': i,
                    'type': 'list_item'
                })
                break
    
    # Detect potential tables (lines with multiple tabs or pipes)
    for i, line in enumerate(lines):
        if line.count('\t') >= 2 or line.count('|') >= 2:
            structure['tables'].append({
                'text': line.strip(),
                'line_number': i,
                'type': 'table_row'
            })
    
    return structure

def validate_text_quality(text: str) -> Dict[str, Any]:
    """Validate text quality and identify potential issues."""
    if not text:
        return {'quality_score': 0, 'issues': ['empty_text']}
    
    issues = []
    quality_metrics = {}
    
    # Check for minimum length
    if len(text) < 100:
        issues.append('text_too_short')
    
    # Check for excessive repetition
    words = tokenize_words_advanced(text.lower())
    if words:
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(1)[0][1]
        repetition_ratio = most_common_freq / len(words)
        
        if repetition_ratio > 0.1:
            issues.append('excessive_repetition')
        quality_metrics['repetition_ratio'] = repetition_ratio
    
    # Check for gibberish (high ratio of non-dictionary words)
    if NLTK_AVAILABLE:
        try:
            from nltk.corpus import words as nltk_words
            english_words = set(nltk_words.words())
            
            text_words = [w.lower() for w in words if w.isalpha()]
            if text_words:
                valid_words = sum(1 for w in text_words if w in english_words)
                gibberish_ratio = 1 - (valid_words / len(text_words))
                
                if gibberish_ratio > 0.3:
                    issues.append('high_gibberish_ratio')
                quality_metrics['gibberish_ratio'] = gibberish_ratio
        except Exception:
            pass
    
    # Check for encoding issues
    try:
        text.encode('utf-8').decode('utf-8')
    except UnicodeError:
        issues.append('encoding_issues')
    
    # Check for excessive punctuation
    punct_count = sum(1 for char in text if char in string.punctuation)
    punct_ratio = punct_count / len(text)
    if punct_ratio > 0.15:
        issues.append('excessive_punctuation')
    quality_metrics['punctuation_ratio'] = punct_ratio
    
    # Calculate overall quality score
    quality_score = 1.0
    quality_score -= len(issues) * 0.2  # Deduct for each issue
    quality_score = max(0, min(1, quality_score))
    
    return {
        'quality_score': quality_score,
        'issues': issues,
        'metrics': quality_metrics
    }

# Utility functions for batch processing
def process_text_batch(
    texts: List[str],
    config: ProcessingConfig = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """Process multiple texts with the same configuration."""
    results = []
    for text in texts:
        processed_text, metadata = preprocess_text_pipeline(text, config)
        results.append((processed_text, metadata))
    return results

def merge_text_features(feature_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge features from multiple text documents."""
    if not feature_dicts:
        return {}
    
    merged = {
        'document_count': len(feature_dicts),
        'combined_stats': {},
        'combined_entities': {},
        'combined_keywords': Counter()
    }
    
    # Merge statistics
    total_chars = sum(f.get('stats', {}).get('char_count', 0) for f in feature_dicts)
    total_words = sum(f.get('stats', {}).get('word_count', 0) for f in feature_dicts)
    total_sentences = sum(f.get('stats', {}).get('sentence_count', 0) for f in feature_dicts)
    
    merged['combined_stats'] = {
        'total_char_count': total_chars,
        'total_word_count': total_words,
        'total_sentence_count': total_sentences,
        'avg_chars_per_doc': total_chars / len(feature_dicts),
        'avg_words_per_doc': total_words / len(feature_dicts)
    }
    
    # Merge entities
    for features in feature_dicts:
        entities = features.get('entities', {})
        for entity_type, entity_list in entities.items():
            if entity_type not in merged['combined_entities']:
                merged['combined_entities'][entity_type] = []
            merged['combined_entities'][entity_type].extend(entity_list)
    
    # Merge keywords
    for features in feature_dicts:
        top_words = features.get('top_words', {})
        merged['combined_keywords'].update(top_words)
    return merged