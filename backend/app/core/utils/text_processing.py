"""
Enhanced Text Processing Utilities for IntelliPDF
Improved version with better NLP capabilities, language detection, 
advanced preprocessing, and multiple library support.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import Counter
from dataclasses import dataclass
import unicodedata
import string
import os
from pathlib import Path

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
        """Download required NLTK datasets with proper error handling."""
        datasets = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('corpora/omw-1.4', 'omw-1.4')
        ]
        
        for path, dataset in datasets:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK dataset: {dataset}")
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
    from langdetect import detect_langs, LangDetectException
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
    lexical_diversity: float

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
    preserve_original_text: bool = True
    chunk_size: int = 10000  # For processing large texts in chunks

class TextProcessingError(Exception):
    """Custom exception for text processing errors."""
    pass

def detect_language(text: str, confidence_threshold: float = 0.8) -> Optional[str]:
    """Detect the language of text with confidence scoring."""
    if not LANGDETECT_AVAILABLE or not text.strip():
        return None
    
    try:
        # Use only first 1000 characters for efficiency
        sample_text = text[:1000] if len(text) > 1000 else text
        languages = detect_langs(sample_text)
        if languages and languages[0].prob >= confidence_threshold:
            return languages[0].lang
    except (LangDetectException, Exception) as e:
        logger.warning(f"Language detection failed: {e}")
    
    return None

def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Normalize unicode characters with configurable normalization form."""
    if not text:
        return ""
    
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
    remove_control_chars: bool = True,
    preserve_structure: bool = True
) -> str:
    """Advanced text cleaning with multiple options."""
    if not text:
        return ""
    
    # Fix common encoding issues
    if fix_encoding:
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            logger.warning(f"Encoding fix failed: {e}")
    
    # Normalize unicode
    text = normalize_unicode(text)
    
    # Normalize quotes and special characters
    if normalize_quotes:
        quote_mapping = {
            '\u2018': "'", '\u2019': "'",  # Smart single quotes
            '\u201c': '"', '\u201d': '"',  # Smart double quotes
            '\u2013': '-', '\u2014': '--',  # En/em dashes
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',  # Non-breaking space
        }
        for old, new in quote_mapping.items():
            text = text.replace(old, new)
    
    # Remove control characters but preserve important ones
    if remove_control_chars:
        if preserve_structure:
            # Keep newlines, tabs, and carriage returns
            text = ''.join(char for char in text 
                          if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
        else:
            text = ''.join(char for char in text 
                          if unicodedata.category(char)[0] != 'C')
    
    # Normalize whitespace
    if remove_extra_whitespace:
        # Normalize line endings
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Clean paragraph breaks (preserve document structure)
        if preserve_structure:
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple empty lines to double
        else:
            text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Multiple spaces to single
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove trailing spaces from lines
        text = re.sub(r' +\n', '\n', text)
        # Remove leading spaces from lines (except intentional indentation)
        if not preserve_structure:
            text = re.sub(r'\n +', '\n', text)
    
    return text.strip()

def tokenize_sentences_advanced(text: str, language: str = "english") -> List[str]:
    """Advanced sentence tokenization with multiple fallbacks."""
    if not text.strip():
        return []
    
    # Try spaCy first (most accurate)
    if SPACY_AVAILABLE and nlp:
        try:
            # Process in chunks for memory efficiency
            chunk_size = 100000
            if len(text) > chunk_size:
                sentences = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    doc = nlp(chunk)
                    sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
                return sentences
            else:
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
    
    # Fallback to improved regex-based splitting
    # Handle common abbreviations
    abbreviations = r'(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|inc|ltd|co|corp|govt|dept|univ|assn|bros|ph\.d|m\.d|b\.a|m\.a|ph\.d|ll\.d|d\.d\.s|m\.b\.a|c\.p\.a|r\.n|m\.d|d\.o|d\.c|o\.d|d\.d\.s|d\.m\.d|d\.v\.m|pharm\.d|m\.p\.h|m\.s\.w|m\.f\.t|o\.t\.r|p\.t|r\.t|m\.t|c\.m\.t|r\.d|c\.c\.c|s\.l\.p|au\.d|ed\.d|psy\.d|d\.b\.a|d\.n\.p|d\.p\.t|pharm\.d|d\.sc|d\.phil|ll\.m|s\.j\.d|m\.c\.j|m\.p\.a|m\.u\.p|m\.arch|m\.l\.s|m\.f\.a|m\.mus|m\.ed|m\.a\.t|m\.s\.n|m\.s\.ed|d\.min|th\.d|d\.d|d\.litt|litt\.d|d\.hum|d\.mus|mus\.d|d\.f\.a|m\.f\.a|b\.f\.a|b\.mus|b\.arch|b\.s\.arch|b\.l\.a|b\.s\.n|b\.s\.ed|b\.s\.w|b\.a\.ed|a\.b|a\.m|s\.b|s\.m|sc\.b|sc\.m|sc\.d|ph\.b|ph\.m|th\.b|th\.m|b\.th|m\.th|b\.d|m\.div|d\.div|j\.d|ll\.b|ll\.m|d\.c\.l|b\.c\.l|m\.c\.l|b\.phil|m\.phil|d\.phil|litt\.b|litt\.m|mus\.b|mus\.m|b\.litt|m\.litt)'
    
    # Protect abbreviations
    text_protected = re.sub(f'({abbreviations})\\.', r'\1<ABBREV>', text, flags=re.IGNORECASE)
    
    # Split on sentence endings
    sentence_endings = r'[.!?]+(?:\s+|$)'
    sentences = re.split(sentence_endings, text_protected)
    
    # Restore abbreviations
    sentences = [s.replace('<ABBREV>', '.').strip() for s in sentences if s.strip()]
    
    return sentences

def tokenize_words_advanced(text: str, language: str = "english", preserve_case: bool = False) -> List[str]:
    """Advanced word tokenization preserving important tokens."""
    if not text.strip():
        return []
    
    # Try spaCy first (handles contractions, punctuation better)
    if SPACY_AVAILABLE and nlp:
        try:
            # Process in chunks for memory efficiency
            chunk_size = 100000
            if len(text) > chunk_size:
                words = []
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    doc = nlp(chunk)
                    chunk_words = [token.text for token in doc if not token.is_space]
                    words.extend(chunk_words)
                return words
            else:
                doc = nlp(text)
                return [token.text for token in doc if not token.is_space]
        except Exception as e:
            logger.warning(f"spaCy word tokenization failed: {e}")
    
    # Try NLTK
    if NLTK_AVAILABLE:
        try:
            return word_tokenize(text, language=language, preserve_line=True)
        except Exception as e:
            logger.warning(f"NLTK word tokenization failed: {e}")
    
    # Fallback to improved regex
    # Handle contractions, hyphenated words, and preserve important punctuation
    word_pattern = r"\b\w+(?:[-']\w+)*\b|\$\d+(?:\.\d+)?|\d+(?:\.\d+)?%?|[^\w\s]"
    words = re.findall(word_pattern, text)
    return words

def remove_stopwords_advanced(
    words: List[str], 
    language: str = "english",
    custom_stopwords: Optional[List[str]] = None,
    preserve_important: bool = True
) -> List[str]:
    """Remove stopwords with custom additions and important word preservation."""
    if not words:
        return []
    
    stopword_set = set()
    
    # Get NLTK stopwords
    if NLTK_AVAILABLE:
        try:
            stopword_set.update(stopwords.words(language))
        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}")
    
    # Add comprehensive English stopwords as fallback
    if not stopword_set or language == "english":
        common_stopwords = {
            # Pronouns
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves',
            # Question words
            'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
            # Demonstratives
            'this', 'that', 'these', 'those',
            # Articles
            'a', 'an', 'the',
            # Auxiliary verbs
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could',
            # Conjunctions
            'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'if', 'because', 'since',
            'although', 'though', 'unless', 'until', 'while', 'whereas', 'whether',
            # Prepositions
            'of', 'at', 'by', 'for', 'with', 'through', 'during', 'to', 'from', 'up',
            'down', 'in', 'out', 'on', 'off', 'over', 'under', 'above', 'below',
            'across', 'between', 'among', 'into', 'onto', 'upon', 'within', 'without',
            'before', 'after', 'around', 'near', 'far', 'against', 'toward', 'towards',
            # Adverbs
            'not', 'no', 'yes', 'very', 'too', 'quite', 'rather', 'really', 'just',
            'only', 'even', 'also', 'still', 'already', 'always', 'never', 'often',
            'sometimes', 'usually', 'again', 'once', 'twice', 'here', 'there', 'where',
            'everywhere', 'anywhere', 'somewhere', 'nowhere', 'then', 'now', 'today',
            'tomorrow', 'yesterday', 'soon', 'later', 'earlier', 'finally', 'first',
            'last', 'next', 'previous', 'more', 'most', 'less', 'least', 'much', 'many',
            'few', 'little', 'some', 'any', 'all', 'both', 'each', 'every', 'either',
            'neither', 'another', 'other', 'such', 'same', 'different', 'new', 'old',
            'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low', 'large',
            'great', 'little', 'own', 'right', 'left', 'sure', 'certain', 'possible',
            'impossible', 'true', 'false', 'real', 'actual', 'general', 'particular',
            'special', 'main', 'important', 'necessary', 'available', 'present', 'current'
        }
        stopword_set.update(common_stopwords)
    
    # Add custom stopwords
    if custom_stopwords:
        stopword_set.update(word.lower() for word in custom_stopwords)
    
    # Words to preserve even if they appear in stopwords (context-dependent)
    preserve_words = set()
    if preserve_important:
        preserve_words = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'}
    
    filtered_words = []
    for word in words:
        word_lower = word.lower()
        if word_lower not in stopword_set or word_lower in preserve_words:
            filtered_words.append(word)
    
    return filtered_words

def stem_words(words: List[str]) -> List[str]:
    """Stem words using Porter Stemmer with error handling."""
    if not NLTK_AVAILABLE or not words:
        return words
    
    try:
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]
    except Exception as e:
        logger.warning(f"Stemming failed: {e}")
        return words

def lemmatize_words(words: List[str]) -> List[str]:
    """Lemmatize words using WordNet Lemmatizer with POS tagging."""
    if not NLTK_AVAILABLE or not words:
        return words
    
    try:
        lemmatizer = WordNetLemmatizer()
        
        # Get POS tags for better lemmatization
        try:
            pos_tags = pos_tag(words)
            
            # Map POS tags to WordNet tags
            def get_wordnet_pos(treebank_tag):
                if treebank_tag.startswith('J'):
                    return 'a'  # adjective
                elif treebank_tag.startswith('V'):
                    return 'v'  # verb
                elif treebank_tag.startswith('N'):
                    return 'n'  # noun
                elif treebank_tag.startswith('R'):
                    return 'r'  # adverb
                else:
                    return 'n'  # default to noun
            
            return [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
                   for word, pos in pos_tags]
        
        except Exception:
            # Fallback to simple lemmatization without POS
            return [lemmatizer.lemmatize(word) for word in words]
    
    except Exception as e:
        logger.warning(f"Lemmatization failed: {e}")
        return words

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using spaCy or NLTK with deduplication."""
    entities = {}
    
    if SPACY_AVAILABLE and nlp:
        try:
            # Process in chunks for large texts
            chunk_size = 100000
            all_entities = {}
            
            if len(text) > chunk_size:
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    doc = nlp(chunk)
                    for ent in doc.ents:
                        if ent.label_ not in all_entities:
                            all_entities[ent.label_] = set()
                        all_entities[ent.label_].add(ent.text.strip())
            else:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ not in all_entities:
                        all_entities[ent.label_] = set()
                    all_entities[ent.label_].add(ent.text.strip())
            
            # Convert sets to lists and filter out empty/short entities
            entities = {
                label: [entity for entity in sorted(entity_set) 
                       if len(entity) > 1 and entity.isalpha()]
                for label, entity_set in all_entities.items()
                if entity_set
            }
            
            return entities
            
        except Exception as e:
            logger.warning(f"spaCy NER failed: {e}")
    
    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Extract proper nouns as potential named entities
            proper_nouns = [word for word, pos in pos_tags if pos in ['NNP', 'NNPS']]
            
            if proper_nouns:
                entities['PERSON'] = list(set(proper_nouns))
            
            return entities
        except Exception as e:
            logger.warning(f"NLTK NER failed: {e}")
    
    return entities

def calculate_text_stats(text: str) -> TextStats:
    """Calculate comprehensive text statistics with lexical diversity."""
    if not text:
        return TextStats(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    
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
    
    # Calculate lexical diversity (type-token ratio)
    unique_words = len(set(word.lower() for word in words if word.isalpha()))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Enhanced complexity score considering multiple factors
    if word_count > 0 and sentence_count > 0:
        avg_word_length = sum(len(word) for word in words) / word_count
        complexity_factors = [
            min(avg_words_per_sentence / 20.0, 1.0),  # Sentence length
            min(avg_word_length / 6.0, 1.0),  # Word length
            min(lexical_diversity * 2, 1.0),  # Vocabulary richness
        ]
        complexity_score = sum(complexity_factors) / len(complexity_factors)
    else:
        complexity_score = 0.0
    
    return TextStats(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_words_per_sentence=avg_words_per_sentence,
        reading_time_minutes=reading_time_minutes,
        complexity_score=complexity_score,
        lexical_diversity=lexical_diversity
    )

def extract_keywords_tfidf(
    texts: List[str],
    max_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.8
) -> Dict[str, float]:
    """Extract keywords using TF-IDF vectorization with improved parameters."""
    if not SKLEARN_AVAILABLE or not texts:
        return {}
    
    try:
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return {}
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Better token pattern
        )
        
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores across all documents
        mean_scores = tfidf_matrix.mean(axis=0).A1
        
        # Sort by score and return top keywords
        keyword_scores = dict(zip(feature_names, mean_scores))
        sorted_keywords = dict(sorted(keyword_scores.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return sorted_keywords
    
    except Exception as e:
        logger.error(f"TF-IDF keyword extraction failed: {e}")
        return {}

def preprocess_text_pipeline(
    text: str,
    config: Optional[ProcessingConfig] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Comprehensive text preprocessing pipeline with memory efficiency.
    Returns processed text and metadata.
    """
    if config is None:
        config = ProcessingConfig()
    
    metadata = {
        'original_length': len(text),
        'language': None,
        'processing_steps': [],
        'processing_time': 0,
        'chunks_processed': 1
    }
    
    if not text:
        return "", metadata
    
    import time
    start_time = time.time()
    
    try:
        # Detect language
        detected_lang = detect_language(text)
        if detected_lang:
            metadata['language'] = detected_lang
            if detected_lang in ['en', 'english']:
                config.language = 'english'
        
        # Clean text
        processed_text = clean_text_advanced(text, preserve_structure=config.preserve_original_text)
        metadata['processing_steps'].append('cleaning')
        
        # Process in chunks if text is very large
        if len(processed_text) > config.chunk_size:
            # Split into chunks for processing
            chunks = []
            chunk_words = []
            
            sentences = tokenize_sentences_advanced(processed_text, config.language)
            metadata['chunks_processed'] = len(sentences)
            
            for sentence in sentences:
                # Convert to lowercase if needed
                if config.to_lowercase:
                    sentence = sentence.lower()
                
                # Tokenize words
                words = tokenize_words_advanced(sentence, config.language)
                
                # Filter by length
                words = [w for w in words if config.min_word_length <= len(w) <= config.max_word_length]
                
                # Remove punctuation
                if config.remove_punctuation:
                    words = [w for w in words if w not in string.punctuation and w.isalnum()]
                
                # Remove stopwords
                if config.remove_stopwords:
                    words = remove_stopwords_advanced(words, config.language)
                
                # Stem words
                if config.stem_words and not config.lemmatize_words:  # Don't do both
                    words = stem_words(words)
                
                # Lemmatize words
                if config.lemmatize_words:
                    words = lemmatize_words(words)
                
                chunk_words.extend(words)
            
            processed_text = ' '.join(chunk_words)
            
        else:
            # Process normally for smaller texts
            if config.to_lowercase:
                processed_text = processed_text.lower()
                metadata['processing_steps'].append('lowercase')
            
            # Tokenize words
            words = tokenize_words_advanced(processed_text, config.language)
            
            # Filter by length
            words = [w for w in words if config.min_word_length <= len(w) <= config.max_word_length]
            
            # Remove punctuation
            if config.remove_punctuation:
                words = [w for w in words if w not in string.punctuation and w.isalnum()]
                metadata['processing_steps'].append('remove_punctuation')
            
            # Remove stopwords
            if config.remove_stopwords:
                words = remove_stopwords_advanced(words, config.language)
                metadata['processing_steps'].append('remove_stopwords')
            
            # Stem words
            if config.stem_words and not config.lemmatize_words:
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
        metadata['word_count'] = len(processed_text.split())
        metadata['compression_ratio'] = (
            metadata['final_length'] / metadata['original_length'] 
            if metadata['original_length'] > 0 else 0
        )
        metadata['processing_time'] = time.time() - start_time
        
        return processed_text, metadata
    
    except Exception as e:
        metadata['processing_time'] = time.time() - start_time
        metadata['error'] = str(e)
        logger.error(f"Text preprocessing pipeline failed: {e}")
        raise TextProcessingError(f"Preprocessing failed: {e}") from e

def extract_text_features(text: str) -> Dict[str, Any]:
    """Extract comprehensive text features for analysis with improved error handling."""
    if not text:
        return {}
    
    features = {}
    
    try:
        # Basic statistics
        stats = calculate_text_stats(text)
        features['stats'] = stats.__dict__
        
        # Language detection
        features['language'] = detect_language(text)
        
        # Named entities
        features['entities'] = extract_named_entities(text)
        
        # Keywords (frequency-based)
        words = tokenize_words_advanced(text.lower())
        if words:
            word_freq = Counter(words)
            features['top_words'] = dict(word_freq.most_common(20))
        else:
            features['top_words'] = {}
        
        # Sentence lengths analysis
        sentences = tokenize_sentences_advanced(text)
        if sentences:
            sentence_lengths = [len(tokenize_words_advanced(sent)) for sent in sentences]
            features['sentence_length_stats'] = {
                'min': min(sentence_lengths),
                'max': max(sentence_lengths),
                'avg': sum(sentence_lengths) / len(sentence_lengths),
                'median': sorted(sentence_lengths)[len(sentence_lengths) // 2]
            }
        else:
            features['sentence_length_stats'] = {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        
        # Text complexity indicators
        if words:
            features['complexity_indicators'] = {
                'avg_word_length': sum(len(word) for word in words) / len(words),
                'unique_words_ratio': len(set(words)) / len(words),
                'long_sentences_ratio': (
                    sum(1 for length in sentence_lengths if length > 20) / len(sentence_lengths)
                    if sentence_lengths else 0
                ),
                'punctuation_density': sum(1 for char in text if char in string.punctuation) / len(text),
                'capitalization_ratio': sum(1 for char in text if char.isupper()) / len(text)
            }
        else:
            features['complexity_indicators'] = {
                'avg_word_length': 0,
                'unique_words_ratio': 0,
                'long_sentences_ratio': 0,
                'punctuation_density': 0,
                'capitalization_ratio': 0
            }
        
        # Document structure analysis
        features['structure'] = extract_document_structure(text)
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        features['extraction_error'] = str(e)
    
    return features

def summarize_text_simple(text: str, max_sentences: int = 3, method: str = 'frequency') -> str:
    """Simple extractive text summarization with multiple methods."""
    if not text:
        return ""
    
    sentences = tokenize_sentences_advanced(text)
    if len(sentences) <= max_sentences:
        return text
    
    try:
        if method == 'frequency':
            # Score sentences by word frequency
            words = tokenize_words_advanced(text.lower())
            word_freq = Counter(words)
            
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_words = tokenize_words_advanced(sentence.lower())
                if sentence_words:
                    score = sum(word_freq[word] for word in sentence_words)
                    sentence_scores[i] = score / len(sentence_words)
                else:
                    sentence_scores[i] = 0
        
        elif method == 'position':
            # Score sentences by their position (beginning and end are important)
            sentence_scores = {}
            total_sentences = len(sentences)
            for i in range(total_sentences):
                if i < total_sentences * 0.3:  # First 30%
                    sentence_scores[i] = 1.0
                elif i > total_sentences * 0.7:  # Last 30%
                    sentence_scores[i] = 0.8
                else:  # Middle
                    sentence_scores[i] = 0.5
        
        elif method == 'length':
            # Score sentences by length (medium-length sentences are preferred)
            sentence_scores = {}
            sentence_lengths = [len(tokenize_words_advanced(sent)) for sent in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            for i, length in enumerate(sentence_lengths):
                # Prefer sentences close to average length
                deviation = abs(length - avg_length)
                sentence_scores[i] = max(0, 1 - (deviation / avg_length))
        
        else:
            # Default to frequency method
            words = tokenize_words_advanced(text.lower())
            word_freq = Counter(words)
            
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_words = tokenize_words_advanced(sentence.lower())
                if sentence_words:
                    score = sum(word_freq[word] for word in sentence_words)
                    sentence_scores[i] = score / len(sentence_words)
                else:
                    sentence_scores[i] = 0
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sentences.sort(key=lambda x: x[0])  # Sort by original order
        
        return ' '.join(sentences[i] for i, _ in top_sentences)
    
    except Exception as e:
        logger.error(f"Text summarization failed: {e}")
        # Fallback to first few sentences
        return ' '.join(sentences[:max_sentences])

def find_similar_sentences(
    query: str, 
    sentences: List[str], 
    top_k: int = 5,
    similarity_threshold: float = 0.1
) -> List[Tuple[str, float]]:
    """Find sentences most similar to query using TF-IDF cosine similarity."""
    if not SKLEARN_AVAILABLE or not sentences or not query.strip():
        return []
    
    try:
        all_texts = [query] + sentences
        
        # Use more sophisticated TF-IDF parameters
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            min_df=1,
            max_df=0.8,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        query_vec = tfidf_matrix[0:1]
        sentence_vecs = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vec, sentence_vecs)[0]
        
        # Filter by threshold and get top similar sentences
        sentence_scores = [
            (sentence, similarity) 
            for sentence, similarity in zip(sentences, similarities)
            if similarity >= similarity_threshold
        ]
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        return sentence_scores[:top_k]
    
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []

def clean_pdf_text(text: str) -> str:
    """Specialized cleaning for PDF-extracted text with improved patterns."""
    if not text:
        return ""
    
    try:
        # Remove common PDF artifacts
        patterns_to_remove = [
            r'(?i)^\s*page\s+\d+\s*,?\s*$',  # Page numbers
            r'(?i)^\s*\d+\s*,?\s*$',  # Standalone numbers
            r'(?i)^\s*chapter\s+\d+\s*$',  # Chapter headers
            r'(?i)^\s*section\s+\d+(\.\d+)*\s*$',  # Section numbers
            r'(?i)^\s*table\s+of\s+contents?\s*$',  # TOC
            r'(?i)^\s*bibliography\s*$',  # Bibliography
            r'(?i)^\s*references?\s*$',  # References
            r'(?i)^\s*appendix\s+[a-z]\s*$',  # Appendix
            r'(?i)^\s*figure\s+\d+\s*:?.*$',  # Figure captions at start of line
            r'(?i)^\s*table\s+\d+\s*:?.*$',  # Table captions at start of line
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Fix broken words (common in PDF extraction)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Hyphenated words across lines
        text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1\2', text)  # Words broken across lines
        
        # Fix spacing issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
        text = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', text)  # Missing spaces after punctuation
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Missing spaces between numbers and letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Missing spaces between letters and numbers
        
        # Clean up headers and footers (repeated text patterns)
        lines = text.split('\n')
        if len(lines) > 10:
            # Remove lines that appear frequently (likely headers/footers)
            line_counts = Counter(line.strip() for line in lines if line.strip())
            frequent_lines = {line for line, count in line_counts.items() 
                            if count > len(lines) * 0.1 and len(line) < 100}
            
            lines = [line for line in lines if line.strip() not in frequent_lines]
            text = '\n'.join(lines)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple empty lines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces
        text = re.sub(r' +\n', '\n', text)  # Trailing spaces
        
        return text.strip()
    
    except Exception as e:
        logger.error(f"PDF text cleaning failed: {e}")
        return text

def extract_document_structure(text: str) -> Dict[str, Any]:
    """Extract document structure with improved pattern recognition."""
    if not text:
        return {}
    
    lines = text.split('\n')
    structure = {
        'headings': [],
        'sections': [],
        'lists': [],
        'tables': [],
        'quotes': [],
        'code_blocks': []
    }
    
    try:
        # Enhanced heading detection patterns
        heading_patterns = [
            r'^\s*[A-Z][A-Z\s]{3,50}$',  # ALL CAPS headings
            r'^\s*\d+\.?\s+[A-Z][a-zA-Z\s]{3,50}$',  # Numbered headings
            r'^\s*[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s*$',  # Title Case headings
            r'^\s*Chapter\s+\d+',  # Chapter headings
            r'^\s*Section\s+\d+',  # Section headings
            r'^\s*Part\s+[IVX]+',  # Part headings with Roman numerals
            r'^\s*[A-Z]\.\s+[A-Z]',  # Lettered sections
        ]
        
        # List patterns (improved)
        list_patterns = [
            r'^\s*[\-\*\+•]\s+',  # Bullet points
            r'^\s*\d+[\.\)]\s+',  # Numbered lists
            r'^\s*[a-zA-Z][\.\)]\s+',  # Lettered lists
            r'^\s*[ivx]+[\.\)]\s+',  # Roman numeral lists
            r'^\s*\(\d+\)\s+',  # Parenthetical numbering
            r'^\s*\([a-zA-Z]\)\s+',  # Parenthetical lettering
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check for headings
            for pattern in heading_patterns:
                if re.match(pattern, line_stripped) and len(line_stripped) < 100:
                    structure['headings'].append({
                        'text': line_stripped,
                        'line_number': i,
                        'type': 'heading',
                        'level': 1  # Could be enhanced to detect levels
                    })
                    break
            
            # Check for lists
            for pattern in list_patterns:
                if re.match(pattern, line_stripped):
                    structure['lists'].append({
                        'text': line_stripped,
                        'line_number': i,
                        'type': 'list_item'
                    })
                    break
            
            # Check for tables (improved detection)
            table_indicators = [
                line_stripped.count('\t') >= 2,
                line_stripped.count('|') >= 2,
                line_stripped.count('   ') >= 3,  # Multiple spaces
                re.search(r'\d+\s+\d+\s+\d+', line_stripped),  # Numbers in columns
            ]
            
            if any(table_indicators):
                structure['tables'].append({
                    'text': line_stripped,
                    'line_number': i,
                    'type': 'table_row'
                })
            
            # Check for quotes
            if (line_stripped.startswith('"') and line_stripped.endswith('"')) or \
               (line_stripped.startswith("'") and line_stripped.endswith("'")):
                structure['quotes'].append({
                    'text': line_stripped,
                    'line_number': i,
                    'type': 'quote'
                })
            
            # Check for code blocks (basic detection)
            code_indicators = [
                line_stripped.startswith('```'),
                line_stripped.startswith('    ') and any(char in line_stripped for char in '{}();'),
                re.search(r'^[a-zA-Z_]\w*\s*[=:]\s*', line_stripped),  # Variable assignments
            ]
            
            if any(code_indicators):
                structure['code_blocks'].append({
                    'text': line_stripped,
                    'line_number': i,
                    'type': 'code'
                })
    
    except Exception as e:
        logger.error(f"Document structure extraction failed: {e}")
    
    return structure

def validate_text_quality(text: str) -> Dict[str, Any]:
    """Enhanced text quality validation with more sophisticated checks."""
    if not text:
        return {'quality_score': 0, 'issues': ['empty_text']}
    
    issues = []
    quality_metrics = {}
    
    try:
        # Check for minimum length
        if len(text) < 100:
            issues.append('text_too_short')
        
        # Check for excessive repetition
        words = tokenize_words_advanced(text.lower())
        if words:
            word_freq = Counter(words)
            most_common_freq = word_freq.most_common(1)[0][1]
            repetition_ratio = most_common_freq / len(words)
            
            if repetition_ratio > 0.15:  # Increased threshold
                issues.append('excessive_repetition')
            quality_metrics['repetition_ratio'] = repetition_ratio
            
            # Check for vocabulary diversity
            unique_words = len(set(words))
            diversity_ratio = unique_words / len(words)
            if diversity_ratio < 0.3:
                issues.append('low_vocabulary_diversity')
            quality_metrics['vocabulary_diversity'] = diversity_ratio
        
        # Check for gibberish using character-level analysis
        alpha_chars = sum(1 for char in text if char.isalpha())
        if alpha_chars > 0:
            vowel_count = sum(1 for char in text.lower() if char in 'aeiou')
            vowel_ratio = vowel_count / alpha_chars
            
            # English text typically has 35-45% vowels
            if vowel_ratio < 0.25 or vowel_ratio > 0.60:
                issues.append('unusual_character_distribution')
            quality_metrics['vowel_ratio'] = vowel_ratio
        
        # Check for encoding issues
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeError:
            issues.append('encoding_issues')
        
        # Check for excessive punctuation
        punct_count = sum(1 for char in text if char in string.punctuation)
        punct_ratio = punct_count / len(text) if text else 0
        if punct_ratio > 0.20:
            issues.append('excessive_punctuation')
        quality_metrics['punctuation_ratio'] = punct_ratio
        
        # Check for sentence structure
        sentences = tokenize_sentences_advanced(text)
        if sentences:
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            if avg_sentence_length < 3:
                issues.append('very_short_sentences')
            elif avg_sentence_length > 50:
                issues.append('very_long_sentences')
            quality_metrics['avg_sentence_length'] = avg_sentence_length
        
        # Check for capitalization patterns
        if text:
            cap_chars = sum(1 for char in text if char.isupper())
            cap_ratio = cap_chars / len(text)
            
            if cap_ratio > 0.50:
                issues.append('excessive_capitalization')
            elif cap_ratio < 0.01 and len(text) > 100:
                issues.append('no_capitalization')
            quality_metrics['capitalization_ratio'] = cap_ratio
        
        # Check for digit density
        digit_count = sum(1 for char in text if char.isdigit())
        digit_ratio = digit_count / len(text) if text else 0
        if digit_ratio > 0.30:
            issues.append('excessive_numbers')
        quality_metrics['digit_ratio'] = digit_ratio
        
        # Calculate overall quality score
        quality_score = 1.0
        
        # Weight different issues differently
        issue_weights = {
            'empty_text': 1.0,
            'text_too_short': 0.3,
            'excessive_repetition': 0.4,
            'low_vocabulary_diversity': 0.3,
            'unusual_character_distribution': 0.5,
            'encoding_issues': 0.6,
            'excessive_punctuation': 0.2,
            'very_short_sentences': 0.2,
            'very_long_sentences': 0.2,
            'excessive_capitalization': 0.3,
            'no_capitalization': 0.1,
            'excessive_numbers': 0.2,
        }
        
        for issue in issues:
            weight = issue_weights.get(issue, 0.2)
            quality_score -= weight
        
        quality_score = max(0, min(1, quality_score))
        
    except Exception as e:
        logger.error(f"Text quality validation failed: {e}")
        issues.append('validation_error')
        quality_metrics['validation_error'] = str(e)
        quality_score = 0.5  # Default middle score on error
    
    return {
        'quality_score': quality_score,
        'issues': issues,
        'metrics': quality_metrics
    }

# Utility functions for batch processing with improved error handling
def process_text_batch(
    texts: List[str],
    config: Optional[ProcessingConfig] = None,
    max_workers: int = 4
) -> List[Tuple[str, Dict[str, Any]]]:
    """Process multiple texts with parallel processing support."""
    if not texts:
        return []
    
    if config is None:
        config = ProcessingConfig()
    
    results = []
    
    # For small batches, process sequentially
    if len(texts) <= 10:
        for text in texts:
            try:
                processed_text, metadata = preprocess_text_pipeline(text, config)
                results.append((processed_text, metadata))
            except Exception as e:
                logger.error(f"Failed to process text: {e}")
                results.append(("", {"error": str(e)}))
    else:
        # For larger batches, could implement parallel processing here
        # For now, process sequentially with progress tracking
        for i, text in enumerate(texts):
            try:
                processed_text, metadata = preprocess_text_pipeline(text, config)
                metadata['batch_index'] = i
                results.append((processed_text, metadata))
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Failed to process text {i}: {e}")
                results.append(("", {"error": str(e), "batch_index": i}))
    
    return results

def merge_text_features(feature_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge features from multiple text documents with improved aggregation."""
    if not feature_dicts:
        return {}
    
    # Filter out invalid feature dictionaries
    valid_features = [f for f in feature_dicts if f and isinstance(f, dict)]
    if not valid_features:
        return {}
    
    merged = {
        'document_count': len(valid_features),
        'combined_stats': {},
        'combined_entities': {},
        'combined_keywords': Counter(),
        'quality_summary': {},
        'language_distribution': Counter()
    }
    
    try:
        # Merge statistics
        stat_fields = ['char_count', 'word_count', 'sentence_count', 'paragraph_count']
        totals = {}
        
        for field in stat_fields:
            totals[field] = sum(
                f.get('stats', {}).get(field, 0) for f in valid_features
            )
        
        merged['combined_stats'] = {
            'total_char_count': totals.get('char_count', 0),
            'total_word_count': totals.get('word_count', 0),
            'total_sentence_count': totals.get('sentence_count', 0),
            'total_paragraph_count': totals.get('paragraph_count', 0),
            'avg_chars_per_doc': totals.get('char_count', 0) / len(valid_features),
            'avg_words_per_doc': totals.get('word_count', 0) / len(valid_features),
            'avg_sentences_per_doc': totals.get('sentence_count', 0) / len(valid_features),
        }
        
        # Merge entities with deduplication
        for features in valid_features:
            entities = features.get('entities', {})
            for entity_type, entity_list in entities.items():
                if entity_type not in merged['combined_entities']:
                    merged['combined_entities'][entity_type] = set()
                merged['combined_entities'][entity_type].update(entity_list)
        
        # Convert sets back to lists
        for entity_type in merged['combined_entities']:
            merged['combined_entities'][entity_type] = list(
                merged['combined_entities'][entity_type]
            )
        
        # Merge keywords with frequency counting
        for features in valid_features:
            top_words = features.get('top_words', {})
            merged['combined_keywords'].update(top_words)
        
        # Get top combined keywords
        merged['top_combined_keywords'] = dict(
            merged['combined_keywords'].most_common(50)
        )
        
        # Language distribution
        for features in valid_features:
            language = features.get('language')
            if language:
                merged['language_distribution'][language] += 1
        
        # Quality summary
        quality_scores = [
            f.get('quality_score', 0) for f in valid_features 
            if 'quality_score' in f
        ]
        
        if quality_scores:
            merged['quality_summary'] = {
                'avg_quality_score': sum(quality_scores) / len(quality_scores),
                'min_quality_score': min(quality_scores),
                'max_quality_score': max(quality_scores),
                'documents_analyzed': len(quality_scores)
            }
    
    except Exception as e:
        logger.error(f"Feature merging failed: {e}")
        merged['merge_error'] = str(e)
    
    return merged

def chunk_text_intelligently(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 100,
    preserve_sentences: bool = True
) -> List[str]:
    """
    Split text into chunks intelligently, preserving sentence boundaries.
    """
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    try:
        if preserve_sentences:
            sentences = tokenize_sentences_advanced(text)
            
            current_chunk = ""
            for sentence in sentences:
                # If adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > overlap:
                        current_chunk = current_chunk[-overlap:] + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        else:
            # Simple character-based chunking
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
    
    except Exception as e:
        logger.error(f"Text chunking failed: {e}")
        # Fallback to simple splitting
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    return chunks

# Export main functions for easier imports
__all__ = [
    'TextStats', 'ProcessingConfig', 'TextProcessingError',
    'detect_language', 'normalize_unicode', 'clean_text_advanced',
    'tokenize_sentences_advanced', 'tokenize_words_advanced',
    'remove_stopwords_advanced', 'stem_words', 'lemmatize_words',
    'extract_named_entities', 'calculate_text_stats',
    'extract_keywords_tfidf', 'preprocess_text_pipeline',
    'extract_text_features', 'summarize_text_simple',
    'find_similar_sentences', 'clean_pdf_text',
    'extract_document_structure', 'validate_text_quality',
    'process_text_batch', 'merge_text_features',
    'chunk_text_intelligently'
]