import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk
from nltk.util import ngrams
from docx import Document
import pandas as pd
import re

# Install NLTK packages (run once)
# nltk.download('punkt')          # tokenizer
# nltk.download('punkt_tab')
# nltk.download('stopwords')      # stop word list
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

# 1. Extract text from DOCX
doc = Document('Tamilaga Vettri Kazhagam .DOCX')
text = '\n'.join([para.text for para in doc.paragraphs])

#Clean text
raw_text = text  # output from your doc extraction

# Patterns/keywords to remove
patterns_to_remove = [
    r'Asian News International \(ANI\)',   # News agency name
    r'Copyright.*?Syndigate Media Inc.*', # Copyright lines
    r'Length:\s*\d+\s*words',             # Length line
    r'Byline:\s*ANI\s*\|',                # Byline line
    r'Classification',                    # Classification headers
    r'Language:\s*ENGLISH',               # Language line
    r'Publication-Type:.*',               # Publication-Type line
    r'Journal Code:\s*\d+',               # Journal code
    r'Load-Date:\s*[A-Za-z]+\s*\d{1,2},\s*\d{4}', # Load-Date line
    r'End of Document',                   # End of Document marker
]

# Remove lines containing specific keywords (like "Body", "Journal Code" etc.)
keywords_to_remove_lines = [
    "Body",
    "Journal Code",
    "Byline",
    "Classification",
    "Load-Date",
    "Publication-Type",
    "Language",
    "End of Document",
]

def clean_text(raw_text):
    # Remove all regex patterns
    for pattern in patterns_to_remove:
        raw_text = re.sub(pattern, '', raw_text, flags=re.IGNORECASE|re.DOTALL)

    # Remove whole lines that contain any of the keywords
    lines = raw_text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not any(keyword.lower() in line.lower() for keyword in keywords_to_remove_lines):
            cleaned_lines.append(line.strip())

    # Remove empty or whitespace-only lines
    cleaned_text = '\n'.join([line for line in cleaned_lines if line])

    return cleaned_text

cleaned_text = clean_text(raw_text)


# 2. Preprocess: tokenize, clean, remove stopwords
tokens = word_tokenize(text.lower())  # lowercase and split into words
# Remove punctuation
tokens = [word for word in tokens if word.isalpha()]
# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]

# 3. Analyze word frequency
freq_dist = nltk.FreqDist(filtered_tokens)
freq_df = pd.DataFrame(freq_dist.most_common(50), columns=['word', 'frequency'])
freq_df.to_csv('word_frequency.csv', index=False)



# 4. POS Tagging

tagged_words = nltk.pos_tag(filtered_tokens)
print(tagged_words[:20])  # Show first 20 tagged words
#Named Entity Recognition (NER)
# NER typically works best on sentences, not just tokens
from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(text)
for sent in sentences[:10]:  # Try the first 10 sentences
    tokens = word_tokenize(sent)
    tags = nltk.pos_tag(tokens)
    tree = ne_chunk(tags)
    print(tree)


#Extract only the named entities:

entities = []
for sent in sentences:
    tree = ne_chunk(nltk.pos_tag(word_tokenize(sent)))
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity_name = " ".join([leaf[0] for leaf in subtree.leaves()])
            entity_type = subtree.label()
            entities.append({'entity': entity_name, 'type': entity_type})
entities_df = pd.DataFrame(entities)
entities_df.to_csv('named_entities.csv', index=False)


#Sentiment Analysis
from transformers import pipeline

# Load a pre-trained sentiment-analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis")

text = "The party's performance in rural development has been outstanding this year."
result = sentiment_analysis(text)
print(result)
# Output might be something like: [{'label': 'POSITIVE', 'score': 0.99}]



#Collocation and Concordance
text_obj = nltk.Text(filtered_tokens)
print(text_obj.collocations())  # Common bigram collocations

text_obj.concordance('development')  # View context for 'development'


#Frequent Word and Noun Phrase Extraction

# Extract frequent bigrams or trigrams (e.g., 'welfare scheme', 'leadership change')
bigrams = ngrams(filtered_tokens, 2)
trigrams = ngrams(filtered_tokens, 3)
four_grams = list(ngrams(filtered_tokens, 4))
five_grams = list(ngrams(filtered_tokens, 5))


bigrams_freq = nltk.FreqDist(bigrams)
trigrams_freq = nltk.FreqDist(trigrams)
four_grams_freq = Counter(four_grams).most_common(10)
five_grams_freq = Counter(five_grams).most_common(10)

bigrams_df = pd.DataFrame([(' '.join(k), v) for k, v in bigrams_freq.most_common(20)], columns=['bigram', 'frequency'])
trigrams_df = pd.DataFrame([(' '.join(k), v) for k, v in trigrams_freq.most_common(20)], columns=['trigram', 'frequency'])
four_grams_df = pd.DataFrame([(' '.join(k), v) for k, v in four_grams_freq.most_common(20)], columns=['fourgram', 'frequency'])
five_grams_df = pd.DataFrame([(' '.join(k), v) for k, v in five_grams_freq.most_common(20)], columns=['fivegram', 'frequency'])


bigrams_df.to_csv('bigram_frequency.csv', index=False)
trigrams_df.to_csv('trigram_frequency.csv', index=False)
four_grams_df.to_csv('fourgram_frequency.csv', index=False)
five_grams_df.to_csv('fivegram_frequency.csv', index=False)


print("Analysis complete. CSV files saved: word_frequency.csv, named_entities.csv, bigram_frequency.csv, trigram_frequency.csv,fourgram_frequency.csv,fivegram_frequency.csv ")
