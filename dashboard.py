import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Party Media Analysis Dashboard")

# Load CSV files
word_freq = pd.read_csv('word_frequency.csv')
entities = pd.read_csv('named_entities.csv')
bigrams = pd.read_csv('bigram_frequency.csv')
trigrams = pd.read_csv('trigram_frequency.csv')
fourgrams = pd.read_csv('fourgram_frequency.csv')
fivegrams = pd.read_csv('fivegram_frequency.csv')

# Section: Word Frequency
st.header("Top 50 Words by Frequency")
st.dataframe(word_freq)

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(word_freq['word'], word_freq['frequency'])
ax.set_xticklabels(word_freq['word'], rotation=45, ha='right')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Section: Named Entities
st.header("Named Entities Extracted")
entity_types = entities['type'].value_counts()
st.subheader("Entity Type Counts")
st.bar_chart(entity_types)

st.subheader("Sample Entities")
st.dataframe(entities.head(20))

# Section: Common Bigrams
st.header("Top 20 Bigrams")
st.dataframe(bigrams)

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.bar(bigrams['bigram'], bigrams['frequency'])
ax2.set_xticklabels(bigrams['bigram'], rotation=45, ha='right')
ax2.set_ylabel('Frequency')
st.pyplot(fig2)

# Section: Common Trigrams
st.header("Top 20 Trigrams")
st.dataframe(trigrams)

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.bar(trigrams['trigram'], trigrams['frequency'])
ax3.set_xticklabels(trigrams['trigram'], rotation=45, ha='right')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)


# Section: Common Fourgrams
st.header("Top 20 Fourgrams")
st.dataframe(fourgrams)

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.bar(fourgrams['fourgram'], fourgrams['frequency'])
ax3.set_xticklabels(fourgrams['trigram'], rotation=45, ha='right')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)


# Section: Common Fivegrams
st.header("Top 20 Fivegrams")
st.dataframe(fivegrams)

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.bar(fivegrams['fivegram'], fivegrams['frequency'])
ax3.set_xticklabels(fivegrams['fivegram'], rotation=45, ha='right')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)
