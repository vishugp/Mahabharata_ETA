# Mahabharata Text Analysis

A computational exploration of the Mahabharata epic using advanced text analysis techniques.

## Project Overview

This project applies modern text analysis methods to the English translation of the Mahabharata (translated by Kisari Mohan Ganguli). Using a structured text analysis pipeline, I've transformed the raw text into a series of analytical models that explore themes, sentiment patterns, character relationships, and linguistic features across this ancient epic.

The analysis follows the Standard Text Analytic Data Model approach, moving from raw text (F0) through parsing (F2), annotation (F3), vectorization (F4), to advanced modeling (F5).

## Key Visualizations & Findings

### PCA Analysis - War vs Peace Themes
![PCA Components 0-1](https://raw.githubusercontent.com/vishugp/Mahabharata_ETA/main/images/pca01pcs.png)

The first principal component clearly separates war and non-war sections of the epic:
- **Positive PC1**: Contains war-related terms like "shafts," "carwarriors," "pierced," "army"
- **Negative PC1**: Contains spiritual terms like "deities," "penances," "brahman," "righteousness"

This visualization effectively maps the timeline progression (Pre-War, War, Post-War) across the narrative space.

### Topic Modeling with LDA
![LDA and PCA](https://raw.githubusercontent.com/vishugp/Mahabharata_ETA/main/images/theta_pca_pcs.png)

Ten distinct topics were identified through Latent Dirichlet Allocation, including:
- Battle and warfare terminology
- Spiritual and philosophical discourse
- Kingdom management and politics
- Family relationships
- Divine weapons and celestial beings

The topics cluster intelligently when projected onto principal component space, with war-related topics appearing in the positive PC1 direction and domestic/kingdom topics in the negative PC1.

### Character Appearances Throughout the Epic
![Character Appearances](https://raw.githubusercontent.com/vishugp/Mahabharata_ETA/main/images/kde_line.png)

This kernel density estimation plot tracks key character appearances across the narrative:
- **Yudhishthira** shows high density in both Pre-War and Post-War phases
- **Karna** appears primarily during the War phase, ending with his death
- **Gandhari** shows peaks after Karna's death and at the epic's conclusion

These patterns align with the known narrative structure while providing quantitative evidence of character prominence.

### Emotional Analysis of Key Characters
![Character Emotions](https://raw.githubusercontent.com/vishugp/Mahabharata_ETA/main/images/emotion_heatmap.png)

This heatmap reveals the emotional profiles of major characters:
- **Bhima, Karna, and Duryodhana** show the highest anger scores
- **Krishna and Yudhishthira** display the highest trust scores
- **Joy** appears most prominently in Yudhishthira, Krishna, and surprisingly, Karna
- **Negative sentiment** is strongest in Bhima, Karna, and Duryodhana

The emotional complexity of these characters supports traditional interpretations while providing numerical evidence of their psychological depth.

### Sentiment Evolution During War
![Character Sentiment](https://raw.githubusercontent.com/vishugp/Mahabharata_ETA/main/images/VADER_characters.png)

This sentiment analysis tracks emotional shifts throughout the narrative:
- Before the war, Krishna maintained positive sentiment while others fluctuated
- During battle, emotions became unstable for all characters with significant overlap
- Karna's sentiment drops dramatically around his death scene (65-70k mark)
- Post-war, all characters settle near neutral emotional territory

The convergence of sentiment during war brilliantly illustrates how conflict blurs the moral lines between heroes and villains.

### Word Embeddings with Word2Vec and t-SNE
![Word Embeddings](https://raw.githubusercontent.com/vishugp/Mahabharata_ETA/main/images/tSNE.png)

Word2Vec embeddings visualized with t-SNE reveal meaningful semantic clusters:
- Emotional terms group together
- Warrior-related terminology forms distinct clusters
- Relational terms (family relationships) appear in proximity

These clusters demonstrate how the embedding model captures the semantic relationships within the epic's vocabulary.

## Technical Implementation

The analysis pipeline included:
- Parsing the text into structured F2 tables (LIB, CORPUS, VOCAB)
- Creating bag-of-words and TF-IDF representations
- Applying dimensionality reduction with PCA
- Topic modeling with LDA
- Sentiment analysis using lexicon-based approaches
- Generating word embeddings with Word2Vec
- Visualizing with t-SNE and other techniques

All code is implemented in Python using libraries including NLTK, scikit-learn, gensim, and various visualization tools.

## Repository Structure

```
Mahabharata_ETA/
├── data/                  # Raw and processed data files
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_create_F2.ipynb # Creating core F2 tables
│   ├── 02_TFIDF.ipynb     # TF-IDF calculation
│   ├── 03_PCA.ipynb       # Principal Component Analysis
│   ├── 04_LDA.ipynb       # Topic Modeling
│   ├── 05_Sentiment_Analysis.ipynb # Sentiment Analysis
│   └── 06_Word2Vec.ipynb  # Word Embeddings
├── images/                # Visualizations and plots
└── README.md              # This file
```

## Future Directions

Potential extensions of this work include:
- Named Entity Recognition to map character relationships more systematically
- Network analysis of character co-occurrences
- Deeper temporal analysis of narrative progression
- Comparative analysis with other translations or epics

## About the Dataset

The analysis uses the English translation of the Mahabharata by Kisari Mohan Ganguli, completed in the late 19th century. This public domain text contains all 18 Parvas (books) of the epic, divided into chapters (Upa-parvas) and sections.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Text as Data course (DS 5001) at the University of Virginia
- Sacred-texts.com for providing the digital corpus
