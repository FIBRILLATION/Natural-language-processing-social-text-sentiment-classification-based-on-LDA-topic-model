81309:Advanced Topics in Machine Learning (K1) § 81310:Advanced Topics in Intelligent Systems (K1) Final project 

allowed to freely choose topics based on machine learning principles (such as computer vision, deep learning, natural language processing), but must use approximate reasoning models and methods of Bayesian reasoning, and not allowed to use libraries such as scikit-learn that provide methods or models.

In this project, my initial idea was to use existing social text data to build a machine learning model, so I chose the field of natural language processing. The goal is to use the lda model to build an algorithm system for sentiment classification and prediction of user comments on social media.

2,000 text data were sampled through crawlers, and in preprocessing, the emotions of these texts were divided into 5 topic variables (0: sadness; 1: happiness; 2: love; 3: anger; 4: fear). The constructed algorithm uses the lda model and Gibbs sampling to encode the text of each topic and predict the 5 words with the highest probability of occurrence. Considering the correlation between text frequency and importance, the TF-IDF model is also imported into the algorithm model to balance the text weight.
