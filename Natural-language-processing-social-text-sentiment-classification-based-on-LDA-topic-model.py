import numpy as np
import pandas as pd
import spacy
from collections import Counter

# データセット導入
file_path = '/Users/wangyuxiang/Downloads/datatsets.csv'
data = pd.read_csv(file_path)
texts = data['text'].tolist()

# ハイパーパラメータの定義
num_topics = 5  # トピックの数
alpha = 0.5  # トピック分散の前にディリクレを調整する
beta = 0.01   # 単語分布の事前ディリクレを調整する
iterations = 1000 # 反復数

# トピックの定義
predefined_topics = {
    0: "sadness",
    1: "happiness",
    2: "love",
    3: "anger",
    4: "fear"
}

# 英語モデルspacyをロードする、テキストクリーニングを実行し、ストップワードとアルファベット以外の文字を削除し、見出し語化を実行する
nlp = spacy.load('en_core_web_sm')


# テキストの前処理
def preprocess_text(texts):
    # 前処理ステップ
    def clean_text(text):
        doc = nlp(text.lower())
        words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token) > 2]
        return words

    # すべてのテキストを処理する
    cleaned_texts = [clean_text(text) for text in texts]
    vocab = list(set([word for text in cleaned_texts for word in text]))
    vocab_size = len(vocab)
    doc_count = len(texts)

    # 低頻度単語と高頻度単語をフィルタリングする
    word_counts = Counter([word for text in cleaned_texts for word in text])
    low_freq_threshold = 5
    high_freq_threshold = doc_count * 0.4
    filtered_vocab = [word for word, count in word_counts.items() if low_freq_threshold <= count <= high_freq_threshold]

    # 文書と単語のマトリックスを構築する
    def build_doc_word_matrix(cleaned_texts, filtered_vocab):
        doc_word_matrix = np.zeros((len(cleaned_texts), len(filtered_vocab)))
        for doc_index, text in enumerate(cleaned_texts):
            for word in text:
                if word in filtered_vocab:
                    word_index = filtered_vocab.index(word)
                    doc_word_matrix[doc_index, word_index] += 1
        return doc_word_matrix

    doc_word_matrix = build_doc_word_matrix(cleaned_texts, filtered_vocab)

    # TF-IDF重みづけを使用する
    def compute_tfidf(doc_word_matrix):
        tf = doc_word_matrix / (np.sum(doc_word_matrix, axis=1, keepdims=True) + 1e-10)  # 添加小常数以防除以零
        df = np.sum(doc_word_matrix > 0, axis=0)
        idf = np.log((doc_word_matrix.shape[0] + 1) / (df + 1)) + 1
        tfidf = tf * idf
        return tfidf

    doc_word_matrix = compute_tfidf(doc_word_matrix)

    return doc_word_matrix, filtered_vocab, len(filtered_vocab), doc_count


def initialize_counts(doc_word_matrix, num_topics, vocab_size, doc_count):
    doc_topic_counts = np.zeros((doc_count, num_topics)) + alpha
    topic_word_counts = np.zeros((num_topics, vocab_size)) + beta
    topic_counts = np.zeros(num_topics) + vocab_size * beta

    # トピックを初期化する
    topic_assignments = []
    for d in range(doc_count):
        current_doc_assignments = []
        for w in range(vocab_size):
            if doc_word_matrix[d, w] > 0:
                topics = np.random.choice(num_topics, int(doc_word_matrix[d, w]))
                current_doc_assignments.extend(topics)
                for t in topics:
                    doc_topic_counts[d, t] += 1
                    topic_word_counts[t, w] += 1
                    topic_counts[t] += 1
        topic_assignments.append(current_doc_assignments)

    return doc_topic_counts, topic_word_counts, topic_counts, topic_assignments


def gibbs_sampling(doc_word_matrix, num_topics, alpha, beta, iterations=1000):
    doc_count, vocab_size = doc_word_matrix.shape
    burn_in = 300
    doc_topic_counts, topic_word_counts, topic_counts, topic_assignments = initialize_counts(doc_word_matrix,
                                                                                             num_topics, vocab_size,
                                                                                             doc_count)

    for it in range(iterations):
        if it % 10 == 0:
            print(f"Iteration {it}")
        for d in range(doc_count):
            for w in range(vocab_size):
                if doc_word_matrix[d, w] > 0:
                    for _ in range(int(doc_word_matrix[d, w])):
                        current_topic = topic_assignments[d].pop()
                        doc_topic_counts[d, current_topic] -= 1
                        topic_word_counts[current_topic, w] -= 1
                        topic_counts[current_topic] -= 1

                        # 新しいトピックの割り当て確率を計算する
                        topic_probs = (doc_topic_counts[d] * topic_word_counts[:, w]) / topic_counts
                        topic_probs = np.maximum(topic_probs, 1e-10)  # 確率がマイナスでもゼロでもないことを確認
                        topic_probs /= topic_probs.sum()

                        new_topic = np.random.choice(num_topics, p=topic_probs)

                        # 更新数とトピックの割り当て
                        topic_assignments[d].append(new_topic)
                        doc_topic_counts[d, new_topic] += 1
                        topic_word_counts[new_topic, w] += 1
                        topic_counts[new_topic] += 1

    return doc_topic_counts, topic_word_counts


def display_top_words(topic_word_counts, vocab, num_top_words=5):
    for topic_idx, topic in enumerate(topic_word_counts):
        top_word_indices = topic.argsort()[-num_top_words:][::-1]
        top_words = [vocab[i] for i in top_word_indices]
        print(f"Topic {topic_idx} ({predefined_topics[topic_idx]}): {' '.join(top_words)}")

# メインの流れ
# テキストの前処理
doc_word_matrix, vocab, vocab_size, doc_count = preprocess_text(texts)

# ギブスサンプリング
doc_topic_counts, topic_word_counts = gibbs_sampling(doc_word_matrix, num_topics, alpha, beta, iterations)

# 各トピックについて最も可能性の高い5つの単語を表示する
display_top_words(topic_word_counts, vocab)
