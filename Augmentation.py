import jieba
import random
import fasttext
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

class SynonymReplacer:
    def __init__(self, model_path='cc.zh.300.bin', stop_words=None, max_workers=8):
        print("正在加载词向量模型...")
        self.model = fasttext.load_model(model_path)
        self.stop_words = stop_words if stop_words else set(["的", "是", "在", "了", "和", "也", "有", "对", "与", "吗", "吧", "哦", "用户", "客服", "<image>"])
        self.max_workers = max_workers

    @lru_cache(maxsize=10000)
    def cached_get_synonyms(self, word, similarity_threshold=0.7):
        try:
            similar_words = self.model.get_nearest_neighbors(word, k=5)
            synonyms = [w for sim, w in similar_words if sim >= similarity_threshold]
            return synonyms
        except Exception:
            return []

    def find_replaceable_words(self, words, similarity_threshold):
        replaceable_indices = []

        def task(word):
            if word not in self.stop_words and any('\u4e00' <= ch <= '\u9fff' for ch in word):
                return self.cached_get_synonyms(word, similarity_threshold)
            return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(task, word): i for i, word in enumerate(words)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                synonyms = future.result()
                if synonyms:
                    replaceable_indices.append((index, synonyms))

        return replaceable_indices

    def synonym_replacement(self, sentence, replace_rate=0.5, similarity_threshold=0.7):
        words = jieba.lcut(sentence)
        new_words = words[:]

        replaceable_indices = self.find_replaceable_words(words, similarity_threshold)

        num_replacements = max(1, int(len(words) * replace_rate))
        for _ in range(num_replacements):
            if not replaceable_indices:
                break
            index, synonyms = random.choice(replaceable_indices)
            new_words[index] = random.choice(synonyms)
            replaceable_indices = [item for item in replaceable_indices if item[0] != index]

        return ''.join(new_words)

    def generate_augmented_sentences(self, sentence, k=5, replace_rate=0.5, similarity_threshold=0.7):
        augmented_sentences = set()
        attempts = 0
        max_attempts = k * 2  # 防止死循环的最大尝试次数

        while len(augmented_sentences) < k - 1 and attempts < max_attempts:
            augmented_sentence = self.synonym_replacement(sentence, replace_rate, similarity_threshold)
            if augmented_sentence != sentence:  # 确保与原句不同
                augmented_sentences.add(augmented_sentence)
            attempts += 1

        return [sentence] + list(augmented_sentences)

if __name__ == "__main__":
    # 示例句子
    sentence = "用户: 这些款式的区别在哪里？哪款手机更好一些，mate60pro还是p70pro？"
    k = 5  # 生成 5 个不同的句子

    replacer = SynonymReplacer()

    augmented_sentences = replacer.generate_augmented_sentences(sentence, k=k)
    print("原句：", sentence)
    print("增广后：")
    for idx, augmented_sentence in enumerate(augmented_sentences, 1):
        print(f"{idx}: {augmented_sentence}")
