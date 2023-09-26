# 윈도우에서 사용 가능한 형태소 분석기는 꼬꼬마 분석기밖에 없었음
# 꼬꼬마 분석기가 반복되는 단어 이모티콘[ex) ㅋㅋ, ㅎㅎ]이 많이 포함된 문자열을 분석할 수 없음 -> soynlp 모듈 사용으로 해결
# 의미 없는 단어[ex) ㅂㄷㅂㄷㅂㄷㅂㄷ]의 반복도 약 18단어 정도 반복되면 프로그램이 멈춤 -> 미해결
import numpy as np
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from soynlp.normalizer import emoticon_normalize


def str_pre_process(string_data):
    normalized = emoticon_normalize(string_data, num_repeats=2)
    replaced = normalized.replace(",", " ").replace(".", "").replace("!", "").replace("?", "").replace(";",
                                                                                                       " ").replace(
        "'", "").replace("\"", "").replace("1", "").replace("|", "")
    return replaced


def str_pre_process_by_morph(string_data):
    replaced = str_pre_process(string_data)
    morphs = kkma.morphs(replaced)
    new_data = " ".join(morphs)
    return new_data


def str_pre_process_by_noun(string_data):
    replaced = str_pre_process(string_data)
    nouns = kkma.nouns(replaced)
    new_data = " ".join(nouns)
    return new_data


if __name__ == "__main__":
    kkma = Kkma()

    train_texts = np.array([])
    train_labels = np.array([])

    with open("dataset.txt", "r", encoding='UTF-8') as f:
        for line in f:
            # 문장 좌우 공백 제거, 훈련 데이터셋의 구분자인 | 를 기준으로 문자열 분할
            data_array = line.strip().split('|')

            processed_morph = str_pre_process_by_morph(data_array[0])
            processed_noun = str_pre_process_by_noun(data_array[0])

            train_texts = np.append(train_texts, data_array[0])
            train_texts = np.append(train_texts, processed_morph)
            train_texts = np.append(train_texts, processed_noun)
            train_labels = np.append(train_labels, data_array[1])
            train_labels = np.append(train_labels, data_array[1])
            train_labels = np.append(train_labels, data_array[1])

            print(data_array[0])
            print(processed_morph)
            print(processed_noun)
            print()

    # 단어의 출현 빈도로 문자열을 벡터화
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_texts)

    # 로지스틱 회귀 모델 생성 및 훈련
    model = LogisticRegression()
    model.fit(X, train_labels)

    # 모델 평가
    score = model.score(X, train_labels)

    # 새로운 텍스트에 대한 예측
    test_dataset = [
        '시발 저걸 못하네',
        '드디어 오늘 졸업했습니다',
        '지랄하지마 진짜 시1발',
        '오늘 날씨가 좋네요',
        '아버지가방에들어가신다',
        '무논리 맘충 존나 많네 요즘',
        '도태한남 재기해',
        '느금마'
    ]

    processed_dataset = []

    for data in test_dataset:
        processed_dataset.append(str_pre_process_by_morph(data))

    new_text_vectorized = vectorizer.transform(processed_dataset)
    prediction = model.predict(new_text_vectorized)
    harmful_text_count = list(train_labels).count('1')

    print("********************************************************************************")
    print(f"Test accuracy                   : {score * 100}%")
    print(f"Harmful text ratio for test     : {harmful_text_count / train_labels.size * 100}%")
    print(f"Test prediction result          : {prediction}")
    print("********************************************************************************")
