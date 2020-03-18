from konlpy.tag import Okt
okt = Okt()

file = open('./ratings_train.txt', 'r', encoding='utf-8-sig')

sentence = []
label = []

for idx, line in enumerate(file):
    if idx == 0:     # 첫 번째 줄은 열의 label이 들어있는 Line
        continue
    line = line.split('\t')
    sentence.append(okt.morphs(line[1].strip()))
    label.append(line[2].strip())

file.close()

file = open('./ratings_test.txt', 'r', encoding='utf-8-sig')

test_sentence = []
test_label = []

for idx, line in enumerate(file):
    if idx == 0:     # 첫 번째 줄은 열의 label이 들어있는 Line
        continue
    line = line.split('\t')
    test_sentence.append(okt.morphs(line[1].strip()))
    test_label.append(line[2].strip())

file.close()

all_sentence = sentence + test_sentence         # train에 비해 test 문장이 더 길면 모델 오류가 나므로 통합적으로 관리

max_len = max([len(i) for i in all_sentence])

vocab = set()

for line in all_sentence:
    for word in line:
        vocab.add(word)
        
vocab_size = len(vocab) + 1
vocab = sorted(list(vocab))

vocab_index = {}

for i in range(len(vocab)):
    vocab_index[vocab[i]] = len(vocab_index)+1

X_train = []
for line in sentence:
    etc = []
    for word in line:
        etc.append(vocab_index[word])
    X_train.append(etc)

Y_train = []
for line in test_sentence:
    etc = []
    for word in line:
        etc.append(vocab_index[word])
    Y_train.append(etc)

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

X_train = pad_sequences(X_train, padding = 'post', maxlen = max_len)
Y_train = pad_sequences(Y_train, padding = 'post', maxlen = max_len)

X_train, X_cv, train_label, cv_label = train_test_split(X_train, label, test_size = 0.1)

from keras.utils.np_utils import to_categorical

train_label = to_categorical(train_label)
cv_label = to_categorical(cv_label)
test_label = to_categorical(test_label)

from keras.layers import Input, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping

input_sentence = Input(shape=(max_len,))

emd = Embedding(vocab_size, 100)(input_sentence)

lstm = LSTM(128, return_sequences=True)(emd)

conv1d = Conv1D(32, 3, activation='relu', strides=1)(lstm)

GMP = GlobalMaxPooling1D()(conv1d)

output = Dense(2, activation='softmax')(GMP)

model = Model(inputs=[input_sentence], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=2, restore_best_weights=True)

model.summary()

model.fit(X_train, train_label, batch_size=256, epochs=30, validation_data=(X_cv, cv_label), callbacks=[es])

evaluation = model.evaluate(Y_train, test_label)

print('Accuracy: '+str(evaluation[1]))
print('Loss: '+str(evaluation[0]))

y_pred = model.predict([Y_train])
Y_pred = y_pred.round()

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(test_label, axis=1)

t = confusion_matrix(y_test, y_pred)

print(classification_report(
    y_test, y_pred,
    target_names=["0","1"]
))