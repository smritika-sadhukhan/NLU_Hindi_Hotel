
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

with open('/content/drive/My Drive/tokenizer.pickle', 'rb') as handle:
    word2id = pickle.load(handle)

with open('/content/drive/My Drive/word_vec.pkl','rb') as f:
    word_vec = pickle.load(f)
  


CNN1=tf.keras.layers.Conv1D(300, 3, activation='relu', name='CNN1', padding='same')
lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 100, 
                                                     return_sequences = True, 
                                                     recurrent_dropout = 0.5), name='lstm1')
lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, 
                                                    return_sequences= False,
                                                    recurrent_dropout = 0.5), name='lstm2')
dense = tf.keras.layers.Dense(7,activation='softmax',name='Dense')

embeddings =  tf.keras.Input((50, 300), dtype="float32")
X=CNN1(embeddings)
X = lstm1(X)
X = lstm2(X)
X = dense(X)
bar_model=tf.keras.models.Model(inputs=embeddings, outputs=X)

model1 = bar_model
model1.load_weights('NLU_Hindi_Hotel/Hindi_HOTEL_NLU/Model/NLU_iNTENT_HOTEL.h5')
model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




def predict_intent(data):
  X_test1=[]
  for i in data:
    list1=[]
    for j in i.split():
      list1.append(word2id.get(j,word2id['unknown']))
    X_test1.append(list1)


  vect = pad_sequences(X_test1, padding='post', maxlen=50)
  list2=[]
  for j in range(0,len(vect)):
    list1=[]
    for i in vect[j]:
      list1.append(list(word_vec[i]))
    list2.append(list1)
  arr = np.array(list2) 
  my_prediction = model1.predict(arr)
  predicted_y =np.argmax(my_prediction, axis=1)

  for i in range(0,len(predicted_y)):
    print(data[i])
    print(max(list(my_prediction[0])))
    if predicted_y[i] == 0:
      print('--> Affirmation')
    elif predicted_y[i] == 1:
      print('--> Affirmation_Payment')
    elif predicted_y[i] == 2:
      print('--> HotelDetail')
    elif predicted_y[i] == 3:
      print('--> HotelSearch')
    elif predicted_y[i] == 4:
       print('--> HotelSearch_Filter')
    elif predicted_y[i] == 5:
      print('--> Negation')
    elif predicted_y[i] == 6:
      print('--> Negation_Payment')
    print(' ')

