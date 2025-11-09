

!pip install numpy librosa scikit-learn tensorflow matplotlib

#
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#
def extract_features(file_path, n_mfcc=40, n_fft=2048, hop_length=512):
    try :
        audio, sr = librosa.load(file_path, sr=None)
        segment_length = int(sr * 3)
        mfcc_list = []

        for start in range(0, len(audio) - segment_length + 1, segment_length):
            segment = audio[start:start + segment_length]
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            if mfcc.shape[0] == 130:
                mfcc_list.append(mfcc)
                break
        return mfcc_list[0] if mfcc_list else None
    except:
        return None

X, y = [], []
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for genre in genres:
    genre_path = f"Data/genres_original/{genre}"
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            features = extract_features(os.path.join(genre_path, file))
            if features is not None:
                X.append(features)
                y.append(genre)
X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} samples")



#
le = LabelEncoder()
y_encoded = to_categorical(le.fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y)

# Converted to RNN model
model = Sequential([
    SimpleRNN(256, return_sequences=True, input_shape=(130,40)),
    BatchNormalization(),
    Dropout(0.3),
    SimpleRNN(128),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, 130, 40))
model.summary()

early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# 6. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# 7. Plot Accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 8. Save Model
model.save('music_genre_rnn.keras')
print("Model saved as 'music_genre_rnn.keras'")


#
def predict_genre(file_path, model, le):
    mfcc_blocks = extract_features(file_path)

    if mfcc_blocks is None or len(mfcc_blocks) == 0:
        return "No features"

    mfcc = mfcc_blocks
    x = np.expand_dims(mfcc, axis=0)
    prob = model.predict(x, verbose=0)[0]
    idx = np.argmax(prob)
    genre = le.inverse_transform([idx])[0]
    confidence = prob[idx]
    return f"{genre} (confidence: {confidence:.2%})"
    
sample_file = "blues.00000.wav"      
print(sample_file,"->",predict_genre(sample_file, model, le))