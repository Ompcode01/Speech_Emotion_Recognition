import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio, display

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Base directory for datasets
base_dir = r"C:\Users\iampr\OneDrive\Desktop\speechrecognition"
Ravdess = os.path.join(base_dir, "Ravdess")
Crema = os.path.join(base_dir, "Crema")
Tess = os.path.join(base_dir, "Tess")
Savee = os.path.join(base_dir, "Savee")

# Check if paths exist
for path in [base_dir, Ravdess, Crema, Tess, Savee]:
    print(f"Path exists: {path} - {os.path.exists(path)}")

# Function to process RAVDESS dataset
def process_ravdess(ravdess_path):
    ravdess_directory_list = os.listdir(ravdess_path)
    file_emotion = []
    file_path = []
    
    for dir in ravdess_directory_list:
        actor_dir = os.path.join(ravdess_path, dir)
        if os.path.isdir(actor_dir):
            for file in os.listdir(actor_dir):
                try:
                    part = file.split('.')[0].split('-')
                    if len(part) >= 3:
                        file_emotion.append(int(part[2]))
                        file_path.append(os.path.join(actor_dir, file))
                    else:
                        print(f"Skipping file due to unexpected format: {file}")
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
    
    if not file_emotion or not file_path:
        print("No valid files found in the RAVDESS dataset.")
        return pd.DataFrame()

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    ravdess_df.Emotions.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
    return ravdess_df

# Function to process CREMA dataset
def process_crema(crema_path):
    crema_directory_list = os.listdir(crema_path)
    file_emotion = []
    file_path = []
    
    for file in crema_directory_list:
        part = file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')
        
        file_path.append(os.path.join(crema_path, file))
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)
    return crema_df

# Function to process TESS dataset
def process_tess(tess_path):
    tess_directory_list = os.listdir(tess_path)
    file_emotion = []
    file_path = []
    
    for dir in tess_directory_list:
        directories = os.path.join(tess_path, dir)
        for file in os.listdir(directories):
            part = file.split('.')[0].split('_')[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(os.path.join(directories, file))
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    tess_df = pd.concat([emotion_df, path_df], axis=1)
    return tess_df

# Function to process SAVEE dataset
def process_savee(savee_path):
    savee_directory_list = os.listdir(savee_path)
    file_emotion = []
    file_path = []
    
    for file in savee_directory_list:
        file_path.append(os.path.join(savee_path, file))
        part = file.split('_')[1]
        if part == 'a':
            file_emotion.append('angry')
        elif part == 'd':
            file_emotion.append('disgust')
        elif part == 'f':
            file_emotion.append('fear')
        elif part == 'h':
            file_emotion.append('happy')
        elif part == 'n':
            file_emotion.append('neutral')
        elif part == 'sa':
            file_emotion.append('sad')
        else:
            file_emotion.append('surprise')
    
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    savee_df = pd.concat([emotion_df, path_df], axis=1)
    return savee_df

# Process all datasets
ravdess_df = process_ravdess(Ravdess)
crema_df = process_crema(Crema)
tess_df = process_tess(Tess)
savee_df = process_savee(Savee)

# Combine all datasets
data_path = pd.concat([ravdess_df, crema_df, tess_df, savee_df], axis=0)
data_path.to_csv("data_path.csv", index=False)
print(data_path.head())

# Function to create waveplot
def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

# Function to create spectrogram
def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

# Example usage for various emotions
for emotion in ['fear', 'angry', 'sad','happy']:
    emotion_data = data_path[data_path.Emotions == emotion]
    
    if len(emotion_data) < 2:
        print(f"Not enough data for emotion '{emotion}'.")
    else:
        # Load the second occurrence of the emotion
        path = emotion_data.Path.iloc[1]
        data, sampling_rate = librosa.load(path)
        
        # Create waveplot and spectrogram
        create_waveplot(data, sampling_rate, emotion)
        create_spectrogram(data, sampling_rate, emotion)
        
        # Play audio
        display(Audio(path))
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    noisy_data = data + noise_amp * np.random.normal(size=data.shape[0])
    return noisy_data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)  # Shifting in milliseconds
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


# Example usage with augmentation
emotion = 'fear'  # Change this to test with different emotions
emotion_data = data_path[data_path.Emotions == emotion]

if len(emotion_data) < 2:
    print(f"Not enough data for emotion '{emotion}'.")
else:
    # Load the second occurrence of the emotion
    path = emotion_data.Path.iloc[1]
    data, sample_rate = librosa.load(path)

    # Apply augmentations
    noisy_data = noise(data)
    stretched_data = stretch(data, rate=0.8)
    shifted_data = shift(data)
    pitched_data = pitch(data, sample_rate, pitch_factor=0.7)

    # Plot original and augmented waveforms
    plt.figure(figsize=(14, 8))

    plt.subplot(5, 1, 1)
    plt.title("Original Waveform")
    librosa.display.waveshow(data, sr=sample_rate)

    plt.subplot(5, 1, 2)
    plt.title("Noisy Waveform")
    librosa.display.waveshow(noisy_data, sr=sample_rate)

    plt.subplot(5, 1, 3)
    plt.title("Stretched Waveform")
    librosa.display.waveshow(stretched_data, sr=sample_rate)

    plt.subplot(5, 1, 4)
    plt.title("Shifted Waveform")
    librosa.display.waveshow(shifted_data, sr=sample_rate)

    plt.subplot(5, 1, 5)
    plt.title("Pitched Waveform")
    librosa.display.waveshow(pitched_data, sr=sample_rate)

    plt.tight_layout()
    plt.show()

    # Play the original audio
    print("Playing original audio:")
    display(Audio(path))

# Define the function to extract features
def extract_features(data, sample_rate):
    # Initialize an empty array for the result
    result = np.array([])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result

def get_features(path):
    # Load the audio data
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Extract features without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array([res1])  # Convert to 2D array for vertical stacking
    
    # Data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically
    
    # Data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically
    
    return result

# Prepare the dataset for features and labels
X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)  # Append emotion for each feature extracted

# Convert X to a numpy array for DataFrame creation
X = np.array(X)

# Create a DataFrame and save to CSV
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)

# Display the first few rows of the DataFrame
print(Features.head())


X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=7, activation='softmax'))  # Changed from 8 to 7 units

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Verify data shapes
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history=model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)


df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(10)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test, y_pred))

model.save('speech_emotion_model.h5')
