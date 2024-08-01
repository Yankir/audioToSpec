import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa

# carrega audio
y1, sr1 = librosa.load("origAudio.wav")
y2, sr2 = librosa.load("fakeAudio.wav")

# MEL-SPECTROGRAM
mel1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
mel2 = librosa.feature.melspectrogram(y=y2, sr=sr2)

# CHROMAGRAM STFT
chroma_stft = librosa.feature.chroma_stft(y=y1, sr=sr1)

# CHROMAGRAM CQT
chroma_cqt = librosa.feature.chroma_cqt(y=y1, sr=sr1)

# MFCC
mfcc = librosa.feature.mfcc(y=y1, sr=sr1)


# plot dos espectrogramas

# mel
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(mel1, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr1, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel - Áudio original')
plt.savefig("spectrograms/mel_og")

# mel fake
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(mel2, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr2, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel - Áudio fake')
plt.savefig("spectrograms/mel_fake")

# chroma stft
fig, ax = plt.subplots()
img = librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Chroma STFT')
plt.savefig("spectrograms/chroma_stft")

# chroma cqt
fig, ax = plt.subplots()
img = librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Chroma CQT')
plt.savefig("spectrograms/chroma_cqt")

# mfcc
fig, ax = plt.subplots()
img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')
plt.savefig("spectrograms/mfcc")


# chroma stft-cqt
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax[0])
ax[0].set(ylabel='Chroma STFT')
ax[0].label_outer()
librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time', ax=ax[1])
ax[1].set(ylabel='Chroma CQT')
plt.savefig("spectrograms/chroma_stft_cqt")

# mel og-fake
fig, ax = plt.subplots(nrows=2, sharex=False)
img = librosa.display.specshow(librosa.power_to_db(mel1, ref=np.max),
                               x_axis='time', y_axis='mel',
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]], format='%+2.0f dB')
ax[0].set(title='Mel Original')
ax[0].label_outer()
img = librosa.display.specshow(librosa.power_to_db(mel2, ref=np.max),
                               x_axis='time', y_axis='mel',
                               ax=ax[1])
fig.colorbar(img, ax=[ax[1]], format='%+2.0f dB')
ax[1].set(title='Mel Fake')
plt.savefig("spectrograms/mel_og_fake")

# multiplot
# fig, ax = plt.subplots(nrows=2, sharex=True)
# img = librosa.display.specshow(librosa.power_to_db(mel1, ref=np.max),
#                                x_axis='time', y_axis='mel',
#                                ax=ax[0])
# fig.colorbar(img, ax=[ax[0]])
# ax[0].set(title='Mel Spectrogram')
# ax[0].label_outer()
# img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
# fig.colorbar(img, ax=[ax[1]])
# ax[1].set(title='MFCC')
# plt.savefig("spectrograms/mel-mfcc")
