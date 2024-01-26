import os

directory = '/home/dabean/spatialaudiogen/spatialaudiogen/data/orig/yt-clean'

for file in os.listdir(directory):
    if file.endswith('.wav'):
        os.remove(os.path.join(directory, file))

print("All .wav files have been removed.")