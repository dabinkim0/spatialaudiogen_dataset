import os
import subprocess

directory = '/home/dabean/spatialaudiogen/spatialaudiogen/data/orig/rec-street'

files = os.listdir(directory)
file_dict = {}


for file in files:
    # Webm to wav
    # if file.endswith('.webm'):
    #     webm_path = os.path.join(directory, file)
    #     wav_path = os.path.join(directory, file.replace('.webm', '.wav'))
    #     command = ['ffmpeg', '-i', webm_path, 
    #                '-af', 'pan=stereo|FL=FL|FR=FR',
    #                '-acodec', 'pcm_s16le', '-ac', '2', wav_path]
    #     subprocess.run(command)

    # Extract YouTube ID
    if file.endswith('.wav') or file.endswith('.mp4'):
        youtube_id = file.split('.')[0]
        file_type = file.split('.')[-1]

        if youtube_id not in file_dict:
            file_dict[youtube_id] = {}

        file_dict[youtube_id][file_type] = file

# Comnine audio and video files
for youtube_id, file_types in file_dict.items():
    audio_file = file_types.get('wav')
    video_file = file_types.get('mp4')
    if audio_file and video_file:
        
        # Set paths and output file name
        audio_path = os.path.join(directory, audio_file)
        video_path = os.path.join(directory, video_file)
        output_path = os.path.join(directory + '/merged', f"{youtube_id}.merged.mp4")
        
        # ffmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-ac', '2',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        
        # Run command
        subprocess.run(command)

print("Merging complete.")