import pandas as pd

import os

textData = {
    'filePath': [],
    'transcript': []
}

directory_path = '/home/qbromley/corover/Speech-Diarization-Whisper-Project-1/Custom_Training_Set'

# Iterating through the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    if (os.path.isdir(file_path) and filename[:9] == "corrected" ):
        textFile_path = file_path+"/transcript.txt"
        
        lines = []
        with open(textFile_path, 'r') as file:
            lines = file.readlines()

        for files in os.listdir(file_path):
            if(files.endswith(".wav")):
                if(files[0:2] == "1_"):
                    continue
                if(files[1] == "_"):
                    textData['filePath'].append(files)
                    if(lines[int(files[0])-1].endswith("\n")  ):
                        textData['transcript'].append(lines[int(files[0])-1][0:-1])
                    else:
                        textData['transcript'].append(lines[int(files[0])-1])
                     
                else: 
                    textData['filePath'].append(files)
                    if(lines[int(files[0])-1].endswith("\n")):
                        textData['transcript'].append(lines[int(files[:2])-1][0:-1])
                    else:
                        textData['transcript'].append(lines[int(files[:2])-1])





df = pd.DataFrame(textData)

df.to_csv('trainingAndTesting.csv', index=False)

input_file_path = 'trainingAndTesting.csv'

df = pd.read_csv(input_file_path)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(0.8 * len(df))

df_80 = df[:split_index]
df_20 = df[split_index:]

df_80.to_csv('training.csv', index=False)
df_20.to_csv('testing.csv', index=False)