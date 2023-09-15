import os
import csv

# Set the directory where the text files are located
directory_path = 'results/'

# Create a list of dictionaries to hold the data from each file
data_list = []

# Loop through all the text files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        # Open the file and read the contents
        with open(os.path.join(directory_path, filename)) as file:
            contents = file.readlines()
            # print(contents)

        # Extract the data from the file contents
        model_name = contents[1].strip().split('-')[0]
        composition = contents[1].strip().split('-')[1]
        precision = contents[2].split(': ')[1]
        f1_score = contents[3].split(': ')[1]
        iou = contents[4].split(': ')[1]
        time = contents[5].split(': ')[1]
        mem = contents[6].split(': ')[1]

        # Create a dictionary to hold the data from this file
        data_dict = {
            'Modelo': model_name,
            'Composição': composition,
            'Precisão': precision,
            'F1 Score': f1_score,
            'mIoU': iou,
            'Tempo': time,
            'Memória': mem
        }

        # Add the dictionary to the list
        data_list.append(data_dict)

# Create a CSV file and write the data to it
with open('results-FromScratch.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file,
                            fieldnames=[
                                'Modelo', 'Composição', 'Precisão', 'F1 Score',
                                'mIoU', 'Tempo', 'Memória'
                            ])
    writer.writeheader()
    writer.writerows(data_list)
