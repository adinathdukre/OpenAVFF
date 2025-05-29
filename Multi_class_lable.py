import os
import csv

# Set the path to your train folder
train_dir = '/l/PathoGen/Adinath/Train_Test_split/val'
output_csv = 'val_videos_labels_MC.csv'

# Define the folder-to-label mapping
folder_to_label = {
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7  # folder 9 → label 7
}

# Collect all video paths and labels
data = []

for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path) and folder in folder_to_label:
        label = folder_to_label[folder]
        for file in os.listdir(folder_path):
            if file.endswith('.mp4'):
                video_path = os.path.join(folder_path, file)
                data.append([video_path, label])

# Write to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['video_name', 'target'])
    writer.writerows(data)

print(f"CSV file saved as {output_csv} with {len(data)} entries.")
