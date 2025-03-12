import numpy as np
from scipy.stats import mode  # For calculating the mode

data = np.load("/Users/dongbaiqi/Desktop/higp-main/data/fish_raw_data.npz")

# Extract the relevant attributes
spks = data['spks']  # Shape: (26370, 7890)
xyz = data['xyz']  # Shape: (26370, 3)
stims = data['stims']  # Shape: (7890,)

# Step 1: Extract specified experimental intervals
experiments_indices = [(99, 899), (2700, 3500), (5300, 6100)]
spks_extracted = []
stims_extracted = []

for start, end in experiments_indices:
    spks_extracted.append(spks[:, start:end])
    extracted_labels = stims[start:end]
    if start == 2700 and end == 3500:  # Replace label 15 with 2 in this segment
        extracted_labels[extracted_labels == 15] = 2
    stims_extracted.append(extracted_labels)

spks = np.hstack(spks_extracted)  # Shape: (100, n_timepoints)
stims = np.hstack(stims_extracted)  # Shape: (n_timepoints,)

# Step 2: Downsample the number of neurons
reduced_neurons = 50  # Keep only the first 100 neurons
spks = spks[:reduced_neurons, :]  # Shape: (100, n_timepoints)
xyz_reduced = xyz[:reduced_neurons, :]  # Shape: (100, 3)

# Step 3: Calculate the Euclidean distance matrix for the reduced neurons
distance_matrix = np.sqrt(
    np.sum((xyz_reduced[:, np.newaxis, :] - xyz_reduced[np.newaxis, :, :]) ** 2, axis=-1)
)  # Shape: (100, 100)

# Save the distance matrix to npz format
output_distance_path = './data/distance_matrix.npy'  # Replace with the actual path
np.save(output_distance_path, distance_matrix)

print(f"Reduced distance matrix saved to {output_distance_path}.")

# Step 4: Divide spks and stims into 10 subjects
num_subjects = 10
subject_length = spks.shape[1] // num_subjects

fold_psg = []  # List to store spks for each subject
fold_label = []  # List to store labels for each subject
fold_len = []  # List to store the number of epochs per subject

for subject_idx in range(num_subjects):
    # Get the subject-specific spks and stims
    spks_subject = spks[:, subject_idx * subject_length:(subject_idx + 1) * subject_length]
    stims_subject = stims[subject_idx * subject_length:(subject_idx + 1) * subject_length]

    # Divide each subject's data into 24 epochs
    epoch_length = subject_length // 24
    spks_epochs = np.array(np.split(spks_subject, 24, axis=1))  # Shape: (24, 100, epoch_length)
    stims_epochs = np.array(np.split(stims_subject, 24))  # Shape: (24, epoch_length)

    # Calculate the label for each epoch (mode of stims)
    labels = mode(stims_epochs, axis=1).mode.flatten()  # Shape: (24,)

    # One-hot encode the labels
    num_classes = 3  # Assuming classes are 0, 1, 2
    one_hot_labels = np.eye(num_classes)[labels]  # Shape: (24, num_classes)

    # Append results for this subject
    fold_psg.append(spks_epochs)  # Shape: (24, 100, epoch_length)
    fold_label.append(one_hot_labels)  # Shape: (24, num_classes)
    fold_len.append(len(labels))  # Should always be 24


# Save to npz file
output_path = './data/fish_data.npz'  # Replace with the actual path
np.savez(output_path,
         Fold_data=fold_psg,
         Fold_label=fold_label,
         Fold_len=fold_len)

print(f"Processed data has been saved to {output_path}.")