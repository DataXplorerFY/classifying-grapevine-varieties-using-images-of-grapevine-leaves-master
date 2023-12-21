import os
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import numpy as np














# import os
# import shutil

def transfer_images(source_folders, destination_folder):
    """
    Transfer all images from source_folders to the destination_folder.

    Args:
        source_folders (list): List of source folder paths.
        destination_folder (str): Destination folder path.

    Returns:
        None
    """
    for folder in source_folders:
        files = os.listdir(folder)
        for file in files:
            src = os.path.join(folder, file)
            shutil.move(src, destination_folder)

# Example usage:
# source_folders = ['/content/Data/negative', '/content/Data/positive']
# destination_folder = '/content/train/'

# transfer_images(source_folders, destination_folder)





# Generating the numbers, corresponding to quantities of images which will be transferred to train, validation and test splits,, for each class

def numbers_of_images_in_splits(total_images_per_class, train_percentage, validation_percentage, test_percentage):
    splits = {}

    for class_label, num_images in total_images_per_class.items():
        num_train = int(num_images * train_percentage)
        num_validation = int(num_images * validation_percentage)
        num_test = int(num_images * test_percentage)

        # Calculate any remaining images that were not assigned due to rounding
        remaining = num_images - (num_train + num_validation + num_test)

        # Distribute remaining images to the training set to minimize waste
        num_train += remaining

        splits[class_label] = {
            'train': num_train,
            'validation': num_validation,
            'test': num_test
        }

    return splits










# import os
# import shutil
# import random

def transfer_images_to_destinations(source_folder, class_to_destinations):
    """
    Transfer images from the source folder to specified destinations according to the class_to_destinations mapping.

    Args:
        source_folder (str): Path to the source folder containing the images.
        class_to_destinations (dict): A dictionary mapping class labels to destination directories.
        split_results (dict): A dictionary containing the number of images to transfer for each class and split.

    Returns:
        None
    """
    # A dictionary to keep track of transferred images
    transferred_images = {}

    # Iterating through each class and copying the specified number of unique images to the destinations
    for class_label, destinations in class_to_destinations.items():
        class_images = [img for img in os.listdir(source_folder) if class_label in img]

        for destination, num_images in destinations:
            if len(class_images) < num_images:
                print(f"Warning: Not enough images with class label '{class_label}' in the source folder for destination '{destination}'.")
                continue

            # Shuffle the images to select a random subset
            random.shuffle(class_images)

            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination):
                os.makedirs(destination)

            # Copy the specified number of unique images to the destination
            transferred_images[class_label] = transferred_images.get(class_label, set())
            images_to_copy = [img for img in class_images if img not in transferred_images[class_label]][:num_images]

            for img_filename in images_to_copy:
                img_source_path = os.path.join(source_folder, img_filename)
                img_destination_path = os.path.join(destination, img_filename)
                shutil.copy(img_source_path, img_destination_path)
                transferred_images[class_label].add(img_filename)

            print(f"{len(images_to_copy)} images from class '{class_label}' copied to '{destination}'.")
        print()

# Example usage:
# source_folder = '/content/train'

# class_to_destinations = {
#     'positive': [
#         ('/content/lung cancer dataset/train/positive',       split_results['positive']['train']),
#         ('/content/lung cancer dataset/validation/positive',  split_results['positive']['validation']),
#         ('/content/lung cancer dataset/test/positive',        split_results['positive']['test']),
#     ],
#     'negative': [
#         ('/content/lung cancer dataset/train/negative',       split_results['negative']['train']),
#         ('/content/lung cancer dataset/validation/negative',  split_results['negative']['validation']),
#         ('/content/lung cancer dataset/test/negative',        split_results['negative']['test']),
#     ],
# }


# transfer_images_to_destinations(source_folder, class_to_destinations, split_results)










# You can use it to delete any folder along with all of it's subfolders on kaggle
# See the last two lines, if you want, you can just delete the sub folders and spare the main folder itself?

# import shutil

# # Define the root folder where the folders to be deleted are located
# root_folder = '/content/Grapevine_Leaves_Image_Dataset/Ala_Idris/cp-0002.ckpt_temp'

# # Walk through the root folder and delete all subfolders and their contents
# for foldername, subfolders, filenames in os.walk(root_folder, topdown=False):
#     for subfolder in subfolders:
#         folder_path = os.path.join(foldername, subfolder)
#         shutil.rmtree(folder_path)

# # Finally, delete the root folder itself (optional)
# shutil.rmtree(root_folder)











def create_class_to_destinations(class_names, directories, split_results):
    """
    Create a dictionary-of-lists-of-tuples that maps class names to destination directories
    and their respective split results.

    Args:
        class_names (list): List of class names.
        directories (list of lists): List of lists of directories for each class.
        split_results (dict): Dictionary of split results for each class.

    Returns:
        dict: Dictionary-of-lists-of-tuples representing class_to_destinations.
    """
    class_to_destinations = {}  # Initialize the output dictionary

    # Iterate through class names, directories, and split results simultaneously
    for class_name, dir_list, split in zip(class_names, directories, split_results.values()):
        class_to_destinations[class_name] = [
            (dir, split[split_type]) for dir, split_type in zip(dir_list, ['train', 'validation', 'test'])
        ]

    return class_to_destinations


# manually you would have to do the following
# class_to_destinations = {
#     'Ak': [
#         (train_Ak_dir,       split_results['Ak']['train']),
#         (validation_Ak_dir,  split_results['Ak']['validation']),
#         (test_Ak_dir,        split_results['Ak']['test']),
#     ],
#     'Ala_Idris': [
#         (train_Ala_Idris_dir,       split_results['Ala_Idris']['train']),
#         (validation_Ala_Idris_dir,  split_results['Ala_Idris']['validation']),
#         (test_Ala_Idris_dir,        split_results['Ala_Idris']['test']),
#     ],

#     'Buzgulu': [
#         (train_Buzgulu_dir,       split_results['Buzgulu']['train']),
#         (validation_Buzgulu_dir,  split_results['Buzgulu']['validation']),
#         (test_Buzgulu_dir,        split_results['Buzgulu']['test']),
#     ],

#     'Dimnit': [
#         (train_Dimnit_dir,       split_results['Dimnit']['train']),
#         (validation_Dimnit_dir,  split_results['Dimnit']['validation']),
#         (test_Dimnit_dir,        split_results['Dimnit']['test']),
#     ],

#     'Nazli': [
#         (train_Nazli_dir,       split_results['Nazli']['train']),
#         (validation_Nazli_dir,  split_results['Nazli']['validation']),
#         (test_Nazli_dir,        split_results['Nazli']['test']),
#     ],
#                       }












# import os

def create_directories(class_names, base_dir):
    """
    Create a list-of-lists-of-directories for each class and split (train, validation, test).

    Args:
        class_names (list): List of class names.
        base_dir (str): Base directory where the class-specific directories will be created.

    Returns:
        list of lists: List-of-lists-of-directories for each class and split.
    """
    directories = []

    for class_name in class_names:
        class_dirs = []  # Create a list to hold directories for the current class

        for split_type in ['train', 'validation', 'test']:
            # Construct the directory path for the current class and split
            dir_path = os.path.join(base_dir, split_type, class_name)
            class_dirs.append(dir_path)

        directories.append(class_dirs)  # Add the class directories to the main list

    return directories
# Manually you would have to do the following by hand:

# train_Ak_dir         = '/content/grapevine_leaves/train/Ak'
# train_Ala_Idris_dir  = '/content/grapevine_leaves/train/Ala_Idris'
# train_Buzgulu_dir    = '/content/grapevine_leaves/train/Buzgulu'
# train_Dimnit_dir     = '/content/grapevine_leaves/train/Dimnit'
# train_Nazli_dir      = '/content/grapevine_leaves/train/Nazli'

# validation_Ak_dir        = '/content/grapevine_leaves/validation/Ak'
# validation_Ala_Idris_dir = '/content/grapevine_leaves/validation/Ala_Idris'
# validation_Buzgulu_dir   = '/content/grapevine_leaves/validation/Buzgulu'
# validation_Dimnit_dir    = '/content/grapevine_leaves/validation/Dimnit'
# validation_Nazli_dir     = '/content/grapevine_leaves/validation/Nazli'

# test_Ak_dir        = '/content/grapevine_leaves/test/Ak'
# test_Ala_Idris_dir = '/content/grapevine_leaves/test/Ala_Idris'
# test_Buzgulu_dir   = '/content/grapevine_leaves/test/Buzgulu'
# test_Dimnit_dir    = '/content/grapevine_leaves/test/Dimnit'
# test_Nazli_dir     = '/content/grapevine_leaves/test/Nazli'

# directories = [
#     [train_Ak_dir, validation_Ak_dir, test_Ak_dir],
#     [train_Ala_Idris_dir, validation_Ala_Idris_dir, test_Ala_Idris_dir],
#     [train_Buzgulu_dir, validation_Buzgulu_dir, test_Buzgulu_dir],
#     [train_Dimnit_dir, validation_Dimnit_dir, test_Dimnit_dir],
#     [train_Nazli_dir, validation_Nazli_dir, test_Nazli_dir]
# ]








def plot_training_history(history):
    """
    Plot training and validation accuracy and loss from a Keras training history.

    Args:
        history (dict): A Keras training history containing 'acc', 'val_acc', 'loss', and 'val_loss' keys.

    Returns:
        None
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plotting training and validation accuracy
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    # Plotting training and validation loss
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

# Example usage:
# history = {
#     'acc': [0.6, 0.7, 0.8, 0.9],
#     'val_acc': [0.5, 0.6, 0.7, 0.8],
#     'loss': [0.5, 0.4, 0.3, 0.2],
#     'val_loss': [0.6, 0.5, 0.4, 0.3]
# }










# import os
# import pandas as pd
# import matplotlib.pyplot as plt

def generate_evaluation_results(
    model, generators, weights_folder, initial_weight_index=0
):
    """
    Generate evaluation results for a set of weights on multiple datasets.

    Args:
        model: A TensorFlow model to evaluate.
        generators: A list of dataset generators, each containing a tuple
            (label, generator) where label is a string and generator is a data generator.
        weights_folder: The path to the folder containing model weights files.
        initial_weight_index: The index of the initial weight to start evaluation.

    Returns:
        df: A pandas DataFrame containing evaluation results.
        checkpoint_names: A list of last two characters of checkpoint names.

    """
    results = {}

    # Get a list of model weights files in the specified folder
    weight_files = sorted([f for f in os.listdir(weights_folder) if f.endswith('.keras')])

    # Handle negative initial_weight_index by ensuring it doesn't exceed the number of available weights
    if initial_weight_index < 0:
        initial_weight_index = max(0, len(weight_files) + initial_weight_index)

    # Select weights starting from the initial_weight_index
    weight_files = weight_files[initial_weight_index:]

    checkpoint_names = []  # List to store last two characters of checkpoint names

    # Loop through each weight file
    for weight_file in weight_files:
        weight_path = os.path.join(weights_folder, weight_file)
        model.load_weights(weight_path)

        metrics = {}
        # Evaluate the model on each dataset and record loss and accuracy
        for dataset_name, generator in generators:
            loss, acc = model.evaluate(generator)
            metrics[dataset_name + ' Loss'] = loss
            metrics[dataset_name + ' Acc'] = acc

        weight_name = os.path.splitext(weight_file)[0]
        checkpoint_names.append(weight_name[-2:])  # Extract the last two characters
        results[weight_name] = metrics

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    df = df.transpose()

    return df, checkpoint_names























# import os
# import pandas as pd
# import plotly.express as px

def plot_evaluation_results(df, checkpoint_names, generators, figure_height=600, figure_width=1000):
    """
    Plot evaluation results for different checkpoints and datasets interactively using Plotly.

    Args:
        df: A pandas DataFrame containing evaluation results.
        checkpoint_names: A list of last two characters of checkpoint names.
        generators: A list of dataset generators, each containing a tuple
            (label, color) where label is a string and color is the line color.
        figure_height: Height of the Plotly figure.
        figure_width: Width of the Plotly figure.

    """
    # Define a list of colors by their names
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    # This creates a list of colors that will be used for different datasets. These colors will be assigned to lines on the plot.

    # Create a Plotly figure for the interactive plot
    fig = px.line()
    # This initializes an empty Plotly figure for creating an interactive line plot.

    
    # Iterate over dataset generators and metric suffixes
    for i, (label, color) in enumerate(generators):
        for suffix in [' Loss', ' Acc']:
            metric = label + suffix
            linestyle = 'dash' if 'Loss' in suffix else 'solid'
    # These nested loops iterate over each dataset generator and each metric suffix (' Loss' and ' Acc').
    # For each combination, it extracts the label and color, then constructs a metric name and linestyle based on the suffix.

            # Add traces for each metric to the figure
            fig.add_scatter(x=checkpoint_names, y=df[metric], name=metric,
                            line=dict(color=colors[i], dash=linestyle))
            # This line adds a trace to the figure. A trace represents a single line on the plot.
            # It specifies the x-axis data (checkpoint names),
            # y-axis data (metric values from the DataFrame), trace name (metric name), and line properties (color and linestyle).

    
    # Configure layout of the figure
    fig.update_layout(
        xaxis_title='Checkpoint Names (Last Two Characters)',
        yaxis_title='Loss / Accuracy',
        title='Model Evaluation Results',
        showlegend=True,
        height=figure_height,  # Set the height of the figure
        width=figure_width    # Set the width of the figure
    )

    # Display the interactive plot in the Jupyter Notebook or in a web browser
    fig.show()

# Example usage
# generators = [('Train', train_data_generator), ('Validation', validation_data_generator), ('Test', test_data_generator)]
# df, checkpoint_names = generate_evaluation_results(model, generators, folder_of_weights, initial_weight_index=2)
# plot_evaluation_results(df, checkpoint_names, generators, figure_height=600, figure_width=1100)













# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def classification_report_and_confusion_matrix(true_labels, predicted_labels):
    """
    Calculate and display a classification report and a confusion matrix.

    Parameters:
    true_labels (array-like): Ground truth (correct) target labels.
    predicted_labels (array-like): Predicted target labels.

    Returns:
    None: The function prints the classification report and displays the confusion matrix.
    """
    # Generating the classification report
    report = classification_report(true_labels, predicted_labels)
    print(report)
    
    # Generating and displaying the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    disp        = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                         display_labels=['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli'])
    disp.plot(cmap='Blues')




# import pandas as pd
# import numpy as np

def probabilities_to_dataframe(probabilities_array, label_replacements):
    """
    Convert a 2D NumPy array of probabilities into a well-structured DataFrame.

    Args:
        probabilities_array (numpy.ndarray): A 2D array with rows representing different samples
            and columns representing class probabilities.
        label_replacements (dict): A dictionary to specify replacements for class labels.

    Returns:
        pandas.DataFrame: A DataFrame with columns for class probabilities, predicted labels, and
        the maximum normalized probability for each sample.

    Example:
            >>> label_replacements = {0: 'Ak', 1: 'Ala_Idris', 2: 'Buzgulu', 3: 'Dimnit', 4: 'Nazli'}
            >>> probabilities_array = np.array([[0.2, 0.2, 0.2, 0.1, 0.3], [0.2, 0.2, 0.2, 0.3, 0.1], [0.2, 0.2, 0.3, 0.1, 0.2]])
            >>> result_df = probabilities_to_dataframe(probabilities_array, label_replacements)
            >>> result_df
    """

    # Creating a DataFrame from the probabilities_array
    class_labels = [f'prob_of_being_{label_replacements[i]}' for i in range(len(label_replacements))]
    df = pd.DataFrame(probabilities_array, columns=class_labels)

    # Normalizing the probabilities in each row
    df = df.div(df.sum(axis=1), axis=0)

    # Adding a 'Predicted_Label' column with the index of the class with the highest probability
    df['predicted_label'] = df.idxmax(axis=1)
 
    # Using the map method to perform the replacements
    df['predicted_label'] = df['predicted_label'].str.replace('prob_of_being_', '')

    # Adding a 'Probability_of_Predicted_Class' column with the value of the highest predicted probability
    df['Probability_of_Predicted_Class'] = df[class_labels].max(axis=1)

    df.index.name = 'sample_no.'

    return df

# # Example usage:
# label_replacements = {0: 'Ak', 1: 'Ala_Idris', 2: 'Buzgulu', 3: 'Dimnit', 4: 'Nazli'}
# probabilities_array = np.array([[0.2, 0.2, 0.2, 0.1, 0.3], [0.2, 0.2, 0.2, 0.3, 0.1], [0.2, 0.2, 0.3, 0.1, 0.2]])
# result_df = probabilities_to_dataframe(probabilities_array, label_replacements)
# result_df



def classification_probability_displayer(probabilities, true_labels, label_replacements):
    """
    Display classification statistics and probabilities based on input probabilities and true labels.

    Args:
        Probabilities (list or array): Predicted class probabilities for each sample.
        True_Labels (list or array): True class labels for each sample.
        label_replacements (dictionary): e.g {0: 'quasar', 1: 'galaxy', & so on...}

    Returns:
        None
    """
    # Converting probabilities to a DataFrame
    data_frame_for_probabilities_of_current_model = probabilities_to_dataframe(probabilities, label_replacements)
    
    # Adding true class labels to the DataFrame
    data_frame_for_probabilities_of_current_model['true_class'] = true_labels
    
    # Defining a dictionary to specify the replacements for class labels
    label_replacements = {0: 'Ak', 1: 'Ala_Idris', 2: 'Buzgulu', 3: 'Dimnit', 4: 'Nazli'}
    
    # Using the map method to perform the replacements for true class labels
    data_frame_for_probabilities_of_current_model['true_class'] = data_frame_for_probabilities_of_current_model['true_class'].map(label_replacements)
    
    # Grouping the DataFrame by predicted labels
    preds_groups_df = data_frame_for_probabilities_of_current_model.groupby('predicted_label')

    # Grouping the DataFrame by true class
    true_groups_df  = data_frame_for_probabilities_of_current_model.groupby('true_class')    
    
    # Calculating the total number of samples
    total_no_of_samples = len(true_labels)
    print(f'total_no_of_samples: {total_no_of_samples}\n')

    # extracting a list of names of classes from the dictionary label_replacemnets
    names_of_classes = []
    for i in label_replacements.values():
        names_of_classes.append(i)

    # Calculating and displaying statistics for each class
    class_statistics_dict = {}
    for i in names_of_classes:
        try:
            no_of_samples_of_current_class  =  len(true_groups_df.get_group(i))
        except:
            no_of_samples_of_current_class  = 0
        try:
            class_statistics_dict[f'predicted_{i}_df']  =  preds_groups_df.get_group(i)
        except:
            class_statistics_dict[f'predicted_{i}_df']  =  0
        try:
            predicted_percentage_of_current_class  =  (len(class_statistics_dict[f'predicted_{i}_df']) / total_no_of_samples) * 100
        except:
            predicted_percentage_of_current_class = 0
        try:
            true_percentage_of_current_class       =  (no_of_samples_of_current_class / total_no_of_samples) * 100
        except:
            true_percentage_of_current_class = 0
            
        print(f'no_of_samples_of_class_{i}:   {no_of_samples_of_current_class}')
        print(f'predicted_percentage_of_{i}:  {predicted_percentage_of_current_class}%')
        print(f'true_percentage_of_{i}:       {true_percentage_of_current_class}\n')

    # Function to calculate statistics for misclassified and correctly classified samples
    def calculate_statistics(class_df, label):
        mask = class_df['predicted_label'] != class_df['true_class']
        indices = preds_groups_df.get_group(label).index[mask].tolist()
        no_of_samples_misclassified = len(indices)
        print(f'no_of_samples_misclassified_as_{label}: {no_of_samples_misclassified}')

        for threshold in [0.6, 0.7, 0.8, 0.9, 0.99]:
            if no_of_samples_misclassified == 0:
                samples_misclassified_with_probability = 0  # Set to 0 or another appropriate value
            else:
                samples_misclassified_with_probability = (np.sum(class_df.loc[indices]['Probability_of_Predicted_Class'] > threshold) / no_of_samples_misclassified) * 100
            print(f'samples_misclassified_as_{label}_with_probability_greater_than_{threshold}: {samples_misclassified_with_probability}')

        
        mask = class_df['predicted_label'] == class_df['true_class']
        indices = preds_groups_df.get_group(label).index[mask].tolist()
        no_of_samples_correctly_classified = len(indices); print()
        print(f'no_of_samples_correctly_classified_as_{label}: {no_of_samples_correctly_classified}')

        for threshold in [0.6, 0.7, 0.8, 0.9, 0.99]:
            if no_of_samples_correctly_classified == 0:
                samples_correctly_classified_with_probability = 0  # Set to 0 or another appropriate value
            else:
                samples_correctly_classified_with_probability = (np.sum(class_df.loc[indices]['Probability_of_Predicted_Class'] > threshold) / no_of_samples_correctly_classified) * 100
            print(f'samples_correctly_classified_as_{label}_with_probability_greater_than_{threshold}: {samples_correctly_classified_with_probability}')
        print('\n\n')

    # Calculating and displaying statistics for misclassified and correctly classified CLASSES
    for i in names_of_classes:
        if i not in preds_groups_df.groups:
            print(f'Zero samples were classified as {i} \n')
        else:
            calculate_statistics(class_statistics_dict[f'predicted_{i}_df'], i)


# label_replacements = {0: 'Ak', 1: 'Ala_Idris', 2: 'Buzgulu', 3: 'Dimnit', 4: 'Nazli'}
# probabilities_array = np.array([
#     [0.2, 0.2, 0.2, 0.1, 0.3], [0.2, 0.2, 0.2, 0.1, 0.3], [0.2, 0.2, 0.2, 0.3, 0.1], [0.2, 0.2, 0.3, 0.1, 0.2],
#     [0.3, 0.2, 0.2, 0.1, 0.2], [0.2, 0.3, 0.2, 0.1, 0.2], [0.2, 0.2, 0.2, 0.1, 0.3], [0.2, 0.2, 0.2, 0.3, 0.1],
#     [0.2, 0.2, 0.3, 0.1, 0.2], [0.2, 0.3, 0.2, 0.1, 0.2]
# ])

# True_Labels = [4, 4, 3, 2, 0, 1, 4, 3, 2, 1]

# a = classification_probability_displayer(probabilities_array, True_Labels, label_replacements)
# a









