import subprocess
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from time import time
def run_r_script(script_path):
    try:
        subprocess.run(['Rscript', script_path], check=True)
        #print("R script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing R script: {e}")

def C2CModelExpansion(coefficientsdf, theta_input):

    #Get the order from the number of coefficients
    order = int((len(coefficientsdf)-1)/2)
    
    cos_theta_pred = coefficientsdf.iloc[0,0] + np.sum([coefficientsdf.iloc[i,0]*np.cos(i*theta_input) + coefficientsdf.iloc[i+order,0]*np.sin(i*theta_input) for i in range(1, order+1)], axis=0)

    sin_theta_pred = coefficientsdf.iloc[0,1] + np.sum([coefficientsdf.iloc[i,1]*np.cos(i*theta_input) + coefficientsdf.iloc[i+order,1]*np.sin(i*theta_input) for i in range(1, order+1)], axis=0)
    
    theta_pred = np.arctan2(sin_theta_pred, cos_theta_pred)
    theta_pred = theta_pred + 2*np.pi*(theta_pred<0)

    return theta_pred, order

def getNullCase(vector):
    # Compute the frequency of each value
    frequency_counts = Counter(vector)
    
    # Total number of elements in the vector
    total_elements = len(vector)
    
    # Compute normalized frequencies
    normalized_frequencies = {key: value / total_elements for key, value in frequency_counts.items()}
    
    # Sum the square of these normalized frequencies
    sum_of_squares = sum(freq ** 2 for freq in normalized_frequencies.values())
    
    return sum_of_squares

N_DOWNSAMPLING = 1
NResamples = 1
num_folds = 3
N_REPETITIONS = 1
time_start = time()
for down in range(N_DOWNSAMPLING):
    
    print(f"Downsampling {down+1}...")
    # Import data
    CurrentFixationAngles = np.genfromtxt("./CCREG/current_fixation_angleSorted.csv", delimiter=',', skip_header=1)
    NextFixationAngles = np.genfromtxt("./CCREG/next_fixation_angleSorted.csv", delimiter=',', skip_header=1)
    
    # Downsample the data to 1/3
    CurrentFixationAngles = CurrentFixationAngles[::NResamples]
    NextFixationAngles = NextFixationAngles[::NResamples]
    
    # Digitize angles into 8 bins between 0 and 2π
    CurrentFixationAnglesClass = np.digitize(NextFixationAngles, np.linspace(0, 2*np.pi, 9)) - 1
    #Null case for the three models
    #CurrentFixationAnglesNullCase = getNullCase(CurrentFixationAnglesClass)

    # Set the number of folds
    kf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=N_REPETITIONS, random_state=42)
    
    accuracy_dict = {
        "NextFixation": [],
    }
    
    confusion_dict = {
        "NextFixation": np.zeros((8, 8)),
    }

    order_dict = {
        "NextFixation": [],
    }


    for angles, labels, output_prefix in [
        (CurrentFixationAngles, CurrentFixationAnglesClass, "NextFixation"),
    ]:
        print(f"PROCESSING {output_prefix}...")

        #Null case for this model
        for train_index, test_index in kf.split(NextFixationAngles, labels):
            #print("     TRAIN:", train_index, "TEST:", test_index)
            #print("     FOLD...")
            X_train, X_test = NextFixationAngles[train_index], NextFixationAngles[test_index]
            y_train, y_test = angles[train_index], angles[test_index]

            #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            #quit()
            # Save the data
            np.savetxt(f"./CCREG/independent_angle_train_fold.csv", X_train, delimiter=",")
            np.savetxt(f"./CCREG/dependent_angle_train_fold.csv", y_train, delimiter=",")
        
            # Run the R script
            run_r_script('your_script.R')
            
            # Load the coefficients
            coefficientsdf = pd.read_csv("./CCREG/fit_fold.csv")
            coefficientsdf.drop(coefficientsdf.columns[0], axis=1, inplace=True)
            y_pred, order_pred = C2CModelExpansion(coefficientsdf, X_test)
            
            # Discretize the predicted and test values to 8 bins between 0 and 2π
            y_pred = np.digitize(y_pred, np.linspace(0, 2*np.pi, 9)) - 1
            y_test = np.digitize(y_test, np.linspace(0, 2*np.pi, 9)) - 1
            
            # Calculate the accuracy
            accuracy = np.mean(y_pred == y_test)
            
            # Update the confusion matrix
            for i in range(len(y_pred)):
                confusion_dict[output_prefix][y_pred[i], y_test[i]] += 1
            
            accuracy_dict[output_prefix].append(accuracy)
            #accuracystd_dict[output_prefix].append(np.std(accuracy))
            # Save the order
            order_dict[output_prefix].append(order_pred)
        
    print(f"Downsampling {down+1} completed in {time()-time_start} seconds.")

# Save the average accuracy and the standard deviation for each angle set
for key in accuracy_dict:
    #ACCURACY
    accuracies = np.array(accuracy_dict[key])
    print(f"{key} - Average accuracy: {np.mean(accuracies)}")
    print(f"{key} - Standard deviation: {np.std(accuracies)}")
    np.savetxt(f"./CCREG/accuracy_{key}.csv", accuracies, delimiter=",")
    #np.savetxt(f"./CCREG/accuracy_{key}_std.csv", np.array(accuracystd_dict[key]), delimiter=",")
    
    #ORDER
    orders = np.array(order_dict[key])
    print(f"{key} - Average order: {np.mean(orders)}")
    print(f"{key} - Standard deviation: {np.std(orders)}")
    np.savetxt(f"./CCREG/order_{key}.csv", orders, delimiter=",")
    #CONFUSION MATRIX
    confusion_matrix = confusion_dict[key]
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # To avoid division by zero
    normalized_confusion_matrix = confusion_matrix / row_sums
    np.savetxt(f"./CCREG/confusion_{key}.csv", confusion_matrix, delimiter=",")
    # Plot the confusion matrix
    plt.imshow(confusion_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f"Confusion Matrix for {key}")
    plt.savefig(f"./CCREG/confusion_{key}.png")
    plt.clf()

print("Processing completed in", time()-time_start, "seconds.")
