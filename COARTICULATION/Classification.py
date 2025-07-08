import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
import pickle as pk
import os



UTILITIES_PATH = "/Users/Mattia/Desktop/PAPER2/UTILITIES/"
with open(UTILITIES_PATH + "subjNames.txt", "r") as f:
    subjNames = f.read().splitlines()
with open(UTILITIES_PATH + "problemList.txt", "r") as f:
    problemList = f.read().splitlines()


pca = pk.load(open("./OUTPUT/pca_model.pkl",'rb'))
nComponents = 10

# Initialize dictionaries to store results
accuracyMeanDict = {}
accuracyStdDict = {}
confusion_matrixDict = {}
lda_coefficientsDict = {}
num_downsamples = 30  # Number of downsampling iterations

# Loop over all cases
for RESTRICTED_CASE, WITHIN_CASE in [(True, True), (True, False), (False, True), (False, False)]:
    case = "WITHIN" if WITHIN_CASE else "BETWEEN"
    if RESTRICTED_CASE:
        case += "_RESTRICTED"
    print("CASE:", case)

    #IMPORT THE CASE-SPECIFIC DATA
    ################################################################################################################################
    # Define file paths
    file_paths = {
        "final_result_matrices": f"./OUTPUT/{case}_final_result_matrices.npy",
        "final_class_labels": f"./OUTPUT/{case}_final_class_labels.npy",
        "final_trial_indices": f"./OUTPUT/{case}_final_trial_indices.npy",
        "final_label_fixes": f"./OUTPUT/{case}_final_label_fixes.npy",
        "final_subj_names": f"./OUTPUT/{case}_final_subj_names.npy",
        "final_eyePosition": f"./OUTPUT/{case}_final_eyePosition.npy"
    }
    
    # Check and load data
    for key, path in file_paths.items():
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            if key == "final_result_matrices":
                final_result_matrices = data
            elif key == "final_class_labels":
                final_class_labels = data
            elif key == "final_trial_indices":
                final_trial_indices = data
            elif key == "final_label_fixes":
                final_label_fixes = data
            elif key == "final_subj_names":
                final_subj_names = data
            elif key == "final_eyePosition":
                final_eyePosition = data
        else:
            print(f"File {path} does not exist. Skipping this file.")
            continue
    
    #### DOWNSAMPLING ################################################################################################################################
    # Count the occurrences of each class
    class_counts = {label: np.sum(final_class_labels == label) for label in set(final_class_labels)}
    # Calculate the target count for downsampling to the smallest class count
    target_count = min(class_counts.get("N", 0), class_counts.get("E", 0), class_counts.get("W", 0))
    # If any class is missing, skip downsampling
    if target_count == 0:
        print(f"Skipping downsampling for case {case} due to missing class")
        continue

    # Downsample the data to balance classes
    downsampled_indices = []
    for label in ["N", "E", "W"]:
        indices = np.where(final_class_labels == label)[0]
        if len(indices) >= target_count:
            downsampled_indices.extend(np.random.choice(indices, size=target_count, replace=False))

    # Convert the list of indices to a numpy array
    downsampled_indices = np.array(downsampled_indices)

    # Create the balanced dataset using the downsampled indices
    final_class_labels = final_class_labels[downsampled_indices]
    final_result_matrices = final_result_matrices[downsampled_indices]
    final_trial_indices = final_trial_indices[downsampled_indices]
    final_label_fixes = final_label_fixes[downsampled_indices]
    final_subj_names = final_subj_names[downsampled_indices]
    final_eyePosition = final_eyePosition[downsampled_indices]


    HAND_ds_mean_accuracies = []
    EYE_ds_mean_accuracies = []

    HAND_all_confusion_matrices = []
    EYE_all_confusion_matrices = []

    ################################################################################################################################
    ### DOWNSAMPLING ###
    ################################################################################################################################
    for _ in range(num_downsamples):
        print(f"Downsample iteration {_ + 1}/{num_downsamples}")

        # Count the occurrences of each class
        class_counts = {label: np.sum(final_class_labels == label) for label in set(final_class_labels)}
        # Calculate the target count for downsampling to the smallest class count
        target_count = min(class_counts.get("N", 0), class_counts.get("E", 0), class_counts.get("W", 0))
        # If any class is missing, skip downsampling
        if target_count == 0:
            print(f"Skipping downsampling for case {case} due to missing class")
            continue

        # Downsample the data to balance classes
        downsampled_indices = []
        for label in ["N", "E", "W"]:
            indices = np.where(final_class_labels == label)[0]
            if len(indices) >= target_count:
                downsampled_indices.extend(np.random.choice(indices, size=target_count, replace=False))

        # Convert the list of indices to a numpy array
        downsampled_indices = np.array(downsampled_indices)

        # Create the balanced dataset using the downsampled indices
        ds_class_labels = final_class_labels[downsampled_indices]
        ds_result_matrices = final_result_matrices[downsampled_indices]
        ds_eyePosition = final_eyePosition[downsampled_indices]

        ################################################################################################################################
        ### CLASSIFICATION: HAND ###
        ################################################################################################################################

        # Perform PCA on the result matrices
        pca_result = pca.transform(ds_result_matrices)[:, :nComponents]

        # Initialize LDA model for classification
        modelTraj = LinearDiscriminantAnalysis()

        # Initialize cross-validation
        kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=100, random_state=42)

        X = pca_result
        y = ds_class_labels

        # Initialize confusion matrix for hand current downsample
        cm_total = np.zeros((len(np.unique(y)), len(np.unique(y))))

        # Perform cross-validation
        HAND_kfold_accuracies = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train and predict using LDA
            modelTraj.fit(X_train, y_train)
            y_pred = modelTraj.predict(X_test)
            
            # Update confusion matrix
            cm_fold = confusion_matrix(y_test, y_pred)
            cm_total += cm_fold

            # Store accuracy for this fold
            fold_accuracy = modelTraj.score(X_test, y_test)
            HAND_kfold_accuracies.append(fold_accuracy)
        
        # Mean accuracy and standard deviation of this downsample
        HAND_ds_mean_accuracies += HAND_kfold_accuracies

        # Store confusion matrix for this downsample
        HAND_all_confusion_matrices.append(cm_total)

        ################################################################################################################################
        ### CLASSIFICATION: EYE ###
        ################################################################################################################################

        X_eye = ds_eyePosition
        y_eye = ds_class_labels

        # Encode class labels into numerical values
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_eye)

        # Initialize LDA model for eye position data
        model_eye = LinearDiscriminantAnalysis()

        # Initialize confusion matrix for current downsample
        cm_total_eye = np.zeros((len(np.unique(y_encoded)), len(np.unique(y_encoded))))

        # LDA COEFFICIENTS
        modelTraj.fit(X, y)
        lda_coefficientsDict[case] = modelTraj.coef_


        # Perform cross-validation
        EYE_kfold_accuracies = []
        for train_index, test_index in kf.split(X_eye, y_encoded):
            X_train_eye, X_test_eye = X_eye[train_index], X_eye[test_index]
            y_train_eye, y_test_eye = y_encoded[train_index], y_encoded[test_index]

            # Train and predict using LDA
            model_eye.fit(X_train_eye, y_train_eye)
            y_pred_eye = model_eye.predict(X_test_eye)

            # Update confusion matrix
            cm_fold_eye = confusion_matrix(y_test_eye, y_pred_eye)
            cm_total_eye += cm_fold_eye

            # Store accuracy for this fold
            fold_accuracy_eye = model_eye.score(X_test_eye, y_test_eye)
            EYE_kfold_accuracies.append(fold_accuracy_eye)

        # Store accuracies and confusion matrix for this downsample
        EYE_ds_mean_accuracies += EYE_kfold_accuracies
    
        EYE_all_confusion_matrices.append(cm_total_eye)


    ################################################################################################################################
    ### AGGREGATE RESULTS ###
    ################################################################################################################################
    # Calculate the mean and standard deviation of accuracies across all downsamples for HAND
    accuracyMeanDict[case], accuracyStdDict[case] = np.mean(HAND_ds_mean_accuracies), np.std(HAND_ds_mean_accuracies)
    # Store final results in dictionaries
    accuracyMeanDict[case + "_EYE"], accuracyStdDict[case + "_EYE"] = np.mean(EYE_ds_mean_accuracies), np.std(EYE_ds_mean_accuracies)

    # Aggregate confusion matrices across all downsamples for HAND
    aggregate_cm = np.sum(HAND_all_confusion_matrices, axis=0)

    # Aggregate confusion matrices across all downsamples for EYE
    aggregate_cm_eye = np.sum(EYE_all_confusion_matrices, axis=0)
    
    # Store confusion matrices
    confusion_matrixDict[case] = aggregate_cm
    confusion_matrixDict[case + "_EYE"] = aggregate_cm_eye


# Save results to file
with open("./OUTPUT/accuracyMeanDict" + str(nComponents) +".pkl", "wb") as f:
    pk.dump(accuracyMeanDict, f)
with open("./OUTPUT/accuracyStdDict" + str(nComponents) +".pkl", "wb") as f:
    pk.dump(accuracyStdDict, f)
with open("./OUTPUT/confusion_matrixDict" + str(nComponents) +".pkl", "wb") as f:
    pk.dump(confusion_matrixDict, f)
with open("./OUTPUT/lda_coefficientsDict" + str(nComponents) +".pkl", "wb") as f:
    pk.dump(lda_coefficientsDict, f)