Wine Classification using KNN

This project implements a K-Nearest Neighbors (KNN) classification algorithm to classify wine types based on various features. The performance of the algorithm is compared using two distance metrics: Euclidean Distance and Manhattan Distance.

Requirements

pandas

numpy

matplotlib

scikit-learn

To install these dependencies, run the following command in your terminal:

pip install pandas numpy matplotlib scikit-learn

Description
Dataset: The wine dataset is loaded from a CSV file, containing multiple features such as Alcohol, Malic Acid, Ash, etc., along with the class (wine type).

Preprocessing: The features are normalized using MinMaxScaler to scale them between 0 and 1.

Model: The KNN algorithm is implemented using custom distance functions (euclidean_distance and manhattan_distance).

Accuracy Comparison: The accuracy is calculated for k values ranging from 1 to 20, and results are plotted for both distance functions.

Confusion Matrix: The confusion matrix for the best-performing model (k=5 and Euclidean distance) is printed.

Usage

Replace the path in wine.data with the correct file location on your machine.

Run the script, and it will plot two graphs comparing the accuracy of KNN for both distance functions.

The confusion matrix for the best model will be printed after the comparison.

Output

Two accuracy plots showing how performance varies with different values of k.

A confusion matrix of the final model.