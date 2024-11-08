
# K-means++ Clustering Project

This project implements the K-means++ algorithm for selecting initial centroids, integrated with the K-means clustering algorithm from a previous assignment. The K-means++ initialization enhances the K-means algorithm by improving the choice of initial centroids, which can lead to better convergence.

## Project Structure

- **kmeans_pp.py**: Python interface for the project, including command-line arguments, data reading, K-means++ initialization, and output.
- **kmeansmodule.c**: C extension containing the K-means algorithm with initial centroids determined by K-means++.
- **setup.py**: Setup file for building the C extension for Python.

## Getting Started

### Prerequisites

This project requires the following:
- **Python 3**
- **NumPy**: For matrix operations.
- **Pandas**: For data handling.
- **C Compiler**: GCC or similar.

### Installation

1. Build the C extension with the following command:
   ```bash
   python3 setup.py build_ext --inplace
   ```

2. Ensure the C components compile without errors using:
   ```bash
   make
   ```

### Usage

To run the K-means++ algorithm and generate clustering results, use the following command structure:

```bash
python3 kmeans_pp.py <K> <iter> <eps> <file_name_1> <file_name_2>
```

- **K**: Number of required clusters.
- **iter**: Maximum iterations for K-means (default is 300).
- **eps**: Convergence threshold.
- **file_name_1**: Path to the first data file (.txt or .csv).
- **file_name_2**: Path to the second data file (.txt or .csv).

Example:
```bash
python3 kmeans_pp.py 3 100 0.01 input_1.txt input_2.txt
```

The program outputs:
1. Indices of initial centroids selected by K-means++.
2. Final centroids calculated by K-means, formatted to four decimal places.


## Assumptions & Notes

1. Data points are unique.
2. Outputs are formatted to four decimal places (e.g., `%.4f`).
3. Error handling:
   - Invalid input results in an appropriate message as per assignment instructions.
   - Other errors result in "An Error Has Occurred" and termination.
4. Use of double in C and float in Python for vector elements.
5. The `key` (first column) is used only as an identifier, not as a clustering feature.

## Building & Running

To build and run the project:
1. Run `setup.py` as mentioned.
2. Ensure `make` runs without errors or warnings.

## References

This project is based on the K-means++ initialization for K-means clustering, which helps improve initial centroid selection.

