#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


double euclidean_distance(double *point1, double *point2, int d) {
    double sum = 0;
    int i;
    double result;
    for (i = 0; i < d; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    result = sqrt(sum);
    return result;
}

void free_2d_double_array(double **array, int len) {
    int j;
    for (j = 0; j < len; j++) {
        free(array[j]);
    }
    free(array);
}

void free_3d_double_array(double ***array, int len1, int len2) {
    int i,j;
    for(i=0; i<len1; i++){
        for (j = 0; j < len2; j++) {
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}



void update_centroid(double **cluster, double *centroid, int cluster_size, int d) {
    int i;
    int j;

    for (i = 0; i < d; i++) {
        centroid[i] = 0.0;
    }
    
    for (i = 0; i < cluster_size; i++) {
        for (j = 0; j < d; j++) {
            centroid[j] += cluster[i][j];
        }
    }
    for (i = 0; i < d; i++) {
        centroid[i] /= cluster_size;
    }
}


void k_means(double **vectors, int n, int d, int k, int iter, double eps, double **centroids, int *cluster_assignments) {

    int i;
    int converged;
    int it;
    int j;
    double ** new_centroids;
    double *** clusters;
    int *cluster_sizes;

    // for (i = 0; i < k; i++) {
    //     for (j = 0; j < d; j++) {
    //         centroids[i][j] = vectors[i][j];
    //     }
    // }

    new_centroids = (double **)malloc(k * sizeof(double *));
    clusters = (double ***)malloc(k * sizeof(double **));
    cluster_sizes = (int *)malloc(k * sizeof(int));
    for (i = 0; i < k; i++) {
        new_centroids[i] = (double *)malloc(d * sizeof(double));
        clusters[i] = (double **)malloc(n * sizeof(double *));
    }

    for (it = 0; it < iter; it++) {
        for (i = 0; i < k; i++) {
            cluster_sizes[i] = 0;
        }


        for (i = 0; i < n; i++) {
            double min_d = euclidean_distance(vectors[i], centroids[0], d);
            int min_ind = 0;
            for (j = 1; j < k; j++) {
                double dist = euclidean_distance(vectors[i], centroids[j], d);
                if (dist < min_d) {
                    min_d = dist;
                    min_ind = j;
                }
            }
            cluster_assignments[i] = min_ind;
            clusters[min_ind][cluster_sizes[min_ind]] = vectors[i];
            cluster_sizes[min_ind] += 1;
        }


        for (i = 0; i < k; i++) {
            if (cluster_sizes[i] > 0) {
                update_centroid(clusters[i], new_centroids[i], cluster_sizes[i], d);
            } else {

                for (j = 0; j < d; j++) {
                    new_centroids[i][j] = centroids[i][j];
                }
            }
        }


         converged = 1;
        for (i = 0; i < k; i++) {
            if (euclidean_distance(new_centroids[i], centroids[i], d) > eps) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            break;
        }


        for (i = 0; i < k; i++) {
            for (j = 0; j < d; j++) {
                centroids[i][j] = new_centroids[i][j];
            }
        }
    }



    free_2d_double_array(new_centroids, k);
    free_2d_double_array((double **)clusters, k);
    free(cluster_sizes);
}

static PyObject* fit(PyObject* self, PyObject* args) {
    PyObject *centroids_list, *vectors_list;
    int k, iter, n, d;
    double eps;
    double **centroids;
    double **vectors;

    int i, j;

    int *cluster_assignments;
    PyObject *result;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "OOiidii", &centroids_list, &vectors_list, &k, &iter, &eps, &n, &d)) {
        return NULL;
    }

    // Allocate memory for C arrays
    centroids = (double **)malloc(k * sizeof(double *));
    vectors = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < k; i++) centroids[i] = (double *)malloc(d * sizeof(double));
    for (i = 0; i < n; i++) vectors[i] = (double *)malloc(d * sizeof(double));

    // Populate the centroids array
    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(centroids_list, i) ,j));
        }
    }

    // Populate the vectors array
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            vectors[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(vectors_list, i) ,j));
        }
    }

    

    // Allocate memory for the cluster assignments
    cluster_assignments = (int *)malloc(n * sizeof(int));


    k_means(vectors, n, d, k, iter, eps, centroids, cluster_assignments);

    

    // For simplicity, we'll return the same centroids we received.
    result = PyList_New(k*d);
    for (i = 0; i < k; i++) {
        for (j = 0; j < d; j++) {
            PyList_SetItem(result, i*d + j, PyFloat_FromDouble(centroids[i][j]));
        }
    }

    

    

    // Free allocated memory
    
    for (i = 0; i < k; i++) free(centroids[i]);
    for (i = 0; i < n; i++) free(vectors[i]);
    free(centroids);
    free(vectors);
    free(cluster_assignments);
    return result;
}



static PyMethodDef kmeansMethods[] = {
    {"fit",                   /* the Python method name that will be used */
      fit, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parameters
accepted for this function */
      PyDoc_STR("Finds the centroids and clusters according to KMeans algorithm")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};


static struct PyModuleDef mykmeansspmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    kmeansMethods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&mykmeansspmodule);
    if (!m) {
        return NULL;
    }
    return m;
}

// int main(int argc, char *argv[]) {
//     int k;
//     int iterations;
//     int i;
//     int j;
//     double **data;
//     int *cluster_assignments;
//     double **centroids;
//     int l;
//     int columns;
//     FILE *file;
//     int lines;
//
//     if (argc < 2 || argc > 3) {
//         printf("An Error Has Occured!\n");
//         return 1;
//     }
//
//     if (!is_int_string(argv[1])) {
//         printf("Invalid number of clusters!\n");
//         return 1;
//     }
//
//     k = atoi(argv[1]);
//
//     if (argc == 2) {
//         iterations = 200;
//     } else {
//         if (!is_int_string(argv[2])) {
//             printf("Invalid number of iterations!\n");
//             return 1;
//         }
//         iterations = atoi(argv[2]);
//         if (iterations >= 1000 || iterations <= 1) {
//             printf("Invalid number of iterations!\n");
//             return 1;
//         }
//     }
//
//     file = stdin;
//
//     lines = line_counter(file);
//     rewind(file);
//     columns = column_counter(file);
//     rewind(file);
//
//     if (k < 1 || k > lines) {
//         printf("Invalid number of clusters!\n");
//         fclose(file);
//         return 1;
//     }
//
//     data = malloc(lines * sizeof(double *));
//     if(data == NULL) {
//         printf("An Error Has Occured");
//         fclose(file);
//         return 1;
//     }
//     for (i = 0; i < lines; i++) {
//         data[i] = malloc(columns * sizeof(double));
//         if(data[i] == NULL) {
//             printf("An Error Has Occured");
//             free_2d_double_array(data, i);
//             fclose(file);
//             return 1;
//         }
//     }
//
//
//     for (l = 0; l < lines; l++) {
//         for (j = 0; j < columns; j++) {
//             fscanf(file, "%lf,", &data[l][j]);
//         }
//     }
//
//     fclose(file);
//
//
//     centroids = malloc(k * sizeof(double *));
//     if(centroids == NULL) {
//         printf("An Error Has Occured");
//         free_2d_double_array(data, lines);
//         return 1;
//     }
//     for (i = 0; i < k; i++) {
//         centroids[i] = malloc(columns * sizeof(double));
//         if(centroids[i] == NULL) {
//             printf("An Error Has Occured");
//             free_2d_double_array(centroids, i);
//             free_2d_double_array(data, lines);
//             return 1;
//         }
//     }
//
//
//     cluster_assignments = (int *)malloc(lines * sizeof(int));
//     if(cluster_assignments == NULL) {
//         printf("An Error Has Occured");
//         free_2d_double_array(centroids, k);
//         free_2d_double_array(data, lines);
//         return 1;
//     }
//
//
//
//     k_means(data, lines, columns, k, iterations, 0.001, centroids, cluster_assignments);
//
//
// for (i = 0; i < k; i++) {
//     for (j = 0; j < columns; j++) {
//         if (j == columns - 1) {
//             printf("%.4f", centroids[i][j]);
//         } else {
//             printf("%.4f,", centroids[i][j]);
//         }
//     }
//     printf("\n");
// }
//
//
//     free_2d_double_array(data, lines);
//     free_2d_double_array(centroids, k);
//     free(cluster_assignments);
//
//     return 0;
// }
