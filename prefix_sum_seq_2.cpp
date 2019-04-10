#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

int main (int argc, char *argv[]) {
    int arr_len = 8;
    int *arr = new int[8];
    srand(time(NULL));

    cout << "Array values: " << endl;
    for (int i = 0; i < arr_len; i++) {
        arr[i] = rand() % 11;
        cout << arr[i] << ", ";
    }
    cout << endl;

    for (int offset = 0; offset < log2(arr_len); offset++) {
        int start = pow(2, offset);
        for (int i = start; i < arr_len; i += pow(2, offset + 1)) {
            arr[i] = arr[i - 1];
            for (int j = i; j < i + offset; j++) {
                cout << j << endl;
                arr[j] += arr[j - 1];
            }
        }
    }

    cout << "Prefix sum: " << endl;
    for (int i = 0; i < arr_len; i++) {
        cout << arr[i] << ", ";
    }
    cout << endl;
}
