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
        int x = pow(2, offset);
        for (int i = arr_len - 1; i > x - 1; i--)
            arr[i] += arr[i - x];
    }

    cout << "Prefix sum: " << endl;
    for (int i = 0; i < arr_len; i++) {
        cout << arr[i] << ", ";
    }
    cout << endl;
}
