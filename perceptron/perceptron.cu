#include "wtime.h"
#include "reader.hpp"
#include "./error_handler.h"

#define DIMENSION 61 

/*__global__ void init_update(float *update_w, float update_b, int size)
{
    for(int i = 0; i < size; i ++)
        update_w[i] = 0;
    update_b = 0;
}*/

__global__ void perceptron_kernel(float *W, float &b, float *data, int count, int dimension, int epoch) {
    int sample_count = count / dimension;

    //W = new float[dimension - 1];
    b = 0;

    for(int i = 0; i < dimension - 1; i++)
        W[i] = 0;


    //batch updates
    float *update_w = new float[dimension -1];
    float update_b;


    for(int trial = 0; trial < epoch; trial ++)
    {
        //init_update(update_w, update_b, dimension - 1);
        for(int i = 0; i < dimension - 1; i ++)
            update_w[i] = 0;


        for(int i = 0; i < count/dimension; i ++)
        {
            float predict = 0;
            float expected = data[i*dimension + dimension - 1];

            //make prediction
            for(int j = 0; j < dimension - 1; j ++)
                predict += W[j] * data[i*dimension +j] + b;
            
            //apply sign function
            if (predict < 0) predict = -1;
            else predict = 1;
            
            //batch the updates to the model
            if (predict != expected) 
            {
                for(int j = 0; j < dimension - 1; j ++)
                    update_w[j] += (data[i*dimension +j] * expected/sample_count);
                update_b += (expected/sample_count);
            }
        }
        //apply updates to the model
        for(int j = 0; j < dimension - 1; j ++)
            W[j] += update_w[j];
        b += update_b;
    }
    delete[] update_w;
}

void predict(float* &W, float &b, float *data, int count, int dimension)
{
    for(int i = 0; i < count/dimension; i++)
    {
        float predict = 0;
        float expected = data[i*dimension + dimension - 1];

        for(int j = 0; j < dimension - 1; j ++)
            predict += W[j] * data[i*dimension + j] + b;

        if (predict < 0) predict = -1;
        else predict = 1;

        std::cout<<"Expect "<<expected<<", predict "<<predict<<"\n";
    }
}

int main(int args, char **argv)
{

    std::cout<<"/path/to/exe epoch\n";

    const int EPOCH = atoi(argv[1]);

    assert(args == 2);
    float *train_data;
    int train_count;

    float *test_data;
    int test_count;
    
    float *weights = new float[DIMENSION - 1];
    float b;

    reader("dataset/train_data.bin", train_data, train_count);
    reader("dataset/test_data.bin", test_data, test_count);
    
    //printer(train_dataset, train_count, DIMENSION);
    //printer(test_dataset, test_count, DIMENSION);

    float *train_d, weights_d;
    HANDLE_ERR(cudaMalloc((void **) &train_d, sizeof (float) * train_count));
    HANDLE_ERR(cudaMalloc((void **) &weights_d, sizeof (float) * (DIMENSION - 1)));

    HANDLE_ERR(cudaMemcpy (train_d, train_data, sizeof (float) * train_count, cudaMemcpyHostToDevice));

    perceptron_kernel <<128, 128>> (weights_d, b, train_d, train_count, DIMENSION, EPOCH);

    HANDLE_ERR(cudaMemcpy (weights, weights_d, sizeof (float) * (DIMENSION - 1), cudaMemcpyDeviceToHost));
    predict(weights, b, test_data, test_count, DIMENSION);
    
    printer(weights, DIMENSION - 1, 1);
    return 0;
}

