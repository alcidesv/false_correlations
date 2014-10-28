
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

template <typename T>
__host__ __device__ T& pitch_get(T* origin, int bytes_pitch, int x, int y)
{
    return *(T*)((uint8_t*)origin + bytes_pitch*y + x*sizeof(T));
}


__device__ void apply_boolean_func(
    // Columns to use..
    uint8_t column_count,
    uint32_t column_mask,
    
    // ... in this data
    uint16_t row_count,
    uint32_t *rows,
    
    // ... with this function 
    // definition.... this is good for 
    // up  to five columns!
    uint32_t function_definition,
    
    // Put the output here
    uint8_t *output_results
    );

extern "C" __global__ void calc_spearman_rho(
    uint16_t  row_count,
    float*    scalar_column_ranks, // has row_count elements...

    uint32_t *rows, // The block of left columns. Each integer
                    // is interpreted as bits, each bit is a separate
                    // column. So this array can hold up to 32 columns.
                    // I'm using only five.

                    // The functions
    uint16_t function_count,
    uint32_t* function_definitions, // See `apply_boolean_func` to 
                                    // understand how each function fits
                                    // inside a single integer

    uint8_t* output_results,  // Properly strided matrix . Each row here contains
                              // the result of computing a function.
                              // So this matrix has row_count _columns_ and 
                              // function_count _rows_.... a bit confusing.  
    size_t results_pitch,     // 

    // Columns to try the functions on
    uint32_t column_mask,

    // Total number of columns of boolean data at the left
    uint16_t column_count,

    // Output: array with one element per function 
    double* rho_values
    )
{
    //printf("Entering\n");
    uint16_t function_index = threadIdx.y + blockIdx.y * blockDim.y;
    if (function_index >= function_count )
    {
        // Not really
        return ;
    }
    
    uint32_t function_definition = function_definitions[
            function_index];

    uint8_t* output_ptr = 
        &pitch_get(output_results, results_pitch, 0, function_index);

    apply_boolean_func(
        column_count,
        column_mask,

        row_count,
        rows,

        function_definition,
        output_ptr
        );

    // Now time to compute  the ranks for the left side of the Spearman 
    // rank correlation. Since we only have 0s and 1s, we only need very 
    // basic maths...
    if (threadIdx.x == 0)
    {
        int zeros_got = 0;
        for (int i=0 ; i < row_count; i++)
        {
            uint8_t output_bit = output_ptr[i];
            //printf("i: %d, bit: %d\n", i, output_bit);
            if (output_bit == 0)
            {
                zeros_got += 1;
            }
        }
        float zero_rank = (zeros_got + 1.f) / 2.f;
        float one_rank = (row_count + zeros_got + 1.f)/2.f;

        float left_sum = 0.0;   
        float right_sum = 0.0;
        for (int i=0; i < row_count; i++)
        {
            uint8_t output_bit = output_ptr[i];
            float left_rank = output_bit ? one_rank : zero_rank ;
            left_sum += left_rank;
            float right_rank = scalar_column_ranks[i];
            right_sum += right_rank;
        }


        float r = row_count;

        float left_avg = left_sum  / r;
        float right_avg = right_sum / r;

        //printf("zero_rank: %f one_rank %f\n", zero_rank, one_rank);

        // Let that floating point unit do some work...
        double sup = 0.0, sx=0.0, sy=0.0;

        for (int i=0 ; i < row_count; i++)
        {
           uint8_t output_bit = output_ptr[i];
           float left_rank = output_bit ? one_rank : zero_rank ;
           float right_rank = scalar_column_ranks[i];
           float ld = left_rank - left_avg ;
           float rd = right_rank - right_avg;

           sup += ld * rd ; sx += ld*ld; sy += rd*rd;

        }

        double rho = sup / sqrt(sx*sy);
        rho_values[function_index] = rho;
    }
}

__device__ void apply_boolean_func(
    
    uint8_t column_count, // Maximun number of columns to use..
    uint32_t column_mask,
    
    // ... in this data
    uint16_t row_count,
    uint32_t *rows,
    
    // ... with this function 
    // definition.... this is good for 
    // up  to five columns!
    uint32_t function_definition,
    
    // Put the output here
    uint8_t *output_results
    )
{
    // Use a kernel row to compute the boolean function 
    // for each data row
    int my_row_index = threadIdx.x;
    if ( my_row_index > row_count)
        return ; // NOn-optimal?
    uint32_t my_row = rows[my_row_index];
    //printf("my_row_index: %d, my_row: %d\n", my_row_index, my_row);

    //printf("Column mask: %d\n", column_mask);
    
    int selector = 0;
    int j = 0;
    for (int i=0; i < 32 ; i++)
    {
        uint32_t ix = 1 << i;
        if ( ix & column_mask)
        {
            int bit = my_row & ix ? 1 : 0;
            selector += bit * (1 << j++);
            if ( j== column_count )
            {
                break ; // GPU-wise optimal? maybe it will free the 
                        // integer unit?
            }
        }
    }
    // Not using more than five columns...
    assert( selector < 32 );
    //printf("Selector:%d\n",selector);
    int result_bit = (function_definition & (1 << selector))?1:0;
    output_results[my_row_index] = result_bit;
}
