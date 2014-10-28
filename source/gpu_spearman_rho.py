import pycuda.autoinit
import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
import numpy as np
import numpy
from scipy.stats import rankdata, spearmanr
from pycuda.compiler import SourceModule
import os.path

current_dir = os.path.dirname(__file__)
kernel_filename = os.path.join(current_dir, "cuda_kernel.cu")
kernel_code = open(kernel_filename).read();
# print(kernel_code.read())
mod = SourceModule(kernel_code, no_extern_c=True)

raw_gpu_func = mod.get_function('calc_spearman_rho')
# extern "C" __global__ void calc_spearman_rho(
#     uint16_t  row_count,
#     float*    scalar_column_ranks, // has row_count elements...

#     uint32_t *rows, // The block of left columns. Each integer
#                     // is interpreted as bits, each bit is a separate
#                     // column. So this array can hold up to 32 columns.
#                     // I'm using only five.

#                     // The functions
#     uint16_t function_count,
#     uint32_t* function_definitions, // See `apply_boolean_func` to 
#                                     // understand how each function fits
#                                     // inside a single integer

#     uint8_t* output_results,  // Properly strided matrix . Each row here contains
#                               // the result of computing a function.
#                               // So this matrix has row_count _columns_ and 
#                               // function_count _rows_.... a bit confusing.  
#     size_t results_pitch,     // 

#     // Columns to try the functions on
#     uint32_t column_mask,

#     // Number of columns of boolean data at the left
#     uint16_t column_count,

#     // Output: array with one element per function 
#     double* rho_values
#     )

class SpearmanRhoCalculator(object):
    
    def __init__(self, 
        left_binary_block,  # 2d array of boolean
        function_definitions, # array of uint32_t
        right_scalars, # Scalar  values at theright
        column_cardinality
        ):
        self._column_cardinality = column_cardinality
        column_count = left_binary_block.shape[1]
        assert column_count < 32
        self._column_count = column_count
        assert function_definitions.dtype == np.uint32
        function_count = function_definitions.shape[0]
        self._function_count = function_count
        
        ranks = (rankdata(right_scalars) ).astype(np.dtype('f4'))
        gpu_ranks =drv.mem_alloc(ranks.nbytes)
        drv.memcpy_htod(gpu_ranks, ranks)
        self._gpu_ranks = gpu_ranks
   
        # How many rows?
        row_count = left_binary_block.shape[0]
        self._row_count = row_count
        
        # Prepare the left block
        left_binary_encoded = np.zeros((row_count,), dtype=np.uint32)
        for i in range(column_count):
            left_binary_encoded += left_binary_block[:,i] << i
        gpu_left_binary_encoded = drv.mem_alloc(left_binary_encoded.nbytes)
        drv.memcpy_htod(gpu_left_binary_encoded, left_binary_encoded)
        self._gpu_left_binary_encoded =gpu_left_binary_encoded
        
        # Function definitions
        gpu_function_definitions = drv.mem_alloc(function_definitions.nbytes)
        drv.memcpy_htod(gpu_function_definitions, function_definitions)
        self._gpu_function_definitions = gpu_function_definitions
        
        # Space for the results
#         print(row_count, function_count)
        gpu_result_space, gpu_result_pitch = drv.mem_alloc_pitch(row_count, function_count, 4)
        self._gpu_result_space = gpu_result_space
        self._gpu_result_pitch = gpu_result_pitch
        gpu_rho_space = drv.mem_alloc(function_count*8)
        self._gpu_rho_space = gpu_rho_space
        
        self._rho_space = np.zeros((function_count,), dtype='f8')
    
    def __call__(
        self,    
        columns_to_take, # A list 
        ):
        """
        Returns a Spearman rho value for each passed in function definition
        """
        assert(len(columns_to_take) == self._column_cardinality)
        # Convert the columns to a bit mask.... 
        column_mask = 0
        for column_index in columns_to_take:
            column_mask += 1 << column_index

        grid_size = (1, self._function_count // 32 + 1 )

        # Invoke...
        raw_gpu_func(
            np.uint16(self._row_count),
            self._gpu_ranks,

            self._gpu_left_binary_encoded,
            np.uint16(self._function_count),
            self._gpu_function_definitions,

            self._gpu_result_space,
            np.uintp(self._gpu_result_pitch),

            np.uint32( column_mask ),
            np.uint16( self._column_count),

            self._gpu_rho_space,

            block = (self._row_count, 32 ,1),
            grid  = grid_size 
        )
        
        drv.memcpy_dtoh(self._rho_space, self._gpu_rho_space)
        return self._rho_space.copy()
        
def test_it_works():    
    calc = SpearmanRhoCalculator(

        # Left block
        np.array([[True],
                  [False], 
                  [True]]),

        # Function definitions
        np.array([1,2], dtype=np.uint32),

        # Right scalars
        [2,
         0,
         2],
        
        # Column cardinality 
        1
    )

    assert str( calc([0]) )== '[-1.  1.]'

    # Truth table
    #
    # 0 0 1
    # 0 1 0
    # 1 0 0
    # 1 1 1
    #

    calc = SpearmanRhoCalculator(

        # Left block
        np.array([[True,True],
                  [False, True], 
                  [True, True]]),

        # Function definitions
        np.array([9], dtype=np.uint32),

        # Right scalars
        [2,
         0,
         2],
        
        # Column cardinality 
        2
    )

    assert str(calc([0,1]))=='[ 1.]'

    # Truth table
    #
    # 0 0 0
    # 0 1 1
    # 1 0 1
    # 1 1 0
    #

    calc = SpearmanRhoCalculator(

        # Left block
        np.array([[True,True],
                  [False, True], 
                  [True, True]]),

        # Function definitions
        np.array([6], dtype=np.uint32),

        # Right scalars
        [2,
         0,
         2],
        
        # Column cardinality 
        2
    )

    assert str(calc([0,1]))=='[-1.]'

if __name__ == '__main__':
    test_it_works()

