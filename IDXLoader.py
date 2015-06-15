import os
from array import array as pyarray
import struct
import numpy

class Loader(object):
    """description of class"""
    def __init__(self):
        return

    def Load(self, filename, output_type=numpy.ubyte):
        print("Loading ",filename)
        # Open the file
        fh = open(filename, 'rb')
        
        #Read magic number
        # The first 4 bytes should be
        # byte 0: 0
        # byte 1: 0
        # byte 2: datatype enum
        # byte 3: dimension
        z1,z2,type_code,dimensions = tuple(ord(i) for i in struct.unpack("4c", fh.read(4)))

                
        # Read size in each dimension
        dim_sizes = [struct.unpack(">I", fh.read(4))[0] for i in range(dimensions)]
        
        # For now we only support type 8 (unsigned byte)
        data_size = numpy.cumprod(dim_sizes)[-1] * 1 #
        
        # Read the data as a C-array
        all_data = pyarray('B', fh.read(data_size))        
        fh.close()

        # Return a vector if dimensions = 1
        if dimensions == 1:
            return numpy.array(all_data, output_type)        
        # Return a matrix if dimension > 1
        reshape_dimension = (dim_sizes[0], data_size / dim_sizes[0])
        return numpy.array(all_data, output_type).reshape(reshape_dimension)

   