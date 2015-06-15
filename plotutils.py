import math
import numpy
import matplotlib.pyplot as plt

def plot_tensor(t):
    dims = len(t.shape)

    if (dims < 2):
        raise ValueError("Tensor should be of at least 2 dimensions")

    nplots = numpy.prod(t.shape[:-2])
    plotdims = numpy.ceil(numpy.sqrt(nplots)).astype(int)

    plt.gray()
    f, axarr = plt.subplots(plotdims, plotdims)

    for i in range(nplots):
        prow = int(math.floor(i / plotdims))
        pcol = i - prow*plotdims
        img_index = get_index_tuple(t, i)
        img_arr = t[img_index]
        axarr[prow, pcol].imshow(img_arr)
    
    plt.show()

def get_index_tuple(t, index):
    result = []
    strides = numpy.cumprod([i for i in reversed(t.shape[:-2])])[::-1]    
    left = index    
    for s in strides[1:]:
        idx = int(math.floor(left / s))       
        left -= idx*s
        result.append(idx)
    result.append(left)
    return tuple(result)

def plot_tensor_image(W):
    nFilters = numpy.prod(W.shape[:-2])
    cols = int(numpy.sqrt(nFilters))
    rows = int(numpy.ceil(nFilters / cols))

    filterSizeY = W.shape[-2]
    filterSizeX = W.shape[-1]

    result = numpy.mean(W)*numpy.ones((filterSizeX*rows, filterSizeY*cols), dtype='float32')

    for i in range(nFilters):
        y = int(i / cols)
        x = int(i - y*cols)
        startY = y*filterSizeY
        startX = x*filterSizeX
        endY = (y+1)*filterSizeY
        endX = (x+1)*filterSizeX
        idx = get_index_tuple(W, i)
        result[startY:endY, startX:endX] = W[idx]
    plt.set_cmap('gray')
    plt.imshow(result)
    plt.show()

def plot_columns(W, filterShape):
    nFilters = W.shape[1]
    cols = int(numpy.sqrt(nFilters))
    rows = int(numpy.ceil(nFilters / cols))

    result = numpy.zeros((filterShape[0]*rows, filterShape[1]*cols), dtype='float32')

    for i in range(nFilters):
        y = int(i / cols)
        x = int(i - y*cols)
        startY = y*filterShape[0]
        startX = x*filterShape[1]
        endY = (y+1)*filterShape[0]
        endX = (x+1)*filterShape[1]
        result[startY:endY, startX:endX] = W[:, i].reshape(filterShape)
    plt.set_cmap('gray')
    plt.imshow(result)
    plt.show()
    
