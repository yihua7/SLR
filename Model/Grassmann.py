import numpy as np


def max_lin_indept(array):
    '''
    Get the Max Linear Independent set of the input array
    :param array:
    :return max linear independent set:
    '''
    assert len(np.shape(array)) == 2, 'Shape Error'
    array = np.array(array)
    rank = np.linalg.matrix_rank(array)
    index = 0
    while index < np.shape(array)[0]:
        new_array = np.delete(array, index, 0)
        new_rank = np.linalg.matrix_rank(new_array)
        if new_rank < rank:
            index += 1
        else:
            array = new_array
    return array


def normalize(array):
    '''
    Input array should not be zero array !!!
    '''
    if len(np.shape(array)) == 1:
        array = array / np.linalg.norm(array)
    else:
        assert len(np.shape(array)) == 2, 'Shape Error'
        array = array / np.transpose([np.linalg.norm(array, axis=1)])
    return array


def _orth_project(array, space):
    '''
    Projection on special case
    :param array: a new array
    :param space: a orthodox space set
    :return: return projection of the array on the space
    '''
    assert np.shape(array)[0] == np.shape(space)[1], 'Shape do not match'
    array = np.reshape(array, [-1, 1])
    coordinate = np.matmul(space, array) / np.reshape(np.power(np.linalg.norm(space, axis=1), 2), [-1, 1])
    recover = space * coordinate
    projection = np.sum(recover, 0)
    return projection


def orthodox(array):
    '''
    Make the array orthonormal
    :param array:
    :return an orthonormal array representing the same subspace:
    '''
    assert len(np.shape(array)) == 2, 'Shape Error'

    # Get the max linear independent set of the array
    array = max_lin_indept(array)

    # Schmidt orthogonalization
    for i in range(1, np.shape(array)[0]):
        space = array[0:i]
        vector = array[i]
        projection = _orth_project(vector, space)
        vector -= projection
        array[i] = vector

    # Normalize the array
    array = normalize(array)
    return array


def angle(array1, array2):
    '''
    Calculate the angle between an angle and a grassmann point (subspace).
    :param array1:
    :param array2:
    :return cosine value:
    '''
    if len(np.shape(array1)) == 1:
        vector = array1
        grassmann = array2
    else:
        vector = array2
        grassmann = array1
    assert len(np.shape(grassmann)) == 2 and len(np.shape(vector)) == 1, 'Shape Error'
    assert np.shape(grassmann)[1] == np.shape(vector)[0], 'Shapes do not match'
    grassmann = orthodox(grassmann)

    # Normalize the vector before reshape it to 2-D
    vector = normalize(vector)
    vector = np.reshape(vector, [-1, 1])

    coordinate = np.matmul(grassmann, vector)
    recover = grassmann * coordinate
    projection = np.sum(recover, axis=0)
    cosine = np.dot(np.squeeze(vector), projection)
    return cosine


def parameterize(grassmann):
    '''
    Parameterize a grassmann point by calculating the angle between it and the standard basis
    :param grassmann: point
    :return: parameters
    '''
    assert len(np.shape(grassmann)) == 2, 'Shape Error'
    dimen = np.shape(grassmann)[1]
    Identity = np.diag(np.ones([dimen]))
    para = np.zeros([dimen])
    for i in range(dimen):
        para[i] = angle(Identity[i], grassmann)
    return para


def svd(matrix, k):
    '''
    Get the most k important singular values and output a grassmann point
    :param matrix: covariance matrix
    :param k: the number of singular values left
    :return: a grassmann point
    '''
    s, v, d = np.linalg.svd(matrix)
    for i in range(k, len(v)):
        v[i] = 0.
    return np.matmul(np.matmul(s, np.diag(v)), d)


def covariance(matrix):
    '''
    Calculate the covariance matrix of the input matrix
    :param matrix: data
    :return: covariance matrix
    '''
    assert len(np.shape(matrix)) == 2, 'Shape Error'
    mean = np.reshape(np.average(matrix, axis=0), [1, -1])
    matrix = matrix - mean
    cov = np.matmul(matrix, np.transpose(matrix))
    return cov


def data2grass(data, k):
    '''
    Process raw data and transfer them to parameters
    :param data: input data
    :param k: svd dimension
    :return parameters:
    '''
    cov = covariance(np.transpose(np.array(data)))
    cov = svd(cov, k)
    output = parameterize(cov)
    return output


if __name__ == '__main__':
    x = np.linspace(1, 9, 9)
    x = x.reshape([3, 3])
    t = max_lin_indept(x)
    print('Max linear independent set: \n', t)
    print(angle(x, [11, 13, 15]))
    print(angle([[0., 1., 0.], [0., 0., 2.], [0., 1., 1.]], [1., 0., 0.]))
    print(svd([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1))
    print(covariance([[1, 2], [3, 4], [5, 6]]))
