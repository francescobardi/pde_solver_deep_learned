# Sams playground doesn't work... though...

def torch_conv_to_matrix(conv):
    t = conv.view(-1).detach().numpy()
    kernel_dim = np.int(np.sqrt(t.shape[0]))
    return t.reshape(kernel_dim, kernel_dim)


def reshape_and_cut_border(x, border_size=1):
    dim = np.int(np.sqrt(x.shape[0]))
    return x.reshape((dim, dim))[border_size:-border_size, border_size:-border_size]\
        .reshape(-1)


def convolution_as_matrix_multiplication(conv, input_dim):
    # following https://dsp.stackexchange.com/questions/35373/convolution-as-a-doubly-block-circulant-matrix-operating-on-a-vector
    # we have k * x = y, x being the input (w) and k being the kernel
    # 1. output size of y is (dim_k + dim_x - 1)^2 => pad k accordingly
    kernel_dim = conv.shape[0]
    N = input_dim
    A = np.zeros((N + kernel_dim - 1, N + kernel_dim - 1)) # N comes from the size of u
    A[-kernel_dim:, 0:kernel_dim] = conv # place the kernel in the lower left corner

    # 2. get circulant matrices for each row of A
    # remember that circulant_rows[0] should be the circulant defined by the last row
    from scipy.linalg import circulant
    circulant_of_rows = [circulant(A[row, :]) for row in reversed(range(A.shape[0]))]

    # 3. construct doubly circulant matrix
    #
    #      X_0 X_{N-1} X_{N-2} ...
    # X =  X_1 X_0 X_{N-1} ...
    #      X_2 X_1 X_0
    #      ...
    #
    # where X_i denotes with circulant matrices from circulant_rows

    doubly_circulant_idx = circulant(range(A.shape[0]))
    X = np.hstack([np.vstack([circulant_of_rows[idx] for idx in doubly_circulant_idx[:, i]])
                   for i in range(doubly_circulant_idx.shape[1])])

    return X


def prep_y_for_conv_mat_mult(y, conv_dim):
    # 4. create corresponding b vector from u
    # how? I placed in the middle...

    # this is special for our case, don't know if this checks out in general
    u_padded_dim = np.int(np.sqrt(conv_dim))
    u_padded = np.zeros((u_padded_dim, u_padded_dim))

    dim = np.int(np.sqrt(y.shape[0]))
    u_padded[1:-1, 1:-1] = y.reshape(dim, dim)
    return u_padded



conv = np.random.rand(3, 3)
X = convolution_as_matrix_multiplication(conv, N)
t = reshape_and_cut_border(X@prep_y_for_conv_mat_mult(u, X.shape[0]).reshape(-1))


from scipy.signal import convolve2d

y_true = reshape_and_cut_border(convolve2d(conv, u.reshape(N, N)).reshape(-1))

t - y_true
