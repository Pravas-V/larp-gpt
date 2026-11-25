
class Tensor:

    def __init__(
            self,
            data,                   # 1D flat lists, element at (i, j) can be accesed using data[i * col + j]
            shape,                  # tuple : (rows, cols)
            _children=(),               
            _op=''                  
        ):  
        self.data = data
        self.shape = shape
        self.grad = [0.0] * len(data)
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op


    def __repr__(self):
        # will implement a better one later
        return f'Tensor(data = {self.data})'


    def __add__(self, other):
        '''Element wise addition'''
        out_data = []
        for a_val, b_val in zip(self.data, other.data):
            out_data.append(a_val + b_val)
        out = Tensor(out_data, self.shape, (self, other), '+')

        def _backward():
            for i in range(len(out_data)):
                self.grad[i] += 1.0 * out.grad[i]   
                other.grad[i] += 1.0 * out.grad[i]
        out._backward = _backward

        return out
    

    def __mul__(self, other):
        '''Element wise multiplication of two identically sized tensors'''
        out_data = []
        for a_val, b_val in zip(self.data, other.data):
            out_data.append(a_val * b_val)
        out = Tensor(out_data, self.shape, (self, other), '*')

        def _backward():
            for i in range(len(out_data)):
                self.grad[i] += other[i] * out.grad[i]
                other.grad[i] += self[i] * out.grad[i]
        out._backward = _backward

        return out


    def __matmul__(self, other):
        '''Matrix multiplication'''
        row1, col1 = self.shape
        row2, col2 = other.shape
        out_data = []
        for i in range(row1):
            for j in range(col2):
                el_data = 0
                for k in range(col1):
                    el_data += self.data[k + i * col1] * other.data[j + k * col2]
                out_data.append(el_data)
        out = Tensor(out_data, (row1, col2), (self, other), '@')

        def _backward():
            for i in range(row1):
                for j in range(col1):
                    for k in range(col2):
                        self.grad[j + i * col1] += other.data[k + j * col2] * out.grad[k + i * col2]

            for j in range(row2):
                for k in range(col2):
                    for i in range(row1):
                        other.grad[k + j * col2] += self.data[j + i * col1] * out.grad[k + i * col2]

        out._backward = _backward

        return out


    @property
    def T(self):
        out_data = []
        row, col = self.shape
        for j in range(col):
            for i in range(row):
                out_data.append(self.data[j + i * col])
        out = Tensor(out_data, (col, row), (self, ), 'T')

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += out.grad[i + j * row]
        out._backward = _backward

        return out

    
    def sum(self):
        row, col = self.shape
        out = Tensor([sum(self.data)], (1, 1), (self, ), 'sum')

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += out.grad[0]
        out._backward = _backward

        return out


    def relu(self):
        row, col = self.shape
        out_data = []
        for val in self.data:
            out_data.append(max(0, val))
        out = Tensor(out_data, self.shape, (self, ), 'RELU')

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += out.grad[j + i * col] * (1 if self.data[j + i * col] > 0 else 0)
        out._backward = _backward

        return out


    def backward(self):
        state = {}
        stack = [self]
        topo = []
        while stack:
            cur = stack.pop()
            if cur not in state:
                state[cur] = 1
                stack.append(cur)
                for child in cur._prev:
                    if child not in state:
                        stack.append(child)

            elif state[cur] == 1:
                state[cur] = 2
                topo.append(cur)

        self.grad = [1.0] * (len(self.data))
        for node in reversed(topo):
            node._backward()

    
