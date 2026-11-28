import math

class Tensor:

    def __init__(
            self,
            data,                   # 1D flat lists, element at (i, j) can be accesed using data[i * col + j]
            shape,                  # tuple : (rows, cols)
            _children=()                  
        ):  
        self.data = data
        self.shape = shape
        self.grad = [0.0] * len(data)
        self._backward = lambda : None
        self._prev = set(_children)


    def __repr__(self):
        # will implement a better one later
        return f'Tensor(data = {self.data})'


    def __add__(self, other):
        '''Element wise addition'''
        row, col = self.shape
        broadcast = other.shape == (1, col) and row != 1

        # dont lose reference to other
        if broadcast:
            other_vals = []
            for _ in range(row):
                other_vals.extend(other.data)
        else:
            other_vals = other.data

        out_data = []
        for a_val, b_val in zip(self.data, other_vals):
            out_data.append(a_val + b_val)
        out = Tensor(out_data, self.shape, (self, other))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += 1.0 * out.grad[j + i * col] 
                    if broadcast:  
                        other.grad[j] += 1.0 * out.grad[j + i * col]
                    else:
                        other.grad[j + i * col] += 1.0 * out.grad[j + i * col]
        out._backward = _backward

        return out
    

    def __mul__(self, other):
        '''Element wise multiplication of two identically sized tensors'''
        if isinstance(other, (float, int)):
            other_data = [other] * len(self.data)
            other = Tensor(other_data, self.shape)

        out_data = []
        for a_val, b_val in zip(self.data, other.data):
            out_data.append(a_val * b_val)
        out = Tensor(out_data, self.shape, (self, other))

        def _backward():
            for i in range(len(out_data)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]
        out._backward = _backward

        return out


    def __rmul__(self, other):
        return self * other


    def __sub__(self, other):
        '''Self - other'''
        out_data = []
        for a_val, b_val in zip(self.data, other.data):
            out_data.append(a_val - b_val)
        out = Tensor(out_data, self.shape, (self, other))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += 1.0 * out.grad[i]
                other.grad[i] += -1.0 * out.grad[i]
        out._backward = _backward

        return out

    def __neg__(self):
        '''Additive inverse of every element in the tensor'''
        out_data = []
        for v in self.data:
            out_data.append(-v)
        out = Tensor(out_data, self.shape, (self, ))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += -out.grad[i]
        out._backward = _backward

        return out


    def __truediv__(self: Tensor, other: Tensor) -> Tensor:
        '''
        self : Tensor (B, V)
        other : Tensor (B, 1)
        Returns a (B, V) Tensor
        '''
        if isinstance(other, (float, int)):
            other_data = [other] * (self.shape[0])
            other = Tensor(other_data, (self.shape[0], 1))
        row, col = self.shape
        out_data = []
        for i in range(row):
            for j in range(col):
                out_data.append(self.data[j + i * col] / other.data[i])
        out = Tensor(out_data, self.shape, (self, other))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += (1 / other.data[i]) * out.grad[j + i * col]
                    other.grad[i] += (-self.data[j + i * col] / other.data[i]**2) * out.grad[j + i * col]
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
        out = Tensor(out_data, (row1, col2), (self, other))

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
        out = Tensor(out_data, (col, row), (self, ))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += out.grad[i + j * row]
        out._backward = _backward

        return out

    
    def sum(self):
        row, col = self.shape
        out = Tensor([sum(self.data)], (1, 1), (self, ))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += out.grad[0]
        out._backward = _backward

        return out


    def exp(self):
        out_data = []
        for v in self.data:
            out_data.append(math.exp(v))
        out = Tensor(out_data, self.shape, (self,))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.data[i] * out.grad[i]
        out._backward = _backward

        return out


    def log(self):
        out_data = []
        for v in self.data:
            out_data.append(math.log(v))
        out = Tensor(out_data, self.shape, (self, ))

        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += (1 / self.data[i]) * out.grad[i]
        out._backward = _backward

        return out


    def sum_rows(self):
        row, col = self.shape
        out_data = []
        for i in range(row):
            s = 0
            for j in range(col):
                s += self.data[j + i * col]
            out_data.append(s)
        out = Tensor(out_data, (row, 1), (self, ))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += 1.0 * out.grad[i]
        out._backward = _backward

        return out
    

    def mean_rows(self) -> Tensor:
        '''
        Returns each row's average : Tensor (D, 1)
        '''
        row, col = self.shape
        out_data = []
        for i in range(row):
            s = 0
            for j in range(col):
                s += self.data[j + i * col]
            out_data.append(s / col)
        out = Tensor(out_data, (row, 1), (self, ))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += 1.0 * out.grad[i] / col
        out._backward = _backward

        return out


    def relu(self):
        row, col = self.shape
        out_data = []
        for val in self.data:
            out_data.append(max(0, val))
        out = Tensor(out_data, self.shape, (self, ))

        def _backward():
            for i in range(row):
                for j in range(col):
                    self.grad[j + i * col] += out.grad[j + i * col] * (1 if self.data[j + i * col] > 0 else 0)
        out._backward = _backward

        return out


    def backward(self):
        '''Toposort on the nodes in the computation graph (exists in memory)'''
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


    def to_list(self):
        '''Takes a Tensor and returns its 2D form'''
        # Write tests for this later if needed
        row, col = self.shape
        out = []
        for i in range(row):
            row = []
            for j in range(col):
                row.append(self.data[j + i * col])
            out.append(row)

        return out

    @classmethod
    def from_list(cls, matrix):
        '''Empty matrices are not allowed'''
        r = len(matrix); c = len(matrix[0])
        out_val = []
        for row in matrix:
            out_val.extend(row)

        return Tensor(out_val, shape=(r, c))

    def rows(self, ids: list[int]) -> Tensor:
        '''
        self : Tensor (V, D)
        '''
        row, col = self.shape
        out_data = []
        for id in ids:
            for j in range(col):
                out_data.append(self.data[j + id * col])
        out = Tensor(out_data, (len(ids), col), (self, ))

        def _backward():
            for i, id in enumerate(ids):
                for j in range(col):
                    self.grad[j + id * col] += 1.0 * out.grad[j + i * col]
        out._backward = _backward

        return out

    

        
        

    
