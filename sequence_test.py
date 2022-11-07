def createFFTLengths(N):
        lengths = [1]
        it = 1
        while len(lengths)<=N:
            lengths.append(2**it)
            lengths.extend(lengths[:-1][::-1])
            it = it+1
        return lengths
N=10
import math
print(createFFTLengths(N))
print([math.gcd(2**j,j) for j in range(1,N)])
