# Ex-1
import math as m

def isPrime(n):
    u = int(m.sqrt(n))
    for i in range(2, u+1):
        if n % i == 0: return False
    return True

n = int(input("Enter an integer in [2, 32767]: "))
print(isPrime(n))

# Ex-2
def makeDict(K, V):
    dict = {}
    for k, v in zip(K, V): dict[k] = v
    return dict

K, V = ('Korean', 'Mathematics', 'English'), (90.3, 85.5, 92.7)
D = makeDict(K, V)
for k in K: print(k, D[k])
