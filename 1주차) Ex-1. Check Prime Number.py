import math

def isPrime(n):
    for i in range(2,int(math.sqrt(n))+1):
        if n%i==0:
            print('i= ',i)
            return 0
        
num=int(input('Enter a integer number (2~32767): '))

result=isPrime(num)

if result==0:
    print('{} is not prime' .format(num))
else:
    print('{} is prime' .format(num))
