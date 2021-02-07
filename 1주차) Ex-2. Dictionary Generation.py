K=('Korean','Mathematics','English')
V=(90.3,85.5,92.7)

def makeDict(K,V):
    D={}
    for i in range(len(K)):
        D.update({K[i]:V[i]})
    return D
result=makeDict(K,V)
print(result)
for i in range(len(result)):
        print(K[i] in result,end=' ')
        
