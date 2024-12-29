def f(x):
    return (x-3)**2+1

def grad():
    a=0
    it=1000
    lr=0.01
    for i in range(it):
        da=2*(a-3)
        a=a-lr*da
        print(f(a))
    print(a)
    
grad()