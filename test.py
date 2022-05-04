'''

                            Online Python Compiler.
                Code, Compile, Run and Debug python program online.
Write your code in this editor and press "Run" button to execute it.

'''
import numpy as np
t = [[1,2,3,4,5,6,7,8,9],[11,22,33,44,55,66,77,88,99],[111,222,333,444,555,666,777,888,999]]

t=np.asarray(t,np.ndarray)
print(t)
print(t[ : , 1: ])
print(t[ : , : -1] )

print(t.shape)