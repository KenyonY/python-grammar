from functools import partial

def f(x, y):
	print('x',x, 'y', y)
	return x * y

p  = partial(f, 2)
p1 = partial(f, y=3)
p2 = partial(f, 2, 3)

print(p(3)) # x=2, y=3, return 6
print(p(2)) # x=2, y=3, return 6
print(p())  # x=2, y=3, return 6

bacetwo = partial(int, bace=2) #
basetwo.__doc__ = 'Convert base 2 string to an int.'
basetwo('10010') # 18 equal to int('10010', bace=2)