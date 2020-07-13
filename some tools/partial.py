from functools import partial

def f(x, y):
	print('x',x, 'y', y)
	return x * y

p  = partial(f, 2)
p1 = partial(f, y=3)
p2 = partial(f, 2, 3)

print(p(3)) # x=2, y=3, return 6
print(p1(2)) # x=2, y=3, return 6
print(p2())  # x=2, y=3, return 6

basetwo = partial(int, base=2) #
basetwo.__doc__ = 'Convert base 2 string to an int.'
print(basetwo('10010')) # 18 equal to int('10010', bace=2)

# 当然，这个我们可以用一个类实现相同的效果 // 2020.5.11 yky
class Partial:
	def __init__(self, func, *args, **kwargs):
		self.func = func
		self.args = args
		self.kwargs = kwargs 
	
	def __call__(self, x):
		return self.func(x, *self.args, **self.kwargs)
	
base2 = Partial(int, base=2)
print(base2('10010'))