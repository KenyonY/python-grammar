from multiprocessing import Pool, TimeoutError
def f(x):
    return x*x

if __name__ == '__main__':

    with Pool(processes=4) as pool:

        multi_res=[pool.apply_async(f, (i,)) for i in range(2, 70)]
        print([res.get() for res in multi_res])

    

    
