from singleton import Params
from example_class_1 import test

def fit():
    Params.size = 9
    Params.length = 10
    
    test_var  = 6
    def test_func():
        print(f"this is a test function!!!!!!! {test_var}")
    Params.test_func = test_func
    
    test()
    
if __name__ == "__main__":
    fit()