import inspect

class Params:
    emb_dim = 5
    attention_heads = 2
    attention_layers = 3
    
    def get_params():
        attributes = inspect.getmembers(Params, lambda a:not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]
        return {
            k:v for (k,v) in attributes
        }
        
        
if __name__ == '__main__':
    print(Params.get_params())