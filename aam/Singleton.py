class BaseSingleton:
    _instance = None
    params = {}
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaseSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_params(self):
        return self.params.copy()
    
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.params and v is not None:
                self.params[k] = v
    


class Regressor_Singleton(BaseSingleton):

    params = {
        "i_table": "./commands.txt", # What should I change this to? for a user perspective.
        "output_dir": "./output", # req
        "m_metadata_column": "example_column", # req
        "m_metadata_file": "./commands.txt", #req
        "p_missing_samples":"error",
        "p_patience": 10,
        "p_epochs": 1000,
        "p_weight_decay": 0.004,
        "p_gen_new_table": True,
        "p_lr": 1e-4,
        "p_attention_layers": 4,
        "p_early_stop_warmup": 50,
        "p_batch_size": 8,
        "p_decay_steps": 1000,
        "p_embedding_dim": 128,
        "p_asv_dropout": 0.0,
        "p_rarefy_depth": 5000,
        "p_attention_heads": 4,
        "p_dropout": 0.0,
        "p_intermediate_activation": "relu",
        "p_asv_limit": 1024,
        "p_max_bp": 150,
        "p_is_categorical": False,
        "i_model": None,
        "p_gotu": False,
        "p_add_token": False,
        "p_warmup_steps": 10000,
        "p_intermediate_size": 1024,
    }

class Fit_Singleton(BaseSingleton):
    params = {
        "p_intermediate_size": 512,
        "p_dropout": 0.0,
        "p_attention_layers": 8,
        "p_decay_steps": 1000,
        "p_lr": 1e-4,
        "p_epochs": 1000,
        "output_dir": "./output", # req
        "p_intermediate_activation": "gelu",
        "p_attention_heads": 4,
        "p_weight_decay": 0.004,
        "p_max_bp": 150,
        "p_embedding_dim": 128
    }

singleton_map = {
    "fit_denoised_unifrac_regressor": Regressor_Singleton,
    "fit_asv_encoder": Fit_Singleton,
    "fit_taxonomy_regressor": Regressor_Singleton,
    "fit_sample_regressor": Regressor_Singleton,
    "predict_sample_regressor": Regressor_Singleton,
    "fit_gotu": Fit_Singleton,
}


def inject_common_params(func):
    
    def wrapper(**kwargs):
        func_name = func.__name__
        unifrac_taxonomy_singleton = singleton_map[func_name]()
        for k,v in kwargs.items():
            print(f"{k}: {v}")
        if(kwargs['use_saved_params']):

            unifrac_taxonomy_singleton.set_params(**kwargs)
            common_params = unifrac_taxonomy_singleton.get_params()
            
            print("Inject common params is running")
            
            kwargs = { **kwargs, **common_params}
        return func(**kwargs)
    return wrapper