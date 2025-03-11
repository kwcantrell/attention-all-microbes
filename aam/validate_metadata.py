import functools
import os

def validate_fit_asv_encoder(func):
    @functools.wraps(func)
    def wrapper(**kwargs):

        try:
            assert kwargs["i_tree"] is not None, "Error: i_tree is missing or None."
            assert os.path.exists(kwargs["i_tree"]), f"File {kwargs['i_tree']} does not exist"
            assert kwargs["output_dir"] is not None, "Error: m_metadata_file is missing or None."
            
            ### Not sure how to handle the output directory for this: 
            
            # assert os.path.exists(kwargs["output_dir"]), f"File {kwargs['output_dir']} does not exist"
            # if os.path.exists(kwargs["output_dir"]):
            #     raise ValueError(f"Error: Output directory {kwargs['output_dir']} already exists.")
        
            print("All Tests Passed")
            return func(**kwargs)
    
        except AssertionError as e:
            print(f"AssertionError: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
    return wrapper

def validate_fit_denoised_unifrac_regressor(func):
    # runner= CliRunner()
    @functools.wraps(func)
    def wrapper(**kwargs):

        try:
            assert kwargs["i_table"] is not None, "Error: i_table is missing or None."
            assert os.path.exists(kwargs["i_table"]), f"File {kwargs['i_table']} does not exist"
            assert kwargs["i_tree"] is not None, "Error: i_tree is missing or None."
            assert os.path.exists(kwargs["i_tree"]), f"File {kwargs['i_tree']} does not exist"
            assert kwargs["m_metadata_file"] is not None, "Error: m_metadata_file is missing or None."
            assert os.path.exists(kwargs["m_metadata_file"]), f"File {kwargs['m_metadata_file']} does not exist"
            assert kwargs["m_metadata_column"] is not None, "Error: m_metadata_column is missing or None."
            assert kwargs["output_dir"] is not None, "Error: m_metadata_file is missing or None."
        
            print("All Tests Passed")
            return func(**kwargs)
    
        except AssertionError as e:
            print(f"AssertionError: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
    return wrapper

def validate_fit_taxonomy_regressor(func):
    # runner= CliRunner()
    @functools.wraps(func)
    def wrapper(**kwargs):

        try:
            assert kwargs["i_table"] is not None, "Error: i_table is missing or None."
            assert os.path.exists(kwargs["i_table"]), f"File {kwargs['i_table']} does not exist"
            assert kwargs["i_taxonomy"] is not None, "Error: i_taxonomy is missing or None."
            assert os.path.exists(kwargs["i_taxonomy"]), f"File {kwargs['i_taxonomy']} does not exist"
            assert kwargs["m_metadata_file"] is not None, "Error: m_metadata_file is missing or None."
            assert os.path.exists(kwargs["m_metadata_file"]), f"File {kwargs['m_metadata_file']} does not exist"
            assert kwargs["m_metadata_column"] is not None, "Error: m_metadata_column is missing or None."
            assert kwargs["p_max_bp"] is not None, "Error: p_max_bp is missing or None."
            assert kwargs["output_dir"] is not None, "Error: m_metadata_file is missing or None."


            ### Not sure how to handle the output directory for this: 

            # assert os.path.exists(kwargs["output_dir"]), f"File {kwargs['output_dir']} does not exist"
            # if os.path.exists(kwargs["output_dir"]):
            #     raise ValueError(f"Error: Output directory {kwargs['output_dir']} already exists.")
        
            print("All Tests Passed")
            return func(**kwargs)
    
        except AssertionError as e:
            print(f"AssertionError: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
    return wrapper

def validate_fit_sample_regressor(func):
    @functools.wraps(func)
    def wrapper(**kwargs):

        try:
            assert kwargs["i_table"] is not None, "Error: i_table is missing or None."
            assert os.path.exists(kwargs["i_table"]), f"File {kwargs['i_table']} does not exist"
            assert kwargs["m_metadata_file"] is not None, "Error: m_metadata_file is missing or None."
            assert os.path.exists(kwargs["m_metadata_file"]), f"File {kwargs['m_metadata_file']} does not exist"
            assert kwargs["m_metadata_column"] is not None, "Error: m_metadata_column is missing or None."
            assert kwargs["output_dir"] is not None, "Error: m_metadata_file is missing or None."
            if os.path.exists(kwargs["output_dir"]):
                raise ValueError(f"Error: Output directory {kwargs['output_dir']} already exists.")
        
            print("All Tests Passed")
            return func(**kwargs)
    
        except AssertionError as e:
            print(f"AssertionError: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
    return wrapper

def validate_predict_sample_regressor(func):
    @functools.wraps(func)
    def wrapper(**kwargs):

        try:
            assert kwargs["i_table"] is not None, "Error: i_table is missing or None."
            assert os.path.exists(kwargs["i_table"]), f"File {kwargs['i_table']} does not exist"
            assert kwargs["i_model_path"] is not None , "Error: i_model_path is missing or None."
            assert os.path.exists(kwargs["i_model_path"]), f"File {kwargs['i_model_path']} does not exist"
            assert kwargs["m_metadata_file"] is not None, "Error: m_metadata_file is missing or None."
            assert os.path.exists(kwargs["m_metadata_file"]), f"File {kwargs['m_metadata_file']} does not exist"
            assert kwargs["m_metadata_column"] is not None, "Error: m_metadata_column is missing or None."
            assert kwargs["output_dir"] is not None, "Error: m_metadata_file is missing or None."
            if os.path.exists(kwargs["output_dir"]):
                raise ValueError(f"Error: Output directory {kwargs['output_dir']} already exists.")
        
            print("All Tests Passed")
            return func(**kwargs)
    
        except AssertionError as e:
            print(f"AssertionError: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
    return wrapper

def validate_gotu_fit_and_infer(func):
    @functools.wraps(func)
    def wrapper(**kwargs):

        try:
            assert kwargs["i_asv_table"] is not None, "Error: i_asv_table is missing or None."
            assert os.path.exists(kwargs["i_asv_table"]), f"File {kwargs['i_asv_table']} does not exist"
            assert kwargs["i_gotu_table"] is not None, "Error: i_gotu_table is missing or None."
            assert os.path.exists(kwargs["i_gotu_table"]), f"File {kwargs['i_gotu_table']} does not exist"
            assert kwargs["i_gotu_tree_index"] is not None, "Error: i_gotu_tree_index is missing or None."
            assert os.path.exists(kwargs["i_gotu_tree_index"]), f"File {kwargs['i_gotu_tree_index']} does not exist"


            assert kwargs["m_metadata_file"] is not None, "Error: m_metadata_file is missing or None."
            assert os.path.exists(kwargs["m_metadata_file"]), f"File {kwargs['m_metadata_file']} does not exist"
            assert kwargs["m_metadata_column"] is not None, "Error: m_metadata_column is missing or None."
            assert kwargs["output_dir"] is not None, "Error: m_metadata_file is missing or None."
            # assert os.path.exists(kwargs["output_dir"]), f"File {kwargs['output_dir']} does not exist"
            if os.path.exists(kwargs["output_dir"]):
                raise ValueError(f"Error: Output directory {kwargs['output_dir']} already exists.")
        
            print("All Tests Passed")
            return func(**kwargs)
    
        except AssertionError as e:
            print(f"AssertionError: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
    return wrapper