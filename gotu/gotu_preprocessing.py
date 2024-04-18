"""gotu_preproccessing.py"""
import os

from typing import Tuple
from bp import parse_newick
import biom

def write_biom_table(biom_table: biom.table.Table,
                     output_path: str,
                     table_name: str,
                     table_desc: str):
    """
    Writes a biom-format table to a file, creating the output directory if it does not exist.

    This function checks if the specified output path exists and creates it if necessary. It then writes the given
    BIOM table to a file within this directory with a specified name and description.

    Args:
        biom_table (biom.table.Table): The BIOM table to write.
        output_path (str): The output directory path where the table should be saved.
        table_name (str): The name of the table file (does not require .biom extension).
        table_desc (str): Metadata description of the table.
    """
    if not os.path.exists(output_path):
        print(f"Error:{output_path} does not exit!")
    file_path = os.path.join(output_path, f"{table_name}.biom")
    with biom.util.biom_open(file_path, 'w') as f:
        biom_table.to_hdf5(f, table_desc)
    

class FormatBiomTables:
    """
    A class to sort and order BIOM tables for use with a specific learning model.
    
    This class is designed to prepare BIOM tables by sorting and filtering them based on the intersection of sample IDs
    between gOTU (genus-level Operational Taxonomic Units) and ASV (Amplicon Sequence Variants) tables and then
    ordering them according to a phylogenetic tree provided in Newick format.
    
    Attributes:
        bp_tree: A tree structure parsed from a Newick file representing the phylogeny of the organisms in the BIOM tables.
        gotu_table: A BIOM table for gOTU data loaded from a specified file path.
        asv_table: A BIOM table for ASV data loaded from a specified file path.
    
    Args:
        newick_fp (str): The file path to the Newick file containing the phylogenetic tree.
        gotu_biom_fp (str): The file path to the BIOM file containing gOTU data.
        asv_biom_fp (str): The file path to the BIOM file containing ASV data.
    
    Raises:
        Exception: If any file paths are invalid or the files cannot be loaded.
    """
    def __init__(self, newick_fp: str, gotu_biom_fp: str, asv_biom_fp: str) -> None:
        """
        Initializes the FormatBiomTables class by loading and parsing the provided Newick and BIOM files.

        Args:
            newick_fp (str): The file path to the Newick file containing the phylogenetic tree.
            gotu_biom_fp (str): The file path to the BIOM file containing gOTU data.
            asv_biom_fp (str): The file path to the BIOM file containing ASV data.

        Raises:
            FileNotFoundError: If any of the provided file paths do not exist.
            IOError: If there is an error opening any of the files.
            ValueError: If there is an issue with the content of the files, such as incorrect format.
        """
        try:
            with open(newick_fp, 'r') as file:
                self.bp_tree = parse_newick(file.read())
        except FileNotFoundError:
            print(f"Error: The Newick file at '{newick_fp}' was not found.")
            raise
        except IOError as e:
            print(f"Error reading Newick file '{newick_fp}': {e}")
            raise
        except ValueError as e:
            print(f"Error in Newick file format '{newick_fp}': {e}")
            raise

        try:
            self.gotu_table = biom.load_table(gotu_biom_fp)
        except FileNotFoundError:
            print(f"Error: The gOTU BIOM file at '{gotu_biom_fp}' was not found.")
            raise
        except (IOError, ValueError) as e:  # Assuming biom.load_table throws these on errors
            print(f"Error loading gOTU BIOM file '{gotu_biom_fp}': {e}")
            raise

        try:
            self.asv_table = biom.load_table(asv_biom_fp)
        except FileNotFoundError:
            print(f"Error: The ASV BIOM file at '{asv_biom_fp}' was not found.")
            raise
        except (IOError, ValueError) as e:  # Assuming biom.load_table throws these on errors
            print(f"Error loading ASV BIOM file '{asv_biom_fp}': {e}")
            raise

        
    def sort_and_filter_biom(self) -> Tuple[biom.table.Table, biom.table.Table]:
        """
        Sorts and filters the BIOM tables by finding the intersection of sample IDs between the gOTU and ASV tables.

        This method identifies common sample IDs between the two tables and filters both tables to only include these
        common samples, thereby synchronizing the sample sets between the gOTU and ASV data.

        Returns:
            A tuple of two BIOM tables (gotu_table_intersect, asv_table_intersect) that have been filtered to include
            only the intersecting sample IDs.
        """
        gotu_table_ids = set(self.gotu_table.ids('sample'))
        asv_table_ids = set(self.asv_table.ids('sample'))
        merged_ids = list(asv_table_ids.intersection(gotu_table_ids))
        gotu_table_intersect = self.gotu_table.filter(merged_ids, axis='sample')
        asv_table_intersect = self.asv_table.filter(merged_ids, axis='sample')
        return (gotu_table_intersect, asv_table_intersect)
    

    def order_biom(self, gotu_table_intersect: biom.table.Table,
                   asv_table_intersect: biom.table.Table) -> Tuple[biom.table.Table, biom.table.Table]:
        """
        Orders the intersected BIOM tables based on the phylogenetic tree order.

        This method sorts the observations in the provided BIOM tables (gOTU and ASV intersected tables) according to
        the order specified by the phylogenetic tree loaded from the Newick file. It ensures that the data in the
        tables are aligned with the phylogenetic relationships of the organisms they represent.

        Args:
            gotu_table_intersect (biom.table.Table): The filtered gOTU table with common sample IDs.
            asv_table_intersect (biom.table.Table): The filtered ASV table with common sample IDs.

        Returns:
            A tuple of two BIOM tables (gotu_ordered_table, asv_ordered_table) that have been ordered according to
            the phylogenetic tree.
        """
        name_list = []
        for i in range(1, ((self.bp_tree.__len__())*2)):
            if self.bp_tree.name(i) is not None:
                name_list.append(self.bp_tree.name(i))
                
        pool = {}
        for name in name_list:
            pool.setdefault(name, len(pool))
            
        gotu_table_intersect_samples = set(gotu_table_intersect.ids('observation'))
        asv_table_intersect_samples = set(asv_table_intersect.ids('observation'))
        
        gotu_index_order = {}
        ids_not_found = []
        for id in gotu_table_intersect_samples:
            if id in pool:
                gotu_index_order.setdefault(pool[id], id)
            else:
                ids_not_found.append(id)
        print(f"Original number of GOTU features: {len(gotu_table_intersect_samples)}")
        print(f"len of index list: {len(gotu_index_order)}")
        print(f"len of IDs not found {len(ids_not_found)}")
        
        asv_index_order = {}
        ids_not_found = []
        for id in asv_table_intersect_samples:
            try:
                if id in pool:
                    asv_index_order.setdefault(pool[id], id)
            except:
                ids_not_found.append(id)
        print(f"Original Number of ASV features: {len(asv_table_intersect_samples)}")
        print(f"len of index list: {len(asv_index_order)}")
        print(f"len of IDs not found {len(ids_not_found)}")
        
        gotu_obs_ordered = []
        asv_obs_ordered = []

        for i in range(1, len(pool) - 1):
            if i in asv_index_order:
                asv_obs_ordered.append(asv_index_order[i])
            elif i in gotu_index_order:
                gotu_obs_ordered.append(gotu_index_order[i])
                
        gotu_ordered_table = gotu_table_intersect.sort_order(gotu_obs_ordered, axis='observation')
        asv_ordered_table = asv_table_intersect.sort_order(asv_obs_ordered, axis='observation')
        
        return (gotu_ordered_table, asv_ordered_table)