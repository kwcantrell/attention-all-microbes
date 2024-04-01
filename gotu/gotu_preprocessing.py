"""gotu_preproccessing.py"""
from typing import Tuple
from bp import parse_newick
import biom

def write_biom_table(biom_table: biom.table.Table,
                     output_path: str,
                     table_name: str,
                     table_desc: str):
    """Writes biom-format table to file

    Args:
        biom_table (biom.table.Table): Biom format table
        output_path (str): output path for table
        table_name (str): table name (does not require .biom ext)
        table_desc (str): Metadata desc. of table
    """
    with biom.util.biom_open(f"{output_path}/{table_name}.biom", 'w') as f:
       biom_table.to_hdf5(f, table_desc)
    

class FormatBiomTables:
    """Sorts and orders biom tables for use with gotu learning model."""
    def __init__(self, 
                 newick_fp: str,
                 gotu_biom_fp: str,
                 asv_biom_fp: str) -> None:
        
        try:
            self.bp_tree = parse_newick(open(newick_fp).read())
            self.gotu_table = biom.load_table(gotu_biom_fp)
            self.asv_table = biom.load_table(asv_biom_fp)
        except:
            print("Error: Invalid file path(s)!")
        
    def sort_and_filter_biom(self) -> Tuple[biom.table.Table, biom.table.Table]:
        gotu_table_ids = set(self.gotu_table.ids('sample'))
        asv_table_ids = set(self.asv_table.ids('sample'))
        merged_ids = list(asv_table_ids.intersection(gotu_table_ids))
        gotu_table_intersect = self.gotu_table.filter(merged_ids, axis='sample')
        asv_table_intersect = self.asv_table.filter(merged_ids, axis='sample')
        return (gotu_table_intersect, asv_table_intersect)
    

    def order_biom(self, gotu_table_intersect: biom.table.Table,
                   asv_table_intersect: biom.table.Table) -> Tuple[biom.table.Table, biom.table.Table]:
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