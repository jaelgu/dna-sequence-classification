import os
import time
import sys
import numpy as np
import pickle

sys.path.append("..")
from config import TOP_K, DEFAULT_TABLE, KMER_K
from logs import LOGGER
from utils import *

def search_in_milvus(table_name, query_sentence, milvus_cli, mysql_cli):
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        kmers = build_kmers(query_sentence,KMER_K)
        query_data = [" ".join(kmers)]
        query_list = encode_seq(query_data)
        LOGGER.info("Searching...")
        results = milvus_cli.search_vectors(table_name,query_list,TOP_K)
        vids = [str(x.id) for x in results[0]]
        print("-----------------", vids)
        ids, results_class, seq_genes = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in results[0]]
        return results_class, seq_genes, distances
    except Exception as e:
        LOGGER.error(" Error with search : {}".format(e))
        sys.exit(1)

"""
from milvus_helpers import MilvusHelper
from mysql_helpers import MySQLHelper
if __name__ == "__main__":
    COLLECTION_NAME = "test"
    query_sentence="ACGTTA"
    MILVUS_CLI = MilvusHelper()
    MYSQL_CLI = MySQLHelper()
    search_in_milvus(COLLECTION_NAME, query_sentence, MILVUS_CLI, MYSQL_CLI)
"""
