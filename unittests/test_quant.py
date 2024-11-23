import unittest
from PyDistances import quantitative
import polars as pl
import numpy as np
from PyMachineLearning.preprocessing import Encoder
from PyDistances.quantitative import (Euclidean_dist, Euclidean_dist_matrix,
                                      Minkowski_dist, Minkowski_dist_matrix,
                                      Canberra_dist, Canberra_dist_matrix,
                                      Pearson_dist_matrix,
                                      Mahalanobis_dist, Mahalanobis_dist_matrix,
                                      Robust_Maha_dist, Robust_Maha_dist_matrix)
from PyDistances.binary import (Sokal_dist, Sokal_dist_matrix,
                                Jaccard_dist, Jaccard_dist_matrix)
from PyDistances.multiclass import (Matching_dist, Matching_dist_matrix)
from PyDistances.mixed import (GG_dist, GG_dist_matrix, RelMS_dist_matrix, S_robust)


def get_df():
    madrid_houses_df = pl.read_csv('Tests/madrid_houses.csv')
    columns_to_exclude = ['', 'id','sq_mt_allotment','floor', 'neighborhood', 'district']
    madrid_houses_df = madrid_houses_df.select(pl.exclude(columns_to_exclude))
    return madrid_houses_df


def get_df_quant():
    madrid_houses_df = get_df()
    binary_cols = ['is_renewal_needed', 'has_lift', 'is_exterior', 'has_parking']
    multi_cols = ['energy_certificate', 'house_type']
    quant_cols = [x for x in madrid_houses_df.columns if x not in binary_cols + multi_cols]

    encoder_ = Encoder(method='ordinal')
    encoded_arr = encoder_.fit_transform(madrid_houses_df[binary_cols + multi_cols])
    cat_df = pl.DataFrame(encoded_arr)
    cat_df.columns = binary_cols + multi_cols
    cat_df = cat_df.with_columns([pl.col(col).cast(pl.Int64) for col in cat_df.columns])
    quant_df = madrid_houses_df[quant_cols]
    return quant_df


class TestTPool(unittest.TestCase):

    def test_euclidean(self):
        madrid_houses_df = get_df_quant()
        x1 = madrid_houses_df[0, :]
        x3 = madrid_houses_df[2, :]
        d = Euclidean_dist(x1, x3)
        self.assertAlmostEqual(d, 59247.00760376004)

    def test_minkowski(self):
        madrid_houses_df = get_df_quant()
        x1 = madrid_houses_df[0, :]
        x3 = madrid_houses_df[2, :]
        d1 = Minkowski_dist(x1, x3, q=1)
        d2 = Minkowski_dist(x1, x3, q=2)
        self.assertAlmostEqual(d1, 59278.0)
        self.assertAlmostEqual(d2, 59247.00760376004)

