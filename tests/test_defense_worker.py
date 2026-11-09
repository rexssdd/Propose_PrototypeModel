"""Behavior checks for the paper-aligned worker; no training dataset required."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('defense_worker',ROOT/'ml/predict.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class DefenseWorkerTest(unittest.TestCase):
    def test_vectorized_forest_matches_individual_paths(self):
        rng=np.random.default_rng(42)
        train=rng.normal(size=(100,9)); rows=rng.normal(size=(37,9))
        rows=np.vstack([rows,train[:3]])
        forest=w.CustomIsolationForest(n_trees=7,sample_size=32).fit(train)
        expected=np.mean([[tree.path_length(row) for row in rows] for tree in forest.trees],axis=0)
        np.testing.assert_allclose(forest.mean_path_length(rows),expected,rtol=0,atol=1e-14)
        np.testing.assert_allclose(forest.anomaly_score(rows),2**(-expected/w.c_factor(32)))

    def test_metrics_match_known_counts_and_tied_score_areas(self):
        y=np.array([0,1,0,1]); pred=np.array([0,1,1,0]); scores=np.array([.1,.8,.8,.2])
        r=w.metrics(y,pred,scores)
        self.assertEqual([r[k] for k in ['tn','fp','fn','tp']],[1,1,1,1])
        self.assertEqual(r['f0_5'],.5)
        self.assertEqual(r['roc_auc'],roc_auc_score(y,scores))
        self.assertEqual(r['average_precision'],average_precision_score(y,scores))

    def test_single_class_has_no_undefined_auc(self):
        r=w.metrics(np.array([0,0]),np.array([0,0]),np.array([.1,.2]))
        self.assertIsNone(r['roc_auc']); self.assertIsNone(r['pr_auc'])

    def test_rejects_invalid_label_before_loading_preprocessors(self):
        for value in [2,-1,.5,None,'invalid']:
            raw=pd.DataFrame([{**dict.fromkeys(w.FEATURES,1),'label':value}])
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'Every label'):
                w.analyse(raw,{'original_features':w.FEATURES})

    def test_rejects_duplicate_and_missing_columns_and_empty_data(self):
        for frame,message in [(pd.DataFrame([[1,2]],columns=['rate',' RATE ']),'Duplicate'),
                              (pd.DataFrame({'rate':[1]}),'Missing'),
                              (pd.DataFrame(columns=w.FEATURES),'no rows')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message):
                w.analyse(frame,{'original_features':w.FEATURES})

if __name__=='__main__': unittest.main()
