#!/usr/bin/env python3
"""Inference worker for the CURRENT Kaggle bundle (9 Shukla features + IF score -> RBF-SVM)."""
import argparse, json, math, sys, traceback, time, hashlib, csv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, roc_curve
import numpy as np
import pandas as pd
import joblib

MAX_ROWS = 300_000
EULER_GAMMA = 0.5772156649015329

# These classes intentionally live in __main__ when this script is executed.
# That lets joblib resolve notebook-pickled __main__.CustomIsolationForest objects.
def c_factor(n):
    n = int(n)
    if n <= 1: return 0.0
    if n == 2: return 1.0
    return 2.0 * (np.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n

class IsolationNode:
    __slots__ = ("feature", "threshold", "left", "right", "size", "is_leaf")
    def __init__(self, feature=None, threshold=None, left=None, right=None, size=0, is_leaf=False):
        self.feature = feature; self.threshold = threshold; self.left = left; self.right = right
        self.size = int(size); self.is_leaf = bool(is_leaf)

class IsolationTree:
    def __init__(self, height_limit, rng):
        self.height_limit = int(height_limit); self.rng = rng; self.root = None
    def fit(self, X):
        self.root = self._build(np.asarray(X, dtype=np.float64), 0); return self
    def _build(self, X, height):
        n = X.shape[0]
        if height >= self.height_limit or n <= 1:
            return IsolationNode(size=n, is_leaf=True)
        mins, maxs = np.min(X, axis=0), np.max(X, axis=0)
        valid = np.where(maxs > mins)[0]
        if len(valid) == 0: return IsolationNode(size=n, is_leaf=True)
        f = int(self.rng.choice(valid)); t = float(self.rng.uniform(mins[f], maxs[f]))
        lm = X[:, f] < t
        if lm.sum() == 0 or (~lm).sum() == 0: return IsolationNode(size=n, is_leaf=True)
        return IsolationNode(f, t, self._build(X[lm], height+1), self._build(X[~lm], height+1), n, False)
    def path_length(self, x):
        node, depth = self.root, 0
        while not node.is_leaf:
            node = node.left if x[node.feature] < node.threshold else node.right; depth += 1
        return float(depth) + (c_factor(node.size) if node.size > 1 else 0.0)

class CustomIsolationForest:
    def __init__(self, n_trees=100, sample_size=256, random_state=42):
        self.n_trees=int(n_trees); self.sample_size=int(sample_size); self.random_state=int(random_state)
        self.trees=[]; self.actual_sample_size=None; self.height_limit=None
    def fit(self, X):
        X=np.asarray(X,dtype=np.float64); n=X.shape[0]; self.actual_sample_size=min(self.sample_size,n)
        self.height_limit=int(math.ceil(math.log2(self.actual_sample_size))); rng=np.random.default_rng(self.random_state); self.trees=[]
        for _ in range(self.n_trees):
            idx=rng.choice(n,size=self.actual_sample_size,replace=False); seed=int(rng.integers(0,np.iinfo(np.int32).max))
            self.trees.append(IsolationTree(self.height_limit,np.random.default_rng(seed)).fit(X[idx]))
        return self
    def mean_path_length(self, X):
        X = np.asarray(X, dtype=np.float64)
        total = np.zeros(len(X))
        # Traverse groups at each node rather than calling Python for each row.
        for tree in self.trees:
            stack = [(tree.root, np.arange(len(X)), 0)]
            while stack:
                node, rows, depth = stack.pop()
                if not len(rows):
                    continue
                if node.is_leaf:
                    total[rows] += depth + c_factor(node.size)
                else:
                    left = X[rows, node.feature] < node.threshold
                    stack.append((node.left, rows[left], depth + 1))
                    stack.append((node.right, rows[~left], depth + 1))
        return total / len(self.trees)
    def anomaly_score(self, X):
        return np.power(2.0, -(self.mean_path_length(X)/c_factor(self.actual_sample_size)))

class IFScoreAugmenter(TransformerMixin, BaseEstimator):
    """Train-only IF and score normalization; cloned inside each CV fold."""
    def __init__(self, n_trees=100, sample_size=256, random_state=42):
        self.n_trees=n_trees; self.sample_size=sample_size; self.random_state=random_state
    def fit(self, X, y=None):
        self.n_features_in_=np.asarray(X).shape[1]
        self.if_scaler_=MinMaxScaler().fit(X)
        values=self.if_scaler_.transform(X)
        self.forest_=CustomIsolationForest(self.n_trees,self.sample_size,self.random_state).fit(values)
        self.score_scaler_=MinMaxScaler(clip=True).fit(self.forest_.anomaly_score(values).reshape(-1,1))
        return self
    def transform(self, X):
        values=np.asarray(X,dtype=float)
        scores=self.forest_.anomaly_score(self.if_scaler_.transform(values))
        normalized=self.score_scaler_.transform(scores.reshape(-1,1))
        return np.column_stack([values,normalized])

FEATURES = ['sttl','ct_state_ttl','dload','rate','dmean','dttl','ackdat','sload','sbytes']

def metrics(y, pred, scores):
    y, pred = np.asarray(y), np.asarray(pred)
    tp=int(((y==1)&(pred==1)).sum()); tn=int(((y==0)&(pred==0)).sum())
    fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum())
    divide=lambda a,b: float(a/b) if b else 0.0
    pre=divide(tp,tp+fp); rec=divide(tp,tp+fn)
    result=dict(tp=tp,tn=tn,fp=fp,fn=fn,accuracy=divide(tp+tn,len(y)),
        precision=pre,recall=rec,f1=divide(2*pre*rec,pre+rec),
        f0_5=divide(1.25*pre*rec,.25*pre+rec),specificity=divide(tn,tn+fp),
        fpr=divide(fp,fp+tn),fnr=divide(fn,fn+tp),
        mcc=divide(tp*tn-fp*fn,math.sqrt(float(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),
        roc_auc=None,pr_auc=None,average_precision=None,roc_curve=[],pr_curve=[])
    if len(np.unique(y)) == 2:
        fpr,tpr,_=roc_curve(y,scores)
        p,r,_=precision_recall_curve(y,scores)
        result.update(roc_auc=float(roc_auc_score(y,scores)),pr_auc=float(auc(r,p)),
            average_precision=float(average_precision_score(y,scores)))
        for key,x,v in [('roc_curve',fpr,tpr),('pr_curve',r,p)]:
            ids=np.unique(np.linspace(0,len(x)-1,min(200,len(x))).astype(int))
            result[key]=[[float(x[i]),float(v[i])] for i in ids]
    return result

def analyse(raw, bundle):
    if 'models' in bundle:
        return analyse_corrected(raw, bundle)
    raise ValueError('This prototype requires the corrected E1_E5_models.joblib. Run the corrected notebook and install its export; the old three-model bundle is incompatible.')

def analyse_legacy(raw, bundle):
    started=time.perf_counter()
    raw=raw.copy(); raw.columns=raw.columns.astype(str).str.strip().str.lower()
    if raw.columns.duplicated().any():
        raise ValueError('Duplicate column names after normalization.')
    if raw.empty: raise ValueError('CSV has no rows.')
    if len(raw)>MAX_ROWS: raise ValueError(f'Limit is {MAX_ROWS:,} rows. Split the file; no records were silently discarded.')
    if bundle.get('original_features') != FEATURES:
        raise ValueError('Bundle must contain the paper’s ordered nine-feature architecture.')
    missing=[c for c in FEATURES if c not in raw]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    y=None
    if 'label' in raw:
        labels=pd.to_numeric(raw['label'],errors='coerce')
        if not labels.isin([0,1]).all(): raise ValueError('Every label must be 0 (Normal) or 1 (Attack).')
        y=labels.to_numpy(dtype=int)
    numeric=raw[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    invalid=numeric.isna()
    clean=numeric.fillna(pd.Series(bundle['training_medians']))
    if not np.isfinite(clean.to_numpy()).all(): raise ValueError('Bundle training medians are missing or invalid.')
    original=clean.to_numpy(dtype=np.float64)
    t=time.perf_counter()
    scaled=bundle['if_scaler'].transform(clean)
    scores=bundle['isolation_forest'].anomaly_score(scaled)
    baseline=(scores<=bundle['if_threshold']).astype(int)
    if_time=time.perf_counter()-t
    hybrid_input=np.column_stack([original,scores])
    if getattr(bundle['hybrid_scaler'],'n_features_in_',0)!=10:
        raise ValueError('Hybrid scaler must expect exactly ten inputs.')
    t=time.perf_counter()
    standardized=bundle['hybrid_scaler'].transform(hybrid_input)
    decisions=np.concatenate([bundle['hybrid_svm'].decision_function(standardized[i:i+2048])
                             for i in range(0,len(raw),2048)])
    hybrid=(decisions>=bundle['hybrid_threshold']).astype(int)
    svm_time=time.perf_counter()-t
    t=time.perf_counter()
    standalone_input=bundle['standalone_scaler'].transform(original)
    standalone_scores=np.concatenate([bundle['standalone_svm'].decision_function(standalone_input[i:i+2048])
                              for i in range(0,len(raw),2048)])
    standalone=(standalone_scores>=bundle['standalone_threshold']).astype(int)
    secondary_time=time.perf_counter()-t
    payload=dict(ok=True,rows=len(raw),labelled=y is not None,truncated=False,
        meta=bundle.get('metadata',{}),official_test_candidate=False,
        provenance='User-uploaded data; held-out provenance must be confirmed against the experiment.',
        validation={'imputed_cells':int(invalid.sum().sum()),'imputed_by_feature':invalid.sum().astype(int).to_dict()},
        thresholds={'baseline':float(bundle['if_threshold']),'hybrid':float(bundle['hybrid_threshold'])},
        timing={'baseline_seconds':if_time,'hybrid_seconds':if_time+svm_time,
                'secondary_svm_seconds':secondary_time,'training_seconds':None},
        disagreements=int((baseline!=hybrid).sum()))
    for name,pred,rank in [('baseline',baseline,-scores),('hybrid',hybrid,decisions),('svm',standalone,standalone_scores)]:
        payload[name]=metrics(y,pred,rank) if y is not None else {}
        payload[name].update(rows=len(raw),attacks=int(pred.sum()),normal=int((pred==0).sum()))
    if y is not None: payload['actual']={'attacks':int(y.sum()),'normal':int((y==0).sum())}
    # Bounded detail transport: all rows are evaluated, first 500 are inspectable.
    payload['preview']=[dict(row=i+1,features={c:(None if pd.isna(numeric.iloc[i][c]) else float(numeric.iloc[i][c])) for c in FEATURES},
        cleaned={c:float(clean.iloc[i][c]) for c in FEATURES},imputed=[c for c in FEATURES if invalid.iloc[i][c]],
        standardized=[float(v) for v in standardized[i]],if_score=float(scores[i]),baseline=int(baseline[i]),
        hybrid=int(hybrid[i]),svm=int(standalone[i]),hybrid_score=float(decisions[i]),
        **({'label':int(y[i])} if y is not None else {})) for i in range(min(500,len(raw)))]
    payload['preview_limit']=500
    payload['timing']['worker_compute_seconds']=time.perf_counter()-started
    return payload


def analyse_corrected(raw, bundle):
    started=time.perf_counter()
    raw=raw.copy(); raw.columns=raw.columns.astype(str).str.strip().str.lower()
    if raw.columns.duplicated().any(): raise ValueError('Duplicate normalized headers.')
    if raw.empty: raise ValueError('CSV has no rows.')
    if len(raw)>MAX_ROWS: raise ValueError('Maximum 300,000 rows per file; split the CSV.')
    if bundle.get('features')!=FEATURES: raise ValueError('Bundle feature order differs from paper.')
    missing=[c for c in FEATURES if c not in raw]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    if set(bundle['models'])!={'E2','E3','E4','E5'}: raise ValueError('Bundle must contain E2–E5.')
    y=None
    if 'label' in raw:
        labels=pd.to_numeric(raw['label'],errors='coerce')
        if not labels.isin([0,1]).all():raise ValueError('Every label must be 0 (Normal) or 1 (Attack).')
        y=labels.to_numpy(dtype=int)
    X=raw[FEATURES].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    invalid=X.isna(); times={}; predictions={}; ranks={}
    t=time.perf_counter()
    e1_values=bundle['e1_scaler'].transform(bundle['e1_imputer'].transform(X))
    e1_score=bundle['e1_forest'].anomaly_score(e1_values)
    predictions['E1']=(e1_score<=bundle['thresholds']['E1']).astype(int)
    ranks['E1']=-e1_score;times['E1']=time.perf_counter()-t
    for name,model in bundle['models'].items():
        t=time.perf_counter()
        ranks[name]=np.concatenate([model.decision_function(X.iloc[i:i+2048]) for i in range(0,len(X),2048)])
        predictions[name]=(ranks[name]>=bundle['thresholds'][name]).astype(int)
        times[name]=time.perf_counter()-t
    configurations={}
    for name in ['E1','E2','E3','E4','E5']:
        pred=predictions[name]
        configurations[name]=metrics(y,pred,ranks[name]) if y is not None else {}
        configurations[name].update(rows=len(X),attacks=int(pred.sum()),normal=int((pred==0).sum()),
            inference_seconds=times[name],features=10 if name in ['E4','E5'] else 9,smote=name in ['E3','E5'])
    mcnemar=None
    if y is not None:
        a=predictions['E2']==y;b=predictions['E4']==y
        bcount=int((a&~b).sum());ccount=int((~a&b).sum())
        pv=float(binomtest(bcount,bcount+ccount,p=.5).pvalue) if bcount+ccount else 1.0
        mcnemar=dict(comparison='E2 vs E4',e2_only_correct=bcount,e4_only_correct=ccount,
                     pvalue=pv,alpha=.05,reject_null=pv<.05,method='exact two-sided McNemar')
    # Inspect the actual final E5 preprocessing, which differs from the E1 reference IF.
    model=bundle['models']['E5']; detail=X.iloc[:500]
    clean=model.named_steps['imputer'].transform(detail)
    augmented=model.named_steps['augment'].transform(clean)
    standardized=model.named_steps['standardize'].transform(augmented)
    preview=[]
    for i in range(len(detail)):
        preview.append(dict(row=i+1,features={c:None if pd.isna(X.iloc[i][c]) else float(X.iloc[i][c]) for c in FEATURES},
            cleaned=dict(zip(FEATURES,map(float,clean[i]))),imputed=[c for c in FEATURES if invalid.iloc[i][c]],
            standardized=standardized[i].tolist(),if_score=float(augmented[i,-1]),e1_if_score=float(e1_score[i]),
            baseline=int(predictions['E1'][i]),svm=int(predictions['E2'][i]),hybrid=int(predictions['E5'][i]),
            experiments={k:int(v[i]) for k,v in predictions.items()},hybrid_score=float(ranks['E5'][i]),
            **({'label':int(y[i])} if y is not None else {})))
    meta=bundle.get('metadata',{}).copy();meta['sklearn_version']=meta.get('sklearn')
    return dict(ok=True,rows=len(X),labelled=y is not None,truncated=False,meta=meta,
        official_test_candidate=False,provenance='Uploaded data; confirm held-out provenance against notebook hashes.',
        validation={'imputed_cells':int(invalid.sum().sum()),'imputed_by_feature':invalid.sum().to_dict()},
        thresholds={'baseline':bundle['thresholds']['E1'],'hybrid':bundle['thresholds']['E5']},
        timing={'baseline_seconds':times['E1'],'hybrid_seconds':times['E5'],
                'training_seconds':None,'worker_compute_seconds':time.perf_counter()-started},
        disagreements=int((predictions['E1']!=predictions['E5']).sum()),
        baseline=configurations['E1'],svm=configurations['E2'],hybrid=configurations['E5'],
        configurations=configurations,mcnemar=mcnemar,preview=preview,preview_limit=500)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--bundle',required=True); p.add_argument('--input',required=True); p.add_argument('--name')
    a=p.parse_args()
    try:
        with open(a.input,encoding='utf-8-sig',newline='') as f:
            header=next(csv.reader(f),[])
        normalized=[c.strip().lower() for c in header]
        if len(set(normalized))!=len(normalized): raise ValueError('Duplicate CSV headers are not allowed.')
        raw=pd.read_csv(a.input,low_memory=False,nrows=MAX_ROWS+1)
        bundle=joblib.load(a.bundle)
        payload=analyse(raw,bundle)
        payload['file']=a.name or 'uploaded.csv'
        for key,path in [('input_sha256',a.input),('bundle_sha256',a.bundle)]:
            digest=hashlib.sha256()
            with open(path,'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
            payload[key]=digest.hexdigest()
        print(json.dumps(payload,allow_nan=False)); return 0
    except (ValueError,KeyError,pd.errors.ParserError,pd.errors.EmptyDataError) as e:
        print(json.dumps({'ok':False,'error':str(e)})); return 2
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'ok':False,'error':'Inference failed. Check the local worker log and model/dependency compatibility.'})); return 1

if __name__=='__main__': sys.exit(main())
