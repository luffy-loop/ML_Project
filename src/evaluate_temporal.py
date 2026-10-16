"""Optional later-presentation holdout evaluation.

Requires prepared cutoff-safe features retaining code_presentation. It trains on
earlier presentations and evaluates on the latest presentation. Do not use this
as evidence until the dataset's presentation chronology and label timing are
verified for the intended prediction point.
"""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def evaluate(features_path, output_path="reports/temporal_metrics.json"):
    df = pd.read_csv(features_path)
    required={"At_Risk","code_presentation","id_student"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Temporal evaluation requires columns: {sorted(missing)}")
    df["At_Risk"]=pd.to_numeric(df["At_Risk"],errors="coerce")
    df=df.dropna(subset=["At_Risk","code_presentation","id_student"]).copy()
    if df["At_Risk"].nunique()!=2: raise ValueError("Dataset must contain both target classes.")
    presentations=sorted(df["code_presentation"].astype(str).unique())
    if len(presentations)<2: raise ValueError("At least two presentations are required.")
    test_presentation=presentations[-1]
    train=df[df["code_presentation"].astype(str)<test_presentation]
    test=df[df["code_presentation"].astype(str)==test_presentation]
    # Drop the presentation field so the identifier for the held-out cohort is
    # not learned as a predictor; retain all other legitimate cutoff-safe fields.
    drop=["At_Risk","id_student","code_presentation"]
    X_train,y_train=train.drop(columns=drop),train["At_Risk"].astype(int)
    X_test,y_test=test.drop(columns=drop),test["At_Risk"].astype(int)
    if y_train.nunique()!=2 or y_test.nunique()!=2:
        raise ValueError("Train and latest-presentation test split must each contain both classes.")
    cat=X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()
    num=[c for c in X_train.columns if c not in cat]
    transformers=[]
    if num: transformers.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num))
    if cat: transformers.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat))
    model=Pipeline([("prep",ColumnTransformer(transformers)),("model",RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42,n_jobs=-1))])
    model.fit(X_train,y_train)
    pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
    result={"evaluation":"train on earlier code_presentation values; test on latest value",
      "train_presentations":sorted(train["code_presentation"].astype(str).unique().tolist()),
      "test_presentation":test_presentation,"train_rows":len(train),"test_rows":len(test),
      "train_students":int(train["id_student"].nunique()),"test_students":int(test["id_student"].nunique()),
      "accuracy":float(accuracy_score(y_test,pred)),"precision_at_risk":float(precision_score(y_test,pred,zero_division=0)),
      "recall_at_risk":float(recall_score(y_test,pred,zero_division=0)),"f1_at_risk":float(f1_score(y_test,pred,zero_division=0)),
      "roc_auc":float(roc_auc_score(y_test,prob)),"pr_auc":float(average_precision_score(y_test,prob)),
      "brier_score":float(brier_score_loss(y_test,prob)),"confusion_matrix":confusion_matrix(y_test,pred,labels=[0,1]).tolist(),
      "warning":"Confirm presentation ordering, feature availability, and prospective label timing. This report is not proof of deployment readiness."}
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--output",default="reports/temporal_metrics.json")
    a=p.parse_args(); print(json.dumps(evaluate(a.features,a.output),indent=2))
