import pandas as pd
import seaborn as sns


def plot_feature_importance(clf):
    feature_importance = list(zip(clf.model, clf.column_names))
    sns.barplot(x=0, y=1, data=pd.DataFrame(sorted(feature_importance, key=lambda x: abs(x[0]), reverse=True)))
