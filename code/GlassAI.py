import shap
import joblib
import lime
import lime.lime_tabular
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance


class Solutions:

    def __init__(self):
        self.methods = ['SHAP', 'LIME', 'permutation_importance']

    def SHAP(self, model_path, data):
        model = joblib.load(model_path)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        shap.force_plot(explainer.expected_value[1], shap_values[1], data, matplotlib=True)

    def LIME(self, model_path, train_data, live_data):
        model = joblib.load(model_path)
        explainer = lime.lime_tabular.LimeTabularExplainer(train_data.values,
                                                           feature_names=train_data.columns.values.tolist(),
                                                           verbose=True, mode='regression')
        exp = explainer.explain_instance(live_data.values, model.predict)
        exp.show_in_notebook(show_table=True)

    def permutation_importance(self, model_path, x_test, y_test):
        model = joblib.load(model_path)
        perm = PermutationImportance(model).fit(x_test, y_test)
        # eli5.show_weights(perm) needs to be in notebook
        print(eli5.format_as_text(eli5.explain_weights(perm)))


