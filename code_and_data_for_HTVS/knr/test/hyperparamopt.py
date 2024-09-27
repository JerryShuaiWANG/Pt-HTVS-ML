#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : QsarTrain.py
@Description  : 
@Time         : 2020/05/27 15:51:50
@Author       : Shengde Zhang
@Version      : 1.0
'''

import os
import hyperopt
import pandas as pd
import numpy as np
from numpy.random import RandomState
import sys
sys.path.append("/home/shuaiw/OLED_ML/quantum_optical_ml/quantum_optical_ml_ks/data/processed/outlier")
from QsarUtils import *
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from hyperopt import hp
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor


def MainRegression(in_file_path, saved_dir, feature_selector_list, select_des_num_list, test_file_path=None, model_list=("RF",), 
         search_max_evals=250, des_type=("all",), smi_column="Smiles", name_column="Name", label_column="", sep=",", group_column="", k=10,
         test_size=0.2,  kfold_type="stratified", random_state=0, search_metric="val_RMSE", greater_is_better=False,):
    """
    :param in_file_path:
    :param saved_dir:
    :param feature_selector_list: list/tup,'RFE', 'f_regression', or 'mutual_info_regression'
    :param select_des_num_list: int/tup,
    :param des_type: str/list/tup, 'rdkit' / 'ecfp4'/ 'macss' / 'rdkfp' / 'ttfp' / 'all'
    :param smi_column:
    :param name_column:
    :param label_column:
    :param group_column:
    :param test_size: float
    :param kfold_type: str, 'normal'(KFold) / 'stratified'(StratifiedKFold) / 'group'(GroupKFold) / 'none'(all train)
    :param random_state:
    :return:
    """
    label = 'Exp. Knr'
    ##dataset = pd.read_csv("C:/Users/Sara/Documents/paper/project/with_des_alldata_latest_211_modified_and_added_V7_emission_Kr.csv")
    # dataset = pd.read_csv(r"C:\Users\Jerry Wang\Desktop\For paper\code\train_after_spxy_published_PBE0_7_3.csv", index_col=0)
    # dataset_test= pd.read_csv(r"C:\Users\Jerry Wang\Desktop\For paper\code\test_after_spxy_published_PBE0_7_3.csv", index_col=0)
    dataset = pd.read_csv("train0.8.csv")
    dataset_test= pd.read_csv("test0.8.csv")
 
    x = dataset.drop(columns=[label])
    #####less feature######
    #x = x.drop(columns=["s1_1_1", "s1_1_2",	"s1_2_1",	"s1_2_2",	"s2_1_1",	"s2_1_2",	"s2_2_1",	"s2_2_2",	"s3_1_1",	"s3_1_2",	"s3_2_1",	"s3_2_2",  "t2_1_1",	"t2_1_2",	"t2_2_1",	"t2_2_2",	"t3_1_1",	"t3_1_2",	"t3_2_1",	"t3_2_2",  "refractive_index",    "electron_density_at_Pt"])
    ######least feature#######
    # x = x.drop(columns=["s1_1_1", "s1_1_2",	"s1_2_1",	"s1_2_2",	"s2_1_1",	"s2_1_2",	"s2_2_1",	"s2_2_2",	"s3_1_1",	"s3_1_2",	"s3_2_1",	"s3_2_2",  "t2_1_1",	"t2_1_2",	"t2_2_1",	"t2_2_2",	"t3_1_1",	"t3_1_2",	"t3_2_1",	"t3_2_2",  "refractive_index",    "electron_density_at_Pt", "atom_type_1",	"atom_type_2",	"atom_type_3",	"atom_type_4",	"distance_1",	"distance_2",	"distance_3",	"distance_4",  "hso"])
    train_X = x.to_numpy()
    train_y = dataset[label].to_numpy()
    # train_index = df_mol_des.index
    
    test_x = dataset_test.drop(columns=[label])
    #####less feature######
    #test_x = test_x.drop(columns=["s1_1_1", "s1_1_2",	"s1_2_1",	"s1_2_2",	"s2_1_1",	"s2_1_2",	"s2_2_1",	"s2_2_2",	"s3_1_1",	"s3_1_2",	"s3_2_1",	"s3_2_2",  "t2_1_1",	"t2_1_2",	"t2_2_1",	"t2_2_2",	"t3_1_1",	"t3_1_2",	"t3_2_1",	"t3_2_2",  "refractive_index",    "electron_density_at_Pt"])
    ######least feature#######
    # test_x = test_x.drop(columns=["s1_1_1", "s1_1_2",	"s1_2_1",	"s1_2_2",	"s2_1_1",	"s2_1_2",	"s2_2_1",	"s2_2_2",	"s3_1_1",	"s3_1_2",	"s3_2_1",	"s3_2_2",  "t2_1_1",	"t2_1_2",	"t2_2_1",	"t2_2_2",	"t3_1_1",	"t3_1_2",	"t3_2_1",	"t3_2_2",  "refractive_index",    "electron_density_at_Pt", "atom_type_1",	"atom_type_2",	"atom_type_3",	"atom_type_4",	"distance_1",	"distance_2",	"distance_3",	"distance_4",  "hso"])
    test_X = test_x.to_numpy()
    test_y = dataset_test[label].to_numpy()


    model = RegressionModel(random_state=random_state)
    model.LoadData(train_X, train_y, test_X, test_y)
    model.ScaleFeature(saved_dir=saved_dir,saved_file_note="_".join(des_type))
    model.KFoldSplit(k=k, kfold_type=kfold_type)

    def Search(params):
        nonlocal model
        nonlocal estimator
        nonlocal search_metric
        nonlocal greater_is_better
        print("#"*20)
        print("params: ",params)
        print("#"*20)
        feature_selector = "f_regression"
        select_des_num = 50
        if "feature_selector" in params:
            feature_selector = params["feature_selector"]
            del params["feature_selector"]
        if "select_des_num" in params:
            select_des_num = int(params["select_des_num"])
            del params["select_des_num"]
        if (model.feature_selector_name != feature_selector) or (model.feature_select_num != select_des_num):
            model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num)
        else:
            pass
        model.Train(estimator,params=params)
        val_metric = model.all_metrics_df.loc["mean",search_metric]
        if greater_is_better:
            return -val_metric
        else:
            return val_metric

    lr_model = LinearRegression()
    lr_params = {}

    xgbr_model = XGBRegressor(objective='reg:squarederror', random_state=random_state)
    xgbr_params = {
        'gamma': hyperopt.hp.uniform("gamma", 0, 0.5),
        'max_depth': hyperopt.hp.uniformint('max_depth', 3, 11),
        'min_child_weight': hyperopt.hp.uniformint('min_child_weight', 1, 20),
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
         'max_delta_step':hyperopt.hp.uniform('max_delta_step', 0.5, 1),
         'reg_alpha': hyperopt.hp.uniform('reg_alpha', 0, 0.5),
         'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.5, 1),
         'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 0, 0.2),
    }

    rfr_model = RandomForestRegressor(random_state=random_state)
    rfr_parms = {'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
                'max_leaf_nodes': hyperopt.hp.uniformint('max_leaf_nodes', 10, 100),
                'min_samples_split': hyperopt.hp.uniformint('min_samples_split', 2, 10),
                'min_samples_leaf': hyperopt.hp.uniformint('min_samples_leaf', 1, 10),
                }

    svr_model = SVR()
    svr_params = {'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                  'gamma': hyperopt.hp.uniform("gamma", 1e-5, 1e2),
                  'epsilon': hyperopt.hp.uniform("epsilon", 1e-5, 1),
                  }

    cat_model = CatBoostRegressor()
    cat_params = {'depth': hyperopt.hp.uniformint('depth', 1, 13),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.0001, 0.2),
            'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 0.5, 1),
            'iterations': hyperopt.hp.uniformint('iterations', 10, 500),
            "bagging_temperature" :hyperopt.hp.uniform('bagging_temperature', 0, 1),
            }

    knnD_model = KNeighborsRegressor(algorithm="auto", weights='distance')#KNUni, KNDist
    knnD_params  = {'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    knnU_model = KNeighborsRegressor(algorithm="auto", weights='uniform')#KNUni, KNDist
    knnU_params  = {'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    lgbm_model = LGBMRegressor()
    lgbm_params  = {'num_leaves': hyperopt.hp.uniformint('num_leaves', 2, 13),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.2),
            'min_child_samples': hyperopt.hp.uniformint('min_child_samples', 0, 50),
            'max_depth': hyperopt.hp.uniformint('max_depth', 0, 13),
            'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
            "bagging_fraction" :hyperopt.hp.uniform('bagging_fraction', 0.5, 1),
            }
    krr_model = KernelRidge()
    krr_params  = {'alpha': hyperopt.hp.uniform('alpha', 0, 3),
            }

    # pls_model = PLSRegression()
    ada_model = AdaBoostRegressor() #0.18
    ada_params  = {'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
        }

    model_param_dict = {"LR": {"estimator": lr_model, "params": lr_params},
                        "XGB": {"estimator": xgbr_model, "params": xgbr_params},
                        "RF":{"estimator": rfr_model, "params": rfr_parms},
                        "SVM":{"estimator": svr_model, "params": svr_params},
                        "CAT":{"estimator": cat_model, "params": cat_params},
                        "KNNU":{"estimator": knnU_model, "params": knnU_params},
                        "KNND":{"estimator": knnD_model, "params": knnD_params},
                        "LGBM":{"estimator": lgbm_model, "params": lgbm_params},
                        "KRR":{"estimator": krr_model, "params": krr_params},
                        "ADA":{"estimator": ada_model, "params": ada_params},

                       }#CAT doesnot work

    for m in model_list:
        estimator = model_param_dict[m]["estimator"]
        model_name = str(estimator)
        model_name = model_name[:model_name.find("(")]
        # feature_selector_list = ["RFE"]
        # select_des_num_list = [n for n in range(10,100,10)]
        # select_des_num_list = [40]
        params_space = {"feature_selector": hyperopt.hp.choice('feature_selector',feature_selector_list),
                        "select_des_num":hyperopt.hp.choice("select_des_num",select_des_num_list)}
        params_space.update(model_param_dict[m]["params"])
        best_params = hyperopt.fmin(Search, space=params_space, algo=hyperopt.tpe.suggest,
                                    max_evals=search_max_evals,rstate=np.random.default_rng(random_state))#RandomState(ran)
        for key,value in params_space.items():
            if value.name == "int":
                best_params[key] = int(best_params[key])
        print("Best params: ",best_params)
        select_des_num = select_des_num_list[best_params["select_des_num"]]
        feature_selector = feature_selector_list[best_params["feature_selector"]]
        model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num, saved_dir=saved_dir, saved_file_note=model_name)
        del best_params["select_des_num"]
        del best_params["feature_selector"]
        model.Train(estimator,params=best_params,saved_dir=saved_dir)
        model.all_metrics_df["model_name"] = model_name
        model.all_metrics_df["feature_num"] = select_des_num
        model.all_metrics_df["random_state"] = random_state
        metrics_out_file = os.path.join(saved_dir,"model_metrics.csv")
        model.all_metrics_df.to_csv(metrics_out_file,mode="a")
        model.SaveTotalModel(saved_dir=saved_dir,saved_file_note=random_state)
        # model.GenerateBallTree(p=1,saved_dir=saved_dir)
        model.DrawScatter(model.val_y_all, model.val_pred_all,  saved_dir=saved_dir, saved_file_note=model_name, data_group="validation")
        if model.test_y is not None:
            model.DrawScatter(model.test_y, model.test_pred_mean, saved_dir=saved_dir, saved_file_note=model_name)
'''
def MainClassification(in_file_path, saved_dir, feature_selector_list, select_des_num_list, model_list=("LRC", ), search_max_evals=30,
                        des_type=("all",), smi_column="Smiles", name_column="Name", label_column="", group_column="", k=5,
                        test_size=0.2,  kfold_type="normal", random_state=0, search_metric="val_CE", greater_is_better=False,):
    """
    :param in_file_path:
    :param saved_dir:
    :param feature_selector_list: list/tup,'RFE', 'f_classif', or 'mutual_info_classif'
    :param select_des_num_list: int/tup,
    :param des_type: str/list/tup, 'rdkit' / 'ecfp4'/ 'macss' / 'rdkfp' / 'ttfp' / 'all'
    :param smi_column:
    :param name_column:
    :param label_column:
    :param group_column:
    :param test_size: float
    :param kfold_type: str, 'normal'(KFold) / 'stratified'(StratifiedKFold) / 'group'(GroupKFold) / 'none'(all train)
    :param random_state:
    :return:
    """

    n_jobs = int(0.8 * os.cpu_count())

    in_df = pd.read_csv(in_file_path)
    print(in_df.shape)

    model = ClassificationModel(random_state=random_state)
    model.LoadData(train_X, train_y, test_X, test_y, train_groups=train_groups, des_type=des_type)
    model.ScaleFeature(saved_dir=saved_dir,saved_file_note="_".join(des_type))
    model.KFoldSplit(k=k, kfold_type=kfold_type)

    def Search(params):
        nonlocal model
        nonlocal estimator
        nonlocal search_metric
        nonlocal greater_is_better
        print("#"*20)
        print("params: ",params)
        print("#"*20)
        feature_selector = "f_classif"
        select_des_num = 50
        if "feature_selector" in params:
            feature_selector = params["feature_selector"]
            del params["feature_selector"]
        if "select_des_num" in params:
            select_des_num = int(params["select_des_num"])
            del params["select_des_num"]
        if (model.feature_selector_name != feature_selector) or (model.feature_select_num != select_des_num):
            model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num)
        else:
            pass
        model.Train(estimator,params=params)
        val_metric = model.all_metrics_df.loc["mean",search_metric]
        if greater_is_better:
            return -val_metric
        else:
            return val_metric

    lrc_model = LogisticRegression(random_state=random_state,n_jobs=n_jobs)
    lrc_params = {'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                   'class_weight': hyperopt.hp.choice('class_weight', ["balanced"])}

    xgbc_model = XGBClassifier(objective='binary:logistic', random_state=random_state, n_jobs=n_jobs)
    xgbc_params = {
        'gamma': hyperopt.hp.uniform("gamma", 0, 0.5),
        'max_depth': hyperopt.hp.uniformint('max_depth', 3, 11),
        'min_child_weight': hyperopt.hp.uniformint('min_child_weight', 1, 20),
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 50, 500),
        'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 0.01, 2),
    }

    rfc_model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    rfc_parms = {'n_estimators': hyperopt.hp.uniformint('n_estimators', 50, 500),
                'max_leaf_nodes': hyperopt.hp.uniformint('max_leaf_nodes', 10, 100),
                'min_samples_split': hyperopt.hp.uniformint('min_samples_split', 2, 10),
                'min_samples_leaf': hyperopt.hp.uniformint('min_samples_leaf', 1, 10),
                'class_weight': hyperopt.hp.choice('class_weight', ["balanced","balanced_subsample", None])
                }

    svc_model = SVC(random_state=random_state)
    svc_params = {'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                  'gamma': hyperopt.hp.uniform("gamma", 1e-5, 1e2),
                  'class_weight': hyperopt.hp.choice('class_weight', ["balanced", None]),}
    
    model_param_dict = {"LR": {"estimator": lrc_model, "params": lrc_params},
                        "XGB": {"estimator": xgbc_model, "params": xgbc_params},
                        "RF":{"estimator": rfc_model, "params": rfc_parms},
                        "SVM":{"estimator": svc_model, "params": svc_params},
                       }


    for m in model_list:
        estimator = model_param_dict[m]["estimator"]
        model_name = str(estimator)
        model_name = model_name[:model_name.find("(")]
        
        params_space = {"feature_selector": hyperopt.hp.choice('feature_selector',feature_selector_list),
                        "select_des_num":hyperopt.hp.choice("select_des_num",select_des_num_list)}
        
        params_space.update(model_param_dict[m]["params"])
        trials = hyperopt.Trials()
        best_params = hyperopt.fmin(Search, space=params_space, algo=hyperopt.tpe.suggest,return_argmin=False,
                                    max_evals=search_max_evals,rstate=RandomState(random_state),trials=trials)
        for key,value in params_space.items():
            if value.name == "int":
                best_params[key] = int(best_params[key])
        print("Best params: ",best_params)

        select_des_num = best_params["select_des_num"]
        feature_selector = best_params["feature_selector"]
        model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num, saved_dir=saved_dir, saved_file_note=model_name)
        del best_params["select_des_num"]
        del best_params["feature_selector"]
        model.Train(estimator,params=best_params,saved_dir=saved_dir)
        model.all_metrics_df["model_name"] = model_name
        model.all_metrics_df["feature_num"] = select_des_num
        model.all_metrics_df["random_state"] = random_state
        metrics_out_file = os.path.join(saved_dir,"model_metrics.csv")
        model.all_metrics_df.to_csv(metrics_out_file,mode="a")

        model.GenerateBallTree(p=1,saved_dir=saved_dir)
        model.SaveTotalModel(saved_dir=saved_dir,saved_file_note=random_state)
'''
if __name__ == "__main__":

###########Regression################
    t0 = time.time()
    data_dir = "./"
    in_file_name = ""
    in_file_path = os.path.join(data_dir, in_file_name)
    os.mkdir("./_/")
    des_type = ('')
    # des_type = ("ecfp4",)
    random_state_list = (63,)
    # random_state_list = (i for i in range(0,501))
    # random_state_list = (42,)
    feature_selector_list = ("RFE",)
    select_des_num_list = (30,)#totally 46 with Cal. Kr, less totally 24, leaset 15
    #model_list = ("LGBM", )
    # model_list = ("SVM", )
    model_list = ("SVM","XGB","LGBM","RF","KNND","KNNU","KRR","ADA", )
    #model_list = ("XGB", "LGBM","RF","SVM",)
    
    smi_column = "cn_smiles"
    name_column = "init_id"
    label_column = "pValue"
    group_column = "cluster_label"
    # group_column = "Exp. Kr"
    test_size = 0
    kfold_type = "normal"
    # kfold_type = "stratified"
    # search_max_evals = 250
    search_max_evals = 10
    search_metric = "val_RMSE"
    # search_metric = "val_R2"
    k = 10

    saved_dir = os.path.join(data_dir, "{}_{}".format(in_file_name[-11:],"_".join(des_type)))
    print(saved_dir)

    for random_state in random_state_list:
        MainRegression(in_file_path, saved_dir, feature_selector_list, select_des_num_list, model_list=model_list, search_max_evals=search_max_evals, 
                        des_type=des_type, smi_column=smi_column, name_column=name_column, label_column=label_column, group_column=group_column, k=k,
                        test_size=test_size, kfold_type=kfold_type, random_state=random_state, search_metric=search_metric, greater_is_better=False)

    print("Time cost: {}".format(Sec2Time(time.time()-t0)))


###########Classification################

    # t0 = time.time()
    # data_dir = "C:/OneDrive/Jupyter_notebook/Data/"
    # in_file_name = "sPLA2_452_descriptors.csv"
    # in_file_path = os.path.join(data_dir, in_file_name)

    # # des_type = ('ecfp4', 'maccs','rdkfp','ttfp')
    # des_type = ("ecfp4",)
    # # random_state_list = (42, 0, 12)
    # random_state_list = (42,)
    # feature_selector_list = ("f_classif",)
    # select_des_num_list = (50,100,150,200,300)
    # model_list = ("LR", )
    # search_max_evals = 50
    # smi_column = "Smiles"
    # name_column = "Name"
    # label_column = "Activity"
    # group_column = "Activity"
    # test_size = 0
    # kfold_type = "stratified"
    # search_metric = "val_CE"

    # saved_dir = os.path.join(data_dir, "{}_{}".format(in_file_name[:-4],"_".join(des_type)))
    # print(saved_dir)

    # for random_state in random_state_list:
    #     MainClassification(in_file_path, saved_dir, feature_selector_list, select_des_num_list, model_list=model_list, k=5,
    #                         search_max_evals=search_max_evals, des_type=des_type, smi_column=smi_column, name_column=name_column, 
    #                         label_column=label_column, group_column=group_column, test_size=test_size, kfold_type=kfold_type, 
    #                         random_state=random_state, search_metric=search_metric, greater_is_better=False)

    # print("Time cost: {}".format(Sec2Time(time.time()-t0)))
