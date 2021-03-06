def get_default_params():
    return { "verbose" : 0,
             "nb_epoch" : 100,
             "standart_ckbs" : "LHESMC",
             "add_callbacks" : [],
             "patience" : 50,
             "batch_size" : 128,
             "log_path" : "./insopt_logs/",
             "train_test_split" : 0.15,
             "normalize" : False,
             "random_state" : 23,
             "cross_val_shuffle" : True,
             "cross_val_by" : None,
             "callbacks_monitor" : "val_loss",
             "mc_monitor" : "val_loss",
             "mc_mode" : "auto",
             "es_monitor" : "val_loss",
             "es_mode" : "auto",
             "metrics" : [],
             "delete_by_col" : True,
             "metric" : "r2",
             "corruption_level" : 0,
             "dump" : True,
             "prev_log_history_path" : None,
             "deep_feature_selection" : False,
             "balance_dataset" : None,
             "oversampling_ratio" : "auto",
             "smote_neighbors" : 5,
             "to_categorical" : False,
             "loss_weights": None
            }