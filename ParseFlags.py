import argparse
import sys

def parse_flags(hparams):
    parser = argparse.ArgumentParser(description="Photovoltaic estimation")
    parser.add_argument("--log_filename", type=str, default="output.txt")
    
    # Flags common to all modes
    all_modes_group = parser.add_argument_group("Flags common to all modes")
    all_modes_group.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    all_modes_group.add_argument("--model", type=str, choices=["RCNN", "RNN", "LSTM"], required=True) 

    # Flags for training only
    training_group = parser.add_argument_group("Flags for training only")
    training_group.add_argument("--save_dir", type=str, default="")
    training_group.add_argument("--aws_dir", type=str, default="./dataset/AWS/")
    training_group.add_argument("--asos_dir", type=str, default="./dataset/ASOS/")
    training_group.add_argument("--solar_dir", type=str, default="./dataset/photovoltaic/GWNU_C9/")
    training_group.add_argument("--loc_ID", type=int, default=678)

    # Flags for validation only
    validation_group = parser.add_argument_group("Flags for validation only")
    validation_group.add_argument("--val_aws_dir", type=str, default="./dataset/AWS/")
    validation_group.add_argument("--val_asos_dir", type=str, default="./dataset/ASOS/")
    validation_group.add_argument("--val_solar_dir", type=str, default="./dataset/photovoltaic/PreSchool/")
    # validation_group.add_argument("--val_solar_dir", type=str, default="./dataset/photovoltaic/GWNU_C3/")
    
    # Flags for test only
    test_group = parser.add_argument_group("Flags for test only")
    test_group.add_argument("--load_path", type=str, default="")
    test_group.add_argument("--tst_samcheok_aws_dir", type=str, default="./dataset/AWS/")
    test_group.add_argument("--tst_samcheok_asos_dir", type=str, default="./dataset/ASOS/")
    test_group.add_argument("--tst_samcheok_solar_dir", type=str, default="./samcheok/data/")
    test_group.add_argument("--tst_samcheok_loc_ID", type=int, default=106)

    test_group.add_argument("--tst_gwnuC3_aws_dir", type=str, default="./dataset/AWS/")
    test_group.add_argument("--tst_gwnuC3_asos_dir", type=str, default="./dataset/ASOS/")
    test_group.add_argument("--tst_gwnuC3_solar_dir", type=str, default="./dataset/photovoltaic/GWNU_C3/")
    test_group.add_argument("--tst_gwnuC3_loc_ID", type=int, default=678)
        
    # Flags for training params
    trn_param_set = parser.add_argument_group("Flags for training paramters")
    trn_param_set.add_argument(
        "--seqLeng", type=int, default=hparams["model"]["seqLeng"]
    )
    trn_param_set.add_argument(
        "--nBatch", type=int, default=hparams["learning"]["nBatch"]
    )
    trn_param_set.add_argument(
        "--max_epoch", type=int, default=hparams["learning"]["max_epoch"]
    )
        
    flags = parser.parse_args()

    hparams["model"]["seqLeng"] = flags.seqLeng
    hparams["learning"]["nBatch"] = flags.nBatch
    hparams["learning"]["max_epoch"] = flags.max_epoch

    # Additional per-mode validation
    try:
        if flags.mode == "train":
            assert flags.save_dir, "Must specify --save_dir"
        elif flags.mode == "test":
            assert flags.load_path, "Must specify --load_path"

    except AssertionError as e:
        print("\nError: ", e, "\n")
        parser.print_help()
        sys.exit(1)

    return flags, hparams, flags.model
