import argparse

def AddModelArgs(parser):

    parser.add_argument("--lora",action="store_true", help="whether use lora fine-tunning")

    parser.add_argument("--prompt_pool",action="store_true")

    parser.add_argument("--ln_grad",action="store_true", help="whether to calculate gradient of LayerNorm ")

    parser.add_argument("--causal", default=0, type=int,
                            help="LLM causal attention")
    
    parser.add_argument("--prompt_prefix", default=None ,type=str, help="whether use prompt or not")


    parser.add_argument("--node_embedding", action="store_true")

    parser.add_argument("--time_token", action="store_true")


    parser.add_argument("--model", default="phi2" ,type=str)

    parser.add_argument("--llm_layers", default=None, type=int)

    parser.add_argument("--dropout", default=0, type=float)

    parser.add_argument("--trunc_k", default=16, type=int)

    parser.add_argument("--t_dim", default=64, type=int)

    parser.add_argument("--node_emb_dim", default=128, type=int)

    parser.add_argument("--sandglassAttn", action="store_true")
    parser.add_argument("--wo_conloss" , action="store_true")
    parser.add_argument("--sag_dim", default=128, type=int)
    parser.add_argument("--sag_tokens", default=128, type=int)


def AddDataArgs(parser):

    parser.add_argument("--dataset" ,type=str)

    parser.add_argument("--data_path" ,type=str)

    parser.add_argument("--adj_filename" ,default=None , type=str)

    parser.add_argument("--sample_len", default=12, type=int)

    parser.add_argument("--predict_len", default=12, type=int)

    # parser.add_argument("--node_num", type=int)

    # parser.add_argument("--features", type=int)

    parser.add_argument("--train_ratio", default=0.6, type=float)

    parser.add_argument("--val_ratio", default=0.6, type=float)

    parser.add_argument("--input_dim", default=1, type=int)

    parser.add_argument("--output_dim", default=1, type=int)

def AddTrainArgs(parser):

    parser.add_argument("--lr", default=0.001, type=float)

    parser.add_argument("--lr_decay", default=0.99, type=float)

    parser.add_argument("--weight_decay", default=0.05, type=float)

    parser.add_argument("--batch_size", default=4, type=int)

    parser.add_argument("--epoch", default=100, type=int)

    parser.add_argument("--val_epoch", default=5, type=int)

    parser.add_argument("--test_epoch", default=5, type=int)

    parser.add_argument("--patience", default=100, type=int)


def InitArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--desc", default='phi2_s_token', type=str,
                            help="description")
    
    parser.add_argument("--log_root", default='../logs', type=str,
                            help="Log root directory")
    
    parser.add_argument("--from_pretrained_model" , default=None ,type=str)

    parser.add_argument("--zero_shot" , action="store_true")

    parser.add_argument("--nni" , action="store_true")

    parser.add_argument("--save_result" , action="store_true")

    parser.add_argument("--few_shot" , default=1, type=float)

    parser.add_argument("--node_shuffle_seed" , default=None, type=int)

    parser.add_argument("--trainset_dynamic_missing" , action="store_true")

    parser.add_argument("--task" , default='prediction' ,choices=['prediction','imputation','all'],type=str)
    
    parser.add_argument("--target_strategy" , default='random' ,choices=['random','hybrid'],type=str)

    AddDataArgs(parser)

    AddModelArgs(parser)

    AddTrainArgs(parser)

    args = parser.parse_args()

    return args