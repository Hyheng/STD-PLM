import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def MAPE_torch(pred, true, mask_value=1e-6):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAPE_torch_node(pred, true, mask_value=1e-6):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = pred*mask
        true = true*mask + (1-mask.float())
        count = mask.sum(dim=-1)
    return torch.sum(torch.abs(torch.div((true - pred)*mask, true)),dim=-1)/count


def cal_metrics(predicts,targets,eval_mask):
    F = targets.shape[-1]

    mae = []
    for f in range(F):
        mask = eval_mask[...,f]
        mae.append(MAE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    rmse = []
    for f in range(F):
        mask = eval_mask[...,f]
        rmse.append(RMSE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    mape = []
    for f in range(F):
        mask = eval_mask[...,f]
        mape.append(MAPE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    mape_10 = []
    for f in range(F):
        mask = eval_mask[...,f]
        mask = mask & (targets[...,0] >= 10)
        mape_10.append(MAPE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())

    mape_20 = []
    for f in range(F):
        mask = eval_mask[...,f]
        mask = mask & (targets[...,0] >= 20)
        mape_20.append(MAPE_torch(pred=predicts[...,f][mask],true=targets[...,f][mask]).item())  

    return mae,rmse,mape,mape_10,mape_20