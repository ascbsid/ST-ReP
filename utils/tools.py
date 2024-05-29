import numpy as np
import torch
import csv
from utils import metrics


def cal_metric_log(logger, expid, args, yhat, realy, pred_len,
                       training_time, inference_time, avg_train_time, avg_inference_time,
                       result_csv_path, result_txt_path, log_csv_path, save_path, flag='torch', train_log_bag=None):
        pre_len = realy.shape[1]

        trainloss_record, bestid, log_in_train_details = [], None, []
        train_metric, valid_metric = None, None
        if train_log_bag is not None:
            trainloss_record, bestid, log_in_train_details, train_metric, valid_metric = train_log_bag
        
        all_amae, all_amape, all_amse, all_armse = [], [], [], []
        MAE_list, MAPE_list, MSE_list, RMSE_list = [], [], [], []
        metric_func ={'torch':metrics.metric_torch,'np':metrics.metric_np}[flag]

        for feature_idx in range(args.pre_dim):
            amae, amape, amse, armse = [], [], [], []
            pred_feature = yhat[..., feature_idx]
            print('pred_feature.shape:',pred_feature.shape)
            real_feature = realy[..., feature_idx]
            mid_print = True
            show_len = min(pre_len, 12)
            for i in range(show_len):
                scores = metric_func(pred_feature[:, i], real_feature[:, i])
                log = 'Evaluate best model on test data for [dim{:d}] horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test MSE: {:.4f}, Test RMSE: {:.4f}'
                if i<12 or i>pre_len-5: 
                    logger.info(log.format(feature_idx, i + 1, scores[0], scores[1], scores[2], scores[3]))
                else: 
                    if mid_print: 
                        logger.info('...')
                        mid_print = False

                amae.append(scores[0])
                amape.append(scores[1])
                amse.append(scores[2])
                armse.append(scores[3])
                
        
            metric_mae_func ={'torch':metrics.mae_torch,'np':metrics.mae_np}[flag]
            metric_mape_func ={'torch':metrics.masked_mape_torch,'np':metrics.masked_mape_np}[flag]
            metric_mse_func ={'torch':metrics.mse_torch,'np':metrics.mse_np}[flag]
            MAE = metric_mae_func(yhat, realy)
            MAPE = metric_mape_func(yhat, realy, mask_val=0)
            MSE = metric_mse_func(yhat, realy)
            if flag == 'torch':
                MAE = MAE.item()
                MAPE = MAPE.item()
                MSE = MSE.item()
            RMSE = MSE ** 0.5


            log = '[dim{:d}] On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test MSE: {:.4f}, Test RMSE: {:.4f}'
            logger.info(log.format(feature_idx, pre_len, MAE, MAPE, MSE, RMSE))

            ## final score 
            MAE_list.append(MAE)
            MAPE_list.append(MAPE)
            MSE_list.append(MSE)
            RMSE_list.append(RMSE)
            
            ## 
            all_amae.append(amae)
            all_amape.append(amape)
            all_amse.append(amse)
            all_armse.append(armse)

        with open(result_csv_path, 'a+', newline='')as f0:
            f_csv = csv.writer(f0)
            row = [expid, args.dataset_name, args.note+f'_{pred_len}', args.M, args.hid_dim, args.n_heads, pred_len, 'test']
            for feature_idx in range(args.pre_dim):
                row.extend([MSE_list[feature_idx], MAE_list[feature_idx]])
            if valid_metric is not None and train_metric is not None:
                row.extend(['val'])
                row.extend(valid_metric)
                row.extend(['train'])
                row.extend(train_metric)
            f_csv.writerow(row)


        logger.info('log and results saved.')

        if bool(args.save_output):
            if flag == 'torch':
                output_path = f'{save_path}/{args.modelid}_{args.dataset_name}_exp{expid}_output.npz'
                np.savez_compressed(output_path, prediction = yhat.cpu().numpy(), truth=realy.cpu().numpy())
                logger.info(f'{output_path}:output npz saved.')
            else:
                output_path = f'{save_path}/{args.modelid}_{args.dataset_name}_exp{expid}_Rep_output.npz'
                np.savez_compressed(output_path, prediction = yhat, truth=realy)
                logger.info(f'{output_path}:output npz saved.')