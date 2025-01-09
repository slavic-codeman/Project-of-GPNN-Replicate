"""
Created on Oct 05, 2017

@author: Siyuan Qi

Description of the file.

"""
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse
import time
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.autograd
import sklearn.metrics

import datasets
import units
import models

import config
import logutil
import utils
from datasets.HICO import metadata
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy
from visualization_utils import visualize_hoi
import cv2
from PIL import Image, ImageDraw, ImageFont


action_class_num = 117
hoi_class_num = 600


def evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=False):
    np_pred_node_labels = pred_node_labels.detach().cpu().numpy()
    np_pred_node_labels_exp = np.exp(np_pred_node_labels)
    np_pred_node_labels = np_pred_node_labels_exp/(np_pred_node_labels_exp+1)  # overflows when x approaches np.inf
    np_node_labels = node_labels.detach().cpu().numpy()

    new_y_true = np.empty((2 * len(det_indices), action_class_num))
    new_y_score = np.empty((2 * len(det_indices), action_class_num))
    for y_i, (batch_i, i, j) in enumerate(det_indices):
        new_y_true[2*y_i, :] = np_node_labels[batch_i, i, :]
        new_y_true[2*y_i+1, :] = np_node_labels[batch_i, j, :]
        new_y_score[2*y_i, :] = np_pred_node_labels[batch_i, i, :]
        new_y_score[2*y_i+1, :] = np_pred_node_labels[batch_i, j, :]

    y_true = np.vstack((y_true, new_y_true))
    y_score = np.vstack((y_score, new_y_score))
    return y_true, y_score


def weighted_loss(output, target):
    weight_mask = torch.autograd.Variable(torch.ones(target.size()))
    if hasattr(args, 'cuda') and args.cuda:
        weight_mask = weight_mask.cuda()
    link_weight = args.link_weight if hasattr(args, 'link_weight') else 1.0
    weight_mask += target * link_weight
    return torch.nn.MultiLabelSoftMarginLoss(weight=weight_mask).cuda()(output, target)


def loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num=[], obj_num=[]):
    pred_adj_mat_prob = torch.nn.Sigmoid()(pred_adj_mat)
    # 检查gt
    np_pred_adj_mat = pred_adj_mat_prob.detach().cpu().numpy()
    # np_gt_adj_mat = adj_mat.detach().cpu().numpy()
    # np_pred_adj_mat = pred_adj_mat.detach().cpu().numpy()
    det_indices = list()
    batch_size = pred_adj_mat.size()[0]
    loss = 0
    for batch_i in range(batch_size):
        valid_node_num = human_num[batch_i] + obj_num[batch_i]\
        # 检查ground truth
        np_pred_adj_mat_batch = np_pred_adj_mat[batch_i, :, :]
        # np_gt_adj_mat_batch = np_gt_adj_mat[batch_i, :, :]

        if len(human_num) != 0:
            human_interval = human_num[batch_i]
            obj_interval = human_interval + obj_num[batch_i]
        max_score = np.max([np.max(np_pred_adj_mat_batch), 0.01])
        mean_score = np.mean(np_pred_adj_mat_batch)
        batch_det_indices = np.where(np_pred_adj_mat_batch > 0.7) # default=0.5
        # batch_det_indices = np.where(np_gt_adj_mat_batch > 0.5)
        for i, j in zip(batch_det_indices[0], batch_det_indices[1]):
            # check validity for H-O interaction instead of O-O interaction
            if len(human_num) != 0:
                if i < human_interval and j < obj_interval:
                    if j >= human_interval:
                        det_indices.append((batch_i, i, j))

        loss = loss + weighted_loss(pred_node_labels[batch_i, :valid_node_num].view(-1, action_class_num), node_labels[batch_i, :valid_node_num].view(-1, action_class_num))
    return det_indices, loss


def compute_mean_avg_prec(y_true, y_score):
    try:

    
        avg_prec = sklearn.metrics.average_precision_score(y_true, y_score, average=None)
       
        mean_avg_prec = np.nansum(avg_prec) / len(avg_prec)
       


    except ValueError:
        mean_avg_prec = 0

    return mean_avg_prec


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    logger = logutil.Logger(os.path.join(args.log_root, timestamp))

    # Load data
    print("Loading data...")
    training_set, valid_set, testing_set, train_loader, valid_loader, test_loader, img_index = utils.get_hico_data(args)

    # Get data size and define model
    edge_features, node_features, adj_mat, node_labels, sequence_id, det_classes, det_boxes, human_num, obj_num = training_set[0]

    edge_feature_size, node_feature_size = edge_features.shape[2], node_features.shape[1]
    message_size = int(edge_feature_size/2)*2
    model_args = {'model_path': args.resume,
                   'edge_feature_size': edge_feature_size,
                    'node_feature_size': node_feature_size,
                       'message_size': message_size, 
                       'link_hidden_size': 512,
                         'link_hidden_layers': 2, 
                         'link_relu': False, 
                         'update_hidden_layers': 1,
                     'update_dropout': False, 
                     'update_bias': True, 
                     'propagate_layers': 3,
                     'hoi_classes': action_class_num, 
                     'resize_feature_to_message_size': False,
                     'update_type':'transformer'}
    
    print("Making model HICO...")
    model = models.GPNN_HICO(model_args)
    del edge_features, node_features, adj_mat, node_labels
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss = torch.nn.MSELoss(size_average=True)
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss(size_average=True)
    if args.cuda:
        model = model.cuda()
        mse_loss = mse_loss.cuda()
        multi_label_loss = multi_label_loss.cuda()

    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf


    for epoch in range(args.start_epoch, args.epochs):
        logger.log_value('learning_rate', args.lr).step()
        # train for one epoch
        train(train_loader, model, mse_loss, multi_label_loss, optimizer, epoch, logger)
        # test on validation set
        epoch_error = validate(valid_loader, model, mse_loss, multi_label_loss, logger)
        epoch_errors.append(epoch_error)
        if len(epoch_errors) == 2:
            new_avg_epoch_error = np.mean(np.array(epoch_errors))
            if avg_epoch_error - new_avg_epoch_error < 0.005:
                print('Learning rate decrease')
            avg_epoch_error = new_avg_epoch_error
            epoch_errors = list()

        if epoch % 5 == 0 and epoch > 0:
            args.lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        is_best = True
        best_epoch_error = min(epoch_error, best_epoch_error)
        datasets.utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                        'best_epoch_error': best_epoch_error, 'avg_epoch_error': avg_epoch_error,
                                        'optimizer': optimizer.state_dict(), },
                                       is_best=is_best, directory=args.resume)
        print('best_epoch_error: {}, avg_epoch_error: {}'.format(best_epoch_error,  avg_epoch_error))

    # For testing
    loaded_checkpoint = datasets.utils.load_best_checkpoint(args, model, optimizer)
    if loaded_checkpoint:
        args, best_epoch_error, avg_epoch_error, model, optimizer = loaded_checkpoint

    validate(test_loader, model, mse_loss, multi_label_loss, test=True)
    # gen_test_result(args, test_loader, model, mse_loss, multi_label_loss, img_index)

    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(train_loader, model, mse_loss, multi_label_loss, optimizer, epoch, logger):
    batch_time = logutil.AverageMeter()
    data_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    # switch to train mode
    model.train()

    end_time = time.time()

    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, det_classes, det_boxes, human_num, obj_num) in enumerate(train_loader):

        data_time.update(time.time() - end_time)
        optimizer.zero_grad()

        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num, obj_num)

        # Log and back propagate
        if len(det_indices) > 0:
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score)

        losses.update(loss.item(), edge_features.size()[0])
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % args.log_interval == 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mean Avg Precision {mean_avg_prec:.4f} ({mean_avg_prec:.4f})\t'
                  'Detected HOIs {y_shape}'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, mean_avg_prec=mean_avg_prec, y_shape=y_true.shape))

    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)

    if logger is not None:
        logger.log_value('train_epoch_loss', losses.avg)
        logger.log_value('train_epoch_map', mean_avg_prec)

    print('Epoch: [{0}] Avg Mean Precision {map:.4f}; Average Loss {loss.avg:.4f}; Avg Time x Batch {b_time.avg:.4f}'
          .format(epoch, map=mean_avg_prec, loss=losses, b_time=batch_time))


def validate(val_loader, model, mse_loss, multi_label_loss, logger=None, test=False):
    if args.visualize:
        # result_folder = os.path.join(args.tmp_root, 'results/HICO/detections/', 'top'+str(args.vis_top_k))
        result_folder = os.path.join(args.tmp_root, 'results/HICO/detections/', 'visualization')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    batch_time = logutil.AverageMeter()
    losses = logutil.AverageMeter()

    y_true = np.empty((0, action_class_num))
    y_score = np.empty((0, action_class_num))

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, det_classes, det_boxes, human_num, obj_num) in enumerate(tqdm(val_loader)):
        if edge_features==None and node_features==None:
            continue
        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)

        try:
            pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        except:
            continue
        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_num, obj_num, args)
        det_indices, loss = loss_fn(pred_adj_mat, adj_mat, pred_node_labels, node_labels, mse_loss, multi_label_loss, human_num, obj_num)
        # print(pred_adj_mat.shape)
        # print(pred_node_labels.shape)
        pred_actions_indeces = torch.argmax(pred_node_labels, axis=2)
        pred_actions = [metadata.action_classes[i] for i in pred_actions_indeces[0]]
        pred_action = None
        for action in pred_actions:
            if action != 'no_interaction':
                pred_action = action
                break
        # print(len(metadata.action_classes))
        # print(human_num)
        # print(obj_num)
        # print(det_boxes[0].shape)
        # print('detection classes', det_classes)
        # print([metadata.coco_classes[i] for i in list(det_classes[0])])
        # print(len(det_indices)) # HOI数
        # for i, (batch, human_indice, obj_indice) in enumerate(det_indices):
        #     print(f"第{i}个HOI：{metadata.coco_classes[det_classes[batch][human_indice]]} 与 {metadata.coco_classes[det_classes[batch][obj_indice]]}")

        # Log
        if len(det_indices) > 0:
            losses.update(loss.item(), len(det_indices))
            y_true, y_score = evaluation(det_indices, pred_node_labels, node_labels, y_true, y_score, test=test)
        
            if args.visualize:
                # visualize_and_save(pred_adj_mat,det_indices, pred_node_labels, node_labels, det_classes, det_boxes, sequence_ids, result_folder, args.vis_top_k)
                for j, image_id in enumerate(sequence_ids):
                    image_path = os.path.join(args.image_root, f"{image_id}.jpg")
                    save_path = os.path.join(result_folder, f"{image_id}.jpg")
                    # image = cv2.imread(image_path)
                    image = Image.open(image_path)
                    image = np.array(image)
                    # image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
                    # image_pil.save('/home/tangjq/WORK/GPNN/gpnn-master/tmp/results/HICO/detections/debug/0.jpg')
                    # import sys
                    # sys.exit()
                    for n, (batch, human_indice, obj_indice) in enumerate(det_indices):
                        classes = [det_classes[batch][human_indice], det_classes[batch][obj_indice]]
                        boxes = np.array([det_boxes[batch][human_indice], det_boxes[batch][obj_indice]])
                        image = visualize_hoi(image=image, boxes=boxes, classes=classes, scores=None, line_thickness=1)
                    if pred_action != None:
                        draw = ImageDraw.Draw(image)
                        font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 24)
                        # text_width, text_height = draw.textsize(pred_action, font=font)
                        # image_width, image_height = image.size
                        # text_x = (image_width - text_width) // 2
                        # text_y = -text_height - 10
                        # draw.text((text_x, text_y), pred_action, fill="black", font=font)
                        # 5. 计算文字大小
                        text_width, text_height = draw.textsize(pred_action, font=font)

                        # 6. 计算矩形框和文字的位置
                        image_width, image_height = image.size
                        text_x = (image_width - text_width) // 2  # 文字水平居中
                        text_y = 10  # 文字距图像顶部10像素
                        padding = 5  # 矩形框与文字之间的边距

                        # 矩形框的坐标
                        rect_x1 = text_x - padding
                        rect_y1 = text_y - padding
                        rect_x2 = text_x + text_width + padding
                        rect_y2 = text_y + text_height + padding

                        # 7. 绘制白色矩形框
                        draw.rectangle([(rect_x1, rect_y1), (rect_x2, rect_y2)], fill="white")

                        # 8. 绘制黑色文字
                        draw.text((text_x, text_y), pred_action, fill="black", font=font)
                    image.save(save_path)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            mean_avg_prec = compute_mean_avg_prec(y_true, y_score)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Mean Avg Precision {mean_avg_prec:.4f} ({mean_avg_prec:.4f})\t'
                  'Detected HOIs {y_shape}'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, mean_avg_prec=mean_avg_prec, y_shape=y_true.shape))
            
    mean_avg_prec = compute_mean_avg_prec(y_true, y_score)

    print(' * Average Mean Precision {mean_avg_prec:.4f}; Average Loss {loss.avg:.4f}'
          .format(mean_avg_prec=mean_avg_prec, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('train_epoch_map', mean_avg_prec)

    return 1.0 - mean_avg_prec


def visualize_and_save(pred_adj_mat,det_indices, pred_node_labels, node_labels, det_classes, det_boxes, sequence_ids, result_folder, top_k):
    """
    Visualize and save the top-k results.
    """
    import cv2
    for idxs in det_indices[:top_k]:
        idx=idxs[0]
        # Extract detection information
        sequence_id = sequence_ids[idx]
        pred_labels = pred_node_labels[idx].cpu().data.numpy()
        gt_labels = node_labels[idx].cpu().data.numpy()
        """Labels:(n,d),n是图中实体个数，d和metadata.action_classes的长度一样，是结果？"""
        mat=pred_adj_mat[idx]
        """mat好像是不同实体（不是box）的0-1邻接矩阵，如果要进一步可视化relationships，可能要用到，甚至更多"""

        # print(mat)
        # import time
        # time.sleep(1000)

        det_class = det_classes[idx]
        # print(det_class)
        # import time
        # time.sleep(1000)
        det_box = det_boxes[idx]

        pred_results=np.argmax(pred_labels,axis=1)
            
      
        # Create a visualization (e.g., draw bounding boxes and labels on an image)
        image_path = os.path.join(args.image_root, f"{sequence_id}.jpg")
        if not os.path.exists(image_path):
            continue
        save_path = os.path.join(result_folder, f"{sequence_id}_topk.jpg")
          
        image = cv2.imread(image_path)
        for i,box in enumerate(det_box):
            x1, y1, x2, y2 = map(int, box)
            """然而bounding box的数量似乎并不总是与实体个数对应？这里会报错
            解决:det class和det box是对应的，这才是box的对应的label，而pred似乎是action的label"""
            try:
                label=metadata.coco_classes[det_class[i]]
                # print(label)
                label_text = f"Class: {label}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            except:
                continue
            #cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save the visualization result
           
        cv2.imwrite(save_path, image)


def get_indices(pred_adj_mat, pred_node_labels, human_num, obj_num, det_class, det_box):

    # Normalize adjacency matrix
    pred_adj_mat_prob = torch.nn.Sigmoid()(pred_adj_mat)
    np_pred_adj_mat = pred_adj_mat_prob.detach().cpu().numpy()
    # Normalize label outputs
    pred_node_labels_prob = torch.nn.Sigmoid()(pred_node_labels)
    np_pred_node_labels = pred_node_labels_prob.detach().cpu().numpy()

    hois = list()
    threshold1 = 0
    threshold2 = 0
    threshold3 = 0
    for h_idx in range(human_num):
        bbox_h = det_box[h_idx]
        h_class = det_class[h_idx]
        for a_idx in range(len(metadata.action_classes)):
            if np_pred_node_labels[h_idx, a_idx] < threshold1:
                continue
            max_score = -np.inf
            selected = -1

            for o_idx in range(human_num + obj_num):
                obj_name = det_class[o_idx]
                if a_idx not in metadata.obj_actions[obj_name]:
                    continue
                if np_pred_node_labels[h_idx, a_idx] < threshold2:
                    continue
                if np_pred_adj_mat[h_idx, o_idx] < threshold3:
                    continue
                score = np_pred_adj_mat[h_idx, o_idx] * np_pred_node_labels[h_idx, a_idx] * np_pred_node_labels[o_idx, a_idx]
                bbox_o = det_box[o_idx]
                rel_idx = metadata.action_to_obj_idx(obj_name, a_idx)
                hois.append((h_class, obj_name, a_idx, (bbox_h, bbox_o, rel_idx, score), (np_pred_node_labels[h_idx, a_idx], np_pred_node_labels[o_idx, a_idx], np_pred_adj_mat[h_idx, o_idx])))

    return hois


def gen_test_result(args, test_loader, model, mse_loss, multi_label_loss, img_index):
    filtered_hoi = dict()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    total_idx = 0
    obj_stats = dict()

    filtered_hoi = dict()

    for i, (edge_features, node_features, adj_mat, node_labels, sequence_ids, det_classes, det_boxes, human_nums, obj_nums) in enumerate(test_loader):
        edge_features = utils.to_variable(edge_features, args.cuda)
        node_features = utils.to_variable(node_features, args.cuda)
        adj_mat = utils.to_variable(adj_mat, args.cuda)
        node_labels = utils.to_variable(node_labels, args.cuda)
        if sequence_ids[0] == 'HICO_test2015_00000396':
            break

        pred_adj_mat, pred_node_labels = model(edge_features, node_features, adj_mat, node_labels, human_nums, obj_nums, args)

        for batch_i in range(pred_adj_mat.size()[0]):
            sequence_id = sequence_ids[batch_i]
            hois_test = get_indices(pred_adj_mat[batch_i], pred_node_labels[batch_i], human_nums[batch_i], obj_nums[batch_i],
                        det_classes[batch_i], det_boxes[batch_i])
            hois_gt = get_indices(adj_mat[batch_i], node_labels[batch_i], human_nums[batch_i],
                                    obj_nums[batch_i],
                                    det_classes[batch_i], det_boxes[batch_i])
            for hoi in hois_test:
                _, o_idx, a_idx, info, _ = hoi
                if o_idx not in filtered_hoi.keys():
                    filtered_hoi[o_idx] = dict()
                if sequence_id not in filtered_hoi[o_idx].keys():
                    filtered_hoi[o_idx][sequence_id] = list()
                filtered_hoi[o_idx][sequence_id].append(info)

        print("finished generating result from " + sequence_ids[0] + " to " + sequence_ids[-1])

    for obj_idx, save_info in filtered_hoi.items():
        obj_start, obj_end = metadata.obj_hoi_index[obj_idx]
        obj_arr = np.empty((obj_end - obj_start + 1, len(img_index)), dtype=object)
        for row in range(obj_arr.shape[0]):
            for col in range(obj_arr.shape[1]):
                obj_arr[row][col] = []
        for id, data_info in save_info.items():
            col_idx = img_index.index(id)
            for pair in data_info:
                row_idx = pair[2]
                bbox_concat = np.concatenate((pair[0], pair[1], [pair[3]]))
                if len(obj_arr[row_idx][col_idx]) > 0:
                    obj_arr[row_idx][col_idx] = np.vstack((obj_arr[row_idx][col_idx], bbox_concat))
                else:
                    obj_arr[row_idx][col_idx] = bbox_concat
        sio.savemat(os.path.join(args.tmp_root, 'results', 'HICO', 'detections_' + str(obj_idx).zfill(2) + '.mat'), {'all_boxes': obj_arr})
        print('finished saving for ' + str(obj_idx))

    return


def parse_arguments():
    # Parser check
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = config.Paths()

    # Path settings
    parser = argparse.ArgumentParser(description='HICO dataset')
    parser.add_argument('--image-root', default="/data1/tangjq/hico_20160224_det/images/test2015/", help='intermediate result path')
    parser.add_argument('--project-root', default=paths.project_root, help='intermediate result path')
    parser.add_argument('--tmp-root', default=paths.tmp_root, help='intermediate result path')
    parser.add_argument('--data-root', default=paths.hico_data_root, help='data path')
    parser.add_argument('--log-root', default=os.path.join(paths.log_root, 'hico/parsing'), help='log files path')
    parser.add_argument('--resume', default=os.path.join(paths.tmp_root, 'checkpoints/hico/parsing'), help='path to latest checkpoint')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize final results')
    parser.add_argument('--vis-top-k', type=int, default=1, metavar='N', help='Top k results to visualize')

    # Optimization Options
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--link-weight', type=float, default=100, metavar='N',
                        help='Loss weight of existing edges')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-5)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.8)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    # i/o
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
  