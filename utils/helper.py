import numpy as np
import cv2
import torch
import collections
import xml.etree.ElementTree as ET
from torchvision import transforms
from torch.autograd import Variable

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.ToTensor(),
   normalize
])




def calc_accuracy(predicted, target, batch_size):
    bitwise_and = np.bitwise_and(predicted,target)*1
    bitwise_or = np.bitwise_or(predicted,target)*1
    num_sum = np.sum(bitwise_and,axis=1)
    den_sum = np.sum(bitwise_or,axis=1)


    return (1/batch_size) * np.sum(num_sum/den_sum)


def get_accuracy(pred, target):
    acc = (pred==target)*1.0
#     print(acc)
    acc = torch.sum(acc, 1).float()
    acc = (acc- acc%20) / 20
    acc = torch.sum(acc) / batch_size
    return acc

def get_prediction(output):
    return torch.round(output)



def returnBbox(cam,image,bbox_real):
    # Grayscale then Otsu's threshold
#     image = cv2.imread('result_new/2008_000003_train_heatmap.jpg',cv2.CV_8UC1)
#     img_out = cv2.imread('result_new/2008_000003_train_CAM.jpg')
#     print(cam)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(cam, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     print(thresh)
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
#     for c in cnts:
    x,y,w,h = cv2.boundingRect(cnts[-1])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
    cv2.rectangle(image, (bbox_real[0], bbox_real[1]), (bbox_real[2],bbox_real[3]), (0,255,0), 2)
    return [x,y,x+w,y+h],image

def find_intersaction(bbox_1, bbox_2):
    x_left = max(bbox_1[0], bbox_2[0])
    y_top = max(bbox_1[1], bbox_2[1])
    x_right = min(bbox_1[2], bbox_2[2])
    y_bottom = min(bbox_1[3], bbox_2[3])
    return (x_left, y_top,  x_right-x_left, y_bottom-y_top)


def calc_iou(bbox_1, bbox_2):
    bbox_inter = find_intersaction(bbox_1, bbox_2)
    bbox_inter_size = (bbox_inter[2] + 1) * (bbox_inter[3] + 1)
    bbox_1_size = (bbox_1[2]-bbox_1[0]+1) * (bbox_1[3]-bbox_1[1]+1)
    bbox_2_size = (bbox_2[2]-bbox_2[0]+1) * (bbox_2[3]-bbox_2[1]+1)
    iou = bbox_inter_size / float(bbox_1_size + bbox_2_size - bbox_inter_size)
    return iou

def calc_iou_accuracy(bbox_pred, bbox_real, thresh_iou=0.5):
    correct = 0
    iou = bb_intersection_over_union(bbox_pred, bbox_real)
    if iou >= thresh_iou:
        correct += 1
    return correct
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
#     print(xA,yA,xB,yB)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
#     print(interArea)
    # return the intersection over union value
    return iou
def parse_voc_xml(node):
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict
def make_one_hot(labels, n=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.nn.functional.one_hot(labels, n)

    target = Variable(one_hot.float())

    return target
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY_INV)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

def draw_bboxes(image,bboxes_computed,bbox_real):
    for i in bboxes_computed:
        for bbox in i:
            if not isinstance(bbox,float) and len(bbox)==4:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2],bbox[3]), (0,0,255), 2)
        break
    cv2.rectangle(image, (bbox_real[0], bbox_real[1]), (bbox_real[2],bbox_real[3]), (0,255,0), 2)
    return image
def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious

def accumulate(scoremap, bbox_real, iou_threshold_list,cam_threshold_list ):
    """
    From a score map, a box is inferred (compute_bboxes_from_scoremaps).
    The box is compared against GT boxes. Count a scoremap as a correct
    prediction if the IOU against at least one box is greater than a certain
    threshold (_IOU_THRESHOLD).

    Args:
        scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
        image_id: string.
    """
    boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
        scoremap=scoremap,
        scoremap_threshold_list=cam_threshold_list,
        multi_contour_eval=True)

    boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

    multiple_iou = calculate_multiple_iou(
        np.array(boxes_at_thresholds),
        np.array(bbox_real))

    idx = 0
    sliced_multiple_iou = []
    for nr_box in number_of_box_list:
        sliced_multiple_iou.append(
            max(multiple_iou.max(1)[idx:idx + nr_box]))
        idx += nr_box

    for _THRESHOLD in self.iou_threshold_list:
        correct_threshold_indices = \
            np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]
        num_correct[_THRESHOLD][correct_threshold_indices] += 1
    cnt += 1
    max_box_acc = []

    for _THRESHOLD in self.iou_threshold_list:
        localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                  float(self.cnt)
        max_box_acc.append(localization_accuracies.max())

    return max_box_acc
def get_PxAP(gt_true_score_hist,gt_false_score_hist):
    num_gt_true = gt_true_score_hist.sum()
    tp = gt_true_score_hist[::-1].cumsum()
    fn = num_gt_true - tp

    num_gt_false = gt_false_score_hist.sum()
    fp = gt_false_score_hist[::-1].cumsum()
    tn = num_gt_false - fp

    if ((tp + fn) <= 0).all():
        raise RuntimeError("No positive ground truth in the eval set.")
    if ((tp + fp) <= 0).all():
        raise RuntimeError("No positive prediction in the eval set.")

    non_zero_indices = (tp + fp) != 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
    return auc
def get_gt_bbox(mask):
    contours = cv2.findContours(
            image=mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    return [x,y,x+w,y+h]
