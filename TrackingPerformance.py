import motmetrics as mm
import pandas as pd

# Load data
gt = pd.read_csv("ground_truth.csv")
pred = pd.read_csv("sperm_tracking_predictions.csv")

acc = mm.MOTAccumulator(auto_id=True)

for frame in sorted(gt['frameid'].unique()):
    gt_frame = gt[gt['frameid'] == frame]
    pred_frame = pred[pred['frameid'] == frame]

    gt_ids = gt_frame['trackerid'].values
    pred_ids = pred_frame['trackerid'].values

    gt_boxes = gt_frame[['x_center', 'y_center', 'width', 'height']].values
    pred_boxes = pred_frame[['x_center', 'y_center', 'width', 'height']].values

    distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1.0)


    acc.update(gt_ids, pred_ids, distances)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'idf1'], name='HybridTracker')

print(summary)