from lvis import LVIS, LVISEval, LVISResults


def val_bnd(result_path, ann_path, val_type):
    lvis_eval = LVISEval(ann_path, result_path, val_type)
    lvis_eval.run()
    lvis_eval.print_results()


if __name__ =='__main__':
    gt_json = 'data/lvis_annotations/lvis_v1_val_cocofied.json'
    dt_json = 'json_results/refine_paste_tiny.json'
    a = val_bnd(dt_json, gt_json, 'segm')
