import sys
import json

import score_segmentations
import score_cvqa

if __name__ == "__main__":

    masks_reference_dir = sys.argv[1] if len(sys.argv)>1 else '/app/input/ref/masks_refs'
    masks_prediction_dir = sys.argv[2] if len(sys.argv)>2 else '/app/input/res/masks_preds'

    reference_fn = sys.argv[3] if len(sys.argv)>3 else '/app/input/ref/test_cvqa.json'
    prediction_fn = sys.argv[4] if len(sys.argv)>4 else '/app/input/res/test_cvqa_sys.json'

    score_dir = sys.argv[5] if len(sys.argv)>5 else '/app/output/'

    #by default looks for a file {IMGID}_mask_sys.tiff, can change this to something else
    sys_suffix = sys.argv[6] if len(sys.argv)>6 else 'sys'

    results = score_segmentations.main( masks_reference_dir, masks_prediction_dir, sys_suffix )
    results_cvqa = score_cvqa.main( reference_fn, prediction_fn  )
    for metric, val in results_cvqa.items() :
        results[metric] = val
    
    with open( '{}/scores.json'.format(score_dir), 'w' ) as f :
        json.dump( results, f, indent=4 )