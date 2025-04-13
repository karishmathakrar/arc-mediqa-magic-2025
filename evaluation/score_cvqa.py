import sys
import json

QIDS = [
    "CQID010-001", # how much of body is affected
    "CQID011-001", # where is the affected area
    "CQID011-002", # where is the affected area
    "CQID011-003", # where is the affected area
    "CQID011-004", # where is the affected area
    "CQID011-005", # where is the affected area
    "CQID011-006", # where is the affected area
    "CQID012-001", # how large are the affected areas
    "CQID012-002", # how large are the affected areas
    "CQID012-003", # how large are the affected areas
    "CQID012-004", # how large are the affected areas
    "CQID012-005", # how large are the affected areas
    "CQID012-006", # how large are the affected areas
    "CQID015-001", # when did the patient first notice the issue
    "CQID020-001", # what label best describes the affected area
    'CQID020-002', # what label best describes the affected area
    'CQID020-003', # what label best describes the affected area
    'CQID020-004', # what label best describes the affected area
    'CQID020-005', # what label best describes the affected area
    'CQID020-006', # what label best describes the affected area
    'CQID020-007', # what label best describes the affected area
    'CQID020-008', # what label best describes the affected area
    'CQID020-009', # what label best describes the affected area
    "CQID025-001", # is there any associated itching with the skin problem
    "CQID034-001", # what is the color of the skin lesion
    "CQID035-001", # how many skin lesions are there
    "CQID036-001", # what is the skin lesion texture
]

QIDS_PARENTS = sorted(list(set([ x.split('-')[0] for x in QIDS ])))


def calculate_accuracy( qid2val_byencounterid_gold, qid2val_byencounterid_sys, qidparents=QIDS_PARENTS) :
    results = {}
    x_all = []
    y_all = []
    encounter_ids = list( qid2val_byencounterid_gold.keys() )
    
    for qid in qidparents :
        goldlist = [ qid2val_byencounterid_gold[encounter_id][qid] for encounter_id in encounter_ids ]
        syslist = [ qid2val_byencounterid_sys[encounter_id][qid] for encounter_id in encounter_ids ]
        x_all.extend( goldlist )
        y_all.extend( syslist )
        results['accuracy_{}'.format(qid)] = get_accuracy_score(goldlist,syslist)
        
    results[ 'accuracy_{}'.format('all') ] = get_accuracy_score( x_all, y_all )
    return results


def get_accuracy_score( gold_items, sys_items ) :
    total = 0
    weight_sum = 0
    for x, y in zip(gold_items,sys_items) :
        weight = len( set(x).intersection(set(y)) )
        weight_sum += weight/max(len(set(x)),len(set(y)))
        total += 1
    return weight_sum/total


def organize_values( data ) :
    qid2val_byencounterid = {}
    for item in data :
        encounter_id = item['encounter_id'].split('-')[0]
        qid2val_byencounterid[encounter_id] = qid2val_byencounterid.get(encounter_id,{})
        for key ,val in item.items() :
            if key == 'encounter_id'  or len(key)!=11:
                continue
            qid, _ = key.split('-')
            qid2val_byencounterid[encounter_id][qid] = qid2val_byencounterid[encounter_id].get(qid,[])
            qid2val_byencounterid[encounter_id][qid].append(val)
    return qid2val_byencounterid


def main( reference_fn, prediction_fn ) :
    with open( reference_fn ) as f :
        data_ref = json.load(f)
    with open( prediction_fn ) as f :
        data_sys = json.load(f)
    
    print('Detected {} instances for reference.'.format(len(data_ref)),file=sys.stderr)
    print('Detected {} instances for predictions.'.format(len(data_sys)),file=sys.stderr)

    if len(data_sys)==0:
        return {
            "accuracy_CQID010": 0.0,
            "accuracy_CQID011": 0.0,
            "accuracy_CQID012": 0.0,
            "accuracy_CQID015": 0.0,
            "accuracy_CQID020": 0.0,
            "accuracy_CQID025": 0.0,
            "accuracy_CQID034": 0.0,
            "accuracy_CQID035": 0.0,
            "accuracy_CQID036": 0.0,
            "accuracy_all": 0.0,
            "number_cvqa_instances": 0
        }

    encounterids_ref =  set([ x['encounter_id'] for x in data_ref ])
    encounterids_sys = set([ x['encounter_id'] for x in data_sys ])
    print( 'ENCOUNTERID-MATCH: {}'.format( encounterids_ref == encounterids_sys ),file=sys.stderr)

    print('Organizing Values by Questionids',file=sys.stderr)
    qid2val_byencounterid_gold = organize_values( data_ref )
    qid2val_byencounterid_sys = organize_values( data_sys )

    print('Calculating Accuracy',file=sys.stderr)
    results = calculate_accuracy( qid2val_byencounterid_gold, qid2val_byencounterid_sys )
    results['number_cvqa_instances'] = len(encounterids_ref)
    return results


if __name__ == "__main__":

    reference_fn = sys.argv[1] if len(sys.argv)>1 else '/app/input/ref/test_cvqa.json'
    prediction_fn = sys.argv[2] if len(sys.argv)>2 else '/app/input/res/test_cvqa_sys.json'
    score_dir = sys.argv[3] if len(sys.argv)>3 else '/app/output/'

    results = main( reference_fn, prediction_fn )
    with open( '{}/scores_cvqa.json'.format(score_dir), 'w' ) as f :
        json.dump( results, f, indent=4 )
