'''
This code is adapted from process steps on eICU of previous works (cited)
https://github.com/Google-Health/records-research/tree/master/graph-convolutional-transformer
'''

import csv
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import sys
import pickle
import argparse
import numpy as np
from sklearn import model_selection
from scipy.sparse import csr_matrix


class EncounterInfo(object):

    def __init__(self, patient_id, encounter_id, encounter_timestamp,
                 readmission):
        self.patient_id = patient_id
        self.encounter_id = encounter_id
        self.encounter_timestamp = encounter_timestamp
        self.readmission = readmission
        self.dx_ids = []
        self.rx_ids = []
        self.labs = {}
        self.physicals = []
        self.treatments = []


def process_patient(infile, encounter_dict, hour_threshold=24):
    inff = open(infile, 'r')
    count = 0
    patient_dict = {}
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        patient_id = line['patienthealthsystemstayid']
        encounter_id = line['patientunitstayid']
        encounter_timestamp = -int(line['hospitaladmitoffset'])
        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append((encounter_timestamp, encounter_id))
    inff.close()
    print('')

    patient_dict_sorted = {}
    for patient_id, time_enc_tuples in patient_dict.items():
        patient_dict_sorted[patient_id] = sorted(time_enc_tuples)

    enc_readmission_dict = {}
    for patient_id, time_enc_tuples in patient_dict_sorted.items():
        for time_enc_tuple in time_enc_tuples[:-1]:
            enc_id = time_enc_tuple[1]
            enc_readmission_dict[enc_id] = True
        last_enc_id = time_enc_tuples[-1][1]
        enc_readmission_dict[last_enc_id] = False

    inff = open(infile, 'r')
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        patient_id = line['patienthealthsystemstayid']
        encounter_id = line['patientunitstayid']
        encounter_timestamp = -int(line['hospitaladmitoffset'])
        discharge_status = line['unitdischargestatus']
        duration_minute = float(line['unitdischargeoffset'])
        readmission = enc_readmission_dict[encounter_id]

        if duration_minute > 60. * hour_threshold:
            continue

        ei = EncounterInfo(patient_id, encounter_id, encounter_timestamp,
                           readmission)
        if encounter_id in encounter_dict:
            print('Duplicate encounter ID!!')
            sys.exit(0)
        encounter_dict[encounter_id] = ei
        count += 1

    inff.close()
    print('')

    return encounter_dict


def process_admission_dx(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        encounter_id = line['patientunitstayid']
        dx_id = line['admitdxpath'].lower()

        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].dx_ids.append(dx_id)
        count += 1
    inff.close()
    print('')
    print('Admission Diagnosis without Encounter ID: %d' % missing_eid)

    return encounter_dict


def process_diagnosis(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()

        encounter_id = line['patientunitstayid']
        dx_id = line['diagnosisstring'].lower()

        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].dx_ids.append(dx_id)
        count += 1
    inff.close()
    print('')
    print('Diagnosis without Encounter ID: %d' % missing_eid)

    return encounter_dict


def process_treatment(infile, encounter_dict):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0

    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        encounter_id = line['patientunitstayid']
        treatment_id = line['treatmentstring'].lower()
        if encounter_id not in encounter_dict:
            missing_eid += 1
            continue
        encounter_dict[encounter_id].treatments.append(treatment_id)
        count += 1
    inff.close()
    print('')
    print('Treatment without Encounter ID: %d' % missing_eid)
    print('Accepted treatments: %d' % count)

    return encounter_dict


def build_seqex(enc_dict,
                skip_duplicate=False,
                min_num_codes=1,
                max_num_codes=50):
    key_list = []
    seqex_list = []
    dx_str2int = {}
    treat_str2int = {}
    num_cut = 0
    num_duplicate = 0
    count = 0
    num_dx_ids = 0
    num_treatments = 0
    num_unique_dx_ids = 0
    num_unique_treatments = 0
    min_dx_cut = 0
    min_treatment_cut = 0
    max_dx_cut = 0
    max_treatment_cut = 0
    num_readmission = 0

    for _, enc in enc_dict.items():
        if skip_duplicate:
            if (len(enc.dx_ids) > len(set(enc.dx_ids)) or
                    len(enc.treatments) > len(set(enc.treatments))):
                num_duplicate += 1
                continue

        if len(set(enc.dx_ids)) < min_num_codes:
            min_dx_cut += 1
            continue

        if len(set(enc.treatments)) < min_num_codes:
            min_treatment_cut += 1
            continue

        if len(set(enc.dx_ids)) > max_num_codes:
            max_dx_cut += 1
            continue

        if len(set(enc.treatments)) > max_num_codes:
            max_treatment_cut += 1
            continue

        count += 1
        num_dx_ids += len(enc.dx_ids)
        num_treatments += len(enc.treatments)
        num_unique_dx_ids += len(set(enc.dx_ids))
        num_unique_treatments += len(set(enc.treatments))

        for dx_id in enc.dx_ids:
            if dx_id not in dx_str2int:
                dx_str2int[dx_id] = len(dx_str2int)

        for treat_id in enc.treatments:
            if treat_id not in treat_str2int:
                treat_str2int[treat_id] = len(treat_str2int)

        seqex = tf.train.SequenceExample()
        seqex.context.feature['patientId'].bytes_list.value.append(
            bytes(enc.patient_id + ':' +enc.encounter_id, 'utf-8'))

        if enc.readmission:
            seqex.context.feature['label'].int64_list.value.append(1)
            num_readmission += 1
        else:
            seqex.context.feature['label'].int64_list.value.append(0)

        dx_ids = seqex.feature_lists.feature_list['dx_ids']
        dx_ids.feature.add().bytes_list.value.extend(list([bytes(s, 'utf-8') for s in set(enc.dx_ids)]))

        dx_int_list = [dx_str2int[item] for item in list(set(enc.dx_ids))]
        dx_ints = seqex.feature_lists.feature_list['dx_ints']
        dx_ints.feature.add().int64_list.value.extend(dx_int_list)

        proc_ids = seqex.feature_lists.feature_list['proc_ids']
        proc_ids.feature.add().bytes_list.value.extend(list([bytes(s, 'utf-8') for s in set(enc.treatments)]))

        proc_int_list = [treat_str2int[item] for item in list(set(enc.treatments))]
        proc_ints = seqex.feature_lists.feature_list['proc_ints']
        proc_ints.feature.add().int64_list.value.extend(proc_int_list)

        seqex_list.append(seqex)
        key = seqex.context.feature['patientId'].bytes_list.value[0]
        key_list.append(key)

    print('Filtered encounters due to duplicate codes: %d' % num_duplicate)
    print('Filtered encounters due to thresholding: %d' % num_cut)
    print('Average num_dx_ids: %f' % (num_dx_ids / count))
    print('Average num_treatments: %f' % (num_treatments / count))
    print('Average num_unique_dx_ids: %f' % (num_unique_dx_ids / count))
    print('Average num_unique_treatments: %f' % (num_unique_treatments / count))
    print('Min dx cut: %d' % min_dx_cut)
    print('Min treatment cut: %d' % min_treatment_cut)
    print('Max dx cut: %d' % max_dx_cut)
    print('Max treatment cut: %d' % max_treatment_cut)
    print('Number of readmission: %d' % num_readmission)

    return key_list, seqex_list, dx_str2int, treat_str2int


def select_train_valid_test(key_list, random_seed=0):
    train_id, val_id = model_selection.train_test_split(
        key_list, test_size=0.2, random_state=random_seed)
    test_id, val_id = model_selection.train_test_split(
        val_id, test_size=0.5, random_state=random_seed)
    return train_id, val_id, test_id


def get_partitions(seqex_list, id_set=None):
    total_visit = 0
    new_seqex_list = []
    for seqex in seqex_list:
        if total_visit % 1000 == 0:
            sys.stdout.write('Visit count: %d\r' % total_visit)
            sys.stdout.flush()
        key = seqex.context.feature['patientId'].bytes_list.value[0]
        if (id_set is not None and key not in id_set):
            total_visit += 1
            continue
        new_seqex_list.append(seqex)
    return new_seqex_list


def parser_fn(serialized_example):
    context_features_config = {
        'patientId': tf.VarLenFeature(tf.string),
        'label': tf.FixedLenFeature([1], tf.int64),
    }
    sequence_features_config = {
        'dx_ints': tf.VarLenFeature(tf.int64),
        'proc_ints': tf.VarLenFeature(tf.int64)
    }
    (batch_context, batch_sequence) = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=context_features_config,
        sequence_features=sequence_features_config)
    labels = tf.squeeze(tf.cast(batch_context['label'], tf.float32))
    return batch_sequence, labels


def tf2csr(output_path, partition, maps):
    num_epochs = 1
    buffer_size = 32
    dataset = tf.data.TFRecordDataset(output_path + partition + ".tfrecord")
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parser_fn, num_parallel_calls=4)
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(16)
    count = 0
    np_data = []
    np_label = []
    for data in dataset:
        count += 1
        np_datum = np.zeros(sum([len(m) for m in maps]))
        dx_pos = tf.sparse.to_dense(data[0]['dx_ints']).numpy().ravel()
        proc_pos = tf.sparse.to_dense(data[0]['proc_ints']).numpy().ravel() + \
                   sum([len(m) for m in maps[:1]])
        np_datum[dx_pos] = 1
        np_datum[proc_pos] = 1
        np_data.append(np_datum)
        np_label.append(data[1].numpy()[0])
        sys.stdout.write('%d\r' % count)
        sys.stdout.flush()
    pickle.dump((csr_matrix(np.array(np_data)), np.array(np_label)), \
                open(output_path + partition + '_csr.pkl', 'wb'))

"""Set <input_path> to where the raw eICU CSV files are located.
Set <output_path> to where you want the output files to be.
"""

def main():
    parser = argparse.ArgumentParser(description='File path')
    parser.add_argument('--input_path', type=str, default='.', help='input path of original dataset')
    parser.add_argument('--output_path', type=str, default='.', help='output path of processed dataset')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    patient_file = input_path + '/patient.csv'
    admission_dx_file = input_path + '/admissionDx.csv'
    diagnosis_file = input_path + '/diagnosis.csv'
    treatment_file = input_path + '/treatment.csv'

    encounter_dict = {}
    print('Processing patient.csv')
    encounter_dict = process_patient(
        patient_file, encounter_dict, hour_threshold=24)
    print(len(encounter_dict))
    print('Processing admission diagnosis.csv')
    encounter_dict = process_admission_dx(admission_dx_file, encounter_dict)
    print('Processing diagnosis.csv')
    encounter_dict = process_diagnosis(diagnosis_file, encounter_dict)
    print('Processing treatment.csv')
    encounter_dict = process_treatment(treatment_file, encounter_dict)

    key_list, seqex_list, dx_map, proc_map = build_seqex(
        encounter_dict, skip_duplicate=False, min_num_codes=1, max_num_codes=50)

    pickle.dump(dx_map, open(output_path + '/dx_map.p', 'wb'), -1)
    pickle.dump(proc_map, open(output_path + '/proc_map.p', 'wb'), -1)

    key_train, key_valid, key_test = select_train_valid_test(key_list)

    train_seqex = get_partitions(seqex_list, set(key_train))
    validation_seqex = get_partitions(seqex_list, set(key_valid))
    test_seqex = get_partitions(seqex_list, set(key_test))

    print("Split done.")

    with tf.io.TFRecordWriter(output_path + '/train.tfrecord') as writer:
        for seqex in train_seqex:
            writer.write(seqex.SerializeToString())

    with tf.io.TFRecordWriter(output_path + '/validation.tfrecord') as writer:
        for seqex in validation_seqex:
            writer.write(seqex.SerializeToString())

    with tf.io.TFRecordWriter(output_path + '/test.tfrecord') as writer:
        for seqex in test_seqex:
            writer.write(seqex.SerializeToString())

    for partition in ['train', 'validation', 'test']:
        tf2csr(output_path, partition, [dx_map, proc_map])
    print('done')


if __name__ == '__main__':
    main()
