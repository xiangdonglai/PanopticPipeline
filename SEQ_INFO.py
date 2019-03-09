import os


class SEQ_INFO:
    def __init__(self, name, calib, start, end, cam_num, captures_nas, processed_nas):
        assert type(name) == str
        assert type(calib) == str
        assert type(start) == int
        assert type(end) == int
        assert type(cam_num) == int
        assert type(captures_nas) == str
        assert type(processed_nas) == str
        self._name = name
        self._calib = calib
        self._start = start
        self._end = end
        self._cam_num = cam_num
        self._captures_nas = captures_nas
        self._processed_nas = processed_nas

    @property
    def name(self):
        return self._name

    @property
    def calib(self):
        return self._calib

    @property
    def calib_path(self):
        return os.path.join('/media/posefs1a/Calibration', self._calib, 'calibFiles')

    @property
    def start_idx(self):
        return self._start

    @property
    def end_idx(self):
        return self._end

    @property
    def cam_num(self):
        return self._cam_num

    @property
    def captures_nas(self):
        return self._captures_nas

    @property
    def processed_nas(self):
        return self._processed_nas

    @property
    def captures_path(self):
        return os.path.join('/media', 'posefs' + self.captures_nas, 'Captures', 'specialEvents', self.name)

    @property
    def processed_path(self):
        return os.path.join('/media', 'posefs' + self.processed_nas, 'Processed', 'specialEvents', self.name)

    def check_path(self):
        assert os.path.isdir(self.captures_path)
        if not os.path.isdir(self.processed_path):
            os.makedirs(self.processed_path)
        assert os.path.isdir(self.processed_path)


def parse_seq(CONFIG):
    # check the data type and length
    assert(type(CONFIG['sequence_names']) == list)
    num_sequence = len(CONFIG['sequence_names'])
    assert(type(CONFIG['calibration_data']) == list and len(CONFIG['calibration_data']) == num_sequence)
    assert(type(CONFIG['start_index']) == list and len(CONFIG['start_index']) == num_sequence)
    assert(type(CONFIG['end_index']) == list and len(CONFIG['end_index']) == num_sequence)
    assert(type(CONFIG['camera_number']) == list and len(CONFIG['camera_number']) == num_sequence)
    assert(type(CONFIG['captures_nas']) == list and len(CONFIG['captures_nas']) == num_sequence)
    assert(type(CONFIG['processed_nas']) == list and len(CONFIG['processed_nas']) == num_sequence)

    seq_infos = []
    for i in range(num_sequence):
        info = SEQ_INFO(CONFIG['sequence_names'][i], CONFIG['calibration_data'][i], CONFIG['start_index'][i], CONFIG['end_index'][i],
                        CONFIG['camera_number'][i], CONFIG['captures_nas'][i], CONFIG['processed_nas'][i])
        info.check_path()
        seq_infos.append(info)
    return seq_infos
