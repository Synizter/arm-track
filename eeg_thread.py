import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, NoiseTypes


class EEGThread(QThread):
    raw_chunk = pyqtSignal(np.ndarray)
    filtered_chunk = pyqtSignal(np.ndarray)

    def __init__(self ,apply_filter = True, port = 'COM3', 
                 board_type = BoardIds.CYTON_DAISY_BOARD):
        super().__init__()
        
        params = BrainFlowInputParams()
        params.serial_port = port
        self.board_shim = BoardShim(board_type, params)
        self.sampling_rate = self.board_shim.get_sampling_rate(board_type)
        self.board_shim.prepare_session()

        #get board data
        self.ch_nums = self.board_shim.get_eeg_channels(board_type)

        #channel map
        self.chs_map = {0: 'FP1', 1: 'FP2', 2: 'C3', 3: 'C4', 4: 'T5', 5: 
                        'T6', 6: 'O1', 7: 'O2', 8: 'P4', 9: 'P3', 10: 'T4', 11: 'T3', 12: 'F4', 13: 'F3', 14: 'F8', 15: 'F7'}
        self._run_flag = True
        self._apply_filter = apply_filter

    def run(self):
        self.board_shim.start_stream()
        data = self.board_shim.get_board_data() #flush buffer
        while self._run_flag:                
            data = self.board_shim.get_board_data()
            # DataFilter.remove_environmental_noise(data[:], self.sampling_rate, NoiseTypes.FIFTY.value)
            if data.shape[1] != 0:
                self.filtered_chunk.emit(data)
            
        
    def flush(self):
        self.board_shim.get_board_data()
        
    def quit(self):
        self.board_shim.release_all_sessions()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.board_shim.get_board_data() #flush
        self.board_shim.stop_stream()
        # self.quit()
        self.wait()

if __name__ == "__main__":
    pass
