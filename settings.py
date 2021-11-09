DISPLAY_CORR_MATRICES = False
SAVE_ENCODED_PICTURES = False
COLUMNS_TO_DROP = [' Source IP', ' Source Port', ' Destination IP', ' Destination Port']
COLUMNS_TO_IGNORE = ["Flow ID", " Label", " Timestamp"]
COLUMNS_TO_DROP_BEFORE_ENCODING =['Flow ID', ' Label']
DUMMIES_COLUMNS = [' Protocol',]
DATA_PATH = r'./data/'
MINIMUM_LOGS_FROM_SESSION = 6
