from datetime import datetime, timedelta
import pickle


class Discretizer:

    def __init__(self, input_specifier='0'):

        # Parse the input
        self.method = None
        self.num_bins = None
        self.time_delta = None
        self.parse_input(input_specifier)

        self.min_date = None
        self.max_date = None
        self.total_range = None
        self.bins = {}

    def parse_input(self, input_specifier):
        """
        Given a string, infer the type of split to apply to the date range:
        split by number of bins or split by time delta between bins.
        The input_specifier string can either be a string containing an
        integer e.g. '100', in which case we will split the date range by
        the specified number of bins, or a string containing a number and
        the relative time unit e.g. '100 seconds', in which case we will
        split the date range using the provided time delta.
        """

        error_msg = "Invalid interval specifier format. Expected format: 'X' or 'X unit', e.g., '100' or '100 seconds'."

        parts = input_specifier.split()
        if len(parts) == 1:
            # Should only contain an integer, split by number of bins
            try:
                self.num_bins = int(parts[0])
                self.method = "num_bins"
            except ValueError:
                raise ValueError(error_msg)
        elif len(parts) == 2:
            # Should only contain a number and a time unit, split by time delta
            try:
                self.time_delta = self.parse_time_delta(input_specifier)
                self.method = "time_delta"
            except ValueError:
                raise ValueError(error_msg)
        else:
            raise ValueError(error_msg)

    @staticmethod
    def parse_time_delta(delta_str):
        parts = delta_str.split()
        if len(parts) != 2:
            raise ValueError("Invalid time delta string format. Expected format: 'X unit', e.g., '5 minutes'.")

        value = int(parts[0])
        unit = parts[1].lower()
        if unit.endswith('s'):
            unit = unit[:-1]  # Remove plural 's'

        if unit == 'second':
            return timedelta(seconds=value)
        elif unit == 'minute':
            return timedelta(minutes=value)
        elif unit == 'hour':
            return timedelta(hours=value)
        elif unit == 'day':
            return timedelta(days=value)
        else:
            raise ValueError(f"Unsupported time unit '{unit}'.")

    @staticmethod
    def string2datetime(string):
        return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

    @staticmethod
    def string2timedelta(string):
        try:
            total_seconds = int(string)
            return timedelta(seconds=total_seconds)
        except ValueError:
            raise ValueError("Invalid input. The string must contain a valid integer representing seconds.")

    def init(self, timestamps):
        
        # Convert timestamps to datetime objects
        # dates = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
        dates = [self.string2datetime(ts) for ts in timestamps]

        # Find minimum and maximum dates
        self.min_date = min(dates)
        self.max_date = max(dates)

        # Compute the total range
        self.total_range = self.max_date - self.min_date
        
        # Split the range so that we have the specified number of bins (self.n is the number of bins)
        if self.method == 'num_bins':

            # Compute the interval for each temporal slot
            total_time_delta = timedelta(days=self.total_range.days, seconds=self.total_range.seconds)
            time_delta_seconds = int(total_time_delta.total_seconds() / self.num_bins)
            self.time_delta = timedelta(seconds=time_delta_seconds)

        # Split the range so that each bin has a certain size (self.n is the size of the bin)
        elif self.method == 'time_delta':

            # Compute the number of bins
            self.num_bins = int(self.total_range / self.time_delta)

        # Create temporal slots and store them in a dictionary
        current_date = self.min_date
        current_bin = 1  # Start from 1 and leave 0 for PAD, EOS, SOS etc.
        self.bins = {}
        while current_date < self.max_date:
            self.bins[current_date] = current_bin
            current_date += self.time_delta
            current_bin += 1

    def find_bin(self, timestamp):
        # Convert the timestamp to a datetime object
        # date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        date = self.string2datetime(timestamp)

        # Find the start date of the bin for the given timestamp
        start_date = self.min_date + int((date - self.min_date) / self.time_delta) * self.time_delta

        # Check if the start date is within the range of bins
        if start_date in self.bins:
            return self.bins[start_date]
        else:
            return None

    def get_slots(self, sequence, len):
        if len(sequence) > len:
            raise RuntimeError(f'Sequence is too long [{len(sequence)} vs max length of {len}].')
        slots = [self.find_bin(elem) for elem in sequence]
        return [0] + slots + [0] * (len(sequence) - len - 1)

    def to_pickle(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'min_date': self.min_date,
                'max_date': self.max_date,
                'num_bins': self.num_bins,
                'delta': self.time_delta,
                'bins': self.bins,
            }, f)

    def from_pickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.min_date = data['min_date']
            self.max_date = data['max_date']
            self.num_bins = data['num_bins']
            self.time_delta = data['time_delta']
            self.bins = data['bins']
            self.total_range = self.max_date - self.min_date

    def to_txt(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'w') as f:
            f.write(f'min_date:\t{self.min_date}\n')
            f.write(f'max_date:\t{self.max_date}\n')
            f.write(f'num_bins:\t{self.num_bins}\n')
            f.write(f'time_delta:\t{int(self.time_delta.total_seconds())}\n')
            for slot, index in self.bins.items():
                f.write(f"{slot}\t{index}\n")

    def from_txt(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            self.min_date = self.string2datetime(lines[0].strip().split('\t')[1])
            self.max_date = self.string2datetime(lines[1].strip().split('\t')[1])
            self.num_bins = int(lines[2].strip().split('\t')[1])
            self.time_delta = self.string2timedelta(lines[3].strip().split('\t')[1])
            self.total_range = self.max_date - self.min_date

            bins = {}
            for line in lines[4:]:
                date, index = line.strip().split('\t')
                bins[self.string2datetime(date)] = int(index)
            self.bins = bins

    @classmethod
    def load(cls, path, driver='pkl'):

        driver = driver.lower()

        if driver == 'infer':
            driver = path.split('.')[-1]

        if driver not in ['pkl', 'pickle', 'txt']:
            raise ValueError(f'Invalid driver: {driver}')

        discretizer = Discretizer()

        if driver == 'pkl' or driver == 'pickle':
            discretizer.from_pickle(path)
        else:
            discretizer.from_txt(path)

        return discretizer


if __name__ == '__main__':

    from data import FoursquareDataModule
    import config
    import os

    def create_discretizer(input_specifier):

        datamodule = FoursquareDataModule(
            config.DATA_DIR,
            config.MAX_SEQ_LEN,
            config.MIN_SEQ_LEN,
            source_tokenizer=None,
            target_tokenizer=None,
            download='infer',
            random_split=False
        )
        datamodule.prepare_data()  # Download the data
        datamodule.setup()  # Setup it

        # Take all the sequences and extract the pois
        sequences = datamodule.sequences_dataset()
        timestamps = [elem for sequence in sequences for elem in sequence['timestamps']]

        # Build a discretizer from the pois
        discretizer = Discretizer(input_specifier)
        discretizer.init(timestamps)

        return discretizer

    # Discretizer path
    discretizer_path = os.path.join(config.DISC_DIR, r'discretizer.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(discretizer_path):
        print(f'Loading tokenizer...')
        discretizer = Discretizer.load(discretizer_path, driver='txt')
    # If not, create it
    else:
        print(f'Creating discretizer...')
        input_specifier = '3 hours'
        discretizer = create_discretizer(input_specifier)
        discretizer.to_txt(discretizer_path)
        print(f'Discretizer saved to {discretizer_path}')

    print(f'Discretizer delta       : {discretizer.time_delta}')
    print(f'Discretizer num bins    : {discretizer.num_bins}')
    print(f'Discretizer min date    : {discretizer.min_date}')
    print(f'Discretizer max date    : {discretizer.max_date}')

    # Find the bin for a given timestamp
    timestamp = '2012-08-18 00:30:00'
    bin_index = discretizer.find_bin(timestamp)
    print(f"Datetime {timestamp} falls into bin with index: {bin_index}")
