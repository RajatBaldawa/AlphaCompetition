import dill as pkl


class AbstractAlpha:
    def __init__(self):
        """
        :param max_lookback: (int) maximum number of previous days data needed for the alpha
        :param factor_list: (list) of factor names used in alpha computation
        :param universe_size: (int) number of tickers in the universe
        """
        # Required variables for AbstractAlpha
        self.name = None
        self.lookback = None
        self.universe_size = None
        self.factor_list = None

        # Variables needed for helper functions
        self.holdings_lock = None

    def generate_day(self, day, data):
        """
        This is the function you need to write for your alpha

        :param data: dictionary keyed by factor names and values of numpy dimension (max_lookback x universe_size)
                     arrays with data with max_lookback days of the factor
        :param day: (int) integer index of the day in the backtest
        :return: A universe_size x 1 vector of holdings
        """
        raise NotImplementedError

    def lock_holdings(self, n):
        if self.holdings_lock is None:
            self.holdings_lock = n-1
            return True

        if self.holdings_lock == 0:
            self.holdings_lock = n-1
            return True
        else:
            self.holdings_lock -= 1
            return False

    def save(self, fname):
        """Saves current alpha object using dill library (based on pickle)

        Args:
            fname: filename to save the alpha object to.

        Returns:
            None
        """
        with open(fname, 'wb') as f:
            pkl.dump(self, f)

    @staticmethod
    def load(fname):
        """Loads Alpha object from a dill pickle file

        Args:
            fname: file location of pickled object

        Returns:
            AbstractAlpha Object
        """
        with open(fname, 'rb') as f:
            obj = pkl.load(f)
            if isinstance(obj, AbstractAlpha):
                return obj
            raise ValueError(f"{fname} is not an instance of AbstractAlpha.")
