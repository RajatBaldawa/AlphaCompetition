"""Includes different class variations of Full backtest type which loads all data into memory
before performing backtest"""
import datetime
import logging
import multiprocessing as mp
import re
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from sif.configs.config import MBT_RESULT, SBT_RESULT, DataStruct, Date
from sif.configs.db_config import ALL_FACTORS
from sif.sifinfra import sif_utils as su
from sif.siftools.abstractalpha import AbstractAlpha
from sif.siftools.backtesters.abstractbacktester import AbstractBacktester

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d (%(levelname)s): %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class FullBacktester(AbstractBacktester, ABC):
    """Child of AbstractBacktester which implements functionality for loading all data initially,
    and then performing the backtest process with all data stored in memory.

    TL;DR Memory expensive, Quick backtest

    Args:
        start_date: starting date of backtesting period
        end_date: ending date of backtesting period
        universe_rebalance_dates: List of month day tuples corresponding to annual calendar dates to
                                  recompute tickers within current trading universe
        max_lookback: the maximum lookback size of an alpha instance using this environment
        universe_size: number of tickers to be in trading universe at each given rebalance date
        factors: list of factors needed for alpha holdings computation. Alpha instance must use
                 a subset of this list of factors in its generate_day method
        meta_dict: Optional parameter to specify metadata fields to filter tickers on.
                   See sif.siftools.sif_utils.get_universe_on_day for more details
        local_data: numpy data loaded from local disk, used to skip API calling. Should not be used
                    by user, but is accessed through the from_local method.
    """
    def __init__(
            self,
            start_date: Date,
            end_date: Date,
            universe_rebalance_dates: Union[List[Tuple[int, int]], str],
            max_lookback: int,
            delay: int = 1,
            *,
            universe_size: int,
            factors: List[str],
            meta_dict: Optional[Dict[str, Union[List[Any], Any]]] = None,
            local_data: Optional[str] = None,
    ):
        # Check invalid parameters specific to FullBacktester
        if universe_size < 1:
            raise AttributeError("Universe size must be >= 1")
        if not factors or not all(factor in ALL_FACTORS for factor in factors):
            raise AttributeError("Factors is empty or contains factor not available in database")

        super(FullBacktester, self).__init__(
            start_date, end_date, universe_rebalance_dates, max_lookback, delay, local_data=local_data
        )
        logging.info('Initializing full backtester settings')
        self.universe_size = universe_size
        self.factors = factors if ['close'] in factors else factors + ['close']

        if local_data is not None:
            logging.info('Loading data and universes from local data')
            self.data, self.universes = self.loaded_data
        else:
            logging.info('Getting universes from api')
            self.universes = self.get_universes(self.universe_size, meta_dict=meta_dict)
            logging.info('Getting data')
            self.data = self.get_data()

    @property
    def settings(self) -> Dict[str, Any]:
        """Creates a dictionary with the current backtester's environment settings. Matches keyword
        argument naming schema

        Returns:
            Dictionary of settings and their values
        """
        settings = {'start_date': self.start_date,
                    'end_date': self.end_date,
                    'universe_rebalance_dates': self.universe_rebalance_dates,
                    'max_lookback': self.max_lookback,
                    'universe_size': self.universe_size,
                    'factors': self.factors}
        return settings

    def save_local(self, file_path: str):
        """Overrides super method, adding data and universes to the saved output

        Args:
            file_path: path to output data to
        Returns:
            0 on success, throws error otherwise
        """
        np.save(file_path, (self.settings, self.trading_days, self.data, self.universes))
        return 0

    @abstractmethod
    def backtest(
            self, alpha: Union[AbstractAlpha, List[AbstractAlpha]], use_env_universe: bool = False
    ) -> Union[SBT_RESULT, MBT_RESULT]:
        raise NotImplementedError(
            'Class inheriting from FullBacktester must implement backtest method'
        )

    def get_data(self) -> DataStruct:
        """
        Loads full dataset needed for the backtest
        Returns:
            dictionary of dataframes for each factor
        """
        tickers = np.unique(self.universes.values.flatten()).tolist()
        # Must pull data + max_lookback data points so full backtest can be conducted out of sample
        logging.info('Finding buffered start from api')
        buffered_start_date = su.trading_days_to_date(
            self.start_date, self.max_lookback + self.delay, lookback=True
        )
        # Loads all factors from each data table for required date range and ticker universe
        data = su.get_data_main(self.factors, buffered_start_date, self.end_date, tickers)
        logging.info('Calculating return dataframe')
        data['ret'] = data['close'].pct_change()  # compute returns to use later
        return data

    def add_factor(self, factor: Union[pd.Series, pd.Series], name: str, overwrite: bool = False) -> DataStruct:
        """Method that allows the user to insert additional data not provided by the API

        The data must have the same format as the other data. This means Series and DataFrames must have the appropriate
        shapes corresponding to the backtester's environment. Additionally, DataFrame columns must match the available
        tickers in the backtester's universe. If a datetime index is not provided, one will be inserted that matches
        the other factors. If the data was not sorted, this would cause a problem.

        Args:
            factor: Series or DataFrame of new factor data being added to the backtester
            name: data key to assign to the factor
            overwrite: If the factor name already exists in the backtester, this flag determines whether to overwrite it

        Returns:
            Updates self.data with the new factor data and returns the resulting DataStruct
        """
        if name in self.data.keys() and not overwrite:
            raise AttributeError(f"{name} is already the name of a factor in this backtesting environment. If you would"
                                 f"like to overwrite this factor, run this method with overwrite=True")
        if isinstance(factor, pd.Series):
            if factor.shape[0] == self.data['ret'].shape[0]:
                if isinstance(factor.index, pd.DatetimeIndex):
                    self.data[name] = factor.sort_index()
                else:
                    self.data[name] = pd.Series(factor, index=factor['ret'].index)
            else:
                raise AttributeError(f"The given Series does not match the length of the backtester's data.\nExpected:"
                                     f" {self.data['ret'].shape[0]}\nGot: {factor.shape[0]}")
        elif isinstance(factor, pd.DataFrame):
            if factor.shape == self.data['ret'].shape and set(factor.columns) == set(self.data['ret'].columns):
                factor = factor.sort_index(axis=1)
                if isinstance(factor.index, pd.DatetimeIndex):
                    self.data[name] = factor.sort_index()
                else:
                    self.data[name] = pd.DataFrame(factor.values, columns=factor.columns, index=self.data['ret'].index)
            else:
                raise AttributeError(f"The given DataFrame does not match the shape of the backtester's data\nExpected:"
                                     f" {self.data['ret'].shape}\nGot: {factor.shape}")
        else:
            raise AttributeError(f"Factor must be of type pandas Series or DataFrame. Got: {type(factor)}")
        self.factors.append(name)
        return self.data

    def eval_factor(self, expr: str, name: Optional[str] = None, overwrite: bool = False, **kwargs):
        """Evaluate a mathematical expression to add new factors to backtesting environment

        Args:
            expr: Expression that is evaluated using current data factors in the backtesting environment
            name: If provided, will set the new factor to this name. If one is not provided, an equals sign must be
            included in the expression, where the left hand side of the equals the name of the new factor
            overwrite: If the factor name already exists in the backtester, this flag determines whether to overwrite it
            **kwargs: Additional keyword arguments to provide to pd.eval(..., kwargs)

        Returns:
            Updates self.data with the new factor data and returns the resulting DataStruct
        """
        OPS = ('+', '-', '*', '/', '**', '%', '//', '(', ')')
        if '=' in expr:
            if name is not None:
                left, expr, *_ = expr.split('=')
            else:
                left, expr, *_ = expr.split('=')
                left = left.strip()
                if not any(invalid in left for invalid in (' ', '\t', '\n') + OPS):
                    name = left
                else:
                    raise ValueError(f"Left side of expression assignment is not a valid factor: {left}\n"
                                     f"Cannot contain whitespace of mathematical operators")
        elif name is None:
            raise AttributeError("No name was provided and = assignment not found in expression")

        if name in self.data.keys() and not overwrite:
            raise AttributeError(f"{name} is already the name of a factor in this backtesting environment. If you would"
                                 f"like to overwrite this factor, run this method with overwrite=True")

        def parse_expr(elem):
            if elem in OPS:  # match operators
                return elem
            elif re.findall(r'^-?\d*\.?\d+$|^-?\d+\.?\d*$', elem):  # match numeric robust
                return elem
            elif elem in self.data:  # match factors
                return f"self.data['{elem}']"
            else:
                raise ValueError(f"Found unsupported operator in eval: {elem}")

        expr = ' '.join(map(parse_expr, expr.split()))
        factor = pd.eval(expr, **kwargs)
        self.data[name] = factor
        self.factors.append(name)
        return self.data


class FullSingleBacktester(FullBacktester):
    """Child of FullBacktester which handles backtesting a single alpha per backtest call"""
    def _get_backtest_universes(self, alpha: AbstractAlpha, use_env_universe: bool) -> pd.DataFrame:
        """Helper function to get the universes for the current alpha and backtest

        Args:
            alpha: AbstractAlpha object that is being backtested
            use_env_universe: Flag which indicates whether the backtest should use the backtester's universe
                              environment, or load the specific universe for the alpha object

        Returns:
            pd.DataFrame: DataFrame where the index is given by the universe rebalance dates, and the columns are
                          N in [1, ..., Universe_Size] with the values being the ticker with marketcap rank N on the
                          given date index
        """
        # Get the universes for this alpha and create count structure for updating the universe
        if use_env_universe or self.universe_size == alpha.universe_size:
            logging.info('Using environment universe')
            universes = self.universes
        else:
            logging.info('Getting alpha universes from api')
            logging.warning(f"Loading alpha universe slows down overall backtest. Consider "
                            f"setting backtest environment to use the same universe.\nBacktest"
                            f" Universe Size: {self.universe_size}\nAlpha Universe Size: "
                            f"{alpha.universe_size}")
            universes = self.get_universes(alpha.universe_size)
        return universes

    def _base_backtest(
            self,
            results_accumulator: Callable[[Any, np.ndarray, datetime.datetime, pd.Series], None],
            init_results: Any,
            universes: pd.DataFrame,
            alpha: AbstractAlpha,
            progress_bar: bool = True
    ) -> None:
        """Base function for backtesting which prepares the environment to backtest on the given AbstractAlpha object.
           Performs backtest process with the resolved environment settings, using the given alpha instance's
           generate_day() function to compute holdings everyday. The backtest calls the results_accumulator on each day,
           passing the holdings and other necessary data to update the init_results

        Args:
            results_accumulator: Function which is called after computing holdings each day in the backtest. The
                                 arguments given to the function are the init_results, holdings for the current day, the
                                 date of the current day, and the universe for the current day of the backtest
            init_results: Initial results data structure that will be modified by the results_accumulator. This is
                          passed to the results_accumulator on each day of the backtest. This data structure should be
                          returned by the parent function calling _base_backtest.
            universes: DataFrame of universes to use throughout the backtest, see _get_backtest_universes
                                      for more details
            alpha: AbstractAlpha instance used to compute holdings
            progress_bar: Default True, if False, progress bar (tqdm) will not be shown during the backtest

        Raises:
            ValueError: The settings of the alpha are not compatible with the backtest environment

        Returns:
            Does not return, updates the mutable data structure, init_results, by calling  results_accumulator each day
        """
        # Initialize alpha settings
        lookback = alpha.lookback
        alpha_factors = alpha.factor_list

        # Check alpha has valid settings
        if lookback > self.max_lookback or set(alpha_factors) - set(self.factors):
            raise ValueError("Alpha settings not compatible with backtest environment.")

        # Get current tickers, universe, and rebalance dates
        current_universe = universes.loc[self.start_date]
        rebalance_dates = list(self.all_rebalance_dates[1:]).copy()

        alpha_data = self.partition_universe(self.data, current_universe, alpha_factors)  # Starting data
        logging.info("Backtesting")
        for i, day in (pbar := tqdm(
                enumerate(self.trading_days), disable=not progress_bar, total=len(self.trading_days))
        ):
            pbar.set_description(f"{day}")
            logging.info(f"{i}: {day}")
            # Rebalance
            if len(rebalance_dates) > 0 and day >= rebalance_dates[0]:
                logging.info('Rebalancing universe')
                while len(rebalance_dates) > 0 and day >= rebalance_dates[0]:
                    rebalance_day = rebalance_dates.pop(0)

                current_universe = universes.loc[rebalance_day]
                alpha_data = AbstractBacktester.partition_universe(self.data, current_universe, alpha_factors)

            # Get data needed for today's computation
            day_index = self.data['ret'].index.get_loc(day)
            daily_data = self.partition_daily(alpha_data, day_index, lookback, self.delay)
            logging.info('Calculating holdings')
            holdings = alpha.generate_day(lookback, daily_data)
            logging.info('Holdings calculated')
            results_accumulator(init_results, holdings, day, current_universe)
        pbar.close()

    def _update_returns_and_holdings(
            self, results: Tuple[pd.Series, pd.DataFrame], holdings: np.ndarray, day: Date, current_universe: pd.Series
    ):
        """Updates the returns and holdings result tables by calculating the returns for the day given the
        computed holdings. Inserts the daily results in the appropriate index of the respective tables

        Note: This does not return anything, but simply updates the multable data structure that is passed in as
              <results>. The creation of the initial results data structure can be seen below

        Args:
            results: A tuple of pd.Series which is the returns table, and a pd.DataFrame which is the holdings table
            holdings: An np.ndarray including the weights returned by the alpha for the given backtest day
            day: The date which the holdings correspond to in the backtest process
            current_universe: The pd.Series of the universe being used on the given date in the backtest process
        """
        returns_results, holdings_results = results
        return_data = self.data['ret'][current_universe].loc[day].values
        ret = np.nansum(return_data * holdings)
        holdings_results.loc[day, current_universe] = holdings
        returns_results.loc[day] = ret

    def backtest(
            self, alpha: AbstractAlpha, use_env_universe: bool = True, progress_bar: bool = True
    ) -> SBT_RESULT:
        """Uses the base backtest method to compute the holdings and returns for the alpha instance on each day of
        the backtest

        Args:
            alpha: AbstractAlpha instance used to compute holdings
            use_env_universe: Default True, will use environment's universe size.
                              If False, environment will load new universe based on alpha
                              universe size <warning - requires API call>
            progress_bar: Default True, if False, progress bar (tqdm) will not be shown during the backtest

        Returns:
            returns and holdings for the given alpha instance over the environment's time range
        """
        universes = self._get_backtest_universes(alpha, use_env_universe)
        unique_tickers = np.unique(universes.values.flatten())
        returns = pd.Series(index=self.trading_days)
        holdings_df = pd.DataFrame(index=self.trading_days, columns=unique_tickers)
        self._base_backtest(self._update_returns_and_holdings, (returns, holdings_df), universes, alpha, progress_bar)
        return returns, holdings_df

    @staticmethod
    def _update_holdings(
            results: Tuple[pd.DataFrame], holdings: np.ndarray, day: Date, current_universe: pd.Series
    ):
        """Updates the holdings result tables by calculating the returns for the day given the
        computed holdings. Inserts the daily results in the appropriate index of the respective tables

        Note: This does not return anything, but simply updates the multable data structure that is passed in as
              <results>. The creation of the initial results data structure can be seen below

        Args:
            results: A tuple of only a pd.DataFrame which is the holdings table
            holdings: An np.ndarray including the weights returned by the alpha for the given backtest day
            day: The date which the holdings correspond to in the backtest process
            current_universe: The pd.Series of the universe being used on the given date in the backtest process
        """
        results.loc[day, current_universe] = holdings

    def compute_holdings_only(
            self, alpha: AbstractAlpha, use_env_universe: bool = True, progress_bar: bool = True
    ) -> pd.DataFrame:
        """Uses the base backtest method to compute the holdings from the alpha instance on each day of the backtest

        Args:
            alpha: AbstractAlpha instance used to compute holdings
            use_env_universe: Default True, will use environment's universe size.
                              If False, environment will load new universe based on alpha
                              universe size <warning - requires API call>
            progress_bar: Default True, if False, progress bar (tqdm) will not be shown during the backtest

        Returns:
            returns and holdings for the given alpha instance over the environment's time range
        """
        universes = self._get_backtest_universes(alpha, use_env_universe)
        unique_tickers = np.unique(universes.values.flatten())
        holdings_df = pd.DataFrame(index=self.trading_days, columns=unique_tickers)
        self._base_backtest(FullSingleBacktester._update_holdings, holdings_df, universes, alpha, progress_bar)
        return holdings_df


class FullMultipleBacktester(FullSingleBacktester):
    """Child of FullSingleBacktester which uses parent backtest function repeatedly to compute
    backtest results for a list of alphas at once. Also uses multiprocessing to significantly
    improve the runtime of backtesting multiple alphas."""
    def single_backtest(
            self, alpha: List[AbstractAlpha],
            use_env_universe: bool = True,
            progress_bar: bool = True,
            names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Performs backtest process over given environment settings, using list of alpha
        instance's given without using any multiprocessing.

        Args:
            alpha: list of alpha instances for holdings computation
            use_env_universe: Default True, will use environment's universe size.
                              If False, environment will load new universe based on alpha
                              universe size <warning - requires API call>
            progress_bar: Default True, if False, progress bar (tqdm) will not be output for
                          completion of alpha computations
            names: names to label returns for alpha instances

        Returns:
            returns and holdings for the given alpha instances over the environment's time range
        """
        logging.info('Getting names for alpha objects')
        # Create data structure for output
        returns = pd.DataFrame(index=self.trading_days, columns=names)
        holdings = []
        # Iterate through alphas and compute results sequentially using single backtest function
        for name, a in (pbar := tqdm(zip(names, alpha), disable=not progress_bar, total=len(alpha))):
            pbar.set_description(name)
            logging.info(f"Backtesting alpha: {name}")
            ret, hold = super().backtest(a, use_env_universe=use_env_universe, progress_bar=False)
            returns[name] = ret
            holdings.append(hold)
        pbar.close()

        return returns, holdings

    def _mp_backtest(self, a, use_env_universe: bool = True):
        """Private function needed because of multiprocessing not supporting lambda functions
        Args:
            a: alpha instance
            use_env_universe: use_env_universe parameter used in passthrough to parent backtest

        Returns:
            Results of single alpha backtest
        """
        return super().backtest(a, use_env_universe=use_env_universe, progress_bar=False)

    def multi_backtest(
            self, alpha: List[AbstractAlpha],
            use_env_universe: bool = True,
            progress_bar: bool = True,
            names: Optional[List[str]] = None,
            processes: int = 0
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Uses multiprocessing to backtest a list of alpha instance's given the environment's
        settings and the number of processes given

        Args:
            alpha: list of alpha instances for holdings computation
            use_env_universe: Default True, will use environment's universe size.
                              If False, environment will load new universe based on alpha
                              universe size <warning - requires API call>
            progress_bar: Default True, if False, progress bar (tqdm) will not be output for
                          completion of alpha computations
            names: names to label returns for alpha instances
            processes: number of CPUs to use for multiprocessing
                       Default 0, uses as many as available

        Returns:
            returns and holdings for the given alpha instances over the environment's time range
        """
        returns = pd.DataFrame(index=self.trading_days, columns=names)
        holdings = []

        # Create partial function that only needs alpha parameter
        bt_func = partial(self._mp_backtest, use_env_universe=use_env_universe)

        logging.info("Running pooled multiprocessing backtest")
        pool = mp.Pool(processes=processes) if processes != 0 else mp.Pool(processes=mp.cpu_count())
        with tqdm(total=len(alpha), disable=not progress_bar) as pbar:
            pbar.set_description(names[0])
            for name, desc, (ret, hold) in zip(names, names[1:] + ['Done!'], pool.imap(bt_func, alpha)):
                pbar.set_description(desc)
                returns[name] = ret
                holdings.append(hold)
                pbar.update(1)

        return returns, holdings

    def backtest(
            self, alpha: List[AbstractAlpha],
            use_env_universe: bool = True,
            progress_bar: bool = True,
            names: Optional[List[str]] = None,
            processes: Optional[int] = 0
    ) -> MBT_RESULT:
        """Overrides the single backtest method to give optional functionality to use
        multiprocessing in backtesting a list of alpha instances using the environment's settings

        Args:
            alpha: list of alpha instances for holdings computation
            use_env_universe: Default True, will use environment's universe size.
                              If False, environment will load new universe based on alpha
                              universe size <warning - requires API call>
            progress_bar: Default True, if False, progress bar (tqdm) will not be output for
                          completion of alpha computations
            names: names to label returns for alpha instances
            processes: number of CPUs to use for multiprocessing, if None, does not use
                       multiprocessing. Default 0, uses all availble CPUs

        Returns:
            returns and holdings for the given alpha instances over the environment's time range
        """
        logging.info('Getting names for alphas')
        if names is None:
            # Get names from attribute of alpha instances
            names = [str(a.name) for a in alpha]
            # If instance names are not unique, fail safe to range
            if len(set(names)) != len(names):
                names = list(map(str, range(len(alpha))))
        else:
            if len(set(names)) != len(alpha):
                raise AttributeError("Length of given alpha names does not match number of alphas "
                                     "or names are not unique")
        if processes is not None:
            if processes < 0:
                raise AttributeError("Number of processes must be a positive number or 0 to use"
                                     "all available cores!")
            logging.info(f"Running multiprocessing backtest using "
                         f"{processes if processes else mp.cpu_count()} CPUs")
            return self.multi_backtest(alpha, use_env_universe, progress_bar, names, processes)

        logging.info('Running backtest with single CPU')
        return self.single_backtest(alpha, use_env_universe, progress_bar, names)
