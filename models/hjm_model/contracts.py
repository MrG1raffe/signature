import numpy as np
import pandas as pd
from numpy import float_
from typing import Union

MONTH_DICT = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12"
}


class ForwardContract:
    name: str
    delivery_start_date: Union[np.datetime64, pd.Timestamp]
    delivery_end_date: Union[np.datetime64, pd.Timestamp]
    forward_type: str
    observation_date: Union[np.datetime64, pd.Timestamp]
    time_to_delivery_start: float
    time_to_delivery_end: float
    F0: float_

    def __init__(
        self,
        name: str,
        observation_date: np.datetime64 = None,
        F0: float_ = None,
    ):
        """
        Creates the forward contract object calculating the delivery period and time to delivery start / end in years.
        from the contract name and the observation date.

        :param name: name of the contract, possible formats "CalYY" for year contracts, "MonYY" for month contracts,
            "QXYY" with X = 1, 2, 3, 4, for quarter contracts, "DX" with X = 1, 2, ... for day contracts,
            "SUMYY" for summer contracts (Q2YY+Q3YY), "WINYY" for winter contracts (Q4YY+Q1(YY+1)).
        :param observation_date: date used to calculate the time to delivery start / end.
        :param F0: forward price as of observation date.
        """
        if len(name) not in (4, 5) and not name.startswith("D"):
            raise ValueError("Length of name is not equal to 4 or 5.")
        self.name = name
        if not name[:3] in MONTH_DICT.keys():
            if name.lower().startswith("cal"):
                year = "20" + name[3:5]
                self.delivery_start_date = pd.Timestamp(np.datetime64(year + '-01-01'))
                self.delivery_end_date = pd.Timestamp(self.delivery_start_date) + pd.DateOffset(years=1)
                self.forward_type = "cal"
            elif name.startswith("Q"):
                year = "20" + name[2:4]
                quart_num = int(name[1]) - 1
                if quart_num not in range(4):
                    raise ValueError("Wrong quarter number was given.")
                self.delivery_start_date = pd.Timestamp(year + '-01-01') + pd.DateOffset(months=3 * quart_num)
                self.delivery_end_date = pd.Timestamp(self.delivery_start_date) + pd.DateOffset(months=3)
                self.forward_type = "quart"
            elif name.startswith("D"):
                days_to_add = int(name[1:])
                self.delivery_start_date = pd.Timestamp(observation_date) + pd.DateOffset(days=days_to_add)
                self.delivery_end_date = self.delivery_start_date + pd.DateOffset(days=1)
                self.forward_type = "daily"
            elif name.startswith('SUM'):
                year = "20" + name[3:5]
                self.delivery_start_date = pd.Timestamp(np.datetime64(year + '-04-01'))
                self.delivery_end_date = pd.Timestamp(np.datetime64(year + '-10-01'))
                self.forward_type = "sum"
            elif name.startswith('WIN'):
                previous_year = "20" + name[3:5]
                final_year = str(int(previous_year) + 1)
                self.delivery_start_date = pd.Timestamp(np.datetime64(previous_year + '-10-01'))
                self.delivery_end_date = pd.Timestamp(np.datetime64(final_year + '-04-01'))
                self.forward_type = "win"
        else:
            month = MONTH_DICT[name[:3]]
            year = "20" + name[3:5]
            self.delivery_start_date = pd.Timestamp(np.datetime64(year + '-' + month + '-' + '01'))
            self.delivery_end_date = pd.Timestamp(self.delivery_start_date) + pd.DateOffset(months=1)
            self.forward_type = "mon"

        if observation_date is not None:
            self.observation_date = observation_date
            self.time_to_delivery_start = pd.Timedelta(self.delivery_start_date - observation_date).days / 365
            self.time_to_delivery_end = pd.Timedelta(self.delivery_end_date - observation_date).days / 365

        self.F0 = F0


def is_strictly_included(contract_name_1: str, contract_name_2: str):
    """
    Check whether the delivery period of contract `contract_name_1` is strictly included into the one of
    contract `contract_name_1`.

    :param contract_name_1: name of the first contract.
    :param contract_name_2: name of the second contract.
    :return: True if the first delivery period is strictly included in the second one.
    """
    if contract_name_1 == contract_name_2:
        return False
    contract_1 = ForwardContract(name=contract_name_1)
    contract_2 = ForwardContract(name=contract_name_2)
    return (contract_1.delivery_start_date >= contract_2.delivery_start_date) and \
           (contract_1.delivery_end_date <= contract_2.delivery_end_date)
