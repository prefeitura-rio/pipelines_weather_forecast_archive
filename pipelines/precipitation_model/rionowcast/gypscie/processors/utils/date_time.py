# -*- coding: utf-8 -*-
# pylint: disable= invalid-name, inconsistent-return-statements, broad-except
"""
ADD
"""
from datetime import date, datetime

from dateutil import parser
from dateutil.relativedelta import relativedelta


def is_valid_date(str_date: str):
    """
    ADD
    """
    parser.parse(str_date).date()
    return True


def day_of_year(date_obj: datetime or date):
    """
    ADD
    """
    assert isinstance(date_obj, (datetime, date))
    return date_obj.timetuple().tm_yday


def str_day_of_year(date_obj: datetime or date, num_digits=3):
    """
    ADD
    """
    assert isinstance(date_obj, (datetime, date))
    str_day = str(day_of_year(date_obj))
    return str_day.zfill(num_digits)


def parse_to_date(str_date: str):
    """
    ADD
    """
    try:
        return parser.parse(str_date).date()
    except Exception as e:
        print(e)


def parse_to_datetime(str_date: str):
    """
    ADD
    """
    try:
        return parser.parse(str_date)
    except Exception as e:
        print(e)


def split_date(dt: date):
    """
    ADD
    """
    return [dt.year, dt.month, dt.day]


def today():
    """
    ADD
    """
    return datetime.today().date()


def next_day(dt: date):
    """
    ADD
    """
    return dt + relativedelta(days=+1)
