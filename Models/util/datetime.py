
import datetime
import time as clock
import numpy as np

from numba import njit, types, typeof, objmode
from numba.experimental import jitclass
from numba.typed import Dict
from typing import Union

INT     = types.int64
FLOAT   = types.float64
STR     = typeof('str')


def unix_to_datetime(unix_time: int, micro=False):
    dt = datetime.datetime.fromtimestamp(int(unix_time))
    if not micro:
        formatted_datetime = dt.strftime("%A, %B %d, %Y - %H:%M:%S")
    else:
        formatted_datetime = dt.strftime("%A, %B %d, %Y - %H:%M:%S.%f")
    return formatted_datetime


def date_to_unix_time(day: int, month: int, year: int, days_offset=0):
    try:
        # Get the required date and time
        input_date = datetime.datetime(year, month, day, 0, 0, 0, 0)

        # Subtract the days_offset to get the desired date
        target_date = input_date - datetime.timedelta(days=days_offset)

        # Convert to Unix time (seconds since epoch)
        unix_time = int(target_date.timestamp())
        return unix_time
    except ValueError as e:
        return f"UtilityError: {e}"


def unix_to_datetime_file(unix_time: int):
    dt = datetime.datetime.fromtimestamp(int(unix_time))
    formatted_datetime = dt.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime


def GetDayStartTime(offset=0):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Subtract the days_offset to get the desired date
    target_date = current_datetime - datetime.timedelta(days=offset)

    # Set the time to the start of the day (midnight)
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert to Unix time (seconds since epoch)
    unix_time = int(start_of_day.timestamp())

    return unix_time


def GetWeekStartTime(offset=0):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Calculate the start of the current week (assuming the week starts on Monday)
    start_of_week = current_datetime - datetime.timedelta(days=current_datetime.weekday())

    # Apply the week offset
    target_week = start_of_week - datetime.timedelta(weeks=offset)

    # Set the time to the start of the day (midnight)
    target_week_start = target_week.replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert to Unix time (seconds since epoch)
    unix_time = int(target_week_start.timestamp())

    return unix_time


def GetMonthStartTime(offset=0):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Calculate the year and month with the offset
    total_months = current_datetime.year * 12 + current_datetime.month - 1 - offset
    target_year = total_months // 12
    target_month = total_months % 12 + 1

    # Create a datetime object for the start of the target month
    start_of_month = datetime.datetime(target_year, target_month, 1)

    # Convert to Unix time (seconds since epoch)
    unix_time = int(start_of_month.timestamp())

    return unix_time


def GetYearStartTime(offset=0):
    # Get the current time in seconds since the epoch
    current_time = clock.time()

    # Get the current year
    current_year = clock.gmtime(current_time).tm_year

    # Calculate the target year
    target_year = current_year - offset

    # Create a struct_time object for the start of the target year -> *(January 1, 00:00:00)
    start_of_year = clock.struct_time((target_year, 1, 1, 0, 0, 0, 0, 0, -1))

    # Convert the struct_time to Unix time
    unix_time = clock.mktime(start_of_year)
    return unix_time


def eta(time_start, units_done, units_total, text=''):
    time_done = clock.perf_counter() - time_start
    progress = round(units_done / units_total * 100)
    eta_ = round(time_done / units_done * (units_total - units_done))
    print(f"\r{text} | {progress}% | eta={eta_}s | ", end='')


# Helper functions for date arithmetic
@njit # (cache=True)
def is_leap_year(year):
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


@njit # (cache=True)
def days_in_month(year, month):
    """Return the number of days in a month for a given year."""
    if month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    else:
        return 31


@njit # (cache=True)
def ConvertUnixTime(timestamp: int):
    """
    Convert Unix timestamp to date and time components without using datetime or time modules.
    This implementation is compatible with Numba njit.
    Can be used to build datetime in yyyy-mm-dd hh:mm:ss format.

    Args:
        timestamp: Unix timestamp (seconds since January 1, 1970)

    Returns:
        Tuple of (year, month, day, hour, minute, second)
    """
    # Extract time components
    seconds = int(timestamp % 60)
    timestamp //= 60
    minutes = int(timestamp % 60)
    timestamp //= 60
    hours = int(timestamp % 24)
    timestamp //= 24

    # Now timestamp is days since epoch (January 1, 1970)
    days = int(timestamp)

    # Account for leap years
    # Algorithm from http://howardhinnant.github.io/date_algorithms.html

    # Convert the days since epoch to days since 0000-03-01
    days += 719468

    # Compute the era and day of era
    era = (days if days >= 0 else days - 146096) // 146097
    day_of_era = days - era * 146097

    # Compute the year of era and day of year
    year_of_era = (day_of_era - day_of_era // 1460 + day_of_era // 36524 - day_of_era // 146096) // 365
    year = year_of_era + era * 400
    day_of_year = day_of_era - (365 * year_of_era + year_of_era // 4 - year_of_era // 100)

    # Compute month and day of month
    mp = (5 * day_of_year + 2) // 153
    day = day_of_year - (153 * mp + 2) // 5 + 1

    # Adjust month
    if mp < 10:
        month = mp + 3
    else:
        month = mp - 9
        year += 1

    return year, month, day, hours, minutes, seconds


@jitclass([
    ('year', INT),
    ('month', INT),
    ('day', INT),
    ('hour', INT),
    ('minute', INT),
    ('second', INT),
    ('_days_mapping', types.DictType(INT, STR)),
    ('_months_mapping', types.DictType(INT, STR)),
])
class DateTime:
    def __init__(self, year, month, day, hour, minute, second):
        self.year: int = year
        self.month: int = month
        self.day: int = day
        self.hour: int = hour
        self.minute: int = minute
        self.second: int = second

        self._days_mapping = Dict([
            (1, 'Monday'),
            (2, 'Tuesday'),
            (3, 'Wednesday'),
            (4, 'Thursday'),
            (5, 'Friday'),
            (6, 'Saturday'),
            (7, 'Sunday')
        ])

        self._months_mapping = Dict([
            (1, 'January'),
            (2, 'February'),
            (3, 'March'),
            (4, 'April'),
            (5, 'May'),
            (6, 'June'),
            (7, 'July'),
            (8, 'August'),
            (9, 'September'),
            (10, 'October'),
            (11, 'November'),
            (12, 'December')
        ])

    @staticmethod
    def format_fill(text: str, min_len: int, value: str = " "):
        fill = value * round(min_len - len(text))
        return fill + text

    def __str__(self):
        return f"{self.year}-" \
               f"{self.format_fill(str(self.month), 2, '0')}-" \
               f"{self.format_fill(str(self.day), 2, '0')} " \
               f"{self.format_fill(str(self.hour), 2, '0')}:" \
               f"{self.format_fill(str(self.minute), 2, '0')}:" \
               f"{self.format_fill(str(self.second), 2, '0')}"

    def dow(self):
        """
        Returns the day of the week as a string (Monday to Sunday)
        Uses Zeller's congruence algorithm to calculate day of week
        """
        # Adjust month for Zeller's congruence (Jan and Feb are 13 and 14 of prev year)
        m = self.month
        y = self.year
        if m < 3:
            m += 12
            y -= 1

        # Calculate day of week (0=Saturday, 1=Sunday, 2=Monday, ..., 6=Friday)
        k = self.day
        j = y // 100
        h = (k + ((13 * (m + 1)) // 5) + y % 100 + ((y % 100) // 4) + (j // 4) - (2 * j)) % 7

        # Convert to 1-7 (Monday to Sunday)
        dow = ((h + 5) % 7) + 1

        return dow

    def day_of_week(self):
        return self._days_mapping[self.dow()]

    def month_of_year(self):
        """
        Returns the month name as a string (January to December)
        """
        if 1 <= self.month <= 12:
            return self._months_mapping[self.month]
        else:
            raise ValueError(f"Invalid month: {self.month}")

    def date(self):
        return f"{self.day_of_week()}, {self.month_of_year()} {self.day}, {self.year}"

    def __eq__(self, other):
        """Equal to comparison operator"""
        if not isinstance(other, DateTime):
            return False
        return (self.year == other.year and
                self.month == other.month and
                self.day == other.day and
                self.hour == other.hour and
                self.minute == other.minute and
                self.second == other.second)

    def __lt__(self, other):
        """Less than comparison operator"""
        if not isinstance(other, DateTime):
            raise TypeError("Can only compare with another DateTime object")

        # Compare components in order of significance
        if self.year != other.year:
            return self.year < other.year
        if self.month != other.month:
            return self.month < other.month
        if self.day != other.day:
            return self.day < other.day
        if self.hour != other.hour:
            return self.hour < other.hour
        if self.minute != other.minute:
            return self.minute < other.minute
        return self.second < other.second

    def __gt__(self, other):
        """Greater than comparison operator"""
        if not isinstance(other, DateTime):
            raise TypeError("Can only compare with another DateTime object")

        # Not equal and not less than means greater than
        return not (self == other or self < other)

    def __le__(self, other):
        """Less than or equal to comparison operator"""
        return self < other or self == other

    def __ge__(self, other):
        """Greater than or equal to comparison operator"""
        return self > other or self == other

    def __add__(self, seconds: int):
        """Add seconds to a DateTime object and return a new DateTime object."""
        if not isinstance(seconds, int):
            raise TypeError("Can only add integer seconds to DateTime")

        # Start with the current date/time components
        year = self.year
        month = self.month
        day = self.day
        hour = self.hour
        minute = self.minute
        second = self.second

        # Add seconds
        second += seconds

        # Handle overflow of seconds to minutes
        minute += second // 60
        second %= 60

        # Handle overflow of minutes to hours
        hour += minute // 60
        minute %= 60

        # Handle overflow of hours to days
        day += hour // 24
        hour %= 24

        # Handle overflow of days to months/years
        while True:
            days_this_month = days_in_month(year, month)
            if day <= days_this_month:
                break

            # Move to next month
            day -= days_this_month
            month += 1

            # Handle overflow of months to years
            if month > 12:
                month = 1
                year += 1

        return DateTime(year, month, day, hour, minute, second)

    def __sub__(self, other: Union[int, 'DateTime']):
        """Subtract another DateTime object or seconds from this DateTime."""
        if isinstance(other, int):
            # Subtracting seconds - negate and use __add__
            return self.__add__(-other)

        # TODO: Use Numba extending to implement DateTime subtraction using other DateTime objects
        # if not isinstance(other, DateTime):
        #     raise TypeError("Can only subtract DateTime object or integer seconds")

        # # For DateTime subtraction, we'll convert both to seconds since epoch
        # # and return the difference in seconds

        # # Helper function to convert DateTime to approximate seconds since epoch
        # def to_seconds(dt: DateTime):
        #     # This is an approximation that works for comparison and basic arithmetic
        #     days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        #     # Adjust for leap years
        #     if is_leap_year(dt.year):
        #         days_per_month[2] = 29

        #     # Calculate days from years (approximate)
        #     days = dt.year * 365 + dt.year // 4 - dt.year // 100 + dt.year // 400

        #     # Add days from months
        #     for m in range(1, dt.month):
        #         days += days_in_month(dt.year, m)

        #     # Add days of current month
        #     days += dt.day - 1  # -1 because we're counting from day 1

        #     # Convert to seconds and add time components
        #     seconds = days * 86400 + dt.hour * 3600 + dt.minute * 60 + dt.second
        #     return seconds

        # # Calculate difference in seconds
        # diff_seconds = to_seconds(self) - to_seconds(other)
        # return diff_seconds

        raise TypeError("Cannot subtract DateTime objects ")

    def __mul__(self, other):
        """Multiplication operator"""
        raise RuntimeError("Cannot multiply DateTime objects")

    def __truediv__(self, other):
        """True division operator"""
        raise RuntimeError("Cannot divide DateTime objects")

    def __floordiv__(self, other):
        """Floor division operator"""
        raise RuntimeError("Cannot floor divide DateTime objects")

    def __mod__(self, other):
        """Modulo operator"""
        raise RuntimeError("Cannot modulo DateTime objects")

    def __pow__(self, other):
        """Power operator"""
        raise RuntimeError("Cannot raise DateTime objects to a power")

    def __lshift__(self, other):
        """Left shift operator"""
        raise RuntimeError("Cannot left shift DateTime objects")

    def __rshift__(self, other):
        """Right shift operator"""
        raise RuntimeError("Cannot right shift DateTime objects")

    def __and__(self, other):
        """Bitwise AND operator"""
        raise NotImplementedError("Cannot perform bitwise AND on DateTime objects yet")

    def __or__(self, other):
        """Bitwise OR operator"""
        raise NotImplementedError("Cannot perform bitwise OR on DateTime objects yet")

    def __xor__(self, other):
        """Bitwise XOR operator"""
        raise NotImplementedError("Cannot perform bitwise XOR on DateTime objects yet")

    def __neg__(self):
        """Negation operator"""
        raise RuntimeError("Cannot negate DateTime objects")

    def __pos__(self):
        """Positive operator"""
        raise RuntimeError("Cannot make DateTime objects positive")

    def to_numpy(self) -> np.datetime64:
        values = (self.year, self.month, self.day, self.hour, self.minute, self.second)
        with objmode(var="types.NPDatetime('s')"):
            var = np.array(["{}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}".format(*values)], dtype="datetime64[s]")[0]
        return var

    def to_unix(self) -> int:
        raise NotImplementedError("Cannot convert DateTime to Unix time, YET!")

    def copy(self):
        return DateTime(self.year, self.month, self.day, self.hour, self.minute, self.second)


@njit
def get_datetime(unix_time: int):
    return DateTime(*ConvertUnixTime(unix_time))


@njit
def add_time(dt: DateTime, seconds=0):
    """Add time (in seconds) to a DateTime object and return a new DateTime object.

        Args:
            dt: DateTime object
            seconds: Number of seconds to add (can be negative)

        Returns:
            A new DateTime object with the time added
    """
    return dt + seconds


def add_time_factors(dt: DateTime, factors: dict[str, int]):
    """
    Adds the specified time factors to the given datetime
    Possible factors: year, month, day, hour, minute, second

    Args:
        dt: The datetime to add the factors to
        factors: A dictionary of time factors to add

    Returns:
        The new DateTime object with the factors added
    """
    # Start with the current date/time components
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    # Add each factor if present
    if 'year' in factors:
        year += factors['year']
    if 'month' in factors:
        month += factors['month']
        # Handle overflow of months to years
        while month > 12:
            month -= 12
            year += 1
        # Handle underflow of months from years
        while month < 1:
            month += 12
            year -= 1
    if 'day' in factors:
        day += factors['day']
    if 'hour' in factors:
        hour += factors['hour']
    if 'minute' in factors:
        minute += factors['minute']
    if 'second' in factors:
        second += factors['second']

    # Create a new DateTime with the base components. TODO: Work on Numba extending in DateTime before using this func
    result = DateTime(year, month, day, hour, minute, second)

    # Normalize the time components (handle overflow)
    # Convert excess seconds to minutes
    excess_minutes = second // 60
    second %= 60
    minute += excess_minutes

    # Convert excess minutes to hours
    excess_hours = minute // 60
    minute %= 60
    hour += excess_hours

    # Convert excess hours to days
    excess_days = hour // 24
    hour %= 24
    day += excess_days

    # Handle days overflow/underflow with months
    while True:
        # Check if day is too large for current month
        days_this_month = days_in_month(year, month)
        if day <= days_this_month:
            break

        # Move to next month
        day -= days_this_month
        month += 1

        # Handle month overflow
        if month > 12:
            month = 1
            year += 1

    # Handle negative days (underflow)
    while day < 1:
        # Move to previous month
        month -= 1

        # Handle month underflow
        if month < 1:
            month = 12
            year -= 1

        day += days_in_month(year, month)

    return DateTime(year, month, day, hour, minute, second)
