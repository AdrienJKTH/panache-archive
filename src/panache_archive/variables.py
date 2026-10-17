"""What the archive holds: CAMS Europe surface concentrations, and how each day sums them up."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Final

UNIT: Final = "μg/m³"
"""Every variable is a mass concentration, written as Panache and Open-Meteo write it."""


class Variable(StrEnum):
    """Surface concentrations, valued with their Open-Meteo names, which Panache uses too."""

    PM10 = "pm10"
    PM2_5 = "pm2_5"
    NITROGEN_DIOXIDE = "nitrogen_dioxide"
    OZONE = "ozone"
    CARBON_MONOXIDE = "carbon_monoxide"
    SULPHUR_DIOXIDE = "sulphur_dioxide"
    AMMONIA = "ammonia"
    DUST = "dust"
    SECONDARY_INORGANIC_AEROSOL = "secondary_inorganic_aerosol"
    PM10_WILDFIRES = "pm10_wildfires"


ADS_NAME: Final[Mapping[Variable, str]] = {
    Variable.PM10: "particulate_matter_10um",
    Variable.PM2_5: "particulate_matter_2.5um",
    Variable.NITROGEN_DIOXIDE: "nitrogen_dioxide",
    Variable.OZONE: "ozone",
    Variable.CARBON_MONOXIDE: "carbon_monoxide",
    Variable.SULPHUR_DIOXIDE: "sulphur_dioxide",
    Variable.AMMONIA: "ammonia",
    Variable.DUST: "dust",
    Variable.SECONDARY_INORGANIC_AEROSOL: "secondary_inorganic_aerosol",
    Variable.PM10_WILDFIRES: "pm10_wildfires",
}
"""Each variable's name in the ADS dataset `cams-europe-air-quality-reanalyses`."""


class Statistic(StrEnum):
    """How a day is summed up from its hours: the column names of the archive."""

    MEAN = "mean"
    """The UTC daily mean: the WHO and EU metric for particles."""
    MAX_8H_MEAN = "max_8h_mean"
    """The day's highest running 8-hour mean: the WHO and EU metric for ozone (Panache's D31)."""


def statistics(variable: Variable) -> tuple[Statistic, ...]:
    """The daily statistics stored for `variable`: the mean, and ozone's 8-hour maximum too."""
    if variable is Variable.OZONE:
        return (Statistic.MEAN, Statistic.MAX_8H_MEAN)
    return (Statistic.MEAN,)


class Reanalysis(StrEnum):
    """The two streams of the reanalysis, valued with their ADS names."""

    VALIDATED = "validated_reanalysis"
    """Final, a year or two behind."""
    INTERIM = "interim_reanalysis"
    """Available sooner, replaced by the validated stream when it comes out."""

    @property
    def label(self) -> str:
        """Short name, as in asset names."""
        return self.value.removesuffix("_reanalysis")
