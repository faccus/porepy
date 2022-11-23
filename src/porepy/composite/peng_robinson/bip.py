"""This modules contains Binary Interaction Parameters for components modelled for the
Peng-Robinson EoS.

BIPs are intended for genuine components (and compounds), but not for pseudo-components.
The effect of pseudo-components is integrated in respective interaction law involving
compounds.

References for these largely heuristic laws and values can be found in respective
implementations.

BIPs are implemented as callable objects. This module provides a map ``BIP_MAP`` which maps
between two components and their respective BIP.

The BIP between a component/compound and itself is assumed to be 0, and hence not given here.

"""
from __future__ import annotations

from typing import Callable

import porepy as pp

from ..phase import VarLike
from .model_components import *

__all__ = [
    "BIP_MAP",
    "get_BIP",
]


def bip_H2O_CO2(T: VarLike, h2o: H2O, co2: CO2) -> pp.ad.Operator:
    """(Constant) BIP for water and carbon dioxide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2O: 7732-18-5
    - CO2: 124-38-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0952)


def bip_H2O_H2S(T: VarLike, h20: H2O, h2s: H2S) -> pp.ad.Operator:
    """(Constant) BIP for water and hydrogen sulfide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2O: 7732-18-5
    - H2S: 7783-06-4

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0394)


def bip_H2O_N2(
    T: VarLike, h2o: H2O, n2: N2
) -> pp.ad.Operator:  # TODO no results found so far
    """(Constant) BIP for water and nitrogen.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2O: 7732-18-5
    - N2: 7727-37-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0)


def bip_CO2_H2S(T: VarLike, co2: CO2, h2s: H2S) -> pp.ad.Operator:
    """(Constant) BIP for carbon dioxide and hydrogen sulfide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - CO2: 124-38-9
    - H2S: 7783-06-4

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.0967)


def bip_CO2_N2(T: VarLike, co2: CO2, n2: N2) -> pp.ad.Operator:
    """(Constant) BIP for carbon dioxide and nitrogen.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - CO2: 124-38-9
    - N2: 7727-37-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(-0.0122)


def bip_N2_H2S(T: VarLike, n2: N2, h2s: H2S) -> pp.ad.Operator:
    """(Constant) BIP for nitrogen and hydrogen sulfide.

    The value is taken from
    `thermo <https://thermo.readthedocs.io/thermo.interaction_parameters.html>`_,
    ``IPDB.get_ip_automatic`` with CAS numbers:

    - H2S: 7783-06-4
    - N2: 7727-37-9

    Returns:
        An AD Operator representing the constant scalar BIP.

    """
    return pp.ad.Scalar(0.1652)


def bip_NaClbrine_H2S(T: VarLike, naclbrine: NaClBrine, h2s: H2S) -> pp.ad.Operator:
    """Temperature- and salinity-dependent BIP for NaCl-brine and hydrogen sulfide.

    The law is taken from
    `Soereide (1992), equation 15 <https://doi.org/10.1016/0378-3812(92)85105-H>`_.

    Returns:
        An AD Operator representing the BIP.

    """
    return -0.20441 + 0.23426 * T / h2s.critical_temperature()


def bip_NaClbrine_CO2(T: VarLike, naclbrine: NaClBrine, co2: CO2) -> pp.ad.Operator:
    """Temperature- and salinity-dependent BIP for NaCl-brine and carbon dioxide.

    The law is taken from
    `Soereide (1992), equation 14 <https://doi.org/10.1016/0378-3812(92)85105-H>`_.

    Returns:
        An AD Operator representing the BIP.

    """
    T_co2_crit = co2.critical_temperature()
    molality = naclbrine.molality_of(naclbrine.NaCl)
    exp = pp.ad.Function(pp.ad.exp, "exp")

    return (
        -0.31092 * (1 + 0.15582 * molality**0.7505)
        + 0.23580 * (1 + 0.17837 * molality**0.979) * T / T_co2_crit
        - 21.2566 * exp(-6.7222 * T / T_co2_crit - molality)
    )


def bip_NaClbrine_N2(T: VarLike, naclbrine: NaClBrine, n2: N2) -> pp.ad.Operator:
    """Temperature- and salinity-dependent BIP for NaCl-brine and nitrogen.

    The law is taken from
    `Soereide (1992), equation 13 <https://doi.org/10.1016/0378-3812(92)85105-H>`_.

    Returns:
        An AD Operator representing the BIP.

    """
    T_n2_crit = n2.critical_temperature()
    molality = naclbrine.molality_of(naclbrine.NaCl)

    return (
        -1.70235 * (1 + 0.25587 * molality**0.75)
        + 0.44338 * (1 + 0.08126 * molality**0.75) * T / T_n2_crit
    )


BIP_MAP: dict[tuple[str, str], Callable] = {
    ("H2O", "CO2"): bip_H2O_CO2,
    ("H2O", "H2S"): bip_H2O_H2S,
    # ("H2O", "N2"): bip_H2O_N2,  # not available, see above TODO
    ("CO2", "H2S"): bip_CO2_H2S,
    ("CO2", "N2"): bip_CO2_N2,
    ("N2", "H2S"): bip_N2_H2S,
    ("NaClBrine", "H2S"): bip_NaClbrine_H2S,
    ("NaClBrine", "CO2"): bip_NaClbrine_CO2,
    ("NaClBrine", "N2"): bip_NaClbrine_N2,
}
"""Contains for a pair of component/compound names (key) the respective
binary interaction parameter in form of a callable.

This map serves the Peng-Robinson composition to assemble the attraction parameter of the
mixture and its intended use is only there.

"""


def get_BIP(component1: str, component2: str) -> tuple[Callable | None, bool]:
    """Returns the callable representing a BIP for two given component names.

    This function is a wrapper for accessing :data:`BIP_MAP`, which is not sensitive
    the order in the 2-tuple containing component names.

    Parameters:
        component1: name of the first component
        component2: name of the second component

    Returns:
        A callable implemented which represents the BIP for given components, if implemented.

        If no BIP is available, returns ``None``.

        The second entry of the tuple is a bool indicating whether the order of input arguments
        fits the order for the BIP arguments. It is ``False``, if the BIP argument order is
        ``component2, component1``.

        If no BIP is found, the bool has no meaning.

    """
    # try input order
    BIP = BIP_MAP.get((component1, component2), None)
    order = True

    # try reverse order
    if BIP is None:
        BIP = BIP_MAP.get((component2, component1), None)
        order = False

    # return what is found, possibly None
    return BIP, order