from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from pathlib import Path
import warnings
from typing import Any

import lxml.etree as ET
import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, RegularGridInterpolator
from scipy.linalg import eigh
from scipy.special import jv

import openmc
import openmc.checkvalue as cv
from openmc.checkvalue import PathLike
from openmc.stats.multivariate import UnitSphere, Spatial
from openmc.stats.univariate import Univariate
from ._xml import get_elem_list, get_text
from .mesh import MeshBase, StructuredMesh, UnstructuredMesh
from .particle_type import ParticleType
from .statepoint import _VERSION_STATEPOINT
from .utility_funcs import input_path


class SourceBase(ABC):
    """Base class for external sources

    Parameters
    ----------
    strength : float
        Strength of the source
    constraints : dict
        Constraints on sampled source particles. Valid keys include 'domains',
        'time_bounds', 'energy_bounds', 'fissionable', and 'rejection_strategy'.
        For 'domains', the corresponding value is an iterable of
        :class:`openmc.Cell`, :class:`openmc.Material`, or
        :class:`openmc.Universe` for which sampled sites must be within. For
        'time_bounds' and 'energy_bounds', the corresponding value is a sequence
        of floats giving the lower and upper bounds on time in [s] or energy in
        [eV] that the sampled particle must be within. For 'fissionable', the
        value is a bool indicating that only sites in fissionable material
        should be accepted. The 'rejection_strategy' indicates what should
        happen when a source particle is rejected: either 'resample' (pick a new
        particle) or 'kill' (accept and terminate).

    Attributes
    ----------
    type : {'independent', 'file', 'compiled', 'mesh', 'tokamak'}
        Indicator of source type.
    strength : float
        Strength of the source
    constraints : dict
        Constraints on sampled source particles. Valid keys include
        'domain_type', 'domain_ids', 'time_bounds', 'energy_bounds',
        'fissionable', and 'rejection_strategy'.

    """

    def __init__(
        self,
        strength: float | None = 1.0,
        constraints: dict[str, Any] | None = None
    ):
        self.strength = strength
        self.constraints = constraints

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, strength):
        cv.check_type('source strength', strength, Real, none_ok=True)
        if strength is not None:
            cv.check_greater_than('source strength', strength, 0.0, True)
        self._strength = strength

    @property
    def constraints(self) -> dict[str, Any]:
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: dict[str, Any] | None):
        self._constraints = {}
        if constraints is None:
            return

        for key, value in constraints.items():
            if key == 'domains':
                cv.check_type('domains', value, Iterable,
                              (openmc.Cell, openmc.Material, openmc.Universe))
                if isinstance(value[0], openmc.Cell):
                    self._constraints['domain_type'] = 'cell'
                elif isinstance(value[0], openmc.Material):
                    self._constraints['domain_type'] = 'material'
                elif isinstance(value[0], openmc.Universe):
                    self._constraints['domain_type'] = 'universe'
                self._constraints['domain_ids'] = [d.id for d in value]
            elif key == 'time_bounds':
                cv.check_type('time bounds', value, Iterable, Real)
                self._constraints['time_bounds'] = tuple(value)
            elif key == 'energy_bounds':
                cv.check_type('energy bounds', value, Iterable, Real)
                self._constraints['energy_bounds'] = tuple(value)
            elif key == 'fissionable':
                cv.check_type('fissionable', value, bool)
                self._constraints['fissionable'] = value
            elif key == 'rejection_strategy':
                cv.check_value('rejection strategy',
                               value, ('resample', 'kill'))
                self._constraints['rejection_strategy'] = value
            else:
                raise ValueError(
                    f'Unknown key in constraints dictionary: {key}')

    @abstractmethod
    def populate_xml_element(self, element):
        """Add necessary source information to an XML element

        Returns
        -------
        element : lxml.etree._Element
            XML element containing source data

        """

    def to_xml_element(self) -> ET.Element:
        """Return XML representation of the source

        Returns
        -------
        element : xml.etree.ElementTree.Element
            XML element containing source data

        """
        element = ET.Element("source")
        element.set("type", self.type)
        if self.strength is not None:
            element.set("strength", str(self.strength))
        self.populate_xml_element(element)
        constraints = self.constraints
        if constraints:
            constraints_elem = ET.SubElement(element, "constraints")
            if "domain_ids" in constraints:
                dt_elem = ET.SubElement(constraints_elem, "domain_type")
                dt_elem.text = constraints["domain_type"]
                id_elem = ET.SubElement(constraints_elem, "domain_ids")
                id_elem.text = ' '.join(str(uid)
                                        for uid in constraints["domain_ids"])
            if "time_bounds" in constraints:
                dt_elem = ET.SubElement(constraints_elem, "time_bounds")
                dt_elem.text = ' '.join(str(t)
                                        for t in constraints["time_bounds"])
            if "energy_bounds" in constraints:
                dt_elem = ET.SubElement(constraints_elem, "energy_bounds")
                dt_elem.text = ' '.join(str(E)
                                        for E in constraints["energy_bounds"])
            if "fissionable" in constraints:
                dt_elem = ET.SubElement(constraints_elem, "fissionable")
                dt_elem.text = str(constraints["fissionable"]).lower()
            if "rejection_strategy" in constraints:
                dt_elem = ET.SubElement(constraints_elem, "rejection_strategy")
                dt_elem.text = constraints["rejection_strategy"]

        return element

    @classmethod
    def from_xml_element(cls, elem: ET.Element, meshes=None) -> SourceBase:
        """Generate source from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element
        meshes : dict
            Dictionary with mesh IDs as keys and openmc.MeshBase instances as
            values

        Returns
        -------
        openmc.SourceBase
            Source generated from XML element

        """
        source_type = get_text(elem, 'type')

        if source_type is None:
            # attempt to determine source type based on attributes
            # for backward compatibility
            if get_text(elem, 'file') is not None:
                return FileSource.from_xml_element(elem)
            elif get_text(elem, 'library') is not None:
                return CompiledSource.from_xml_element(elem)
            else:
                return IndependentSource.from_xml_element(elem)
        else:
            if source_type == 'independent':
                return IndependentSource.from_xml_element(elem, meshes)
            elif source_type == 'compiled':
                return CompiledSource.from_xml_element(elem)
            elif source_type == 'file':
                return FileSource.from_xml_element(elem)
            elif source_type == 'mesh':
                return MeshSource.from_xml_element(elem, meshes)
            elif source_type == 'tokamak':
                return TokamakSource.from_xml_element(elem)
            else:
                raise ValueError(
                    f'Source type {source_type} is not recognized')

    @staticmethod
    def _get_constraints(elem: ET.Element) -> dict[str, Any]:
        # Find element containing constraints
        constraints_elem = elem.find("constraints")
        elem = constraints_elem if constraints_elem is not None else elem

        constraints = {}
        domain_type = get_text(elem, "domain_type")
        if domain_type is not None:
            domain_ids = get_elem_list(elem, "domain_ids", int)

            # Instantiate some throw-away domains that are used by the
            # constructor to assign IDs
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', openmc.IDWarning)
                if domain_type == 'cell':
                    domains = [openmc.Cell(uid) for uid in domain_ids]
                elif domain_type == 'material':
                    domains = [openmc.Material(uid) for uid in domain_ids]
                elif domain_type == 'universe':
                    domains = [openmc.Universe(uid) for uid in domain_ids]
            constraints['domains'] = domains

        time_bounds = get_elem_list(elem, "time_bounds", float)
        if time_bounds is not None:
            constraints['time_bounds'] = time_bounds

        energy_bounds = get_elem_list(elem, "energy_bounds", float)
        if energy_bounds is not None:
            constraints['energy_bounds'] = energy_bounds

        fissionable = get_text(elem, "fissionable")
        if fissionable is not None:
            constraints['fissionable'] = fissionable in ('true', '1')

        rejection_strategy = get_text(elem, "rejection_strategy")
        if rejection_strategy is not None:
            constraints['rejection_strategy'] = rejection_strategy

        return constraints


class IndependentSource(SourceBase):
    """Distribution of phase space coordinates for source sites.

    .. versionadded:: 0.14.0

    Parameters
    ----------
    space : openmc.stats.Spatial
        Spatial distribution of source sites
    angle : openmc.stats.UnitSphere
        Angular distribution of source sites
    energy : openmc.stats.Univariate
        Energy distribution of source sites
    time : openmc.stats.Univariate
        time distribution of source sites
    strength : float
        Strength of the source
    particle : str or int or openmc.ParticleType
        Source particle type (name, PDG number, or type)
    domains : iterable of openmc.Cell, openmc.Material, or openmc.Universe
        Domains to reject based on, i.e., if a sampled spatial location is not
        within one of these domains, it will be rejected.

        .. deprecated:: 0.15.0
            Use the `constraints` argument instead.
    constraints : dict
        Constraints on sampled source particles. Valid keys include 'domains',
        'time_bounds', 'energy_bounds', 'fissionable', and 'rejection_strategy'.
        For 'domains', the corresponding value is an iterable of
        :class:`openmc.Cell`, :class:`openmc.Material`, or
        :class:`openmc.Universe` for which sampled sites must be within. For
        'time_bounds' and 'energy_bounds', the corresponding value is a sequence
        of floats giving the lower and upper bounds on time in [s] or energy in
        [eV] that the sampled particle must be within. For 'fissionable', the
        value is a bool indicating that only sites in fissionable material
        should be accepted. The 'rejection_strategy' indicates what should
        happen when a source particle is rejected: either 'resample' (pick a new
        particle) or 'kill' (accept and terminate).

    Attributes
    ----------
    space : openmc.stats.Spatial or None
        Spatial distribution of source sites
    angle : openmc.stats.UnitSphere or None
        Angular distribution of source sites
    energy : openmc.stats.Univariate or None
        Energy distribution of source sites
    time : openmc.stats.Univariate or None
        time distribution of source sites
    strength : float
        Strength of the source
    type : str
        Indicator of source type: 'independent'

        .. versionadded:: 0.14.0
    particle : str or int or openmc.ParticleType
        Source particle type (alias, PDG number, or GNDS nuclide name)
    constraints : dict
        Constraints on sampled source particles. Valid keys include
        'domain_type', 'domain_ids', 'time_bounds', 'energy_bounds',
        'fissionable', and 'rejection_strategy'.

    """

    def __init__(
        self,
        space: openmc.stats.Spatial | None = None,
        angle: openmc.stats.UnitSphere | None = None,
        energy: openmc.stats.Univariate | None = None,
        time: openmc.stats.Univariate | None = None,
        strength: float = 1.0,
        particle: str | int | ParticleType = 'neutron',
        domains: Sequence[openmc.Cell | openmc.Material |
                          openmc.Universe] | None = None,
        constraints: dict[str, Any] | None = None
    ):
        if domains is not None:
            warnings.warn("The 'domains' arguments has been replaced by the "
                          "'constraints' argument.", FutureWarning)
            constraints = {'domains': domains}

        super().__init__(strength=strength, constraints=constraints)

        self._space = None
        self._angle = None
        self._energy = None
        self._time = None

        if space is not None:
            self.space = space
        if angle is not None:
            self.angle = angle
        if energy is not None:
            self.energy = energy
        if time is not None:
            self.time = time
        self.particle = particle

    @property
    def type(self) -> str:
        return 'independent'

    def __getattr__(self, name):
        cls_names = {'file': 'FileSource', 'library': 'CompiledSource',
                     'parameters': 'CompiledSource'}
        if name in cls_names:
            raise AttributeError(
                f'The "{name}" attribute has been deprecated on the '
                f'IndependentSource class. Please use the {cls_names[name]} class.')
        else:
            super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in ('file', 'library', 'parameters'):
            # Ensure proper AttributeError is thrown
            getattr(self, name)
        else:
            super().__setattr__(name, value)

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        cv.check_type('spatial distribution', space, Spatial)
        self._space = space

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        cv.check_type('angular distribution', angle, UnitSphere)
        self._angle = angle

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        cv.check_type('energy distribution', energy, Univariate)
        self._energy = energy

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        cv.check_type('time distribution', time, Univariate)
        self._time = time

    @property
    def particle(self) -> ParticleType:
        return self._particle

    @particle.setter
    def particle(self, particle):
        self._particle = ParticleType(particle)

    def populate_xml_element(self, element):
        """Add necessary source information to an XML element

        Returns
        -------
        element : lxml.etree._Element
            XML element containing source data

        """
        element.set("particle", str(self.particle))
        if self.space is not None:
            element.append(self.space.to_xml_element())
        if self.angle is not None:
            element.append(self.angle.to_xml_element())
        if self.energy is not None:
            element.append(self.energy.to_xml_element('energy'))
        if self.time is not None:
            element.append(self.time.to_xml_element('time'))

    @classmethod
    def from_xml_element(cls, elem: ET.Element, meshes=None) -> SourceBase:
        """Generate source from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element
        meshes : dict
            Dictionary with mesh IDs as keys and openmc.MeshBase instaces as
            values

        Returns
        -------
        openmc.Source
            Source generated from XML element

        """
        constraints = cls._get_constraints(elem)
        source = cls(constraints=constraints)

        strength = get_text(elem, 'strength')
        if strength is not None:
            source.strength = float(strength)

        particle = get_text(elem, 'particle')
        if particle is not None:
            source.particle = particle

        space = elem.find('space')
        if space is not None:
            source.space = Spatial.from_xml_element(space, meshes)

        angle = elem.find('angle')
        if angle is not None:
            source.angle = UnitSphere.from_xml_element(angle)

        energy = elem.find('energy')
        if energy is not None:
            source.energy = Univariate.from_xml_element(energy)

        time = elem.find('time')
        if time is not None:
            source.time = Univariate.from_xml_element(time)

        return source


class MeshSource(SourceBase):
    """A source with a spatial distribution over mesh elements

    This class represents a mesh-based source in which random positions are
    uniformly sampled within mesh elements and each element can have independent
    angle, energy, and time distributions. The element sampled is chosen based
    on the relative strengths of the sources applied to the elements. The
    strength of the mesh source as a whole is the sum of all source strengths
    applied to the elements.

    .. versionadded:: 0.15.0

    Parameters
    ----------
    mesh : openmc.MeshBase
        The mesh over which source sites will be generated.
    sources : sequence of openmc.SourceBase
        Sources for each element in the mesh. Sources must be specified as
        either a 1-D array in the order of the mesh indices or a
        multidimensional array whose shape matches the mesh shape. If spatial
        distributions are set on any of the source objects, they will be ignored
        during source site sampling.
    constraints : dict
        Constraints on sampled source particles. Valid keys include 'domains',
        'time_bounds', 'energy_bounds', 'fissionable', and 'rejection_strategy'.
        For 'domains', the corresponding value is an iterable of
        :class:`openmc.Cell`, :class:`openmc.Material`, or
        :class:`openmc.Universe` for which sampled sites must be within. For
        'time_bounds' and 'energy_bounds', the corresponding value is a sequence
        of floats giving the lower and upper bounds on time in [s] or energy in
        [eV] that the sampled particle must be within. For 'fissionable', the
        value is a bool indicating that only sites in fissionable material
        should be accepted. The 'rejection_strategy' indicates what should
        happen when a source particle is rejected: either 'resample' (pick a new
        particle) or 'kill' (accept and terminate).

    Attributes
    ----------
    mesh : openmc.MeshBase
        The mesh over which source sites will be generated.
    sources : numpy.ndarray of openmc.SourceBase
        Sources to apply to each element
    strength : float
        Strength of the source
    type : str
        Indicator of source type: 'mesh'
    constraints : dict
        Constraints on sampled source particles. Valid keys include
        'domain_type', 'domain_ids', 'time_bounds', 'energy_bounds',
        'fissionable', and 'rejection_strategy'.

    """

    def __init__(
            self,
            mesh: MeshBase,
            sources: Sequence[SourceBase],
            constraints: dict[str, Any] | None = None,
    ):
        super().__init__(strength=None, constraints=constraints)
        self.mesh = mesh
        self.sources = sources

    @property
    def type(self) -> str:
        return "mesh"

    @property
    def mesh(self) -> MeshBase:
        return self._mesh

    @property
    def strength(self) -> float:
        return sum(s.strength for s in self.sources)

    @property
    def sources(self) -> np.ndarray:
        return self._sources

    @mesh.setter
    def mesh(self, m):
        cv.check_type('source mesh', m, MeshBase)
        self._mesh = m

    @sources.setter
    def sources(self, s):
        cv.check_iterable_type('mesh sources', s, SourceBase, max_depth=3)

        s = np.asarray(s)

        if isinstance(self.mesh, StructuredMesh):
            if s.size != self.mesh.n_elements:
                raise ValueError(
                    f'The length of the source array ({s.size}) does not match '
                    f'the number of mesh elements ({self.mesh.n_elements}).')

            # If user gave a multidimensional array, flatten in the order
            # of the mesh indices
            if s.ndim > 1:
                s = s.ravel(order='F')

        elif isinstance(self.mesh, UnstructuredMesh):
            if s.ndim > 1:
                raise ValueError(
                    'Sources must be a 1-D array for unstructured mesh')

        self._sources = s
        for src in self._sources:
            if isinstance(src, IndependentSource) and src.space is not None:
                warnings.warn('Some sources on the mesh have spatial '
                              'distributions that will be ignored at runtime.')
                break

    @strength.setter
    def strength(self, val):
        if val is not None:
            cv.check_type('mesh source strength', val, Real)
            self.set_total_strength(val)

    def set_total_strength(self, strength: float):
        """Scales the element source strengths based on a desired total strength.

        Parameters
        ----------
        strength : float
            Total source strength

        """
        current_strength = self.strength if self.strength != 0.0 else 1.0

        for s in self.sources:
            s.strength *= strength / current_strength

    def normalize_source_strengths(self):
        """Update all element source strengths such that they sum to 1.0."""
        self.set_total_strength(1.0)

    def populate_xml_element(self, elem: ET.Element):
        """Add necessary source information to an XML element

        Returns
        -------
        element : lxml.etree._Element
            XML element containing source data

        """
        elem.set("mesh", str(self.mesh.id))

        # write in the order of mesh indices
        for s in self.sources:
            elem.append(s.to_xml_element())

    @classmethod
    def from_xml_element(cls, elem: ET.Element, meshes) -> openmc.MeshSource:
        """
        Generate MeshSource from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element
        meshes : dict
            A dictionary with mesh IDs as keys and openmc.MeshBase instances as
            values

        Returns
        -------
        openmc.MeshSource
            MeshSource generated from the XML element
        """
        mesh_id = int(get_text(elem, 'mesh'))
        mesh = meshes[mesh_id]

        sources = [SourceBase.from_xml_element(
            e) for e in elem.iterchildren('source')]
        constraints = cls._get_constraints(elem)
        return cls(mesh, sources, constraints=constraints)


def Source(*args, **kwargs):
    """
    A function for backward compatibility of sources. Will be removed in the
    future. Please update to IndependentSource.
    """
    warnings.warn(
        "This class is deprecated in favor of 'IndependentSource'", FutureWarning)
    return openmc.IndependentSource(*args, **kwargs)


class CompiledSource(SourceBase):
    """A source based on a compiled shared library

    .. versionadded:: 0.14.0

    Parameters
    ----------
    library : path-like
        Path to a compiled shared library
    parameters : str
        Parameters to be provided to the compiled shared library function
    strength : float
        Strength of the source
    constraints : dict
        Constraints on sampled source particles. Valid keys include 'domains',
        'time_bounds', 'energy_bounds', 'fissionable', and 'rejection_strategy'.
        For 'domains', the corresponding value is an iterable of
        :class:`openmc.Cell`, :class:`openmc.Material`, or
        :class:`openmc.Universe` for which sampled sites must be within. For
        'time_bounds' and 'energy_bounds', the corresponding value is a sequence
        of floats giving the lower and upper bounds on time in [s] or energy in
        [eV] that the sampled particle must be within. For 'fissionable', the
        value is a bool indicating that only sites in fissionable material
        should be accepted. The 'rejection_strategy' indicates what should
        happen when a source particle is rejected: either 'resample' (pick a new
        particle) or 'kill' (accept and terminate).

    Attributes
    ----------
    library : pathlib.Path
        Path to a compiled shared library
    parameters : str
        Parameters to be provided to the compiled shared library function
    strength : float
        Strength of the source
    type : str
        Indicator of source type: 'compiled'
    constraints : dict
        Constraints on sampled source particles. Valid keys include
        'domain_type', 'domain_ids', 'time_bounds', 'energy_bounds',
        'fissionable', and 'rejection_strategy'.

    """

    def __init__(
        self,
        library: PathLike,
        parameters: str | None = None,
        strength: float = 1.0,
        constraints: dict[str, Any] | None = None
    ) -> None:
        super().__init__(strength=strength, constraints=constraints)
        self.library = library
        self._parameters = None
        if parameters is not None:
            self.parameters = parameters

    @property
    def type(self) -> str:
        return "compiled"

    @property
    def library(self) -> Path:
        return self._library

    @library.setter
    def library(self, library_name: PathLike):
        cv.check_type('library', library_name, PathLike)
        self._library = input_path(library_name)

    @property
    def parameters(self) -> str:
        return self._parameters

    @parameters.setter
    def parameters(self, parameters_path):
        cv.check_type('parameters', parameters_path, str)
        self._parameters = parameters_path

    def populate_xml_element(self, element):
        """Add necessary compiled source information to an XML element

        Returns
        -------
        element : lxml.etree._Element
            XML element containing source data

        """
        element.set("library", str(self.library))

        if self.parameters is not None:
            element.set("parameters", self.parameters)

    @classmethod
    def from_xml_element(cls, elem: ET.Element) -> openmc.CompiledSource:
        """Generate a compiled source from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element
        meshes : dict
            Dictionary with mesh IDs as keys and openmc.MeshBase instances as
            values

        Returns
        -------
        openmc.CompiledSource
            Source generated from XML element

        """
        kwargs = {'constraints': cls._get_constraints(elem)}
        kwargs['library'] = get_text(elem, 'library')

        source = cls(**kwargs)

        strength = get_text(elem, 'strength')
        if strength is not None:
            source.strength = float(strength)

        parameters = get_text(elem, 'parameters')
        if parameters is not None:
            source.parameters = parameters

        return source


class FileSource(SourceBase):
    """A source based on particles stored in a file

    .. versionadded:: 0.14.0

    Parameters
    ----------
    path : path-like
        Path to the source file from which sites should be sampled
    strength : float
        Strength of the source (default is 1.0)
    constraints : dict
        Constraints on sampled source particles. Valid keys include 'domains',
        'time_bounds', 'energy_bounds', 'fissionable', and 'rejection_strategy'.
        For 'domains', the corresponding value is an iterable of
        :class:`openmc.Cell`, :class:`openmc.Material`, or
        :class:`openmc.Universe` for which sampled sites must be within. For
        'time_bounds' and 'energy_bounds', the corresponding value is a sequence
        of floats giving the lower and upper bounds on time in [s] or energy in
        [eV] that the sampled particle must be within. For 'fissionable', the
        value is a bool indicating that only sites in fissionable material
        should be accepted. The 'rejection_strategy' indicates what should
        happen when a source particle is rejected: either 'resample' (pick a new
        particle) or 'kill' (accept and terminate).

    Attributes
    ----------
    path : Pathlike
        Source file from which sites should be sampled
    strength : float
        Strength of the source
    type : str
        Indicator of source type: 'file'
    constraints : dict
        Constraints on sampled source particles. Valid keys include
        'domain_type', 'domain_ids', 'time_bounds', 'energy_bounds',
        'fissionable', and 'rejection_strategy'.

    """

    def __init__(
        self,
        path: PathLike,
        strength: float = 1.0,
        constraints: dict[str, Any] | None = None
    ):
        super().__init__(strength=strength, constraints=constraints)
        self.path = path

    @property
    def type(self) -> str:
        return "file"

    @property
    def path(self) -> PathLike:
        return self._path

    @path.setter
    def path(self, p: PathLike):
        cv.check_type('source file', p, PathLike)
        self._path = input_path(p)

    def populate_xml_element(self, element):
        """Add necessary file source information to an XML element

        Returns
        -------
        element : lxml.etree._Element
            XML element containing source data

        """
        if self.path is not None:
            element.set("file", str(self.path))

    @classmethod
    def from_xml_element(cls, elem: ET.Element) -> openmc.FileSource:
        """Generate file source from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element
        meshes : dict
            Dictionary with mesh IDs as keys and openmc.MeshBase instances as
            values

        Returns
        -------
        openmc.FileSource
            Source generated from XML element

        """
        kwargs = {'constraints': cls._get_constraints(elem)}
        kwargs['path'] = get_text(elem, 'file')
        strength = get_text(elem, 'strength')
        if strength is not None:
            kwargs['strength'] = float(strength)

        return cls(**kwargs)

class TokamakSource(SourceBase):
    r"""A source representing neutron emission from a tokamak plasma.

    This source samples neutron positions from a tokamak plasma geometry using
    Miller-style flux surface parameterization. The user provides an emission
    profile S(r/a) as a function of normalized minor radius, along with one or
    more energy distributions, and either scalar or profile-dependent elongation
    and triangularity.

    The flux surface parameterization is:

    .. math::

        \begin{aligned}
        R(r, \alpha) &= R_0 + r \cos\left(\alpha + \delta(r) \sin\alpha\right)
                     + \Delta \left[1 - \left(\frac{r}{a}\right)^2\right] \\
        Z(r, \alpha) &= Z_\mathrm{shift} + \kappa(r) r \sin\alpha
        \end{aligned}

    where :math:`R_0` is the major radius, :math:`a` is the minor radius,
    :math:`\kappa(r)` is the plasma elongation (scalar or 1D profile),
    :math:`\delta(r)` is the plasma triangularity (scalar or 1D profile),
    :math:`\Delta` is the Shafranov shift, and :math:`Z_\mathrm{shift}` is
    the vertical shift.

    Parameters
    ----------
    major_radius : float
        Major radius R0 in [cm] (must be > 0)
    minor_radius : float
        Minor radius a in [cm] (must be > 0 and < major_radius)
    elongation : float or Sequence[float]
        Plasma elongation κ (must be > 0). Can be provided as a single scalar
        or as a 1D array of values corresponding to each ``r_over_a`` grid point.
    triangularity : float or Sequence[float]
        Plasma triangularity δ (must be in [-1, 1]). Can be provided as a single
        scalar or as a 1D array of values corresponding to each ``r_over_a`` grid point.
    shafranov_shift : float
        Shafranov shift Δ in [cm] (must be >= 0 and < a/2)
    r_over_a : numpy.ndarray or Sequence[float]
        Normalized minor radius grid points, must start at 0 and end at 1.
    emission_density : numpy.ndarray or Sequence[float]
        Emission density S(r) at each r/a point (arbitrary units, must be >= 0).
        Must have the same length as ``r_over_a`` and contain at least one positive value.
    energy : openmc.stats.Univariate or Sequence[openmc.stats.Univariate]
        Energy distribution(s). Either a single distribution used at all radii,
        or one distribution per ``r_over_a`` grid point (stochastic interpolation).
    method : {'fourier', 'bernstein'}, optional
        Poloidal angle sampling algorithm (default: 'fourier'):
        - 'fourier': Rejection-free Fourier-Bessel expansion supporting both scalar
          and 1D profile inputs for elongation and triangularity.
        - 'bernstein': 6-component Bernstein polynomial mixture sampling (requires
          scalar elongation and triangularity).
    time : openmc.stats.Univariate, optional
        Time distribution of the source. If None, particles are born at :math:`t=0`.
    phi_start : float, optional
        Starting toroidal angle in [rad] (default: 0.0)
    phi_extent : float, optional
        Toroidal angle extent in [rad] (default: 2π)
    n_alpha : int, optional
        Number of poloidal angle grid points for CDF tabulation (default: 101)
    vertical_shift : float, optional
        Vertical shift of the plasma center in [cm] (default: 0.0)
    strength : float, optional
        Strength of the source (default: 1.0)
    constraints : dict, optional
        Constraints on sampled source particles. See :class:`SourceBase` for valid keys.

    Attributes
    ----------
    major_radius : float
        Major radius R0 in [cm]
    minor_radius : float
        Minor radius a in [cm]
    elongation : float or numpy.ndarray
        Plasma elongation κ (scalar or 1D array)
    triangularity : float or numpy.ndarray
        Plasma triangularity δ (scalar or 1D array)
    shafranov_shift : float
        Shafranov shift Δ in [cm]
    r_over_a : numpy.ndarray
        Normalized minor radius grid points
    emission_density : numpy.ndarray
        Emission density S(r) at each r/a point
    energy : list of openmc.stats.Univariate
        Energy distribution(s)
    time : openmc.stats.Univariate or None
        Time distribution of the source
    phi_start : float
        Starting toroidal angle in [rad]
    phi_extent : float
        Toroidal angle extent in [rad]
    n_alpha : int
        Number of poloidal angle grid points
    vertical_shift : float
        Vertical shift of the plasma center in [cm]
    strength : float
        Strength of the source
    type : str
        Indicator of source type: 'tokamak'
    constraints : dict
        Constraints on sampled source particles
    """


    def __init__(
        self,
        major_radius: float,
        minor_radius: float,
        elongation: float | Sequence[float],
        triangularity: float | Sequence[float],
        shafranov_shift: float,
        r_over_a: Sequence[float],
        emission_density: Sequence[float],
        energy: Univariate | Sequence[Univariate],
        method: str = 'fourier',
        time: Univariate | None = None,
        phi_start: float = 0.0,
        phi_extent: float = 2.0 * np.pi,
        n_alpha: int = 101,
        vertical_shift: float = 0.0,
        strength: float = 1.0,
        constraints: dict[str, Any] | None = None
    ):
        super().__init__(strength=strength, constraints=constraints)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.r_over_a = np.asarray(r_over_a)
        self.shafranov_shift = shafranov_shift
        if method not in ('fourier', 'bernstein'):
            raise ValueError(f"Invalid method '{method}'. Valid options are 'fourier' or 'bernstein'.")
        self._method = method
        if method == 'bernstein':
            if isinstance(elongation, Iterable) or isinstance(triangularity, Iterable):
                raise ValueError("Bernstein method requires scalar elongation and triangularity.")
            self._is_profile = False
            self.elongation = float(elongation)
            self.triangularity = float(triangularity)
        else:  # 'fourier' (default for both profiles and scalars)
            self._is_profile = True
            is_elong_iter = isinstance(elongation, Iterable)
            is_triang_iter = isinstance(triangularity, Iterable)
            self.elongation = np.asarray(elongation, dtype=float) if is_elong_iter else np.full_like(self.r_over_a, float(elongation))
            self.triangularity = np.asarray(triangularity, dtype=float) if is_triang_iter else np.full_like(self.r_over_a, float(triangularity))
            if len(self.elongation) != len(self.r_over_a):
                raise ValueError(
                    f"elongation profile (length {len(self.elongation)}) must "
                    f"have the same length as r_over_a (length {len(self.r_over_a)})")
            if len(self.triangularity) != len(self.r_over_a):
                raise ValueError(
                    f"triangularity profile (length {len(self.triangularity)}) must "
                    f"have the same length as r_over_a (length {len(self.r_over_a)})")
            if is_elong_iter or is_triang_iter:
                bc = ((1, 0.0), 'not-a-knot')
                self._kappa_prime = (
                    CubicSpline(self.r_over_a, self.elongation, bc_type=bc).derivative()(self.r_over_a)
                    if is_elong_iter else np.zeros_like(self.r_over_a)
                )
                self._delta_prime = (
                    CubicSpline(self.r_over_a, self.triangularity, bc_type=bc).derivative()(self.r_over_a)
                    if is_triang_iter else np.zeros_like(self.r_over_a)
                )
            else:
                self._kappa_prime = np.zeros_like(self.r_over_a)
                self._delta_prime = np.zeros_like(self.r_over_a)
        self.emission_density = emission_density
        self.phi_start = phi_start
        self.phi_extent = phi_extent
        self.n_alpha = n_alpha
        self.vertical_shift = vertical_shift
        self.energy = energy
        self.time = time

        self._validate()

    def _validate(self):
        """Validate relationships between tokamak source parameters."""
        if self.minor_radius >= self.major_radius:
            raise ValueError(
                f"minor_radius ({self.minor_radius}) must be smaller than "
                f"major_radius ({self.major_radius})")
        if self.shafranov_shift >= 0.5 * self.minor_radius:
            raise ValueError(
                f"shafranov_shift ({self.shafranov_shift}) must be smaller "
                f"than half the minor_radius ({0.5 * self.minor_radius})")
        if len(self.emission_density) != len(self.r_over_a):
            raise ValueError(
                f"emission_density (length {len(self.emission_density)}) must "
                f"have the same length as r_over_a (length {len(self.r_over_a)})")
        if not np.any(self.emission_density > 0.0):
            raise ValueError("emission_density must contain a positive value")
        if len(self.energy) not in (1, len(self.r_over_a)):
            raise ValueError(
                f"Number of energy distributions ({len(self.energy)}) must be "
                f"either 1 or equal to the number of r_over_a grid points "
                f"({len(self.r_over_a)})")
        if self._is_profile and hasattr(self, '_kappa_prime'):
            if np.any(self.elongation + self.r_over_a * self._kappa_prime <= 0.0):
                raise ValueError(
                    "Non-positive vertical Jacobian (kappa + r * kappa' <= 0) detected; "
                    "flux surfaces overlap."
                )

    @property
    def type(self) -> str:
        return "tokamak"

    @property
    def method(self) -> str:
        return self._method

    @property
    def major_radius(self) -> float:
        return self._major_radius

    @major_radius.setter
    def major_radius(self, value: float):
        cv.check_type('major radius', value, Real)
        cv.check_greater_than('major radius', value, 0.0)
        self._major_radius = value

    @property
    def minor_radius(self) -> float:
        return self._minor_radius

    @minor_radius.setter
    def minor_radius(self, value: float):
        cv.check_type('minor radius', value, Real)
        cv.check_greater_than('minor radius', value, 0.0)
        self._minor_radius = value

    @property
    def elongation(self) -> float | np.ndarray:
        return self._elongation

    @elongation.setter
    def elongation(self, value: float | Sequence[float]):
        if isinstance(value, Iterable):
            val_arr = np.asarray(value, dtype=float)
            if np.any(val_arr <= 0.0):
                raise ValueError("elongation values must be > 0")
            self._elongation = val_arr
        else:
            cv.check_type('elongation', value, Real)
            cv.check_greater_than('elongation', value, 0.0)
            self._elongation = float(value)

    @property
    def triangularity(self) -> float | np.ndarray:
        return self._triangularity

    @triangularity.setter
    def triangularity(self, value: float | Sequence[float]):
        if isinstance(value, Iterable):
            val_arr = np.asarray(value, dtype=float)
            if np.any(val_arr < -1.0) or np.any(val_arr > 1.0):
                raise ValueError("triangularity values must be in [-1, 1]")
            self._triangularity = val_arr
        else:
            cv.check_type('triangularity', value, Real)
            cv.check_greater_than('triangularity', value, -1.0, equality=True)
            cv.check_less_than('triangularity', value, 1.0, equality=True)
            self._triangularity = float(value)

    @property
    def shafranov_shift(self) -> float:
        return self._shafranov_shift

    @shafranov_shift.setter
    def shafranov_shift(self, value: float):
        cv.check_type('Shafranov shift', value, Real)
        cv.check_greater_than('Shafranov shift', value, 0.0, equality=True)
        self._shafranov_shift = value

    @property
    def r_over_a(self) -> np.ndarray:
        return self._r_over_a

    @r_over_a.setter
    def r_over_a(self, value: Sequence[float]):
        value = np.asarray(value, dtype=float)
        if value.ndim != 1 or len(value) < 2:
            raise ValueError("r_over_a must be a 1-D array with at least 2 points")
        if value[0] != 0.0:
            raise ValueError("r_over_a must start at 0")
        if value[-1] != 1.0:
            raise ValueError("r_over_a must end at 1")
        if not np.all(np.diff(value) > 0):
            raise ValueError("r_over_a must be strictly increasing")
        self._r_over_a = value

    @property
    def emission_density(self) -> np.ndarray:
        return self._emission_density

    @emission_density.setter
    def emission_density(self, value: Sequence[float]):
        value = np.asarray(value, dtype=float)
        if value.ndim != 1:
            raise ValueError("emission_density must be a 1-D array")
        if np.any(value < 0):
            raise ValueError("emission_density values cannot be negative")
        self._emission_density = value

    @property
    def energy(self) -> list[Univariate]:
        return self._energy

    @energy.setter
    def energy(self, value: Univariate | Sequence[Univariate]):
        if isinstance(value, Univariate):
            self._energy = [value]
        else:
            cv.check_iterable_type('energy distributions', value, Univariate)
            self._energy = list(value)

    @property
    def time(self) -> Univariate | None:
        return self._time

    @time.setter
    def time(self, value: Univariate | None):
        if value is not None:
            cv.check_type('time distribution', value, Univariate)
        self._time = value

    @property
    def phi_start(self) -> float:
        return self._phi_start

    @phi_start.setter
    def phi_start(self, value: float):
        cv.check_type('phi_start', value, Real)
        self._phi_start = value

    @property
    def phi_extent(self) -> float:
        return self._phi_extent

    @phi_extent.setter
    def phi_extent(self, value: float):
        cv.check_type('phi_extent', value, Real)
        cv.check_greater_than('phi_extent', value, 0.0)
        cv.check_less_than('phi_extent', value, 2.0 * np.pi, equality=True)
        self._phi_extent = value

    @property
    def n_alpha(self) -> int:
        return self._n_alpha

    @n_alpha.setter
    def n_alpha(self, value: int):
        cv.check_type('n_alpha', value, Integral)
        cv.check_greater_than('n_alpha', value, 2)
        if value < 51:
            warnings.warn(
                "n_alpha values below 51 may introduce noticeable "
                "discretization bias in tokamak source sampling", stacklevel=2)
        self._n_alpha = value

    @property
    def vertical_shift(self) -> float:
        return self._vertical_shift

    @vertical_shift.setter
    def vertical_shift(self, value: float):
        cv.check_type('vertical shift', value, Real)
        self._vertical_shift = value

    def populate_xml_element(self, element):
        """Add necessary tokamak source information to an XML element

        Returns
        -------
        element : lxml.etree._Element
            XML element containing source data

        """
        self._validate()

        # Geometry parameters
        ET.SubElement(element, "major_radius").text = str(self.major_radius)
        ET.SubElement(element, "minor_radius").text = str(self.minor_radius)
        if self._is_profile:
            ET.SubElement(element, "elongation").text = ' '.join(str(k) for k in self.elongation)
            ET.SubElement(element, "elongation_prime").text = ' '.join(str(kp) for kp in self._kappa_prime)
            ET.SubElement(element, "triangularity").text = ' '.join(str(d) for d in self.triangularity)
            ET.SubElement(element, "triangularity_prime").text = ' '.join(str(dp) for dp in self._delta_prime)
        else:
            ET.SubElement(element, "elongation").text = str(self.elongation)
            ET.SubElement(element, "triangularity").text = str(self.triangularity)
        ET.SubElement(element, "shafranov_shift").text = str(self.shafranov_shift)

        # Toroidal angle bounds
        ET.SubElement(element, "phi_start").text = str(self.phi_start)
        ET.SubElement(element, "phi_extent").text = str(self.phi_extent)

        # Poloidal sampling resolution
        ET.SubElement(element, "n_alpha").text = str(self.n_alpha)

        # Vertical shift
        if self.vertical_shift != 0.0:
            ET.SubElement(element, "vertical_shift").text = str(self.vertical_shift)

        # Emission profile
        ET.SubElement(element, "r_over_a").text = ' '.join(str(r) for r in self.r_over_a)
        ET.SubElement(element, "emission_density").text = ' '.join(str(s) for s in self.emission_density)

        # Energy distribution(s)
        for dist in self.energy:
            element.append(dist.to_xml_element('energy'))

        # Time distribution
        if self.time is not None:
            element.append(self.time.to_xml_element('time'))

    @classmethod
    def from_xml_element(cls, elem: ET.Element) -> TokamakSource:
        """Generate tokamak source from an XML element

        Parameters
        ----------
        elem : lxml.etree._Element
            XML element

        Returns
        -------
        openmc.TokamakSource
            Source generated from XML element

        """
        # Read geometry parameters
        major_radius = float(get_text(elem, 'major_radius'))
        minor_radius = float(get_text(elem, 'minor_radius'))
        elong_txt = get_text(elem, 'elongation').split()
        elongation = [float(x) for x in elong_txt] if len(elong_txt) > 1 else float(elong_txt[0])
        triang_txt = get_text(elem, 'triangularity').split()
        triangularity = [float(x) for x in triang_txt] if len(triang_txt) > 1 else float(triang_txt[0])

        shafranov_shift = float(get_text(elem, 'shafranov_shift'))

        # Read optional parameters
        phi_start_text = get_text(elem, 'phi_start')
        phi_start = float(phi_start_text) if phi_start_text else 0.0

        phi_extent_text = get_text(elem, 'phi_extent')
        phi_extent = float(phi_extent_text) if phi_extent_text else 2.0 * np.pi

        n_alpha_text = get_text(elem, 'n_alpha')
        n_alpha = int(n_alpha_text) if n_alpha_text else 101

        vertical_shift_text = get_text(elem, 'vertical_shift')
        vertical_shift = float(vertical_shift_text) if vertical_shift_text else 0.0

        # Read emission profile
        r_over_a = np.array([float(x) for x in get_text(elem, 'r_over_a').split()])
        emission_density = np.array([float(x) for x in get_text(elem, 'emission_density').split()])

        # Read energy distributions
        energy = [Univariate.from_xml_element(e) for e in elem.findall('energy')]
        if len(energy) == 1:
            energy = energy[0]

        # Read time distribution
        time_elem = elem.find('time')
        time = Univariate.from_xml_element(time_elem) if time_elem is not None else None

        # Read constraints and strength
        constraints = cls._get_constraints(elem)
        strength_text = get_text(elem, 'strength')
        strength = float(strength_text) if strength_text else 1.0

        elong_prime_elem = elem.find('elongation_prime')
        triang_prime_elem = elem.find('triangularity_prime')
        has_prime = elong_prime_elem is not None or triang_prime_elem is not None
        is_profile = isinstance(elongation, list) or isinstance(triangularity, list)
        method = 'fourier' if (has_prime or is_profile) else 'bernstein'

        src = cls(
            major_radius=major_radius,
            minor_radius=minor_radius,
            elongation=elongation,
            triangularity=triangularity,
            shafranov_shift=shafranov_shift,
            r_over_a=r_over_a,
            emission_density=emission_density,
            energy=energy,
            method=method,
            time=time,
            phi_start=phi_start,
            phi_extent=phi_extent,
            n_alpha=n_alpha,
            vertical_shift=vertical_shift,
            strength=strength,
            constraints=constraints
        )
        if elong_prime_elem is not None and elong_prime_elem.text:
            src._kappa_prime = np.array([float(x) for x in elong_prime_elem.text.split()])
        if triang_prime_elem is not None and triang_prime_elem.text:
            src._delta_prime = np.array([float(x) for x in triang_prime_elem.text.split()])
        if has_prime and np.any(src.elongation + src.r_over_a * src._kappa_prime <= 0.0):
            raise ValueError(
                "Non-positive vertical Jacobian (kappa + r * kappa' <= 0) detected; "
                "flux surfaces overlap."
            )
        return src
    
    @classmethod
    def from_profiles(
        cls,
        r_over_a: Sequence[float],
        temperature: Sequence[float],
        density_D: Sequence[float],
        density_T: Sequence[float],
        major_radius: float,
        minor_radius: float,
        elongation: float | Sequence[float],
        triangularity: float | Sequence[float],
        shafranov_shift: float = 0.0,
        reaction: str = 'DT',
        method: str = 'fourier',
        time: Univariate | None = None,
        phi_start: float = 0.0,
        phi_extent: float = 2.0 * np.pi,
        n_alpha: int = 101,
        vertical_shift: float = 0.0,
        strength: float = 1.0,
        constraints: dict[str, Any] | None = None
    ) -> TokamakSource:

        """Create a TokamakSource from 1D plasma profiles on the normalized minor radius grid.

        Parameters
        ----------
        r_over_a : Sequence[float]
            Normalized minor radius grid points (r_tilde = r/a), from 0.0 to 1.0.
        temperature : Sequence[float]
            Ion temperature profile T_i in [eV] on the ``r_over_a`` grid.
        density_D : Sequence[float]
            Deuterium ion density profile n_D in [m^-3] on the ``r_over_a`` grid.
        density_T : Sequence[float]
            Tritium ion density profile n_T in [m^-3] on the ``r_over_a`` grid.
        major_radius : float
            Major radius R0 in [cm] (must be > 0)
        minor_radius : float
            Minor radius a in [cm] (must be > 0 and < major_radius)
        elongation : float or Sequence[float]
            Plasma elongation κ (scalar or 1D array on ``r_over_a``)
        triangularity : float or Sequence[float]
            Plasma triangularity δ (scalar or 1D array on ``r_over_a``)
        shafranov_shift : float, optional
            Shafranov shift Δ in [cm] (default: 0.0)
        reaction : {'DT', 'DD', 'TT'}, optional
            Fusion reaction type (default: 'DT')
        method : {'fourier', 'bernstein'}, optional
            Poloidal angle sampling algorithm (default: 'fourier')
        time : openmc.stats.Univariate, optional
            Time distribution of the source. If None, particles are born at t=0.
        phi_start : float, optional
            Starting toroidal angle in [rad] (default: 0.0)
        phi_extent : float, optional
            Toroidal angle extent in [rad] (default: 2π)
        n_alpha : int, optional
            Number of poloidal angle grid points (default: 101)
        vertical_shift : float, optional
            Vertical shift of the plasma center in [cm] (default: 0.0)
        strength : float, optional
            Strength of the source (default: 1.0)
        constraints : dict, optional
            Constraints on sampled source particles.

        Returns
        -------
        TokamakSource
            Initialized TokamakSource with calculated emission density and
            energy distributions.
        """
        r_arr = np.asarray(r_over_a, dtype=float)
        T_arr = np.asarray(temperature, dtype=float)
        nD_arr = np.asarray(density_D, dtype=float)
        nT_arr = np.asarray(density_T, dtype=float)

        if len(T_arr) != len(r_arr) or len(nD_arr) != len(r_arr) or len(nT_arr) != len(r_arr):
            raise ValueError(
                f"Length mismatch: temperature ({len(T_arr)}), density_D ({len(nD_arr)}), "
                f"and density_T ({len(nT_arr)}) must match r_over_a ({len(r_arr)})"
            )

        # 1. Calculate Bosch-Hale volumetric fusion emissivity S(r)
        emission_density = cls._calculate_fusion_emissivity(nD_arr, nT_arr, T_arr, reaction=reaction)

        # 2. Calculate Ballabio Doppler-broadened energy distribution per radial point
        if reaction in ('DT', 'DD'):
            energy_dist = cls._ballabio_energy_spectrum(T_arr, reaction=reaction)
        elif reaction == 'TT':
            energy_dist = cls._get_tt_energy_spectrum()
        else:
            energy_dist = cls._ballabio_energy_spectrum(T_arr, reaction='DT')

        return cls(
            major_radius=major_radius,
            minor_radius=minor_radius,
            elongation=elongation,
            triangularity=triangularity,
            shafranov_shift=shafranov_shift,
            r_over_a=r_arr,
            emission_density=emission_density,
            energy=energy_dist,
            method=method,
            time=time,
            phi_start=phi_start,
            phi_extent=phi_extent,
            n_alpha=n_alpha,
            vertical_shift=vertical_shift,
            strength=strength,
            constraints=constraints
        )

    @staticmethod
    def _bosch_hale_reactivity(T_i: float | np.ndarray, reaction: str = 'DT') -> float | np.ndarray:
        """Calculates fusion reactivity <sigma*v> in m^3/s using Bosch-Hale parameterization."""
        T_ev = np.asarray(T_i, dtype=float)
        T_kev = np.maximum(T_ev / 1000.0, 1e-6)

        PARAMS = {
            'DT': {
                'BG': 34.3827, 'mrc2': 1124656.0, 'C1': 1.17302e-9,
                'C2': 1.51361e-2, 'C3': 7.51886e-2, 'C4': 4.60643e-3,
                'C5': 1.35000e-2, 'C6': -1.06750e-4, 'C7': 1.36600e-5
            },
            'DD_n': {
                'BG': 31.3970, 'mrc2': 937814.0, 'C1': 5.43360e-12,
                'C2': 5.85778e-3, 'C3': 7.68222e-3, 'C4': 0.0,
                'C5': -2.96400e-6, 'C6': 0.0, 'C7': 0.0
            },
            'DD_p': {
                'BG': 31.3970, 'mrc2': 937814.0, 'C1': 5.65718e-12,
                'C2': 3.41267e-3, 'C3': 1.99167e-3, 'C4': 0.0,
                'C5': 1.05060e-5, 'C6': 0.0, 'C7': 0.0
            },
            'TT': {
                'BG': 38.6300, 'mrc2': 1409120.0, 'C1': 3.43470e-12,
                'C2': 6.09650e-3, 'C3': 1.07750e-2, 'C4': 0.0,
                'C5': -1.22270e-5, 'C6': 0.0, 'C7': 0.0
            }
        }

        if reaction == 'DD':
            return TokamakSource._bosch_hale_reactivity(T_i, 'DD_n')
        if reaction == 'DD_total':
            return TokamakSource._bosch_hale_reactivity(T_i, 'DD_n') + TokamakSource._bosch_hale_reactivity(T_i, 'DD_p')

        if reaction not in PARAMS:
            raise ValueError(f"Unknown reaction '{reaction}'. Valid options: 'DT', 'DD', 'DD_n', 'DD_p', 'DD_total', 'TT'")

        if np.any(T_kev > 100.0):
            warnings.warn(
                f"Ion temperature exceeds 100 keV (max={float(np.max(T_kev)):.1f} keV), "
                f"which is outside the calibrated validity range [0.2, 100] keV of the "
                f"Bosch-Hale {reaction} reactivity parameterization.",
                UserWarning,
                stacklevel=2,
            )

        p = PARAMS[reaction]
        num = T_kev * (p['C2'] + T_kev * (p['C4'] + T_kev * p['C6']))
        den = 1.0 + T_kev * (p['C3'] + T_kev * (p['C5'] + T_kev * p['C7']))
        theta = T_kev / (1.0 - num / den)

        xi = (p['BG']**2 / (4.0 * theta))**(1.0 / 3.0)
        sigmav_cm3_s = p['C1'] * theta * np.sqrt(xi / (p['mrc2'] * T_kev**3)) * np.exp(-3.0 * xi)
        return np.maximum(0.0, sigmav_cm3_s * 1e-6)

    @staticmethod
    def _calculate_fusion_emissivity(
        nD: np.ndarray,
        nT: np.ndarray,
        Ti: np.ndarray,
        reaction: str = 'DT'
    ) -> np.ndarray:
        """Calculates volumetric fusion emission density S(r) in neutrons / m^3 / s."""
        nD_arr = np.asarray(nD, dtype=float)
        nT_arr = np.asarray(nT, dtype=float)
        Ti_arr = np.asarray(Ti, dtype=float)

        if reaction == 'DT':
            return nD_arr * nT_arr * TokamakSource._bosch_hale_reactivity(Ti_arr, 'DT')
        elif reaction in ('DD', 'DD_n'):
            return 0.5 * (nD_arr**2) * TokamakSource._bosch_hale_reactivity(Ti_arr, 'DD_n')
        elif reaction == 'DD_p':
            return 0.5 * (nD_arr**2) * TokamakSource._bosch_hale_reactivity(Ti_arr, 'DD_p')
        elif reaction == 'DD_total':
            return 0.5 * (nD_arr**2) * TokamakSource._bosch_hale_reactivity(Ti_arr, 'DD_total')
        elif reaction == 'TT':
            # T(t, 2n)alpha produces 2 neutrons per reaction event:
            # S_n = 2 * (1/2 * n_T^2 * <sigma*v>_TT) = n_T^2 * <sigma*v>_TT
            return (nT_arr**2) * TokamakSource._bosch_hale_reactivity(Ti_arr, 'TT')
        else:
            raise ValueError(f"Invalid reaction '{reaction}'. Valid options: 'DT', 'DD', 'DD_n', 'DD_p', 'DD_total', 'TT'")

    @staticmethod
    def _ballabio_energy_spectrum(T_i: float | Sequence[float], reaction: str = 'DT') -> Any:
        """Generates Ballabio relativistic Gaussian distributions (openmc.stats.Normal)."""
        T_i_arr = np.asarray(T_i, dtype=float)
        T_kev = np.maximum(T_i_arr * 1e-3, 1e-6)

        if reaction == 'DD':
            E_0 = 2449734.0
            w0 = 82.542
            a1, a2, a3, a4 = 4.69515, -0.040729, 0.47, 0.81844
            b1, b2, b3, b4 = 1.7013e-3, 0.16888, 0.49, 7.9460e-4
            a5, a6 = 18.225, 2.1525
            b5, b6 = 8.4619e-3, 8.3241e-4
        elif reaction == 'DT':
            E_0 = 14028448.0
            w0 = 177.259
            a1, a2, a3, a4 = 5.30509, 2.4736e-3, 1.84, 1.3818
            b1, b2, b3, b4 = 5.1068e-4, 7.6223e-3, 1.78, 8.7691e-5
            a5, a6 = 37.771, 0.92181
            b5, b6 = 2.0199e-3, 5.9501e-5
        else:
            raise ValueError("Invalid reaction for Ballabio spectrum. Choose 'DT' or 'DD'.")

        low_mask = (T_kev <= 40.0)
        Delta_E = np.where(low_mask, a1 / (1.0 + a2 * T_kev**a3) * T_kev**(2.0/3.0) + a4 * T_kev, a5 + a6 * T_kev)
        delta_w = np.where(low_mask, b1 / (1.0 + b2 * T_kev**b3) * T_kev**(2.0/3.0) + b4 * T_kev, b5 + b6 * T_kev)

        mean_eV = E_0 + Delta_E * 1e3
        fwhm_eV = (w0 * (1.0 + delta_w) * np.sqrt(T_kev)) * 1e3
        sigma_eV = fwhm_eV / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        if T_i_arr.ndim == 0:
            return openmc.stats.Normal(float(mean_eV), float(sigma_eV))
        elif T_i_arr.ndim == 1:
            return [openmc.stats.Normal(float(m), float(s)) for m, s in zip(mean_eV, sigma_eV)]

    @staticmethod
    def _get_tt_energy_spectrum() -> openmc.stats.Tabular:
        """Returns continuous Tabular spectrum for the T(t, 2n)alpha 3-body phase-space continuum."""
        e_max = 9.443e6  # 5/6 * Q (Q = 11.332 MeV)
        tt_energies = np.linspace(1e4, e_max, 100)
        x = tt_energies / e_max
        tt_pdf = np.sqrt(np.maximum(0.0, x * (1.0 - x)))
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        tt_pdf /= trapz_fn(tt_pdf, tt_energies)
        return openmc.stats.Tabular(tt_energies, tt_pdf, interpolation='linear-linear')

    @classmethod
    def from_imas(
        cls,
        imas_input: Any,
        reaction: str = 'DT',
        n_points: int = 101,
        method: str = 'fourier',
        time: Univariate | None = None,
        phi_start: float = 0.0,
        phi_extent: float = 2.0 * np.pi,
        n_alpha: int = 101,
        vertical_shift: float | None = None,
        strength: float = 1.0,
        scale_to_cm: float | None = None,
        constraints: dict[str, Any] | None = None,
        time_index: int | None = None,
        time_s: float | None = None
    ) -> TokamakSource:
        """Create a deterministic TokamakSource directly from an IMAS equilibrium and core profiles.

        Parameters
        ----------
        imas_input : PathLike, str, omas.ODS, or dict
            Path to a native IMAS dataset directory (containing equilibrium.nc, core_profiles.nc, etc.),
            a native IMAS NetCDF-4 file with IDS groups, a legacy flat OMAS file (.nc, .h5, .hdf5),
            or an existing in-memory ODS/dict object.
        reaction : {'DT', 'DD', 'TT'}, optional
            Fusion reaction type (default: 'DT').
        n_points : int, optional
            Number of radial grid points for the uniform r_over_a grid (default: 101).
        method : {'fourier', 'bernstein'}, optional
            Poloidal angle sampling algorithm (default: 'fourier').
        time : openmc.stats.Univariate, optional
            Time distribution of the source. If None, particles are born at t=0.
        phi_start : float, optional
            Starting toroidal angle in [rad] (default: 0.0).
        phi_extent : float, optional
            Toroidal angle extent in [rad] (default: 2π).
        n_alpha : int, optional
            Number of poloidal angle grid points (default: 101).
        vertical_shift : float, optional
            Vertical shift of the plasma center in [cm]. If None (default), automatically
            extracted from the IMAS equilibrium magnetic axis vertical position.
        strength : float, optional
            Strength of the source (default: 1.0).
        scale_to_cm : float, optional
            Scaling factor to convert spatial coordinates to cm (default: 100.0 if R0 < 20 m).
        constraints : dict, optional
            Constraints on sampled source particles.
        time_index : int, optional
            Time slice index to extract from time-dependent profiles.
        time_s : float, optional
            Target time in seconds to extract from time-dependent profiles (selects closest slice).

        Returns
        -------
        TokamakSource
            Initialized TokamakSource with calculated emission density and
            energy distributions.
        """
        ods = cls._load_ods(imas_input, time_index=time_index, time_s=time_s)
        data = cls._extract_and_map_imas_profiles(
            ods,
            n_points=n_points,
            scale_to_cm=scale_to_cm,
            reaction=reaction
        )

        if reaction == 'DT' and (not np.any(data['density_D'] > 0.0) or not np.any(data['density_T'] > 0.0)):
            missing = 'Tritium' if np.any(data['density_D'] > 0.0) else 'Deuterium'
            raise ValueError(
                f"Cannot construct a 'DT' TokamakSource because {missing} density is zero "
                "in core_profiles. For a pure-fuel discharge, specify reaction='DD' or 'TT'."
            )

        resolved_vertical_shift = (
            data.get('vertical_shift', 0.0) if vertical_shift is None else float(vertical_shift)
        )

        return cls.from_profiles(
            r_over_a=data['r_over_a'],
            temperature=data['temperature'],
            density_D=data['density_D'],
            density_T=data['density_T'],
            major_radius=data['major_radius'],
            minor_radius=data['minor_radius'],
            elongation=data['elongation'],
            triangularity=data['triangularity'],
            shafranov_shift=data['shafranov_shift'],
            reaction=reaction,
            method=method,
            time=time,
            phi_start=phi_start,
            phi_extent=phi_extent,
            n_alpha=n_alpha,
            vertical_shift=resolved_vertical_shift,
            strength=strength,
            constraints=constraints
        )

    @staticmethod
    def _has_imas_data(node: Any) -> bool:
        """Check if an IMAS leaf or node contains valid, populated data."""
        if node is None:
            return False
        if hasattr(node, 'has_value') and not node.has_value:
            return False
        if hasattr(node, 'size'):
            return bool(node.size > 0)
        try:
            return len(node) > 0
        except TypeError:
            return True

    @staticmethod
    def _get_imas_field(node: Any, key: str, default: Any = None) -> Any:
        """Retrieve a field from an IMAS/OMAS mapping or object attribute."""
        if hasattr(node, 'get'):
            try:
                if key in node:
                    return node[key]
            except Exception:
                pass
            return node.get(key, default)
        return getattr(node, key, default)

    @staticmethod
    def _find_fuel_ion_densities(cp: Any, reaction: str = 'DT') -> tuple[np.ndarray, np.ndarray]:
        """Extract Deuterium and Tritium fuel ion density profiles by name, label, or composition.

        Parameters
        ----------
        cp : dict or omas.ODS
            1D core profiles Interface Data Structure (IDS).
        reaction : {'DT', 'DD', 'TT'}, optional
            Target fusion reaction type (default: 'DT').

        Returns
        -------
        tuple of numpy.ndarray
            (density_D, density_T) raw radial density arrays.

        Raises
        ------
        ValueError
            If fuel ion densities cannot be resolved from the core profiles.
        """
        def _fallback_from_electrons(msg: str, stacklevel: int) -> tuple[np.ndarray, np.ndarray]:
            ne = np.asarray(cp['electrons']['density'], dtype=float)
            warnings.warn(msg, UserWarning, stacklevel=stacklevel)
            if reaction == 'DD':
                return ne, np.zeros_like(ne)
            if reaction == 'TT':
                return np.zeros_like(ne), ne
            return 0.5 * ne, 0.5 * ne

        has_ne = 'electrons' in cp and 'density' in cp['electrons'] and len(cp['electrons']['density']) > 0
        if 'ion' not in cp or len(cp['ion']) == 0:
            if has_ne:
                return _fallback_from_electrons(
                    "No ion species found in core_profiles. Assuming 50:50 D-T plasma from electron density.",
                    stacklevel=3,
                )
            raise ValueError("No ion species or electron density found in core_profiles.")

        ions_raw = cp['ion']
        ions = (
            list(ions_raw.values()) if hasattr(ions_raw, 'values')
            else list(ions_raw) if isinstance(ions_raw, (list, tuple))
            else [ions_raw[i] for i in range(len(ions_raw))]
        )

        nD = None
        nT = None
        any_named = False
        get_f = TokamakSource._get_imas_field

        for ion in ions:
            raw_name = get_f(ion, 'name') or get_f(ion, 'label') or ''
            name = str(raw_name).strip().upper()
            raw_dens = get_f(ion, 'density')
            if raw_dens is None:
                continue
            dens = np.asarray(raw_dens, dtype=float)
            if dens.size == 0:
                continue

            # Check element composition (Z=1, A=2 for D; Z=1, A=3 for T)
            z_n = None
            a = None
            elements = get_f(ion, 'element')
            if elements is not None and len(elements) > 0:
                first_el = (
                    list(elements.values())[0] if hasattr(elements, 'values')
                    else elements[0]
                )
                z_n = get_f(first_el, 'z_n')
                a = get_f(first_el, 'a')

            if name or (elements is not None and len(elements) > 0):
                any_named = True

            is_d_by_comp = (
                z_n is not None and round(float(z_n)) == 1 and
                a is not None and round(float(a)) == 2
            )
            is_t_by_comp = (
                z_n is not None and round(float(z_n)) == 1 and
                a is not None and round(float(a)) == 3
            )

            if name in ('D', 'DEUTERIUM', 'DEUTERON', 'D+', 'D1+') or is_d_by_comp:
                nD = dens
            elif name in ('T', 'TRITIUM', 'TRITON', 'T+', 'T1+') or is_t_by_comp:
                nT = dens

        if nD is not None and nT is not None:
            return nD, nT

        # Reaction-specific resolution for single-species plasmas
        if reaction == 'DD' and nD is not None:
            return nD, np.zeros_like(nD)
        if reaction == 'TT' and nT is not None:
            return np.zeros_like(nT), nT

        # Single-species plasma resolution for DT reaction (e.g. pure D or pure T experiments)
        if reaction == 'DT':
            if nD is not None and nT is None:
                warnings.warn(
                    "Deuterium density was found in core_profiles, but Tritium density is missing. "
                    "Assuming Tritium density is zero.",
                    UserWarning,
                    stacklevel=2,
                )
                return nD, np.zeros_like(nD)
            if nT is not None and nD is None:
                warnings.warn(
                    "Tritium density was found in core_profiles, but Deuterium density is missing. "
                    "Assuming Deuterium density is zero.",
                    UserWarning,
                    stacklevel=2,
                )
                return np.zeros_like(nT), nT
        if reaction == 'DD' and nD is None and any_named:
            raise ValueError("Deuterium density was not found in core_profiles for DD reaction.")
        if reaction == 'TT' and nT is None and any_named:
            raise ValueError("Tritium density was not found in core_profiles for TT reaction.")

        # Fallback 1: Unlabelled 2-species test fixtures (backward compatibility only if NO ion is named)
        if not any_named and len(ions) >= 2 and nD is None and nT is None:
            dens0 = get_f(ions[0], 'density')
            dens1 = get_f(ions[1], 'density')
            if dens0 is not None and dens1 is not None and len(dens0) > 0 and len(dens1) > 0:
                warnings.warn(
                    "Ion species in core_profiles are unlabelled. Assuming ion[0] is Deuterium and ion[1] is Tritium.",
                    UserWarning,
                    stacklevel=2,
                )
                return np.asarray(dens0, dtype=float), np.asarray(dens1, dtype=float)

        # Fallback 2: 50:50 DT plasma assumption from electron density (only if ions were unlabelled or missing)
        if not any_named and has_ne:
            return _fallback_from_electrons(
                "No fuel ion species found in core_profiles. Assuming 50:50 D-T plasma from electron density.",
                stacklevel=2,
            )

        raise ValueError(
            f"Unable to resolve Deuterium and Tritium fuel densities from core_profiles. "
            f"Found {len(ions)} ion species, but could not identify required fuel species for '{reaction}' reaction."
        )

    @classmethod
    def _read_imas_python(
        cls,
        path: Path,
        ods: Any,
        time_index: int | None = None,
        time_s: float | None = None
    ) -> bool:
        """Read IMAS NetCDF-4 dataset using the official imas-python library.

        Parameters
        ----------
        path : Path
            Path to a directory containing IMAS NetCDF-4 files or a single IMAS .nc file.
        ods : omas.ODS
            Target OMAS Data Structure to populate.
        time_index : int, optional
            Time slice index to extract from time-dependent profiles (default: 0).
        time_s : float, optional
            Target time in seconds to extract from time-dependent profiles.

        Returns
        -------
        bool
            True if IMAS dataset was successfully read via imas-python; False otherwise.
        """
        try:
            import imas
        except ImportError:
            return False

        files = sorted(path.glob('*.nc')) if path.is_dir() else (
            [path] if path.suffix.lower() == '.nc' else []
        )
        if not files:
            return False

        target_ids = ('core_profiles', 'equilibrium', 'summary')
        loaded_ids = set()

        for f in files:
            try:
                with imas.DBEntry(str(f), 'r') as db:
                    candidate_ids = (f.stem,) if f.stem in target_ids else target_ids
                    for ids_name in candidate_ids:
                        try:
                            ids_obj = db.get(ids_name)
                        except Exception:
                            continue

                        file_time_index = time_index if time_index is not None else 0
                        if time_s is not None and time_index is None and hasattr(ids_obj, 'time'):
                            if cls._has_imas_data(ids_obj.time):
                                times = np.asarray(ids_obj.time)
                                if times.size > 0:
                                    file_time_index = int(np.argmin(np.abs(times - time_s)))

                        if ids_name == 'core_profiles':
                            if hasattr(ids_obj, 'profiles_1d'):
                                if len(ids_obj.profiles_1d) <= file_time_index:
                                    raise IndexError(
                                        f"Requested time_index {file_time_index} exceeds available profiles_1d slices "
                                        f"({len(ids_obj.profiles_1d)}) in '{f.name}'."
                                    )
                                p1d = ids_obj.profiles_1d[file_time_index]
                                if hasattr(p1d, 'grid') and hasattr(p1d.grid, 'rho_tor_norm') and cls._has_imas_data(p1d.grid.rho_tor_norm):
                                    ods['core_profiles.profiles_1d[0].grid.rho_tor_norm'] = np.asarray(p1d.grid.rho_tor_norm)
                                if hasattr(p1d, 'electrons') and hasattr(p1d.electrons, 'density') and cls._has_imas_data(p1d.electrons.density):
                                    ods['core_profiles.profiles_1d[0].electrons.density'] = np.asarray(p1d.electrons.density)
                                if hasattr(p1d, 'electrons') and hasattr(p1d.electrons, 'temperature') and cls._has_imas_data(p1d.electrons.temperature):
                                    ods['core_profiles.profiles_1d[0].electrons.temperature'] = np.asarray(p1d.electrons.temperature)
                                if hasattr(p1d, 't_i_average') and cls._has_imas_data(p1d.t_i_average):
                                    ods['core_profiles.profiles_1d[0].t_i_average'] = np.asarray(p1d.t_i_average)
                                if hasattr(p1d, 'ion') and len(p1d.ion) > 0:
                                    for i, ion in enumerate(p1d.ion):
                                        name = str(getattr(ion, 'name', getattr(ion, 'label', f'ion_{i}')))
                                        ods[f'core_profiles.profiles_1d[0].ion[{i}].name'] = name
                                        ods[f'core_profiles.profiles_1d[0].ion[{i}].label'] = name
                                        if hasattr(ion, 'density') and cls._has_imas_data(ion.density):
                                            ods[f'core_profiles.profiles_1d[0].ion[{i}].density'] = np.asarray(ion.density)
                                        if hasattr(ion, 'temperature') and cls._has_imas_data(ion.temperature):
                                            ods[f'core_profiles.profiles_1d[0].ion[{i}].temperature'] = np.asarray(ion.temperature)
                                        if hasattr(ion, 'element') and len(ion.element) > 0:
                                            el = ion.element[0]
                                            if hasattr(el, 'z_n') and cls._has_imas_data(el.z_n):
                                                ods[f'core_profiles.profiles_1d[0].ion[{i}].element[0].z_n'] = float(el.z_n)
                                            if hasattr(el, 'a') and cls._has_imas_data(el.a):
                                                ods[f'core_profiles.profiles_1d[0].ion[{i}].element[0].a'] = float(el.a)
                                if hasattr(ids_obj, 'covariance'):
                                    cov = ids_obj.covariance
                                    if hasattr(cov, 'data') and hasattr(cov, 'rows_uri') and cls._has_imas_data(cov.data):
                                        c_data = np.asarray(cov.data)
                                        if c_data.size > 0 and len(cov.rows_uri) > 0:
                                            ods['core_profiles.covariance.data'] = c_data
                                            ods['core_profiles.covariance.rows_uri'] = [str(u) for u in cov.rows_uri]
                                loaded_ids.add('core_profiles')

                        elif ids_name == 'equilibrium':
                            if hasattr(ids_obj, 'time_slice'):
                                if len(ids_obj.time_slice) <= file_time_index:
                                    raise IndexError(
                                        f"Requested time_index {file_time_index} exceeds available time_slice slices "
                                        f"({len(ids_obj.time_slice)}) in '{f.name}'."
                                    )
                                ts = ids_obj.time_slice[file_time_index]
                                p1d = getattr(ts, 'profiles_1d', None)
                                if p1d is not None:
                                    if hasattr(p1d, 'r_outboard') and cls._has_imas_data(p1d.r_outboard):
                                        ods['equilibrium.time_slice[0].profiles_1d.r_outboard'] = np.asarray(p1d.r_outboard)
                                    if hasattr(p1d, 'r_inboard') and cls._has_imas_data(p1d.r_inboard):
                                        ods['equilibrium.time_slice[0].profiles_1d.r_inboard'] = np.asarray(p1d.r_inboard)
                                    if hasattr(p1d, 'rho_tor_norm') and cls._has_imas_data(p1d.rho_tor_norm):
                                        ods['equilibrium.time_slice[0].profiles_1d.rho_tor_norm'] = np.asarray(p1d.rho_tor_norm)
                                    if hasattr(p1d, 'elongation') and cls._has_imas_data(p1d.elongation):
                                        ods['equilibrium.time_slice[0].profiles_1d.elongation'] = np.asarray(p1d.elongation)
                                    if hasattr(p1d, 'triangularity_upper') and cls._has_imas_data(p1d.triangularity_upper):
                                        ods['equilibrium.time_slice[0].profiles_1d.triangularity_upper'] = np.asarray(p1d.triangularity_upper)
                                    if hasattr(p1d, 'triangularity_lower') and cls._has_imas_data(p1d.triangularity_lower):
                                        ods['equilibrium.time_slice[0].profiles_1d.triangularity_lower'] = np.asarray(p1d.triangularity_lower)
                                    if hasattr(p1d, 'triangularity') and cls._has_imas_data(p1d.triangularity):
                                        ods['equilibrium.time_slice[0].profiles_1d.triangularity'] = np.asarray(p1d.triangularity)
                                    if hasattr(p1d, 'geometric_axis') and hasattr(p1d.geometric_axis, 'r') and cls._has_imas_data(p1d.geometric_axis.r):
                                        ods['equilibrium.time_slice[0].profiles_1d.geometric_axis.r'] = np.asarray(p1d.geometric_axis.r)
                                gq = getattr(ts, 'global_quantities', None)
                                if gq is not None and hasattr(gq, 'magnetic_axis') and hasattr(gq.magnetic_axis, 'r') and cls._has_imas_data(gq.magnetic_axis.r):
                                    ods['equilibrium.time_slice[0].global_quantities.magnetic_axis.r'] = np.asarray(gq.magnetic_axis.r)
                                if gq is not None and hasattr(gq, 'magnetic_axis') and hasattr(gq.magnetic_axis, 'z') and cls._has_imas_data(gq.magnetic_axis.z):
                                    ods['equilibrium.time_slice[0].global_quantities.magnetic_axis.z'] = np.asarray(gq.magnetic_axis.z)
                                loaded_ids.add('equilibrium')

                        elif ids_name == 'summary':
                            loaded_ids.add('summary')

            except IndexError:
                raise
            except Exception as e:
                warnings.warn(f"Failed to read native IMAS data via imas-python from '{f}': {e}", UserWarning, stacklevel=2)
                continue

        return 'core_profiles' in loaded_ids and 'equilibrium' in loaded_ids

    @classmethod
    def _read_imas_hdf5(
        cls,
        path: Path,
        ods: Any,
        time_index: int | None = None,
        time_s: float | None = None
    ) -> bool:
        """Parse native IMAS Access Layer multi-HDF5 dataset files into an OMAS Data Structure (ODS).

        Parameters
        ----------
        path : Path
            Path to a directory containing native IMAS HDF5 files (*.h5) or to a single IMAS HDF5 file.
        ods : omas.ODS
            Target OMAS Data Structure to populate in-place.
        time_index : int, optional
            Time slice index to extract from time-dependent profiles.
        time_s : float, optional
            Target time in seconds to extract from time-dependent profiles (selects closest slice).

        Returns
        -------
        bool
            True if native IMAS HDF5 datasets were detected and successfully ingested;
            False otherwise.
        """
        files = sorted(path.glob('*.h5')) if path.is_dir() else (
            [path] if path.suffix.lower() in ('.h5', '.hdf5') else []
        )
        if not files:
            return False

        target_ids = ('core_profiles', 'equilibrium', 'summary')
        found_target = False
        for f in files:
            if any(f.name.startswith(t) for t in target_ids):
                found_target = True
                break
        if not found_target:
            return False

        # Resolve time index independently per target IDS if time_s is provided
        ids_time_indices: dict[str, int] = {}
        ids_actual_times: dict[str, float] = {}
        if time_s is not None and time_index is None:
            for f in files:
                try:
                    with h5py.File(str(f), 'r') as h5f:
                        for target_id in target_ids:
                            if target_id in ids_time_indices:
                                continue
                            for time_k in (f'{target_id}/time', f'{target_id}/time_slice/time', 'time'):
                                if time_k == 'time' and len(files) > 1 and not (f.stem.startswith(target_id) or f.name.startswith(target_id)):
                                    continue
                                if time_k in h5f and isinstance(h5f[time_k], h5py.Dataset):
                                    times = np.asarray(h5f[time_k][:])
                                    if times.size > 0:
                                        t_idx = int(np.argmin(np.abs(times - time_s)))
                                        ids_time_indices[target_id] = t_idx
                                        ids_actual_times[target_id] = float(times[t_idx])
                                        break
                except Exception:
                    continue

        loaded_ids = set()
        for f in files:
            try:
                with h5py.File(str(f), 'r') as h5f:
                    keys = []
                    h5f.visit(keys.append)
                    for k in keys:
                        if not isinstance(h5f[k], h5py.Dataset):
                            continue
                        if k.endswith('_SHAPE') or k.endswith('&AOS_SHAPE'):
                            continue

                        parts = k.split('/', 1)
                        if len(parts) == 2:
                            ids_name, subpath = parts[0], parts[1]
                        else:
                            ids_name = f.stem
                            subpath = parts[0]

                        if ids_name not in target_ids:
                            continue

                        resolved_time_index = (
                            time_index if time_index is not None
                            else ids_time_indices.get(ids_name, 0)
                        )
                        actual_time = ids_actual_times.get(ids_name, None)

                        # Check convergence flag
                        if ids_name == 'equilibrium' and subpath == 'code&output_flag':
                            flag_val = h5f[k][resolved_time_index] if h5f[k].ndim > 0 else h5f[k][()]
                            if int(flag_val) < 0:
                                time_str = f" at t={actual_time:.3f} s" if actual_time is not None else ""
                                warnings.warn(
                                    f"Equilibrium time slice {resolved_time_index}{time_str} has output_flag={flag_val} (< 0), "
                                    "indicating reconstruction may not have converged.",
                                    UserWarning,
                                    stacklevel=2,
                                )

                        val = h5f[k][()] if h5f[k].shape == () else h5f[k][:]
                        if getattr(val, 'dtype', None) is not None and val.dtype.kind in ('O', 'S', 'U'):
                            continue

                        # Handle time slicing along dimension 0
                        tokens = subpath.split('&')
                        if 'time_slice[]' in tokens or 'profiles_1d[]' in tokens:
                            if hasattr(val, 'ndim') and val.ndim >= 1 and val.shape[0] > resolved_time_index:
                                val = val[resolved_time_index]
                            elif hasattr(val, 'ndim') and val.ndim >= 1 and val.shape[0] <= resolved_time_index:
                                raise IndexError(
                                    f"Requested time_index {resolved_time_index} exceeds available slices "
                                    f"({val.shape[0]}) for '{k}' in '{f.name}'."
                                )
                        elif subpath in ('time', 'vacuum_toroidal_field&b0'):
                            if hasattr(val, 'ndim') and val.ndim >= 1 and val.shape[0] > resolved_time_index:
                                val = val[resolved_time_index]

                        ods_subpath = subpath.replace('[]', '.0').replace('&', '.')
                        ods_path = f'{ids_name}.{ods_subpath}'
                        try:
                            ods[ods_path] = val
                            loaded_ids.add(ids_name)
                        except Exception:
                            pass
            except IndexError:
                raise
            except Exception as e:
                warnings.warn(f"Failed to read native IMAS HDF5 data from '{f}': {e}", UserWarning, stacklevel=2)
                continue

        return 'core_profiles' in loaded_ids and 'equilibrium' in loaded_ids

    @classmethod
    def _read_imas_netcdf(
        cls,
        path: Path,
        ods: Any,
        time_index: int | None = None,
        time_s: float | None = None
    ) -> bool:
        """Parse native IMAS NetCDF-4 dataset files into an OMAS Data Structure (ODS).

        IMAS netCDF files (as written by IMAS-Python or the IMAS NetCDF backend) store
        each IDS occurrence in a group hierarchy:
            root -> <ids_name> -> <occurrence> (e.g. '0') -> <dot_separated_variables>

        This fallback reader directly traverses NetCDF-4 groups without requiring the
        official ITER Access Layer / imas-python library.

        Parameters
        ----------
        path : Path
            Path to a directory containing native IMAS files (*.nc) or to a single
            native IMAS NetCDF-4 file.
        ods : omas.ODS
            Target OMAS Data Structure to populate in-place.
        time_index : int, optional
            Time slice index to extract from time-dependent profiles (default: 0).
        time_s : float, optional
            Target time in seconds to extract from time-dependent profiles (selects closest slice).

        Returns
        -------
        bool
            True if native IMAS NetCDF-4 groups were detected and successfully ingested;
            False otherwise (e.g., if files are flat legacy OMAS).
        """
        try:
            import netCDF4
        except ImportError:
            return False

        files = sorted(path.glob('*.nc')) if path.is_dir() else (
            [path] if path.suffix.lower() == '.nc' else []
        )
        if not files:
            return False

        target_ids = ('core_profiles', 'equilibrium', 'summary')
        loaded_ids = set()

        # Resolve time_index independently per target IDS from time_s if requested
        ids_time_indices: dict[str, int] = {}
        if time_s is not None and time_index is None:
            for f in files:
                try:
                    with netCDF4.Dataset(str(f), 'r') as ds:
                        for ids_name, grp in ds.groups.items():
                            if ids_name in target_ids and ids_name not in ids_time_indices:
                                occ_groups = grp.groups.values() if grp.groups else [grp]
                                for occ_grp in occ_groups:
                                    for v_name, var in occ_grp.variables.items():
                                        if v_name in ('time', 'profiles_1d.time', 'equilibrium.time', f'{ids_name}.time'):
                                            times = np.asarray(var[:])
                                            if times.size > 0:
                                                ids_time_indices[ids_name] = int(np.argmin(np.abs(times - time_s)))
                                                break
                except Exception:
                    pass

        for f in files:
            try:
                with netCDF4.Dataset(str(f), 'r') as ds:
                    if not ds.groups:
                        continue  # Flat NetCDF file (legacy OMAS), skip native group reader

                    for ids_name, grp in ds.groups.items():
                        if ids_name not in target_ids:
                            continue
                        resolved_time_index = (
                            time_index if time_index is not None
                            else ids_time_indices.get(ids_name, 0)
                        )
                        occ_groups = grp.groups.values() if grp.groups else [grp]
                        for occ_grp in occ_groups:
                            for var_name, var in occ_grp.variables.items():
                                if '.' not in var_name and var.shape == ():
                                    continue
                                try:
                                    val = var[:]
                                except Exception:
                                    continue
                                if np.ma.is_masked(val):
                                    if val.mask.all():
                                        continue
                                    val = np.ma.filled(val, np.nan)

                                arr = np.asarray(val)
                                if arr.size == 0:
                                    continue

                                # Slicing along the dimension explicitly named 'time'
                                time_dim_idx = None
                                for d_idx, d_name in enumerate(var.dimensions):
                                    if d_name == 'time' or d_name.endswith('.time'):
                                        time_dim_idx = d_idx
                                        break

                                if time_dim_idx is not None and arr.ndim > time_dim_idx:
                                    if arr.shape[time_dim_idx] > resolved_time_index:
                                        slc = [slice(None)] * arr.ndim
                                        slc[time_dim_idx] = resolved_time_index
                                        arr = arr[tuple(slc)]
                                    else:
                                        raise IndexError(
                                            f"Requested time_index {resolved_time_index} exceeds available "
                                            f"time slices ({arr.shape[time_dim_idx]}) for "
                                            f"variable '{var_name}' in '{f.name}'."
                                        )

                                parts = var_name.split('.')

                                # Handle multi-species ion AoS: profiles_1d.ion.<property>
                                if parts[0] == 'profiles_1d' and len(parts) >= 3 and parts[1] == 'ion':
                                    prop = '.'.join(parts[2:])
                                    if arr.ndim >= 2:
                                        for ion_idx in range(arr.shape[0]):
                                            try:
                                                ods[f'{ids_name}.profiles_1d.0.ion.{ion_idx}.{prop}'] = arr[ion_idx]
                                                loaded_ids.add(ids_name)
                                            except (KeyError, ValueError, TypeError):
                                                pass
                                    elif arr.ndim == 1:
                                        for ion_idx in range(len(arr)):
                                            try:
                                                ods[f'{ids_name}.profiles_1d.0.ion.{ion_idx}.{prop}'] = arr[ion_idx]
                                                loaded_ids.add(ids_name)
                                            except (KeyError, ValueError, TypeError):
                                                pass
                                    continue

                                # Map time-sliced profiles into occurrence index 0
                                if parts[0] in ('time_slice', 'profiles_1d'):
                                    mapped_path = f'{ids_name}.{parts[0]}.0.' + '.'.join(parts[1:])
                                else:
                                    mapped_path = f'{ids_name}.{var_name}'

                                try:
                                    ods[mapped_path] = arr
                                    loaded_ids.add(ids_name)
                                except (KeyError, ValueError, TypeError):
                                    pass
            except IndexError:
                raise
            except Exception as e:
                warnings.warn(f"Failed to read native IMAS NetCDF data from '{f}': {e}", UserWarning, stacklevel=2)
                continue

        return 'core_profiles' in loaded_ids and 'equilibrium' in loaded_ids

    @classmethod
    def _load_ods(
        cls,
        imas_input: PathLike | str | Any,
        time_index: int | None = None,
        time_s: float | None = None
    ) -> Any:
        """Loads an IMAS dataset from path or returns existing in-memory ODS instance.

        This loader prioritizes official imas-python reading for native IMAS NetCDF-4 files,
        supports standalone native IMAS Access Layer multi-HDF5 files, falls back to
        standalone netCDF4 group parsing if imas-python is unavailable, and accommodates
        legacy flat OMAS formats (.nc, .h5, .hdf5) and in-memory ODS/dict.

        Parameters
        ----------
        imas_input : PathLike, str, omas.ODS, or dict
            Path to an IMAS directory or file, flat OMAS file (.nc, .h5, .hdf5),
            or an existing in-memory ODS/dict object.
        time_index : int, optional
            Time slice index to extract from time-dependent profiles (default: 0).
        time_s : float, optional
            Target time in seconds to extract from time-dependent profiles (selects closest slice).

        Returns
        -------
        omas.ODS or dict
            Populated data structure containing equilibrium and core_profiles IDSs.

        Raises
        ------
        ImportError
            If required parsing libraries are not installed.
        FileNotFoundError
            If the specified file or directory does not exist or contains no valid datasets.
        ValueError
            If the format or file extension is unsupported.
        """
        try:
            from omas import ODS, load_omas_h5, load_omas_nc
        except ImportError:
            raise ImportError(
                "The 'omas' package is required for IMAS/OMAS data extraction. "
                "Install it via: pip install omas"
            )

        if time_index is not None and time_s is not None:
            warnings.warn(
                "Both 'time_index' and 'time_s' were provided; 'time_index' takes precedence.",
                UserWarning,
                stacklevel=2,
            )

        # 0. Pass-through in-memory representations
        if isinstance(imas_input, (ODS, dict)):
            if time_index is not None or time_s is not None:
                warnings.warn(
                    "'time_index' and 'time_s' parameters are ignored when imas_input is an in-memory ODS or dict.",
                    UserWarning,
                    stacklevel=2,
                )
            return imas_input

        path = Path(imas_input)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: '{path}'")

        # 1. Primary: Official imas-python reader
        ods_py = ODS(consistency_check=False)
        if cls._read_imas_python(path, ods_py, time_index=time_index, time_s=time_s):
            return ods_py

        # 2. Native IMAS Access Layer multi-HDF5 reader
        ods_h5 = ODS(consistency_check=False)
        if cls._read_imas_hdf5(path, ods_h5, time_index=time_index, time_s=time_s):
            return ods_h5

        # 3. Portable Fallback: Standalone NetCDF-4 group reader (no imas-python required)
        ods_nc = ODS(consistency_check=False)
        if cls._read_imas_netcdf(path, ods_nc, time_index=time_index, time_s=time_s):
            return ods_nc

        # 4. Legacy Fallback: Single flat OMAS files (.nc, .h5, .hdf5)
        if path.is_file():
            ext = path.suffix.lower()
            if ext == '.nc':
                return load_omas_nc(str(path))
            elif ext in ('.h5', '.hdf5'):
                return load_omas_h5(str(path))
            else:
                raise ValueError(
                    f"Unsupported file extension '{ext}' for IMAS input. "
                    "Expected directory or '.nc', '.h5', or '.hdf5' file."
                )

        # If it is a directory containing .nc files and netCDF4 was missing
        if path.is_dir() and any(path.glob('*.nc')):
            try:
                import netCDF4
            except ImportError:
                raise ImportError(
                    f"Directory '{path}' appears to contain native IMAS files, but reading "
                    "them requires either the 'imas' package or 'netCDF4'. "
                    "Install netCDF4 via: pip install netCDF4"
                )

        raise FileNotFoundError(f"No valid IMAS or OMAS datasets found in path: '{path}'")

    @staticmethod
    def _resolve_length_scale_to_cm(major_radius: float, scale_to_cm: float | None = None) -> float:
        """Return multiplicative factor to convert length units to cm (IMAS uses m, OpenMC uses cm)."""
        if scale_to_cm is not None:
            return float(scale_to_cm)
        return 100.0 if major_radius < 20.0 else 1.0

    @classmethod
    def _extract_and_map_imas_profiles(
        cls,
        ods: Any,
        n_points: int = 101,
        scale_to_cm: float | None = None,
        reaction: str = 'DT'
    ) -> dict[str, Any]:
        """Extracts 1D profiles and geometry from IMAS and maps to a uniform r_over_a grid."""
        cp = ods['core_profiles']['profiles_1d'][0]
        eq = ods['equilibrium']['time_slice'][0]['profiles_1d']
        ts = ods['equilibrium']['time_slice'][0]

        def _fallback_eq_grid() -> np.ndarray:
            if 'rho_tor_norm' in eq and cls._has_imas_data(eq['rho_tor_norm']):
                return np.asarray(eq['rho_tor_norm'], dtype=float)
            if 'psi_norm' in eq and cls._has_imas_data(eq['psi_norm']):
                return np.sqrt(np.clip(np.asarray(eq['psi_norm'], dtype=float), 0.0, 1.0))
            n_eq_geom = (
                len(eq['elongation']) if ('elongation' in eq and cls._has_imas_data(eq['elongation']))
                else len(eq['triangularity']) if ('triangularity' in eq and cls._has_imas_data(eq['triangularity']))
                else n_points
            )
            return np.linspace(0.0, 1.0, n_eq_geom)

        # 1. 1D boundary radii and geometric mapping r_tilde(rho)
        r_in = None
        r_out = None
        if (
            'r_inboard' in eq and 'r_outboard' in eq
            and cls._has_imas_data(eq['r_inboard']) and cls._has_imas_data(eq['r_outboard'])
        ):
            r_in = np.asarray(eq['r_inboard'], dtype=float)
            r_out = np.asarray(eq['r_outboard'], dtype=float)
            r_geom = 0.5 * (r_out - r_in)
            a_minor = float(r_geom[-1])
            if a_minor <= 0.0:
                raise ValueError(
                    f"Invalid minor radius calculated from equilibrium boundary: a = {a_minor} <= 0."
                )
            r_tilde_eq = r_geom / a_minor
            R0 = float(0.5 * (r_out[-1] + r_in[-1]))
        elif (
            'profiles_2d' in ts and len(ts['profiles_2d']) > 0
            and 'grid' in ts['profiles_2d'][0] and 'psi' in ts['profiles_2d'][0]
            and 'dim1' in ts['profiles_2d'][0]['grid'] and 'dim2' in ts['profiles_2d'][0]['grid']
            and cls._has_imas_data(ts['profiles_2d'][0]['grid']['dim1'])
            and cls._has_imas_data(ts['profiles_2d'][0]['grid']['dim2'])
            and cls._has_imas_data(ts['profiles_2d'][0]['psi'])
            and 'global_quantities' in ts and 'magnetic_axis' in ts['global_quantities']
            and 'r' in ts['global_quantities']['magnetic_axis']
            and 'z' in ts['global_quantities']['magnetic_axis']
            and cls._has_imas_data(ts['global_quantities']['magnetic_axis']['r'])
            and cls._has_imas_data(ts['global_quantities']['magnetic_axis']['z'])
            and 'psi_magnetic_axis' in ts['global_quantities']
            and cls._has_imas_data(ts['global_quantities']['psi_magnetic_axis'])
            and 'boundary' in ts and 'psi' in ts['boundary']
            and cls._has_imas_data(ts['boundary']['psi'])
        ):
            r_grid = np.squeeze(np.asarray(ts['profiles_2d'][0]['grid']['dim1'], dtype=float))
            z_grid = np.squeeze(np.asarray(ts['profiles_2d'][0]['grid']['dim2'], dtype=float))
            psi_2d = np.squeeze(np.asarray(ts['profiles_2d'][0]['psi'], dtype=float))

            mag_r = float(np.asarray(ts['global_quantities']['magnetic_axis']['r']).flat[0])
            mag_z = float(np.asarray(ts['global_quantities']['magnetic_axis']['z']).flat[0])
            psi_axis = float(np.asarray(ts['global_quantities']['psi_magnetic_axis']).flat[0])
            psi_bnd = float(np.asarray(ts['boundary']['psi']).flat[0])

            # Determine grid orientation (dim1=R, dim2=Z vs dim1=Z, dim2=R)
            if psi_2d.shape == (len(r_grid), len(z_grid)) and len(r_grid) != len(z_grid):
                interp = RegularGridInterpolator((r_grid, z_grid), psi_2d, method='linear', bounds_error=False, fill_value=None)
                fine_r = np.linspace(r_grid[0], r_grid[-1], 1000)
                pts = np.column_stack([fine_r, np.full_like(fine_r, mag_z)])
            elif psi_2d.shape == (len(z_grid), len(r_grid)) and len(r_grid) != len(z_grid):
                interp = RegularGridInterpolator((z_grid, r_grid), psi_2d, method='linear', bounds_error=False, fill_value=None)
                fine_r = np.linspace(r_grid[0], r_grid[-1], 1000)
                pts = np.column_stack([np.full_like(fine_r, mag_z), fine_r])
            elif psi_2d.shape == (len(r_grid), len(z_grid)) and len(r_grid) == len(z_grid):
                # Square grid: determine whether orientation is (R, Z) or (Z, R)
                # by evaluating psi at off-axis points (boundary outline or midplane points)
                interp_rz = RegularGridInterpolator((r_grid, z_grid), psi_2d, method='linear', bounds_error=False, fill_value=np.nan)
                interp_zr = RegularGridInterpolator((z_grid, r_grid), psi_2d, method='linear', bounds_error=False, fill_value=np.nan)

                test_pts_rz = []
                if (
                    'boundary' in ts and 'outline' in ts['boundary']
                    and 'r' in ts['boundary']['outline'] and 'z' in ts['boundary']['outline']
                    and cls._has_imas_data(ts['boundary']['outline']['r'])
                    and cls._has_imas_data(ts['boundary']['outline']['z'])
                ):
                    r_outl = np.asarray(ts['boundary']['outline']['r'], dtype=float)
                    z_outl = np.asarray(ts['boundary']['outline']['z'], dtype=float)
                    step = max(1, len(r_outl) // 8)
                    for ro, zo in zip(r_outl[::step], z_outl[::step]):
                        test_pts_rz.append((float(ro), float(zo), 1.0))

                a_cand = (
                    float(np.asarray(ts['boundary']['minor_radius']).flat[0])
                    if ('boundary' in ts and 'minor_radius' in ts['boundary'] and cls._has_imas_data(ts['boundary']['minor_radius']))
                    else 0.25 * (r_grid[-1] - r_grid[0])
                )
                r_mid = (
                    float(np.asarray(ts['boundary']['geometric_axis']['r']).flat[0])
                    if ('boundary' in ts and 'geometric_axis' in ts['boundary'] and 'r' in ts['boundary']['geometric_axis'] and cls._has_imas_data(ts['boundary']['geometric_axis']['r']))
                    else mag_r
                )
                test_pts_rz.append((r_mid + a_cand, mag_z, 1.0))
                test_pts_rz.append((r_mid - a_cand, mag_z, 1.0))
                test_pts_rz.append((r_mid + 0.5 * a_cand, mag_z, 0.25))
                test_pts_rz.append((r_mid - 0.5 * a_cand, mag_z, 0.25))

                err_rz_list = []
                err_zr_list = []
                denom = (psi_bnd - psi_axis) if psi_bnd != psi_axis else 1.0
                for r_pt, z_pt, target_psin in test_pts_rz:
                    v_rz = float(interp_rz([[r_pt, z_pt]])[0])
                    v_zr = float(interp_zr([[z_pt, r_pt]])[0])
                    if not np.isnan(v_rz):
                        psin_rz = (v_rz - psi_axis) / denom if psi_bnd != psi_axis else v_rz
                        err_rz_list.append(abs(psin_rz - target_psin))
                    else:
                        err_rz_list.append(100.0)

                    if not np.isnan(v_zr):
                        psin_zr = (v_zr - psi_axis) / denom if psi_bnd != psi_axis else v_zr
                        err_zr_list.append(abs(psin_zr - target_psin))
                    else:
                        err_zr_list.append(100.0)

                fine_r = np.linspace(r_grid[0], r_grid[-1], 1000)
                if np.mean(err_rz_list) <= np.mean(err_zr_list):
                    interp = interp_rz
                    pts = np.column_stack([fine_r, np.full_like(fine_r, mag_z)])
                else:
                    interp = interp_zr
                    pts = np.column_stack([np.full_like(fine_r, mag_z), fine_r])
            else:
                raise ValueError(
                    f"2D equilibrium grid shape mismatch: psi_2d {psi_2d.shape} does not match "
                    f"grid dimensions (dim1={len(r_grid)}, dim2={len(z_grid)})."
                )

            psi_mid_n = (interp(pts) - psi_axis) / (psi_bnd - psi_axis)

            # Extract monotonic branches radiating outward from magnetic axis mag_r
            outboard_mask = (fine_r >= mag_r)
            r_out_cand = fine_r[outboard_mask]
            psi_out_cand = psi_mid_n[outboard_mask]

            inboard_mask = (fine_r <= mag_r)
            r_in_cand = fine_r[inboard_mask][::-1]  # starts at mag_r, decreasing R
            psi_in_cand = psi_mid_n[inboard_mask][::-1]

            # Build strictly increasing monotonic branch from axis for outboard
            valid_out = [0]
            for i in range(1, len(r_out_cand)):
                if psi_out_cand[i] > psi_out_cand[valid_out[-1]]:
                    valid_out.append(i)
                    if psi_out_cand[i] >= 1.05:
                        break
                elif psi_out_cand[valid_out[-1]] >= 1.0:
                    break
            r_out_clean = r_out_cand[valid_out]
            psi_out_clean = psi_out_cand[valid_out]
            psi_out_clean[0] = 0.0
            r_out_clean[0] = mag_r

            # Build strictly increasing monotonic branch from axis for inboard
            valid_in = [0]
            for i in range(1, len(r_in_cand)):
                if psi_in_cand[i] > psi_in_cand[valid_in[-1]]:
                    valid_in.append(i)
                    if psi_in_cand[i] >= 1.05:
                        break
                elif psi_in_cand[valid_in[-1]] >= 1.0:
                    break
            r_in_clean = r_in_cand[valid_in]
            psi_in_clean = psi_in_cand[valid_in]
            psi_in_clean[0] = 0.0
            r_in_clean[0] = mag_r

            # Ensure branches cover up to normalized flux 1.0
            if psi_out_clean[-1] < 1.0 and len(psi_out_clean) >= 2:
                slope = (r_out_clean[-1] - r_out_clean[-2]) / (psi_out_clean[-1] - psi_out_clean[-2])
                r_out_clean = np.append(r_out_clean, r_out_clean[-1] + slope * (1.0 - psi_out_clean[-1]))
                psi_out_clean = np.append(psi_out_clean, 1.0)
            if psi_in_clean[-1] < 1.0 and len(psi_in_clean) >= 2:
                slope = (r_in_clean[-1] - r_in_clean[-2]) / (psi_in_clean[-1] - psi_in_clean[-2])
                r_in_clean = np.append(r_in_clean, r_in_clean[-1] + slope * (1.0 - psi_in_clean[-1]))
                psi_in_clean = np.append(psi_in_clean, 1.0)

            if 'psi_norm' in eq and cls._has_imas_data(eq['psi_norm']):
                psi_n_1d = np.asarray(eq['psi_norm'], dtype=float)
            elif 'psi' in eq and cls._has_imas_data(eq['psi']):
                psi_n_1d = (np.asarray(eq['psi'], dtype=float) - psi_axis) / (psi_bnd - psi_axis)
            elif 'rho_tor_norm' in eq and cls._has_imas_data(eq['rho_tor_norm']):
                psi_n_1d = (np.asarray(eq['rho_tor_norm'], dtype=float)) ** 2
            else:
                psi_n_1d = np.linspace(0.0, 1.0, n_points)

            psi_n_1d = np.clip(psi_n_1d, 0.0, 1.0)
            r_in = np.interp(psi_n_1d, psi_in_clean, r_in_clean)
            r_out = np.interp(psi_n_1d, psi_out_clean, r_out_clean)
            r_geom = 0.5 * (r_out - r_in)
            r_geom[0] = 0.0

            if (
                'boundary' in ts and 'minor_radius' in ts['boundary']
                and cls._has_imas_data(ts['boundary']['minor_radius'])
            ):
                a_minor = float(np.asarray(ts['boundary']['minor_radius']).flat[0])
            else:
                a_minor = float(r_geom[-1])

            if a_minor <= 0.0:
                raise ValueError(
                    f"Invalid minor radius calculated from 2D equilibrium: a = {a_minor} <= 0."
                )

            r_tilde_eq = np.clip(r_geom / a_minor, 0.0, 1.0)
            r_tilde_eq[0] = 0.0
            r_tilde_eq[-1] = 1.0
            for i in range(1, len(r_tilde_eq)):
                if r_tilde_eq[i] <= r_tilde_eq[i - 1]:
                    r_tilde_eq[i] = r_tilde_eq[i - 1] + 1e-7
            r_tilde_eq[-1] = max(r_tilde_eq[-1], 1.0)

            if (
                'boundary' in ts and 'geometric_axis' in ts['boundary']
                and 'r' in ts['boundary']['geometric_axis']
                and cls._has_imas_data(ts['boundary']['geometric_axis']['r'])
            ):
                R0 = float(np.asarray(ts['boundary']['geometric_axis']['r']).flat[0])
            elif (
                'boundary' in ts and 'outline' in ts['boundary']
                and 'r' in ts['boundary']['outline']
                and cls._has_imas_data(ts['boundary']['outline']['r'])
            ):
                outline_r = np.asarray(ts['boundary']['outline']['r'], dtype=float)
                R0 = float(0.5 * (outline_r.max() + outline_r.min()))
            else:
                R0 = float(0.5 * (r_out[-1] + r_in[-1]))
        elif (
            'boundary' in ts and 'minor_radius' in ts['boundary']
            and cls._has_imas_data(ts['boundary']['minor_radius'])
        ):
            a_minor = float(np.asarray(ts['boundary']['minor_radius']).flat[0])
            if a_minor <= 0.0:
                raise ValueError(
                    f"Invalid minor radius found in equilibrium boundary: a = {a_minor} <= 0."
                )

            # Resolve major radius R0 from geometric axis, outline, or magnetic axis
            if (
                'geometric_axis' in ts['boundary'] and 'r' in ts['boundary']['geometric_axis']
                and cls._has_imas_data(ts['boundary']['geometric_axis']['r'])
            ):
                R0 = float(np.asarray(ts['boundary']['geometric_axis']['r']).flat[0])
            elif (
                'outline' in ts['boundary'] and 'r' in ts['boundary']['outline']
                and cls._has_imas_data(ts['boundary']['outline']['r'])
            ):
                outline_r = np.asarray(ts['boundary']['outline']['r'], dtype=float)
                R0 = float(0.5 * (outline_r.max() + outline_r.min()))
            elif (
                'global_quantities' in ts and 'magnetic_axis' in ts['global_quantities']
                and 'r' in ts['global_quantities']['magnetic_axis']
                and cls._has_imas_data(ts['global_quantities']['magnetic_axis']['r'])
            ):
                R0 = float(np.asarray(ts['global_quantities']['magnetic_axis']['r']).flat[0])
            else:
                raise ValueError(
                    "Unable to resolve major radius R0 from equilibrium boundary data."
                )

            warnings.warn(
                f"No 1D 'r_inboard' and 'r_outboard' profiles found in equilibrium. "
                f"Using boundary scalar dimensions: major_radius R0={R0:.3f} m, minor_radius a={a_minor:.3f} m.",
                UserWarning,
                stacklevel=2,
            )
            r_tilde_eq = _fallback_eq_grid()
        elif (
            'boundary' in ts and 'outline' in ts['boundary'] and 'r' in ts['boundary']['outline']
            and cls._has_imas_data(ts['boundary']['outline']['r'])
        ):
            outline_r = np.asarray(ts['boundary']['outline']['r'], dtype=float)
            R0 = float(0.5 * (outline_r.max() + outline_r.min()))
            a_minor = float(0.5 * (outline_r.max() - outline_r.min()))
            if a_minor <= 0.0:
                raise ValueError(
                    f"Invalid minor radius calculated from boundary outline: a = {a_minor} <= 0."
                )
            warnings.warn(
                f"No 1D 'r_inboard'/'r_outboard' or 'boundary.minor_radius' found in equilibrium. "
                f"Derived dimensions from boundary outline: major_radius R0={R0:.3f} m, minor_radius a={a_minor:.3f} m.",
                UserWarning,
                stacklevel=2,
            )
            r_tilde_eq = _fallback_eq_grid()
        else:
            raise ValueError(
                "Unable to determine plasma dimensions (major_radius R0 and minor_radius a) from equilibrium. "
                "Neither 1D radial profiles ('r_inboard', 'r_outboard') nor boundary scalar metrics "
                "('boundary.minor_radius', 'boundary.geometric_axis.r', 'boundary.outline.r') were found."
            )

        scale_to_cm = cls._resolve_length_scale_to_cm(R0, scale_to_cm)
        major_radius_cm = R0 * scale_to_cm
        minor_radius_cm = a_minor * scale_to_cm

        r_over_a = np.linspace(0.0, 1.0, n_points)

        # 2. Resample Mean Profiles using PCHIP
        if 't_i_average' in cp and cls._has_imas_data(cp['t_i_average']):
            T_mean_raw = np.asarray(cp['t_i_average'], dtype=float)
        elif 'ion' in cp and any(cls._has_imas_data(cls._get_imas_field(ion, 'temperature')) for ion in (cp['ion'].values() if hasattr(cp['ion'], 'values') else cp['ion'])):
            ions_list = list(cp['ion'].values()) if hasattr(cp['ion'], 'values') else list(cp['ion'])
            t_ion = next(np.asarray(cls._get_imas_field(ion, 'temperature'), dtype=float) for ion in ions_list if cls._has_imas_data(cls._get_imas_field(ion, 'temperature')))
            warnings.warn("No 't_i_average' found in core_profiles; using species ion temperature from 'ion[].temperature'.", UserWarning, stacklevel=2)
            T_mean_raw = t_ion
        elif 'electrons' in cp and 'temperature' in cp['electrons'] and cls._has_imas_data(cp['electrons']['temperature']):
            warnings.warn(
                "No ion temperature ('t_i_average' or 'ion[].temperature') found in core_profiles. "
                "Assuming thermal equilibrium T_i = T_e from electron temperature.",
                UserWarning,
                stacklevel=2
            )
            T_mean_raw = np.asarray(cp['electrons']['temperature'], dtype=float)
        else:
            raise ValueError(
                "Unable to resolve plasma temperature from core_profiles. "
                "Neither 't_i_average', 'ion[].temperature', nor 'electrons.temperature' was found."
            )

        nD_mean_raw, nT_mean_raw = cls._find_fuel_ion_densities(cp, reaction=reaction)
        ne_mean_raw = (
            np.asarray(cp['electrons']['density'], dtype=float)
            if 'electrons' in cp and 'density' in cp['electrons'] and len(cp['electrons']['density']) > 0
            else (nD_mean_raw + nT_mean_raw)
        )

        # Determine coordinate grids for equilibrium and core_profiles
        has_eq_rho = (
            'rho_tor_norm' in eq and len(eq['rho_tor_norm']) == len(r_tilde_eq)
        )
        has_cp_rho = (
            'grid' in cp and 'rho_tor_norm' in cp['grid']
            and len(cp['grid']['rho_tor_norm']) == len(T_mean_raw)
        )

        if has_eq_rho and has_cp_rho:
            rho_eq = np.asarray(eq['rho_tor_norm'], dtype=float)
            rho_cp = np.asarray(cp['grid']['rho_tor_norm'], dtype=float)
            r_tilde_cp = PchipInterpolator(rho_eq, r_tilde_eq)(rho_cp)
        elif has_cp_rho and not has_eq_rho:
            warnings.warn(
                "Missing 'rho_tor_norm' in equilibrium profiles; assuming uniform radial flux spacing.",
                UserWarning,
                stacklevel=2,
            )
            rho_eq = np.linspace(0.0, 1.0, len(r_tilde_eq))
            rho_cp = np.asarray(cp['grid']['rho_tor_norm'], dtype=float)
            r_tilde_cp = PchipInterpolator(rho_eq, r_tilde_eq)(rho_cp)
        elif has_eq_rho and not has_cp_rho:
            warnings.warn(
                "Missing 'grid.rho_tor_norm' in core_profiles; assuming uniform radial flux spacing.",
                UserWarning,
                stacklevel=2,
            )
            rho_eq = np.asarray(eq['rho_tor_norm'], dtype=float)
            rho_cp = np.linspace(0.0, 1.0, len(T_mean_raw))
            r_tilde_cp = PchipInterpolator(rho_eq, r_tilde_eq)(rho_cp)
        else:
            if len(r_tilde_eq) == len(T_mean_raw):
                r_tilde_cp = r_tilde_eq.copy()
            else:
                warnings.warn(
                    "Missing 'rho_tor_norm' in both equilibrium and core_profiles with different grid sizes; "
                    "assuming uniform radial spacing.",
                    UserWarning,
                    stacklevel=2,
                )
                rho_eq = np.linspace(0.0, 1.0, len(r_tilde_eq))
                rho_cp = np.linspace(0.0, 1.0, len(T_mean_raw))
                r_tilde_cp = PchipInterpolator(rho_eq, r_tilde_eq)(rho_cp)

        # Do NOT force boundary clamping unless the source grid actually spans [0, 1]
        r_tilde_cp = np.clip(r_tilde_cp, 0.0, 1.0)
        if has_cp_rho:
            if np.isclose(rho_cp[0], 0.0, atol=1e-3):
                r_tilde_cp[0] = 0.0
            if np.isclose(rho_cp[-1], 1.0, atol=1e-3):
                r_tilde_cp[-1] = 1.0
            elif r_tilde_cp[-1] < 1.0:
                warnings.warn(
                    f"Core profiles grid terminates at rho={rho_cp[-1]:.3f} (r_tilde={r_tilde_cp[-1]:.3f}). "
                    "Profile values will be extrapolated to the plasma boundary r/a=1.0.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            r_tilde_cp[0] = 0.0
            r_tilde_cp[-1] = 1.0

        for i in range(1, len(r_tilde_cp)):
            if r_tilde_cp[i] <= r_tilde_cp[i - 1]:
                r_tilde_cp[i] = r_tilde_cp[i - 1] + 1e-7

        r_eval_cp = np.clip(r_over_a, r_tilde_cp[0], r_tilde_cp[-1])
        temperature = np.maximum(0.0, PchipInterpolator(r_tilde_cp, T_mean_raw)(r_eval_cp))
        density_D = np.maximum(0.0, PchipInterpolator(r_tilde_cp, nD_mean_raw)(r_eval_cp))
        density_T = np.maximum(0.0, PchipInterpolator(r_tilde_cp, nT_mean_raw)(r_eval_cp))

        # 3. Geometry Profiles
        if 'elongation' in eq and cls._has_imas_data(eq['elongation']):
            kappa_raw = np.asarray(eq['elongation'], dtype=float)
            if len(kappa_raw) == 1:
                elongation = 1.0 + (float(kappa_raw[0]) - 1.0) * (r_over_a ** 2)
            else:
                elongation = np.maximum(1.0, PchipInterpolator(r_tilde_eq, kappa_raw)(r_over_a))
        elif 'boundary' in ts and 'elongation' in ts['boundary'] and cls._has_imas_data(ts['boundary']['elongation']):
            kappa_edge = float(np.asarray(ts['boundary']['elongation']).flat[0])
            elongation = 1.0 + (kappa_edge - 1.0) * (r_over_a ** 2)
        else:
            elongation = np.ones_like(r_over_a)

        if (
            'triangularity_upper' in eq and 'triangularity_lower' in eq
            and cls._has_imas_data(eq['triangularity_upper']) and cls._has_imas_data(eq['triangularity_lower'])
        ):
            delta_u = np.asarray(eq['triangularity_upper'], dtype=float)
            delta_l = np.asarray(eq['triangularity_lower'], dtype=float)
            if not np.allclose(delta_u, delta_l, atol=1e-3):
                warnings.warn(
                    "OpenMC TokamakSource currently assumes up-down symmetric Miller geometry. "
                    "Upper and lower triangularities differ; averaging them: delta = (delta_u + delta_l)/2.",
                    UserWarning,
                    stacklevel=2,
                )
            delta_raw = 0.5 * (delta_u + delta_l)
            if len(delta_raw) == 1:
                triangularity = float(delta_raw[0]) * r_over_a
            else:
                triangularity = PchipInterpolator(r_tilde_eq, delta_raw)(r_over_a)
        elif 'triangularity' in eq and cls._has_imas_data(eq['triangularity']):
            delta_raw = np.asarray(eq['triangularity'], dtype=float)
            if len(delta_raw) == 1:
                triangularity = float(delta_raw[0]) * r_over_a
            else:
                triangularity = PchipInterpolator(r_tilde_eq, delta_raw)(r_over_a)
        elif 'boundary' in ts and 'triangularity' in ts['boundary'] and cls._has_imas_data(ts['boundary']['triangularity']):
            delta_edge = float(np.asarray(ts['boundary']['triangularity']).flat[0])
            triangularity = delta_edge * r_over_a
        elif (
            'boundary' in ts and 'triangularity_upper' in ts['boundary'] and 'triangularity_lower' in ts['boundary']
            and cls._has_imas_data(ts['boundary']['triangularity_upper']) and cls._has_imas_data(ts['boundary']['triangularity_lower'])
        ):
            delta_u = float(np.asarray(ts['boundary']['triangularity_upper']).flat[0])
            delta_l = float(np.asarray(ts['boundary']['triangularity_lower']).flat[0])
            triangularity = 0.5 * (delta_u + delta_l) * r_over_a
        else:
            triangularity = np.zeros_like(r_over_a)

        triangularity[0] = 0.0  # Triangularity vanishes at the core axis

        # 4. Shafranov and vertical shifts from magnetic/geometric axis
        def _resolve_axis_coord(coord: str, check_boundary_geom: bool = False) -> float | None:
            if (
                'global_quantities' in ts and 'magnetic_axis' in ts['global_quantities']
                and coord in ts['global_quantities']['magnetic_axis']
                and cls._has_imas_data(ts['global_quantities']['magnetic_axis'][coord])
            ):
                return float(np.asarray(ts['global_quantities']['magnetic_axis'][coord]).flat[0])
            flat_mag = f'global_quantities.magnetic_axis.{coord}'
            if flat_mag in ts and np.asarray(ts[flat_mag]).size > 0:
                return float(np.asarray(ts[flat_mag]).flat[0])
            if (
                check_boundary_geom and 'boundary' in ts and 'geometric_axis' in ts['boundary']
                and coord in ts['boundary']['geometric_axis']
                and cls._has_imas_data(ts['boundary']['geometric_axis'][coord])
            ):
                return float(np.asarray(ts['boundary']['geometric_axis'][coord]).flat[0])
            flat_geom = f'geometric_axis.{coord}'
            if flat_geom in eq and np.asarray(eq[flat_geom]).size > 0:
                return float(np.asarray(eq[flat_geom]).flat[0])
            return None

        R_axis = _resolve_axis_coord('r', check_boundary_geom=False)
        shafranov_shift = max(0.0, float(R_axis - R0) * scale_to_cm) if R_axis is not None else 0.0

        z_axis = _resolve_axis_coord('z', check_boundary_geom=True)
        vertical_shift_detected = float(z_axis * scale_to_cm) if z_axis is not None else 0.0

        return {
            'r_over_a': r_over_a,
            'r_tilde_mapped': r_tilde_cp,
            'temperature': temperature,
            'density_D': density_D,
            'density_T': density_T,
            'major_radius': major_radius_cm,
            'minor_radius': minor_radius_cm,
            'elongation': elongation,
            'triangularity': triangularity,
            'shafranov_shift': shafranov_shift,
            'vertical_shift': vertical_shift_detected,
            'R0': R0,
            'a_minor': a_minor,
            'T_mean_raw': T_mean_raw,
            'nD_mean_raw': nD_mean_raw,
            'nT_mean_raw': nT_mean_raw,
            'ne_mean_raw': ne_mean_raw,
            'r_inboard': r_in * scale_to_cm if r_in is not None else None,
            'r_outboard': r_out * scale_to_cm if r_out is not None else None,
        }

class TokamakSourceEnsemble:
    """An ensemble representing stochastic realizations of a tokamak plasma source for Uncertainty Quantification (UQ).

    This class manages Monte Carlo realizations of tokamak plasma profiles (ion temperature,
    deuterium density, tritium density) and simultaneously evaluates all three primary fusion
    reaction channels:
    1. D-T: D + T -> n (14.1 MeV) + alpha (3.5 MeV)
    2. D-D: D + D -> n (2.45 MeV) + He-3 (0.82 MeV)
    3. T-T: T + T -> 2n (continuum up to 9.5 MeV) + alpha (3.5 MeV)

    Each realization in the ensemble represents an independent physical state of the tokamak
    plasma core, accounting for epistemic uncertainties in temperature and density measurements
    or transport models (e.g. from IMAS covariance matrices decomposed via Karhunen-Loeve expansion).

    The spatial flux surfaces for every realization are parameterized by:

        R(r, alpha) = R_0 + r * cos(alpha + delta(r) * sin(alpha)) + Delta * [1 - (r/a)^2]
        Z(r, alpha) = Z_shift + kappa(r) * r * sin(alpha)

    where R_0 is the major radius, a is the minor radius, kappa(r) is the plasma elongation,
    delta(r) is the plasma triangularity, Delta is the Shafranov shift, and Z_shift is the vertical shift.

    Total neutron source strength for each reaction channel and realization is calculated using 2D Miller volume integration:

        I_reaction = int_0^1 S_reaction(r) * (dV/dr) * dr

    where S_reaction(r) is the volumetric fusion emissivity (neutrons / m^3 / s) evaluated using
    Bosch-Hale parameterization, and dV/dr is the differential volume element in m^3.

    Parameters
    ----------
    major_radius : float
        Major radius R0 in [cm] (must be > 0).
    minor_radius : float
        Minor radius a in [cm] (must be > 0 and < major_radius).
    elongation : float or Sequence[float]
        Plasma elongation kappa (must be > 0). Can be provided as a single scalar
        or as a 1D array of values corresponding to each ``r_over_a`` grid point.
    triangularity : float or Sequence[float]
        Plasma triangularity delta (must be in [-1, 1]). Can be provided as a single
        scalar or as a 1D array of values corresponding to each ``r_over_a`` grid point.
    r_over_a : Sequence[float]
        Normalized minor radius grid points (r_tilde = r/a), strictly from 0.0 to 1.0.
    temperature : Sequence[Sequence[float]]
        Ensemble of ion temperature realizations T_i in [eV] of shape (n_samples, len(r_over_a)).
    density_D : Sequence[Sequence[float]]
        Ensemble of deuterium ion density realizations n_D in [m^-3] of shape (n_samples, len(r_over_a)).
    density_T : Sequence[Sequence[float]]
        Ensemble of tritium ion density realizations n_T in [m^-3] of shape (n_samples, len(r_over_a)).
    shafranov_shift : float, optional
        Shafranov shift Delta in [cm] (must be >= 0 and < a/2, default: 0.0).
    emission_density_DT : Sequence[Sequence[float]], optional
        Ensemble of D-T fusion emissivity profiles in [neutrons / m^3 / s].
    emission_density_DD : Sequence[Sequence[float]], optional
        Ensemble of D-D fusion emissivity profiles in [neutrons / m^3 / s].
    emission_density_TT : Sequence[Sequence[float]], optional
        Ensemble of T-T fusion emissivity profiles in [neutrons / m^3 / s].
    energy_DT : Sequence[Sequence[openmc.stats.Univariate]], optional
        Ensemble of radial Ballabio relativistic Doppler Gaussian distributions for D-T neutrons (~14.1 MeV).
    energy_DD : Sequence[Sequence[openmc.stats.Univariate]], optional
        Ensemble of radial Ballabio relativistic Doppler Gaussian distributions for D-D neutrons (~2.45 MeV).
    energy_TT : openmc.stats.Tabular, optional
        Tabular 3-body continuum distribution for T-T neutrons (0 to 9.5 MeV).
    strength_DT : Sequence[float], optional
        Integrated D-T source strength in [neutrons / s] for each realization.
    strength_DD : Sequence[float], optional
        Integrated D-D source strength in [neutrons / s] for each realization.
    strength_TT : Sequence[float], optional
        Integrated T-T source strength in [neutrons / s] for each realization.
    method : {'fourier', 'bernstein'}, optional
        Poloidal angle sampling algorithm (default: 'fourier').
    time : openmc.stats.Univariate, optional
        Time distribution of the source. If None, particles are born at t=0.
    phi_start : float, optional
        Starting toroidal angle in [rad] (default: 0.0).
    phi_extent : float, optional
        Toroidal angle extent in [rad] (default: 2*pi).
    n_alpha : int, optional
        Number of poloidal angle grid points for CDF tabulation (default: 101).
    vertical_shift : float, optional
        Vertical shift of the plasma center in [cm] (default: 0.0).
    constraints : dict, optional
        Constraints on sampled source particles.

    Attributes
    ----------
    major_radius : float
        Major radius R0 in [cm].
    minor_radius : float
        Minor radius a in [cm].
    elongation : float or numpy.ndarray
        Plasma elongation kappa (scalar or 1D array).
    triangularity : float or numpy.ndarray
        Plasma triangularity delta (scalar or 1D array).
    shafranov_shift : float
        Shafranov shift Delta in [cm].
    r_over_a : numpy.ndarray
        Normalized minor radius grid points in [0.0, 1.0].
    temperature : numpy.ndarray
        2D array of ion temperature realizations in [eV] of shape (n_samples, n_points).
    density_D : numpy.ndarray
        2D array of deuterium density realizations in [m^-3] of shape (n_samples, n_points).
    density_T : numpy.ndarray
        2D array of tritium density realizations in [m^-3] of shape (n_samples, n_points).
    emission_density_DT : numpy.ndarray
        2D array of D-T fusion emissivity realizations in [neutrons / m^3 / s].
    emission_density_DD : numpy.ndarray
        2D array of D-D fusion emissivity realizations in [neutrons / m^3 / s].
    emission_density_TT : numpy.ndarray
        2D array of T-T fusion emissivity realizations in [neutrons / m^3 / s].
    emission_density : numpy.ndarray
        2D array of total fusion emissivity realizations (DT + DD + TT) in [neutrons / m^3 / s].
    energy_DT : list of list of openmc.stats.Univariate
        Ensemble of radial D-T energy distributions.
    energy_DD : list of list of openmc.stats.Univariate
        Ensemble of radial D-D energy distributions.
    energy_TT : openmc.stats.Tabular
        Continuous Tabular T-T 3-body energy spectrum.
    strength_DT : numpy.ndarray
        1D array of integrated D-T source strengths in [neutrons / s] for each realization.
    strength_DD : numpy.ndarray
        1D array of integrated D-D source strengths in [neutrons / s] for each realization.
    strength_TT : numpy.ndarray
        1D array of integrated T-T source strengths in [neutrons / s] for each realization.
    strengths : numpy.ndarray
        1D array of total integrated source strengths (DT + DD + TT) in [neutrons / s] for each realization.
    weight_DT : numpy.ndarray
        1D array of normalized relative emission fractions for D-T fusion.
    weight_DD : numpy.ndarray
        1D array of normalized relative emission fractions for D-D fusion.
    weight_TT : numpy.ndarray
        1D array of normalized relative emission fractions for T-T fusion.
    weights : numpy.ndarray
        2D array of normalized reaction weights of shape (n_samples, 3).
    mean_strength : float
        Expected total source strength in [neutrons / s] across realizations.
    std_strength : float
        Standard deviation of total source strength in [neutrons / s] across realizations.
    n_samples : int
        Number of realizations in the ensemble.
    """

    def __init__(
        self,
        major_radius: float,
        minor_radius: float,
        elongation: float | Sequence[float],
        triangularity: float | Sequence[float],
        r_over_a: Sequence[float],
        temperature: Sequence[Sequence[float]],
        density_D: Sequence[Sequence[float]],
        density_T: Sequence[Sequence[float]],
        shafranov_shift: float = 0.0,
        emission_density_DT: Sequence[Sequence[float]] | None = None,
        emission_density_DD: Sequence[Sequence[float]] | None = None,
        emission_density_TT: Sequence[Sequence[float]] | None = None,
        energy_DT: Sequence[Sequence[Univariate]] | None = None,
        energy_DD: Sequence[Sequence[Univariate]] | None = None,
        energy_TT: Univariate | None = None,
        strength_DT: Sequence[float] | None = None,
        strength_DD: Sequence[float] | None = None,
        strength_TT: Sequence[float] | None = None,
        method: str = 'fourier',
        time: Univariate | None = None,
        phi_start: float = 0.0,
        phi_extent: float = 2.0 * np.pi,
        n_alpha: int = 101,
        vertical_shift: float = 0.0,
        constraints: dict[str, Any] | None = None
    ):
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.r_over_a = np.asarray(r_over_a, dtype=float)
        self.shafranov_shift = float(shafranov_shift)
        self.elongation = np.asarray(elongation, dtype=float) if isinstance(elongation, Iterable) else float(elongation)
        self.triangularity = np.asarray(triangularity, dtype=float) if isinstance(triangularity, Iterable) else float(triangularity)

        self.temperature = np.asarray(temperature, dtype=float)
        self.density_D = np.asarray(density_D, dtype=float)
        self.density_T = np.asarray(density_T, dtype=float)

        n_samples = len(self.temperature)
        n_points = len(self.r_over_a)

        # Dimension validation
        if self.temperature.shape != (n_samples, n_points):
            raise ValueError(
                f"temperature shape {self.temperature.shape} must match (n_samples, n_points)=({n_samples}, {n_points})"
            )
        if self.density_D.shape != (n_samples, n_points):
            raise ValueError(
                f"density_D shape {self.density_D.shape} must match (n_samples, n_points)=({n_samples}, {n_points})"
            )
        if self.density_T.shape != (n_samples, n_points):
            raise ValueError(
                f"density_T shape {self.density_T.shape} must match (n_samples, n_points)=({n_samples}, {n_points})"
            )

        self.method = method
        self.time = time
        self.phi_start = phi_start
        self.phi_extent = phi_extent
        self.n_alpha = n_alpha
        self.vertical_shift = vertical_shift
        self.constraints = constraints

        # 1. Compute or assign emission density profiles for DT, DD, TT (vectorized over realizations)
        self.emission_density_DT = (
            TokamakSource._calculate_fusion_emissivity(
                self.density_D, self.density_T, self.temperature, reaction='DT'
            ) if emission_density_DT is None else np.asarray(emission_density_DT, dtype=float)
        )
        self.emission_density_DD = (
            TokamakSource._calculate_fusion_emissivity(
                self.density_D, self.density_T, self.temperature, reaction='DD'
            ) if emission_density_DD is None else np.asarray(emission_density_DD, dtype=float)
        )
        self.emission_density_TT = (
            TokamakSource._calculate_fusion_emissivity(
                self.density_D, self.density_T, self.temperature, reaction='TT'
            ) if emission_density_TT is None else np.asarray(emission_density_TT, dtype=float)
        )

        # Combined total emissivity profile
        self.emission_density = self.emission_density_DT + self.emission_density_DD + self.emission_density_TT

        # 2. Compute or assign radial energy distributions for DT, DD, TT
        self.energy_DT = (
            [TokamakSource._ballabio_energy_spectrum(self.temperature[i], reaction='DT') for i in range(n_samples)]
            if energy_DT is None else list(energy_DT)
        )
        self.energy_DD = (
            [TokamakSource._ballabio_energy_spectrum(self.temperature[i], reaction='DD') for i in range(n_samples)]
            if energy_DD is None else list(energy_DD)
        )
        self.energy_TT = TokamakSource._get_tt_energy_spectrum() if energy_TT is None else energy_TT

        # 3. Compute or assign integrated source strengths in [neutrons / s] (vectorized over realizations)
        geom_args = (
            self.r_over_a, self.major_radius, self.minor_radius,
            self.elongation, self.triangularity, self.shafranov_shift
        )
        self.strength_DT = (
            self._calculate_integrated_source_strength(self.emission_density_DT, *geom_args)
            if strength_DT is None else np.asarray(strength_DT, dtype=float)
        )
        self.strength_DD = (
            self._calculate_integrated_source_strength(self.emission_density_DD, *geom_args)
            if strength_DD is None else np.asarray(strength_DD, dtype=float)
        )
        self.strength_TT = (
            self._calculate_integrated_source_strength(self.emission_density_TT, *geom_args)
            if strength_TT is None else np.asarray(strength_TT, dtype=float)
        )

        # Total integrated source strength
        self._strengths = self.strength_DT + self.strength_DD + self.strength_TT

        # Karhunen-Loève expansion metadata (populated when loaded via from_imas)
        self.kl_info: dict[str, Any] | None = None

    def __len__(self) -> int:
        """Returns the number of realizations in the ensemble."""
        return len(self.temperature)

    def __getitem__(self, index: int | slice) -> Any:
        """Access a realization [source_DT, source_DD, source_TT] or slice a sub-ensemble."""
        if isinstance(index, (int, np.integer)):
            return self.sample(int(index))
        elif isinstance(index, slice):
            sub_ens = TokamakSourceEnsemble(
                major_radius=self.major_radius,
                minor_radius=self.minor_radius,
                elongation=self.elongation,
                triangularity=self.triangularity,
                r_over_a=self.r_over_a,
                temperature=self.temperature[index],
                density_D=self.density_D[index],
                density_T=self.density_T[index],
                shafranov_shift=self.shafranov_shift,
                emission_density_DT=self.emission_density_DT[index],
                emission_density_DD=self.emission_density_DD[index],
                emission_density_TT=self.emission_density_TT[index],
                energy_DT=self.energy_DT[index],
                energy_DD=self.energy_DD[index],
                energy_TT=self.energy_TT,
                strength_DT=self.strength_DT[index],
                strength_DD=self.strength_DD[index],
                strength_TT=self.strength_TT[index],
                method=self.method,
                time=self.time,
                phi_start=self.phi_start,
                phi_extent=self.phi_extent,
                n_alpha=self.n_alpha,
                vertical_shift=self.vertical_shift,
                constraints=self.constraints
            )
            sub_ens.kl_info = self.kl_info
            return sub_ens
        else:
            raise TypeError(f"Invalid index type: {type(index)}. Expected int or slice.")

    def __iter__(self):
        """Yields [source_DT, source_DD, source_TT] realization lists sequentially."""
        for i in range(len(self)):
            yield self.sample(i)

    def sample(self, index: int, normalize_strength: bool = False) -> list[TokamakSource]:
        """Constructs and returns the 3 TokamakSource objects [source_DT, source_DD, source_TT] for realization index.

        Parameters
        ----------
        index : int
            Index of the realization (0 <= index < len(ensemble)).
        normalize_strength : bool, optional
            If True, sets the source strengths to relative weights summing to 1.0.
            If False (default), sets strengths to absolute neutron rates in neutrons / second.

        Returns
        -------
        list of openmc.TokamakSource
            List of 3 sources [source_DT, source_DD, source_TT] representing the simultaneous
            fusion reaction channels for this realization.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Sample index {index} out of range for ensemble of size {len(self)}.")

        tot = self._strengths[index] if normalize_strength and self._strengths[index] > 0 else 1.0
        sources = []

        channels = (
            (self.strength_DT[index], self.emission_density_DT[index], self.energy_DT[index]),
            (self.strength_DD[index], self.emission_density_DD[index], self.energy_DD[index]),
            (self.strength_TT[index], self.emission_density_TT[index], self.energy_TT),
        )
        for s_raw, em_dens, energy_dist in channels:
            if s_raw > 0.0 and np.any(em_dens > 0.0):
                s_val = float(s_raw / tot) if normalize_strength else float(s_raw)
                sources.append(TokamakSource(
                    major_radius=self.major_radius,
                    minor_radius=self.minor_radius,
                    elongation=self.elongation,
                    triangularity=self.triangularity,
                    shafranov_shift=self.shafranov_shift,
                    r_over_a=self.r_over_a,
                    emission_density=em_dens,
                    energy=energy_dist,
                    method=self.method,
                    time=self.time,
                    phi_start=self.phi_start,
                    phi_extent=self.phi_extent,
                    n_alpha=self.n_alpha,
                    vertical_shift=self.vertical_shift,
                    strength=s_val,
                    constraints=self.constraints
                ))

        if not sources:
            raise ValueError(
                f"No active fusion reaction channels with non-zero emission density for realization {index}."
            )

        return sources

    @property
    def strengths(self) -> np.ndarray:
        """Array of total integrated source strengths (DT + DD + TT) in [neutrons / s] for each realization."""
        return self._strengths

    @property
    def mean_strength(self) -> float:
        """Expected total source strength in [neutrons / s] across realizations."""
        return float(np.mean(self._strengths))

    @property
    def std_strength(self) -> float:
        """Standard deviation of total source strength in [neutrons / s] across realizations."""
        return float(np.std(self._strengths, ddof=1)) if len(self) > 1 else 0.0

    @property
    def weights(self) -> np.ndarray:
        """2D array of normalized reaction weights (DT, DD, TT) of shape (n_samples, 3)."""
        tot = np.maximum(self._strengths, 1e-30)[:, np.newaxis]
        return np.column_stack((self.strength_DT, self.strength_DD, self.strength_TT)) / tot

    @property
    def weight_DT(self) -> np.ndarray:
        """Normalized emission weight fraction for D-T fusion per realization."""
        return self.weights[:, 0]

    @property
    def weight_DD(self) -> np.ndarray:
        """Normalized emission weight fraction for D-D fusion per realization."""
        return self.weights[:, 1]

    @property
    def weight_TT(self) -> np.ndarray:
        """Normalized emission weight fraction for T-T fusion per realization."""
        return self.weights[:, 2]

    @staticmethod
    def _calculate_integrated_source_strength(
        S_profile: np.ndarray,
        r_over_a: np.ndarray,
        major_radius: float,
        minor_radius: float,
        elongation: float | np.ndarray,
        triangularity: float | np.ndarray,
        shafranov_shift: float = 0.0
    ) -> float | np.ndarray:
        """Calculates total neutron emission rate (neutrons / second) across the tokamak plasma volume.

        Parameters
        ----------
        S_profile : numpy.ndarray
            Volumetric fusion emissivity profile S(r) in [neutrons / m^3 / s] (1D or 2D array).
        r_over_a : numpy.ndarray
            Normalized minor radius grid points in [0.0, 1.0].
        major_radius : float
            Major radius R0 in [cm] (or [m]).
        minor_radius : float
            Minor radius a in [cm] (or [m]).
        elongation : float or numpy.ndarray
            Plasma elongation kappa (scalar or 1D array on r_over_a).
        triangularity : float or numpy.ndarray
            Plasma triangularity delta (scalar or 1D array on r_over_a).
        shafranov_shift : float, optional
            Shafranov shift Delta in [cm] (or [m]).

        Returns
        -------
        float or numpy.ndarray
            Total neutron emission rate in [neutrons / s].
        """
        # Convert dimensions to meters for volume element.
        scale = 0.01 / TokamakSource._resolve_length_scale_to_cm(major_radius)
        R0_m = major_radius * scale
        a_m = minor_radius * scale
        shift_m = shafranov_shift * scale

        r_tilde = np.asarray(r_over_a, dtype=float)
        eps = a_m / R0_m if R0_m > 0 else 0.0
        delta_tilde = shift_m / a_m if a_m > 0 else 0.0

        delta = np.asarray(triangularity, dtype=float)
        kappa = np.asarray(elongation, dtype=float)

        if kappa.ndim > 0 and kappa.size > 1:
            kp = CubicSpline(r_tilde, kappa, bc_type=((1, 0.0), 'not-a-knot')).derivative()(r_tilde)
        else:
            kp = np.zeros_like(r_tilde)

        if delta.ndim > 0 and delta.size > 1:
            dp = CubicSpline(r_tilde, delta, bc_type=((1, 0.0), 'not-a-knot')).derivative()(r_tilde)
        else:
            dp = np.zeros_like(r_tilde)

        A = kappa * delta - kappa * r_tilde * dp + kp * r_tilde * delta
        T1 = (
            2.0 * np.pi * (kappa + 0.5 * kp * r_tilde) * jv(0, delta)
            - np.pi * kp * r_tilde * jv(2, delta)
            + 0.5 * np.pi * A * (jv(1, delta) + jv(3, delta))
        )
        T2 = (
            -(kappa + 0.5 * kp * r_tilde) * jv(1, 2.0 * delta)
            + 0.5 * kp * r_tilde * jv(3, 2.0 * delta)
            + 0.25 * A * (jv(0, 2.0 * delta) - jv(4, 2.0 * delta))
            - 2.0 * kappa * delta_tilde * r_tilde * (jv(0, delta) + jv(2, delta))
        )
        dV_dr = (
            2.0 * np.pi * R0_m * (a_m**2) * r_tilde
            * np.maximum(
                0.0,
                (1.0 + eps * delta_tilde - eps * delta_tilde * r_tilde**2) * T1
                + eps * r_tilde * np.pi * T2
            )
        )

        S_arr = np.asarray(S_profile, dtype=float)
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        strength_n_per_sec = np.maximum(0.0, trapz_fn(S_arr * dV_dr, r_tilde, axis=-1))
        return float(strength_n_per_sec) if S_arr.ndim == 1 else strength_n_per_sec

    @staticmethod
    def _perform_kl_expansion(
        cov_matrix: np.ndarray,
        kl_components: int | float = 0.99,
        grid: Sequence[float] | None = None,
        weights: Sequence[float] | None = None
    ) -> dict[str, Any]:
        """Performs Karhunen-Loeve spectral decomposition on a covariance matrix.

        When a non-uniform spatial ``grid`` or quadrature ``weights`` vector is
        supplied, solves the symmetric Nyström integral eigenvalue problem
        :math:`W^{1/2} C W^{1/2} u_k = \\lambda_k u_k` with :math:`\\phi_k = W^{-1/2} u_k`
        so that dense grid regions do not bias the spectral mode basis.
        """
        cov = np.asarray(cov_matrix, dtype=float)
        n_grid = cov.shape[0]

        if weights is not None:
            w = np.maximum(np.asarray(weights, dtype=float), 1e-30)
        elif grid is not None and len(grid) == n_grid and n_grid > 1:
            x = np.asarray(grid, dtype=float)
            w = np.empty(n_grid, dtype=float)
            w[0] = 0.5 * (x[1] - x[0])
            w[1:-1] = 0.5 * (x[2:] - x[:-2])
            w[-1] = 0.5 * (x[-1] - x[-2])
            w = np.maximum(w, 1e-30)
        else:
            w = None

        if w is not None:
            sqrt_w = np.sqrt(w)
            eigenvalues, eigenvectors = eigh(cov * np.outer(sqrt_w, sqrt_w))
        else:
            eigenvalues, eigenvectors = eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(0.0, eigenvalues[idx])
        eigenvectors = eigenvectors[:, idx]

        total_var = np.sum(eigenvalues)
        cum_var = np.cumsum(eigenvalues) / total_var if total_var > 0 else np.ones(n_grid)

        if isinstance(kl_components, (int, np.integer)):
            k_modes = min(int(kl_components), n_grid)
        elif isinstance(kl_components, (float, np.floating)):
            k_modes = int(np.searchsorted(cum_var, float(kl_components) - 1e-12) + 1)
            k_modes = min(k_modes, n_grid)
        else:
            raise ValueError("kl_components must be an int or float.")

        k_evals = eigenvalues[:k_modes]
        k_evecs = eigenvectors[:, :k_modes]
        if w is not None:
            k_evecs = k_evecs / sqrt_w[:, np.newaxis]
        mode_basis = k_evecs * np.sqrt(k_evals)

        return {
            'n_modes': k_modes,
            'explained_variance_ratio': float(cum_var[k_modes - 1]),
            'eigenvalues': k_evals,
            'eigenvectors': k_evecs,
            'mode_basis': mode_basis
        }

    @staticmethod
    def _sample_kl_expansion(
        mean_profile: np.ndarray,
        mode_basis: np.ndarray,
        n_samples: int = 100,
        rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Generates stochastic realizations using the KL mode basis."""
        mean_arr = np.asarray(mean_profile, dtype=float)
        basis_arr = np.asarray(mode_basis, dtype=float)
        k_modes = basis_arr.shape[1]

        if rng is None:
            rng = np.random.default_rng()

        xi = rng.normal(loc=0.0, scale=1.0, size=(n_samples, k_modes))
        samples = mean_arr + np.dot(xi, basis_arr.T)
        return np.maximum(0.0, samples)

    @classmethod
    def from_imas(
        cls,
        imas_input: Any,
        n_samples: int = 100,
        kl_components: int | float = 0.99,
        n_points: int = 101,
        method: str = 'fourier',
        time: Univariate | None = None,
        phi_start: float = 0.0,
        phi_extent: float = 2.0 * np.pi,
        n_alpha: int = 101,
        vertical_shift: float | None = None,
        random_seed: int | np.random.Generator | None = None,
        scale_to_cm: float | None = None,
        constraints: dict[str, Any] | None = None,
        time_index: int | None = None,
        time_s: float | None = None
    ) -> TokamakSourceEnsemble:
        """Create a TokamakSourceEnsemble directly from an IMAS equilibrium and core profiles with covariance matrices.

        This method performs the upfront coordinate transformation from flux coordinate rho
        to normalized minor radius r_tilde before Karhunen-Loeve expansion by interpolating
        eigenvector modes onto the regular r_tilde grid using CubicSpline.

        Parameters
        ----------
        imas_input : PathLike, str, omas.ODS, or dict
            Path to a native IMAS dataset directory (containing equilibrium.nc, core_profiles.nc, etc.),
            a native IMAS NetCDF-4 file with IDS groups, a legacy flat OMAS file (.nc, .h5, .hdf5),
            or an existing in-memory ODS/dict object.
        n_samples : int, optional
            Number of Monte Carlo realizations to generate (default: 100).
        kl_components : int or float, optional
            Number of KL modes or cumulative variance threshold (default: 0.99).
        n_points : int, optional
            Number of radial grid points for the uniform r_over_a grid (default: 101).
        method : {'fourier', 'bernstein'}, optional
            Poloidal angle sampling algorithm (default: 'fourier').
        time : openmc.stats.Univariate, optional
            Time distribution of the source.
        phi_start : float, optional
            Starting toroidal angle in [rad] (default: 0.0).
        phi_extent : float, optional
            Toroidal angle extent in [rad] (default: 2*pi).
        n_alpha : int, optional
            Number of poloidal angle grid points (default: 101).
        vertical_shift : float, optional
            Vertical shift of the plasma center in [cm]. If None (default), automatically
            extracted from the IMAS equilibrium magnetic axis vertical position.
        random_seed : int or numpy.random.Generator, optional
            Random seed or generator for reproducible sampling.
        scale_to_cm : float, optional
            Scaling factor to convert spatial coordinates to cm (default: 100.0 if R0 < 20 m).
        constraints : dict, optional
            Constraints on sampled source particles.
        time_index : int, optional
            Time slice index to extract from time-dependent profiles.
        time_s : float, optional
            Target time in seconds to extract from time-dependent profiles (selects closest slice).

        Returns
        -------
        TokamakSourceEnsemble
            Initialized TokamakSourceEnsemble managing stochastic plasma realizations.
        """
        ods = TokamakSource._load_ods(imas_input, time_index=time_index, time_s=time_s)
        data = TokamakSource._extract_and_map_imas_profiles(
            ods,
            n_points=n_points,
            scale_to_cm=scale_to_cm,
            reaction='DT'
        )

        r_over_a = data['r_over_a']
        r_tilde_mapped = data['r_tilde_mapped']
        temperature_mean = data['temperature']
        density_D_mean = data['density_D']
        density_T_mean = data['density_T']
        T_mean_raw = data['T_mean_raw']
        nD_mean_raw = data['nD_mean_raw']
        nT_mean_raw = data['nT_mean_raw']
        ne_mean_raw = data['ne_mean_raw']

        # Extract Covariance Submatrices and Map via Eigenvector Mode Splines
        if 'covariance' not in ods['core_profiles'] or 'data' not in ods['core_profiles']['covariance']:
            raise ValueError(
                "No covariance data found in IMAS core_profiles ('core_profiles.covariance.data'). "
                "TokamakSourceEnsemble.from_imas requires a covariance matrix to perform "
                "Karhunen-Loève profile sampling."
            )

        full_cov = np.asarray(ods['core_profiles']['covariance']['data'], dtype=float)
        if 'rows_uri' not in ods['core_profiles']['covariance']:
            raise ValueError(
                "No 'rows_uri' metadata found in IMAS core_profiles covariance."
            )
        rows_uri = [str(u) for u in ods['core_profiles']['covariance']['rows_uri']]

        ne_idx = [i for i, u in enumerate(rows_uri) if 'electrons.density' in u]
        ti_idx = [i for i, u in enumerate(rows_uri) if 't_i_average' in u]

        if len(ne_idx) == 0:
            raise ValueError(
                "No electron density covariance found in IMAS covariance matrix ('electrons.density')."
            )
        if len(ti_idx) == 0:
            raise ValueError(
                "No ion temperature covariance found in IMAS covariance matrix ('t_i_average')."
            )

        cov_ne_raw = full_cov[np.ix_(ne_idx, ne_idx)]
        cov_Ti_raw = full_cov[np.ix_(ti_idx, ti_idx)]
        cov_Ti_ne_raw = full_cov[np.ix_(ti_idx, ne_idx)]

        # Scale electron density perturbations for D and T species (preserving quasi-neutrality)
        fD_raw = nD_mean_raw / np.maximum(ne_mean_raw, 1e-30)
        fT_raw = nT_mean_raw / np.maximum(ne_mean_raw, 1e-30)

        if isinstance(random_seed, np.random.Generator):
            main_rng = random_seed
        elif random_seed is not None:
            main_rng = np.random.default_rng(random_seed)
        else:
            main_rng = np.random.default_rng()

        if len(r_tilde_mapped) > 1:
            w_grid = np.empty(len(r_tilde_mapped), dtype=float)
            w_grid[0] = 0.5 * (r_tilde_mapped[1] - r_tilde_mapped[0])
            w_grid[1:-1] = 0.5 * (r_tilde_mapped[2:] - r_tilde_mapped[:-2])
            w_grid[-1] = 0.5 * (r_tilde_mapped[-1] - r_tilde_mapped[-2])
            w_grid = np.maximum(w_grid, 1e-30)
        else:
            w_grid = None

        if np.any(np.abs(cov_Ti_ne_raw) > 0.0):
            # Joint KL expansion on normalized (T_i, n_e) correlation matrix to preserve cross-covariances
            n_ti = len(ti_idx)
            joint_idx = ti_idx + ne_idx
            cov_joint_raw = full_cov[np.ix_(joint_idx, joint_idx)]
            std_joint = np.sqrt(np.maximum(0.0, np.diag(cov_joint_raw)))
            inv_std = np.where(std_joint > 0.0, 1.0 / std_joint, 0.0)
            corr_joint = cov_joint_raw * np.outer(inv_std, inv_std)

            w_joint = (
                np.concatenate([w_grid, w_grid])
                if w_grid is not None and 2 * len(w_grid) == len(joint_idx)
                else None
            )
            kl_joint = cls._perform_kl_expansion(
                corr_joint, kl_components=kl_components, weights=w_joint
            )
            basis_joint_raw = kl_joint['mode_basis'] * std_joint[:, np.newaxis]
            basis_T_raw = basis_joint_raw[:n_ti, :]
            basis_ne_raw = basis_joint_raw[n_ti:, :]
            kl_T_info = {'n_modes': kl_joint['n_modes'], 'explained_variance': kl_joint['explained_variance_ratio']}
            kl_ne_info = {'n_modes': kl_joint['n_modes'], 'explained_variance': kl_joint['explained_variance_ratio']}
            seed_T = int(main_rng.integers(0, 2**31 - 1))
            seed_ne = seed_T
        else:
            # Separate KL expansions for T_i and n_e
            kl_T = cls._perform_kl_expansion(
                cov_Ti_raw, kl_components=kl_components, grid=r_tilde_mapped
            )
            kl_ne = cls._perform_kl_expansion(
                cov_ne_raw, kl_components=kl_components, grid=r_tilde_mapped
            )
            basis_T_raw = kl_T['mode_basis']
            basis_ne_raw = kl_ne['mode_basis']
            kl_T_info = {'n_modes': kl_T['n_modes'], 'explained_variance': kl_T['explained_variance_ratio']}
            kl_ne_info = {'n_modes': kl_ne['n_modes'], 'explained_variance': kl_ne['explained_variance_ratio']}
            seed_T = int(main_rng.integers(0, 2**31 - 1))
            seed_ne = int(main_rng.integers(0, 2**31 - 1))

        basis_nD_raw = basis_ne_raw * fD_raw[:, np.newaxis]
        basis_nT_raw = basis_ne_raw * fT_raw[:, np.newaxis]

        # Interpolate mode basis onto regular r_over_a grid with neutral edge clamping
        r_eval = np.clip(r_over_a, r_tilde_mapped[0], r_tilde_mapped[-1])
        mode_basis_T_rtilde = PchipInterpolator(r_tilde_mapped, basis_T_raw, axis=0)(r_eval)
        mode_basis_nD_rtilde = PchipInterpolator(r_tilde_mapped, basis_nD_raw, axis=0)(r_eval)
        mode_basis_nT_rtilde = PchipInterpolator(r_tilde_mapped, basis_nT_raw, axis=0)(r_eval)

        T_ens = cls._sample_kl_expansion(
            temperature_mean, mode_basis_T_rtilde, n_samples=n_samples, rng=np.random.default_rng(seed_T)
        )
        # Share seed_ne between D and T so delta_nD / nD == delta_nT / nT == delta_ne / ne
        nD_ens = cls._sample_kl_expansion(
            density_D_mean, mode_basis_nD_rtilde, n_samples=n_samples, rng=np.random.default_rng(seed_ne)
        )
        nT_ens = cls._sample_kl_expansion(
            density_T_mean, mode_basis_nT_rtilde, n_samples=n_samples, rng=np.random.default_rng(seed_ne)
        )

        resolved_vertical_shift = (
            data.get('vertical_shift', 0.0) if vertical_shift is None else float(vertical_shift)
        )

        ens = cls(
            major_radius=data['major_radius'],
            minor_radius=data['minor_radius'],
            elongation=data['elongation'],
            triangularity=data['triangularity'],
            shafranov_shift=data['shafranov_shift'],
            r_over_a=r_over_a,
            temperature=T_ens,
            density_D=nD_ens,
            density_T=nT_ens,
            method=method,
            time=time,
            phi_start=phi_start,
            phi_extent=phi_extent,
            n_alpha=n_alpha,
            vertical_shift=resolved_vertical_shift,
            constraints=constraints
        )
        ens.kl_info = {
            'T_i': dict(kl_T_info),
            'n_D': dict(kl_ne_info),
            'n_T': dict(kl_ne_info)
        }
        return ens

class SourceParticle:
    """Source particle

    This class can be used to create source particles that can be written to a
    file and used by OpenMC

    Parameters
    ----------
    r : iterable of float
        Position of particle in Cartesian coordinates
    u : iterable of float
        Directional cosines
    E : float
        Energy of particle in [eV]
    time : float
        Time of particle in [s]
    wgt : float
        Weight of the particle
    delayed_group : int
        Delayed group particle was created in (neutrons only)
    surf_id : int
        Surface ID where particle is at, if any.
    particle : ParticleType or str or int
        Type of the particle (type, name, or PDG number)

    """

    def __init__(
        self,
        r: Iterable[float] = (0., 0., 0.),
        u: Iterable[float] = (0., 0., 1.),
        E: float = 1.0e6,
        time: float = 0.0,
        wgt: float = 1.0,
        delayed_group: int = 0,
        surf_id: int = 0,
        particle: ParticleType | str | int = ParticleType.NEUTRON
    ):

        self.r = tuple(r)
        self.u = tuple(u)
        self.E = float(E)
        self.time = float(time)
        self.wgt = float(wgt)
        self.delayed_group = delayed_group
        self.surf_id = surf_id
        self.particle = particle

    @property
    def particle(self) -> ParticleType:
        return self._particle

    @particle.setter
    def particle(self, particle):
        self._particle = ParticleType(particle)

    def __repr__(self):
        return f'<SourceParticle: {str(self.particle)} at E={self.E:.6e} eV>'

    def to_tuple(self) -> tuple:
        """Return source particle attributes as a tuple

        Returns
        -------
        tuple
            Source particle attributes

        """
        return (self.r, self.u, self.E, self.time, self.wgt,
                self.delayed_group, self.surf_id, self.particle.pdg_number)


def write_source_file(
    source_particles: Iterable[SourceParticle],
    filename: PathLike, **kwargs
):
    """Write a source file using a collection of source particles

    Parameters
    ----------
    source_particles : iterable of SourceParticle
        Source particles to write to file
    filename : str or path-like
        Path to source file to write
    **kwargs
        Keyword arguments to pass to :class:`h5py.File`

    See Also
    --------
    openmc.SourceParticle

    """
    cv.check_iterable_type(
        "source particles", source_particles, SourceParticle)
    pl = ParticleList(source_particles)
    pl.export_to_hdf5(filename, **kwargs)


class ParticleList(list):
    """A collection of SourceParticle objects.

    Parameters
    ----------
    particles : list of SourceParticle
        Particles to collect into the list

    """
    @classmethod
    def from_hdf5(cls, filename: PathLike) -> ParticleList:
        """Create particle list from an HDF5 file.

        Parameters
        ----------
        filename : path-like
            Path to source file to read.

        Returns
        -------
        ParticleList instance

        """
        with h5py.File(filename, 'r') as fh:
            filetype = fh.attrs['filetype']
            arr = fh['source_bank'][...]

        if filetype != b'source':
            raise ValueError(f'File {filename} is not a source file')

        source_particles = [
            SourceParticle(*params, ParticleType(particle))
            for *params, particle in arr
        ]
        return cls(source_particles)

    @classmethod
    def from_mcpl(cls, filename: PathLike) -> ParticleList:
        """Create particle list from an MCPL file.

        Parameters
        ----------
        filename : path-like
            Path to MCPL file to read.

        Returns
        -------
        ParticleList instance

        """
        import mcpl
        # Process .mcpl file
        particles = []
        with mcpl.MCPLFile(filename) as f:
            for particle in f.particles:
                particle_type = ParticleType(particle.pdgcode)

                # Create a source particle instance. Note that MCPL stores
                # energy in MeV and time in ms.
                source_particle = SourceParticle(
                    r=tuple(particle.position),
                    u=tuple(particle.direction),
                    E=1.0e6*particle.ekin,
                    time=1.0e-3*particle.time,
                    wgt=particle.weight,
                    particle=particle_type
                )
                particles.append(source_particle)

        return cls(particles)

    def __getitem__(self, index):
        """
        Return a new ParticleList object containing the particle(s)
        at the specified index or slice.

        Parameters
        ----------
        index : int, slice or list
            The index, slice or list to select from the list of particles

        Returns
        -------
        openmc.ParticleList or openmc.SourceParticle
            A new object with the selected particle(s)
        """
        if isinstance(index, int):
            # If it's a single integer, return the corresponding particle
            return super().__getitem__(index)
        elif isinstance(index, slice):
            # If it's a slice, return a new ParticleList object with the
            # sliced particles
            return ParticleList(super().__getitem__(index))
        elif isinstance(index, list):
            # If it's a list of integers, return a new ParticleList object with
            # the selected particles. Note that Python 3.10 gets confused if you
            # use super() here, so we call list.__getitem__ directly.
            return ParticleList([list.__getitem__(self, i) for i in index])
        else:
            raise TypeError(f"Invalid index type: {type(index)}. Must be int, "
                            "slice, or list of int.")

    def to_dataframe(self) -> pd.DataFrame:
        """A dataframe representing the source particles

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the source particles attributes.
        """
        # Extract the attributes of the source particles into a list of tuples
        data = [(sp.r[0], sp.r[1], sp.r[2], sp.u[0], sp.u[1], sp.u[2],
                 sp.E, sp.time, sp.wgt, sp.delayed_group, sp.surf_id,
                 str(sp.particle)) for sp in self]

        # Define the column names for the DataFrame
        columns = ['x', 'y', 'z', 'u_x', 'u_y', 'u_z', 'E', 'time', 'wgt',
                   'delayed_group', 'surf_id', 'particle']

        # Create the pandas DataFrame from the data
        return pd.DataFrame(data, columns=columns)

    def export_to_hdf5(self, filename: PathLike, **kwargs):
        """Export particle list to an HDF5 file.

        This method write out an .h5 file that can be used as a source file in
        conjunction with the :class:`openmc.FileSource` class.

        Parameters
        ----------
        filename : path-like
            Path to source file to write
        **kwargs
            Keyword arguments to pass to :class:`h5py.File`

        See Also
        --------
        openmc.FileSource

        """
        # Create compound datatype for source particles
        pos_dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        source_dtype = np.dtype([
            ('r', pos_dtype),
            ('u', pos_dtype),
            ('E', '<f8'),
            ('time', '<f8'),
            ('wgt', '<f8'),
            ('delayed_group', '<i4'),
            ('surf_id', '<i4'),
            ('particle', '<i4'),
        ])

        # Create array of source particles
        arr = np.array([s.to_tuple() for s in self], dtype=source_dtype)

        # Write array to file
        kwargs.setdefault('mode', 'w')
        with h5py.File(filename, **kwargs) as fh:
            fh.attrs['filetype'] = np.bytes_("source")
            fh.attrs['version'] = np.array([_VERSION_STATEPOINT, 2])
            fh.create_dataset('source_bank', data=arr, dtype=source_dtype)


def read_source_file(filename: PathLike) -> ParticleList:
    """Read a source file and return a list of source particles.

    .. versionadded:: 0.15.0

    Parameters
    ----------
    filename : str or path-like
        Path to source file to read

    Returns
    -------
    openmc.ParticleList

    See Also
    --------
    openmc.SourceParticle

    """
    filename = Path(filename)
    if filename.suffix not in ('.h5', '.mcpl'):
        raise ValueError('Source file must have a .h5 or .mcpl extension.')

    if filename.suffix == '.h5':
        return ParticleList.from_hdf5(filename)
    else:
        return ParticleList.from_mcpl(filename)


def read_collision_track_hdf5(filename):
    """Read a collision track file in HDF5 format.

    Parameters
    ----------
    filename : str or path-like
        Path to the HDF5 collision track file.

    Returns
    -------
    numpy.ndarray
        Structured array containing collision track data.

    See Also
    --------
    read_collision_track_mcpl
    read_collision_track_file
    """

    with h5py.File(filename, 'r') as file:
        data = file['collision_track_bank'][:]

    return data


def read_collision_track_mcpl(file_path):
    """Read a collision track file in MCPL format.

    Parameters
    ----------
    file_path : str or path-like
        Path to the MCPL collision track file.

    Returns
    -------
    numpy.ndarray
        Structured array of particle collision track information, including
        position, direction, energy, weight, reaction data, and identifiers.

    See Also
    --------
    read_collision_track_hdf5
    read_collision_track_file
    """
    import mcpl
    myfile = mcpl.MCPLFile(file_path)
    data = {
        'r': [],  # for position (x, y, z)
        'u': [],  # for direction (ux, uy, uz)
        'E': [], 'dE': [], 'time': [],
        'wgt': [], 'event_mt': [], 'delayed_group': [],
        'cell_id': [], 'nuclide_id': [], 'material_id': [],
        'universe_id': [], 'n_collision': [], 'particle': [],
        'parent_id': [], 'progeny_id': []
    }

    # Read and collect data from the MCPL file
    for i, p in enumerate(myfile.particles):
        if f'blob_{i}' in myfile.blobs:
            blob_data = myfile.blobs[f'blob_{i}']
            decoded_str = blob_data.decode('utf-8')
            pairs = decoded_str.split(';')
            values_dict = {k.strip(): v.strip()
                           for k, v in (pair.split(':') for pair in pairs if pair.strip())}

            data['r'].append((p.x, p.y, p.z))  # Append as tuple
            data['u'].append((p.ux, p.uy, p.uz))  # Append as tuple
            data['E'].append(p.ekin * 1e6)
            data['dE'].append(float(values_dict.get('dE', 0)))
            data['time'].append(p.time * 1e-3)
            data['wgt'].append(p.weight)
            data['event_mt'].append(int(values_dict.get('event_mt', 0)))
            data['delayed_group'].append(
                int(values_dict.get('delayed_group', 0)))
            data['cell_id'].append(int(values_dict.get('cell_id', 0)))
            data['nuclide_id'].append(int(values_dict.get('nuclide_id', 0)))
            data['material_id'].append(int(values_dict.get('material_id', 0)))
            data['universe_id'].append(int(values_dict.get('universe_id', 0)))
            data['n_collision'].append(int(values_dict.get('n_collision', 0)))
            data['particle'].append(ParticleType(p.pdgcode))
            data['parent_id'].append(int(values_dict.get('parent_id', 0)))
            data['progeny_id'].append(int(values_dict.get('progeny_id', 0)))

    dtypes = [
        ('r', [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]),
        ('u', [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]),
        ('E', 'f8'), ('dE', 'f8'), ('time', 'f8'), ('wgt', 'f8'),
        ('event_mt', 'f8'), ('delayed_group', 'i4'), ('cell_id', 'i4'),
        ('nuclide_id', 'i4'), ('material_id', 'i4'), ('universe_id', 'i4'),
        ('n_collision', 'i4'), ('particle', 'i4'),
        ('parent_id', 'i8'), ('progeny_id', 'i8')
    ]

    structured_array = np.zeros(len(data['r']), dtype=dtypes)
    for key in data:
        structured_array[key] = data[key]  # Assign data

    return structured_array


def read_collision_track_file(filename):
    """Read a collision track file (HDF5 or MCPL) and return its data.

    Parameters
    ----------
    filename : str or path-like
        Path to the collision track file to read. Must end with
        ``.h5`` or ``.mcpl``.

    Returns
    -------
    numpy.ndarray
        Structured array containing collision track data.

    See Also
    --------
    read_collision_track_hdf5
    read_collision_track_mcpl
    """

    filename = Path(filename)
    if filename.suffix not in ('.h5', '.mcpl'):
        raise ValueError('Collision track file must have a .h5 or .mcpl extension.')

    if filename.suffix == '.h5':
        return read_collision_track_hdf5(filename)
    else:
        return read_collision_track_mcpl(filename)
