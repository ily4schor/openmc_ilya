from pathlib import Path

import numpy as np
import pytest

import openmc
import openmc.stats

from tests.unit_tests import assert_sample_mean


def make_source(**kwargs):
    """Build a valid TokamakSource, overriding defaults via kwargs."""
    r_over_a = np.linspace(0.0, 1.0, 10)
    params = dict(
        major_radius=620.0,
        minor_radius=200.0,
        elongation=1.8,
        triangularity=0.45,
        shafranov_shift=10.0,
        r_over_a=r_over_a,
        emission_density=(1.0 - r_over_a**2),
        energy=openmc.stats.muir(e0=14.08e6, m_rat=5.0, kt=2.0e4),
    )
    params.update(kwargs)
    return openmc.TokamakSource(**params)


def test_tokamak_source_roundtrip():
    src = make_source(
        phi_start=0.1, phi_extent=np.pi, n_alpha=51, vertical_shift=5.0,
        strength=2.0, time=openmc.stats.Uniform(0.0, 1e-6))

    elem = src.to_xml_element()
    assert elem.get('type') == 'tokamak'

    new = openmc.SourceBase.from_xml_element(elem)
    assert isinstance(new, openmc.TokamakSource)
    assert new.major_radius == src.major_radius
    assert new.minor_radius == src.minor_radius
    np.testing.assert_allclose(new.elongation, src.elongation)
    np.testing.assert_allclose(new.triangularity, src.triangularity)
    assert new.shafranov_shift == src.shafranov_shift
    assert new.phi_start == src.phi_start
    assert new.phi_extent == src.phi_extent
    assert new.n_alpha == src.n_alpha
    assert new.vertical_shift == src.vertical_shift
    assert new.strength == src.strength
    np.testing.assert_allclose(new.r_over_a, src.r_over_a)
    np.testing.assert_allclose(new.emission_density, src.emission_density)
    assert len(new.energy) == 1
    assert isinstance(new.time, openmc.stats.Uniform)
    assert new.time.a == src.time.a
    assert new.time.b == src.time.b


def test_tokamak_source_default_time():
    src = make_source()
    assert src.time is None

    new = openmc.SourceBase.from_xml_element(src.to_xml_element())
    assert new.time is None

    with pytest.raises(TypeError):
        make_source(time=1.0)


def test_tokamak_source_multiple_energies():
    r_over_a = np.linspace(0.0, 1.0, 5)
    energies = [openmc.stats.muir(e0=14.08e6, m_rat=5.0, kt=kt)
                for kt in (1.0e4, 1.5e4, 2.0e4, 2.5e4, 3.0e4)]
    src = make_source(r_over_a=r_over_a,
                      emission_density=np.ones_like(r_over_a),
                      energy=energies)
    assert len(src.energy) == len(r_over_a)

    new = openmc.SourceBase.from_xml_element(src.to_xml_element())
    assert len(new.energy) == len(r_over_a)


@pytest.mark.parametrize("kwargs, match", [
    (dict(minor_radius=700.0), "smaller than major_radius"),
    (dict(shafranov_shift=150.0), "half the minor_radius"),
    (dict(emission_density=np.ones(5)), "same length as r_over_a"),
    (dict(energy=[openmc.stats.muir(14.08e6, 5.0, 2.0e4)] * 2),
     "Number of energy distributions"),
    (dict(r_over_a=np.linspace(0.1, 1.0, 10)), "must start at 0"),
    (dict(r_over_a=np.linspace(0.0, 0.9, 10)), "must end at 1"),
    (dict(emission_density=-np.linspace(0.0, 1.0, 10)), "cannot be negative"),
    (dict(emission_density=np.zeros(10)), "must contain a positive value"),
])
def test_tokamak_source_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_source(**kwargs)


@pytest.mark.parametrize("value", [-1.5, 1.5])
def test_tokamak_source_invalid_triangularity(value):
    with pytest.raises(ValueError):
        make_source(triangularity=value)


def test_tokamak_source_invalid_n_alpha():
    with pytest.raises(ValueError):
        make_source(n_alpha=2)


@pytest.mark.parametrize(("attribute", "value", "match"), [
    ("minor_radius", 700.0, "smaller than major_radius"),
    ("shafranov_shift", 150.0, "half the minor_radius"),
    ("emission_density", np.ones(5), "same length as r_over_a"),
    ("energy", [openmc.stats.delta_function(1.0)] * 2,
     "Number of energy distributions"),
])
def test_tokamak_source_mutation_validation(attribute, value, match):
    src = make_source()
    setattr(src, attribute, value)
    with pytest.raises(ValueError, match=match):
        src.to_xml_element()


@pytest.mark.parametrize(("emission_density", "expected_mean"), [
    ([1.0, 1.0], 2.0 / 3.0),
    ([1.0, 0.0], 0.5),
])
def test_tokamak_source_radial_sampling(
    run_in_tmpdir, emission_density, expected_mean
):
    """Check radial sampling for profiles on the coarsest valid grid."""
    major_radius = 620.0
    minor_radius = 200.0
    src = make_source(
        major_radius=major_radius,
        minor_radius=minor_radius,
        elongation=1.0,
        triangularity=0.0,
        shafranov_shift=0.0,
        r_over_a=[0.0, 1.0],
        emission_density=emission_density,
        energy=openmc.stats.delta_function(14.07e6),
    )

    sphere = openmc.Sphere(r=2000.0, boundary_type='vacuum')
    model = openmc.Model(
        geometry=openmc.Geometry([openmc.Cell(region=-sphere)]),
        settings=openmc.Settings(
            particles=100, batches=1, run_mode='fixed source', source=src),
    )

    sites = model.sample_external_source(20_000)
    xyz = np.array([site.r for site in sites])
    major_r = np.hypot(xyz[:, 0], xyz[:, 1])
    r_over_a = np.hypot(major_r - major_radius, xyz[:, 2]) / minor_radius
    assert_sample_mean(r_over_a, expected_mean)


def test_tokamak_source_poloidal_sampling(run_in_tmpdir):
    """Check linear-linear poloidal sampling on a coarse internal grid."""
    major_radius = 620.0
    minor_radius = 200.0
    with pytest.warns(UserWarning, match="below 51"):
        src = make_source(
            major_radius=major_radius,
            minor_radius=minor_radius,
            elongation=1.0,
            triangularity=0.0,
            shafranov_shift=0.0,
            emission_density=np.ones(10),
            n_alpha=3,
            method='bernstein',
            energy=openmc.stats.delta_function(14.07e6),
        )

    sphere = openmc.Sphere(r=2000.0, boundary_type='vacuum')
    model = openmc.Model(
        geometry=openmc.Geometry([openmc.Cell(region=-sphere)]),
        settings=openmc.Settings(
            particles=100, batches=1, run_mode='fixed source', source=src),
    )

    sites = model.sample_external_source(100_000)
    xyz = np.array([site.r for site in sites])
    major_r = np.hypot(xyz[:, 0], xyz[:, 1])

    # With three alpha points, linear interpolation of cos(alpha) on [0, pi]
    # gives 1 - 2*alpha/pi. Integrating the resulting density gives this mean.
    expected_R = (
        major_radius
        + 2.0 * minor_radius**2 / (np.pi**2 * major_radius)
    )
    assert_sample_mean(major_r, expected_R)


@pytest.mark.flaky(reruns=1)
def test_tokamak_source_sampling(run_in_tmpdir):
    """Exercise the compiled C++ sampling path and check invariants.

    Sampled moments are compared against direct numerical quadrature of the
    exact source density S(r)*R*|J|, where the Jacobian J of the flux-surface
    map is computed from analytic partial derivatives. This check is
    independent of the Bernstein-mixture factorization used by the
    implementation.
    """
    R0, a, kappa, delta = 620.0, 200.0, 1.8, -0.5
    shafranov, zshift = 40.0, 25.0
    phi_start, phi_extent = 0.5, np.pi / 2

    # Fine grids to make discretization error negligible relative to
    # statistical uncertainty
    r_over_a = np.linspace(0.0, 1.0, 200)
    src = make_source(
        major_radius=R0, minor_radius=a, elongation=kappa,
        triangularity=delta, shafranov_shift=shafranov, vertical_shift=zshift,
        r_over_a=r_over_a, emission_density=1.0 - r_over_a**2,
        phi_start=phi_start, phi_extent=phi_extent, n_alpha=201,
        energy=openmc.stats.delta_function(14.07e6))

    sphere = openmc.Sphere(r=2000.0, boundary_type='vacuum')
    cell = openmc.Cell(region=-sphere)
    settings = openmc.Settings(
        particles=100, batches=1, run_mode='fixed source', source=src)
    model = openmc.Model(geometry=openmc.Geometry([cell]), settings=settings)

    n_samples = 20_000
    sites = model.sample_external_source(n_samples)

    xyz = np.array([s.r for s in sites])
    R = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]

    # Energy, weight, and time invariants
    assert np.all([s.E == 14.07e6 for s in sites])
    assert np.all([s.wgt == 1.0 for s in sites])
    assert np.all([s.time == 0.0 for s in sites])

    # Toroidal angle within the requested sector
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    assert phi.min() >= phi_start
    assert phi.max() <= phi_start + phi_extent

    # Positions bounded by the last closed flux surface
    assert R.min() >= R0 - a
    assert R.max() <= R0 + a + shafranov
    assert np.abs(z - zshift).max() <= kappa * a

    # Up-down symmetry about the vertical shift
    assert_sample_mean(z, zshift)

    # Reference moments by 2D quadrature of the exact density
    r = np.linspace(0.0, 1.0, 1001)[:, np.newaxis]  # r/a
    alpha = np.linspace(0.0, 2 * np.pi, 2001)[np.newaxis, :]
    psi = alpha + delta * np.sin(alpha)
    R_map = R0 + a * r * np.cos(psi) + shafranov * (1.0 - r**2)
    Z_map = kappa * a * r * np.sin(alpha)
    dR_dr = a * np.cos(psi) - 2.0 * shafranov * r
    dR_da = -a * r * np.sin(psi) * (1.0 + delta * np.cos(alpha))
    dZ_dr = kappa * a * np.sin(alpha) * np.ones_like(psi)
    dZ_da = kappa * a * r * np.cos(alpha)
    jac = np.abs(dR_dr * dZ_da - dR_da * dZ_dr)
    dens = (1.0 - r**2) * R_map * jac
    norm = dens.sum()
    expected_R = (R_map * dens).sum() / norm
    expected_z2 = (Z_map**2 * dens).sum() / norm

    assert_sample_mean(R, expected_R)
    assert_sample_mean((z - zshift)**2, expected_z2)


def test_tokamak_source_profile_roundtrip():
    """Verify XML serialization and deserialization for 1D profile geometries."""
    r_over_a = np.linspace(0.0, 1.0, 15)
    elongation = 1.8 - 0.2 * r_over_a**2
    triangularity = 0.45 * r_over_a
    src = make_source(
        r_over_a=r_over_a,
        emission_density=np.ones_like(r_over_a),
        elongation=elongation,
        triangularity=triangularity,
    )
    elem = src.to_xml_element()
    assert elem.find('elongation_prime') is not None
    assert elem.find('triangularity_prime') is not None

    new = openmc.SourceBase.from_xml_element(elem)
    assert isinstance(new, openmc.TokamakSource)
    np.testing.assert_allclose(new.elongation, src.elongation)
    np.testing.assert_allclose(new.triangularity, src.triangularity)
    np.testing.assert_allclose(new.r_over_a, src.r_over_a)


@pytest.mark.parametrize("kwargs, match", [
    (dict(elongation=np.ones(5)), "elongation profile"),
    (dict(triangularity=np.ones(5)), "triangularity profile"),
    (dict(elongation=[-1.0] * 10), "elongation values must be > 0"),
    (dict(triangularity=[1.5] * 10), "triangularity values must be in"),
    (dict(elongation=np.ones(10), method='bernstein'), "Bernstein method requires scalar"),
])
def test_tokamak_source_profile_invalid(kwargs, match):
    """Verify input validation rules for 1D profile geometries."""
    with pytest.raises(ValueError, match=match):
        make_source(**kwargs)


def test_tokamak_source_profile_sampling(run_in_tmpdir):
    """Exercise Fourier-Bessel sampling with 1D profile geometries and verify spatial bounds."""
    R0, a = 620.0, 200.0
    shafranov, zshift = 20.0, 15.0
    r_over_a = np.linspace(0.0, 1.0, 50)
    kappa = 1.8 - 0.3 * r_over_a**2
    delta = 0.4 * r_over_a

    src = make_source(
        major_radius=R0,
        minor_radius=a,
        elongation=kappa,
        triangularity=delta,
        shafranov_shift=shafranov,
        vertical_shift=zshift,
        r_over_a=r_over_a,
        emission_density=1.0 - r_over_a**2,
        energy=openmc.stats.delta_function(14.08e6),
    )

    sphere = openmc.Sphere(r=2000.0, boundary_type='vacuum')
    cell = openmc.Cell(region=-sphere)
    model = openmc.Model(
        geometry=openmc.Geometry([cell]),
        settings=openmc.Settings(particles=100, batches=1, run_mode='fixed source', source=src),
    )

    sites = model.sample_external_source(2000)
    xyz = np.array([s.r for s in sites])
    R = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]

    # Verify positions are bounded by the plasma boundary
    assert R.min() >= R0 - a
    assert R.max() <= R0 + a + shafranov
    assert np.abs(z - zshift).max() <= np.max(kappa) * a


def test_tokamak_source_ensemble_kl_and_slicing():
    """Verify TokamakSourceEnsemble creation, KL expansion, slicing, and reaction channels."""
    n_samples = 4
    n_points = 10
    r_over_a = np.linspace(0.0, 1.0, n_points)
    temperature = np.full((n_samples, n_points), 2.0e4)
    density_D = np.full((n_samples, n_points), 5.0e19)
    density_T = np.full((n_samples, n_points), 5.0e19)

    ensemble = openmc.TokamakSourceEnsemble(
        major_radius=620.0,
        minor_radius=200.0,
        elongation=1.8,
        triangularity=0.45,
        shafranov_shift=10.0,
        r_over_a=r_over_a,
        temperature=temperature,
        density_D=density_D,
        density_T=density_T,
    )

    assert len(ensemble) == n_samples
    assert len(ensemble.strengths) == n_samples
    assert np.all(ensemble.strengths > 0)
    assert ensemble.mean_strength > 0

    # Test indexing and realization sampling
    sources = ensemble[0]
    assert len(sources) == 3
    assert all(isinstance(s, openmc.TokamakSource) for s in sources)

    # Test slicing
    sub_ens = ensemble[0:2]
    assert isinstance(sub_ens, openmc.TokamakSourceEnsemble)
    assert len(sub_ens) == 2

    # Test normalized sampling
    sources_norm = ensemble.sample(0, normalize_strength=True)
    assert len(sources_norm) == 3
    assert np.isclose(sum(s.strength for s in sources_norm), 1.0)

    # Test KL expansion helper method directly
    cov = np.eye(5) * 0.1
    kl_res = openmc.TokamakSourceEnsemble._perform_kl_expansion(cov, kl_components=0.99)
    assert kl_res['n_modes'] > 0
    assert 'mode_basis' in kl_res
    assert kl_res['mode_basis'].shape[0] == 5


def test_tokamak_source_model_run_uq(run_in_tmpdir):
    """Verify Model.run_uq execution, tally extraction, and Law of Total Variance."""
    sphere = openmc.Sphere(r=2000.0, boundary_type='vacuum')
    cell = openmc.Cell(region=-sphere)
    tally = openmc.Tally(name='flux_tally')
    tally.scores = ['flux']

    model = openmc.Model(
        geometry=openmc.Geometry([cell]),
        settings=openmc.Settings(particles=10, batches=2, run_mode='fixed source'),
        tallies=openmc.Tallies([tally]),
    )

    r_over_a = np.linspace(0.0, 1.0, 10)
    ensemble = openmc.TokamakSourceEnsemble(
        major_radius=620.0,
        minor_radius=200.0,
        elongation=1.8,
        triangularity=0.45,
        shafranov_shift=10.0,
        r_over_a=r_over_a,
        temperature=np.full((2, 10), 2.0e4),
        density_D=np.full((2, 10), 5.0e19),
        density_T=np.full((2, 10), 5.0e19),
    )

    uq_res = model.run_uq(
        ensemble=ensemble,
        n_realizations=2,
        cwd="uq_runs",
        keep_statepoints=False,
    )

    assert isinstance(uq_res, openmc.model.UQResult)
    assert uq_res.n_realizations == 2
    assert 'flux_tally' in uq_res.realization_means

    # Test Law of Total Variance breakdown
    stats_nom = uq_res.get_tally('flux_tally', mode='nominal')
    assert stats_nom['mean'] > 0.0
    assert stats_nom['std_dev_total'] >= 0.0
    assert stats_nom['std_dev_mc'] >= 0.0
    assert stats_nom['std_dev_parameters'] >= 0.0

    # Test per-particle and absolute modes
    stats_pp = uq_res.get_tally('flux_tally', mode='per_particle')
    stats_abs = uq_res.get_tally('flux_tally', mode='absolute')
    assert stats_pp['mean'] > 0.0
    assert stats_abs['mean'] > 0.0

    # Test DataFrame export
    df = uq_res.to_dataframe()
    assert len(df) == 2
    assert 'flux_tally' in df['tally'].values


def test_tokamak_source_ensemble_from_imas_missing_covariance():
    """Verify that TokamakSourceEnsemble.from_imas raises ValueError when covariance is absent."""
    fake_ods = {
        'core_profiles': {
            'profiles_1d': [{
                'grid': {'rho_tor_norm': np.linspace(0, 1, 10)},
                't_i_average': np.full(10, 1.0e4),
                'ion': [{'density': np.full(10, 1.0e19)}, {'density': np.full(10, 1.0e19)}],
                'electrons': {'density': np.full(10, 2.0e19)}
            }]
        },
        'equilibrium': {
            'time_slice': [{
                'profiles_1d': {
                    'r_inboard': np.linspace(2.0, 1.0, 10),
                    'r_outboard': np.linspace(4.0, 5.0, 10)
                }
            }]
        }
    }
    with pytest.raises(ValueError, match="No covariance data found in IMAS core_profiles"):
        openmc.TokamakSourceEnsemble.from_imas(fake_ods)


def test_find_fuel_ion_densities_by_name():
    """Verify fuel ion extraction correctly identifies species by name, order, and ignores impurities."""
    grid = np.linspace(0, 1, 10)
    dens_D = np.full(10, 5.0e19)
    dens_T = np.full(10, 4.0e19)
    dens_W = np.full(10, 1.0e16)
    dens_B = np.full(10, 2.0e17)

    # 1. Multi-species plasma in normal order [D, T, W, B]
    cp_normal = {
        'ion': [
            {'name': 'D', 'density': dens_D},
            {'name': 'T', 'density': dens_T},
            {'name': 'W', 'density': dens_W},
            {'name': 'B', 'density': dens_B},
        ]
    }
    nD, nT = openmc.TokamakSource._find_fuel_ion_densities(cp_normal)
    np.testing.assert_allclose(nD, dens_D)
    np.testing.assert_allclose(nT, dens_T)

    # 2. Reversed order [T, D]
    cp_reversed = {
        'ion': [
            {'name': 'T', 'density': dens_T},
            {'name': 'D', 'density': dens_D},
        ]
    }
    nD_rev, nT_rev = openmc.TokamakSource._find_fuel_ion_densities(cp_reversed)
    np.testing.assert_allclose(nD_rev, dens_D)
    np.testing.assert_allclose(nT_rev, dens_T)


def test_find_fuel_ion_densities_by_composition():
    """Verify fuel ion extraction correctly identifies species by atomic number Z and mass A."""
    dens_D = np.full(10, 5.0e19)
    dens_T = np.full(10, 4.0e19)

    cp_comp = {
        'ion': [
            {'element': [{'z_n': 1, 'a': 3}], 'density': dens_T},
            {'element': [{'z_n': 1, 'a': 2}], 'density': dens_D},
        ]
    }
    nD, nT = openmc.TokamakSource._find_fuel_ion_densities(cp_comp)
    np.testing.assert_allclose(nD, dens_D)
    np.testing.assert_allclose(nT, dens_T)


def test_find_fuel_ion_densities_reaction_specific():
    """Verify single-species identification for pure DD or TT fusion reactions."""
    dens_D = np.full(10, 5.0e19)
    cp_pure_D = {
        'ion': [{'name': 'D', 'density': dens_D}]
    }
    nD, nT = openmc.TokamakSource._find_fuel_ion_densities(cp_pure_D, reaction='DD')
    np.testing.assert_allclose(nD, dens_D)
    assert np.all(nT == 0.0)

    dens_T = np.full(10, 4.0e19)
    cp_pure_T = {
        'ion': [{'name': 'T', 'density': dens_T}]
    }
    nD_t, nT_t = openmc.TokamakSource._find_fuel_ion_densities(cp_pure_T, reaction='TT')
    assert np.all(nD_t == 0.0)
    np.testing.assert_allclose(nT_t, dens_T)


def test_find_fuel_ion_densities_fallbacks_and_errors():
    """Verify fallback behavior for legacy unlabelled fixtures and missing species errors."""
    dens1 = np.full(10, 1.0e19)
    dens2 = np.full(10, 2.0e19)

    # Legacy unlabelled fixture
    cp_unlabelled = {'ion': [{'density': dens1}, {'density': dens2}]}
    nD, nT = openmc.TokamakSource._find_fuel_ion_densities(cp_unlabelled)
    np.testing.assert_allclose(nD, dens1)
    np.testing.assert_allclose(nT, dens2)

    # Electron-density only fixture
    ne = np.full(10, 4.0e19)
    cp_electrons = {'electrons': {'density': ne}}
    nD_e, nT_e = openmc.TokamakSource._find_fuel_ion_densities(cp_electrons)
    np.testing.assert_allclose(nD_e, 0.5 * ne)
    np.testing.assert_allclose(nT_e, 0.5 * ne)

    # Unresolvable species
    cp_unresolvable = {'ion': [{'name': 'He4', 'density': dens1}]}
    with pytest.raises(ValueError, match="Unable to resolve Deuterium and Tritium"):
        openmc.TokamakSource._find_fuel_ion_densities(cp_unresolvable)


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_native_imas_netcdf_reader(tmp_path):
    """Verify native IMAS NetCDF-4 group hierarchy reader with time-slicing and AoS mapping."""
    netCDF4 = pytest.importorskip("netCDF4")

    nc_file = tmp_path / "imas_test.nc"
    with netCDF4.Dataset(str(nc_file), "w", format="NETCDF4") as ds:
        # Create core_profiles IDS group
        cp_grp = ds.createGroup("core_profiles").createGroup("0")
        cp_grp.createDimension("time", 2)
        cp_grp.createDimension("rho_dim", 10)
        cp_grp.createDimension("ion_dim", 2)

        v_rho = cp_grp.createVariable("profiles_1d.grid.rho_tor_norm", "f8", ("time", "rho_dim"))
        v_rho[:] = np.tile(np.linspace(0, 1, 10), (2, 1))

        v_ne = cp_grp.createVariable("profiles_1d.electrons.density", "f8", ("time", "rho_dim"))
        v_ne[:] = np.tile(np.full(10, 5.0e19), (2, 1))

        v_ti = cp_grp.createVariable("profiles_1d.t_i_average", "f8", ("time", "rho_dim"))
        v_ti[:] = np.tile(np.full(10, 1.5e4), (2, 1))

        v_ion_name = cp_grp.createVariable("profiles_1d.ion.name", str, ("time", "ion_dim"))
        v_ion_name[:] = np.array([["D", "T"], ["D", "T"]], dtype=object)

        v_ion_dens = cp_grp.createVariable("profiles_1d.ion.density", "f8", ("time", "ion_dim", "rho_dim"))
        v_ion_dens[0, 0, :] = np.full(10, 2.5e19)
        v_ion_dens[0, 1, :] = np.full(10, 2.5e19)
        v_ion_dens[1, 0, :] = np.full(10, 2.5e19)
        v_ion_dens[1, 1, :] = np.full(10, 2.5e19)

        # Create equilibrium IDS group
        eq_grp = ds.createGroup("equilibrium").createGroup("0")
        eq_grp.createDimension("time", 2)
        eq_grp.createDimension("grid_dim", 10)

        v_rout = eq_grp.createVariable("time_slice.profiles_1d.r_outboard", "f8", ("time", "grid_dim"))
        v_rout[:] = np.tile(np.linspace(3.1, 5.0, 10), (2, 1))

        v_rin = eq_grp.createVariable("time_slice.profiles_1d.r_inboard", "f8", ("time", "grid_dim"))
        v_rin[:] = np.tile(np.linspace(3.1, 1.0, 10), (2, 1))

        v_elong = eq_grp.createVariable("time_slice.profiles_1d.elongation", "f8", ("time", "grid_dim"))
        v_elong[:] = np.tile(np.full(10, 1.7), (2, 1))

        v_mag = eq_grp.createVariable("time_slice.global_quantities.magnetic_axis.r", "f8", ("time",))
        v_mag[:] = [3.1, 3.1]

    # Test reading via TokamakSource.from_imas
    src = openmc.TokamakSource.from_imas(nc_file)
    assert isinstance(src, openmc.TokamakSource)
    assert np.isclose(src.major_radius, 300.0)  # R0 = (5.0 + 1.0)/2 = 3.0 m = 300 cm
    assert np.isclose(src.minor_radius, 200.0)  # a = (5.0 - 1.0)/2 = 2.0 m = 200 cm
    assert np.isclose(src.shafranov_shift, 10.0)  # R_mag - R0 = 3.1 - 3.0 = 0.1 m = 10 cm


def test_from_imas_te_and_boundary_fallback():
    """Verify fallback to Te when Ti is absent, and boundary dimensions when r_inboard is absent."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['core_profiles.profiles_1d.0.electrons.density'] = np.full(10, 2e20)
    ods['core_profiles.profiles_1d.0.electrons.temperature'] = np.full(10, 3000.0)

    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['equilibrium.time_slice.0.boundary.geometric_axis.r'] = 0.68
    ods['equilibrium.time_slice.0.boundary.minor_radius'] = 0.22
    ods['equilibrium.time_slice.0.boundary.elongation'] = 1.6
    ods['equilibrium.time_slice.0.boundary.triangularity'] = 0.4
    ods['equilibrium.time_slice.0.global_quantities.magnetic_axis.r'] = 0.70

    with pytest.warns(UserWarning, match='thermal equilibrium T_i = T_e'):
        with pytest.warns(UserWarning, match='Using boundary scalar dimensions'):
            src = openmc.TokamakSource.from_imas(ods)

    assert isinstance(src, openmc.TokamakSource)
    assert np.isclose(src.major_radius, 68.0)
    assert np.isclose(src.minor_radius, 22.0)
    assert np.isclose(src.shafranov_shift, 2.0)


def test_from_imas_missing_dimensions_error():
    """Verify ValueError is raised when equilibrium dimensions cannot be resolved (no ITER default)."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['core_profiles.profiles_1d.0.electrons.density'] = np.full(10, 2e20)
    ods['core_profiles.profiles_1d.0.electrons.temperature'] = np.full(10, 3000.0)
    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = np.linspace(0, 1, 10)

    with pytest.raises(ValueError, match="Unable to determine plasma dimensions"):
        openmc.TokamakSource.from_imas(ods)


def test_load_ods_error_handling(tmp_path):
    """Verify error handling in _load_ods for invalid paths and extensions."""
    with pytest.raises(FileNotFoundError, match="Path not found"):
        openmc.TokamakSource._load_ods(tmp_path / "non_existent_file.nc")

    invalid_ext = tmp_path / "invalid_file.txt"
    invalid_ext.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        openmc.TokamakSource._load_ods(invalid_ext)


def test_find_fuel_ion_densities_named_non_fuel_rejection():
    """Verify that explicitly named non-fuel impurities (e.g. He, Ne) are rejected."""
    dens_he = np.full(10, 1.0e19)
    dens_ne = np.full(10, 2.0e18)

    cp = {
        'ion': [
            {'name': 'He', 'density': dens_he},
            {'name': 'Ne', 'density': dens_ne},
        ]
    }
    with pytest.raises(ValueError, match="could not identify required fuel species"):
        openmc.TokamakSource._find_fuel_ion_densities(cp, reaction='DT')


def test_find_fuel_ion_densities_single_fuel_species():
    """Verify that if D is present but T is missing in DT reaction, it warns and sets nT=0 (not overwritten with ne)."""
    dens_d = np.full(10, 4.0e19)
    dens_he = np.full(10, 1.0e19)
    ne = np.full(10, 6.0e19)

    cp = {
        'ion': [
            {'name': 'D', 'density': dens_d},
            {'name': 'He', 'density': dens_he},
        ],
        'electrons': {'density': ne}
    }
    with pytest.warns(UserWarning, match="Assuming Tritium density is zero"):
        nD, nT = openmc.TokamakSource._find_fuel_ion_densities(cp, reaction='DT')
    np.testing.assert_allclose(nD, dens_d)
    assert np.all(nT == 0.0)


def test_grid_mismatch_equilibrium_core_profiles():
    """Verify that equilibrium and core_profiles with different grid sizes are correctly aligned and values match."""
    # 21 equilibrium grid points with linear mapping r_tilde = rho
    n_eq = 21
    r_in = np.linspace(3.0, 1.0, n_eq)
    r_out = np.linspace(3.0, 5.0, n_eq)
    rho_eq = np.linspace(0.0, 1.0, n_eq)
    elong = np.full(n_eq, 1.6)

    # 11 core_profiles grid points: parabolic temperature T(rho) = 10000 * (1 - rho^2)
    n_cp = 11
    rho_cp = np.linspace(0.0, 1.0, n_cp)
    ti = 10000.0 * (1.0 - rho_cp**2)
    nd = np.full(n_cp, 4.0e19)
    nt = np.full(n_cp, 4.0e19)

    ods = {
        'core_profiles': {
            'profiles_1d': [{
                'grid': {'rho_tor_norm': rho_cp},
                't_i_average': ti,
                'ion': [
                    {'name': 'D', 'density': nd},
                    {'name': 'T', 'density': nt},
                ]
            }]
        },
        'equilibrium': {
            'time_slice': [{
                'profiles_1d': {
                    'r_inboard': r_in,
                    'r_outboard': r_out,
                    'rho_tor_norm': rho_eq,
                    'elongation': elong,
                    'triangularity': np.zeros(n_eq),
                },
                'global_quantities': {
                    'magnetic_axis': {'r': 3.0}
                }
            }]
        }
    }

    data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=51)
    assert len(data['temperature']) == 51
    assert len(data['density_D']) == 51
    assert len(data['elongation']) == 51
    assert len(data['r_tilde_mapped']) == n_cp
    assert np.isclose(data['major_radius'], 300.0)
    assert np.isclose(data['minor_radius'], 200.0)
    # Check value accuracy at r/a = 0.5 (index 25 on 51-point grid): T should be ~7500 eV
    assert np.isclose(data['temperature'][25], 7500.0, rtol=1e-2)


def test_grid_mismatch_rho_cp_terminates_early():
    """Verify that core profiles terminating before rho=1.0 do not get erroneously stretched."""
    n_eq = 21
    r_in = np.linspace(3.0, 1.0, n_eq)
    r_out = np.linspace(3.0, 5.0, n_eq)
    rho_eq = np.linspace(0.0, 1.0, n_eq)

    # Core profiles only extend to rho=0.9
    n_cp = 10
    rho_cp = np.linspace(0.0, 0.9, n_cp)
    ti = np.full(n_cp, 10.0e3)
    nd = np.full(n_cp, 4.0e19)
    nt = np.full(n_cp, 4.0e19)

    ods = {
        'core_profiles': {
            'profiles_1d': [{
                'grid': {'rho_tor_norm': rho_cp},
                't_i_average': ti,
                'ion': [
                    {'name': 'D', 'density': nd},
                    {'name': 'T', 'density': nt},
                ]
            }]
        },
        'equilibrium': {
            'time_slice': [{
                'profiles_1d': {
                    'r_inboard': r_in,
                    'r_outboard': r_out,
                    'rho_tor_norm': rho_eq,
                },
                'global_quantities': {'magnetic_axis': {'r': 3.0}}
            }]
        }
    }

    with pytest.warns(UserWarning, match="Core profiles grid terminates at rho=0.900"):
        data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=51)
    # r_tilde_mapped[-1] should accurately reflect rho=0.9, NOT be clamped to 1.0
    assert np.isclose(data['r_tilde_mapped'][-1], 0.9, atol=1e-2)


def test_elongation_without_r_inboard():
    """Verify that elongation with missing r_inboard/r_outboard falls back gracefully without length mismatch."""
    n_eq = 15
    elong = np.full(n_eq, 1.6)
    n_cp = 10
    ods = {
        'core_profiles': {
            'profiles_1d': [{
                'grid': {'rho_tor_norm': np.linspace(0, 1, n_cp)},
                't_i_average': np.full(n_cp, 10.0e3),
                'ion': [
                    {'name': 'D', 'density': np.full(n_cp, 4.0e19)},
                    {'name': 'T', 'density': np.full(n_cp, 4.0e19)},
                ]
            }]
        },
        'equilibrium': {
            'time_slice': [{
                'profiles_1d': {'elongation': elong},
                'boundary': {
                    'minor_radius': 1.13,
                    'geometric_axis': {'r': 3.3},
                }
            }]
        }
    }

    with pytest.warns(UserWarning, match="No 1D 'r_inboard' and 'r_outboard' profiles found in equilibrium"):
        data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=51)
    assert len(data['elongation']) == 51
    assert np.allclose(data['elongation'], 1.6)
    assert np.isclose(data['a_minor'], 1.13)
    assert np.isclose(data['R0'], 3.3)


def test_invalid_minor_radius_raises():
    """Verify that degenerate boundary radii (a <= 0) raise a ValueError."""
    ods = {
        'core_profiles': {
            'profiles_1d': [{
                't_i_average': np.full(10, 10.0e3),
                'ion': [{'name': 'D', 'density': np.full(10, 4.0e19)}, {'name': 'T', 'density': np.full(10, 4.0e19)}]
            }]
        },
        'equilibrium': {
            'time_slice': [{
                'profiles_1d': {
                    'r_inboard': np.array([3.0, 3.0]),
                    'r_outboard': np.array([3.0, 3.0]),  # r_out - r_in = 0 => a = 0
                }
            }]
        }
    }
    with pytest.raises(ValueError, match="Invalid minor radius"):
        openmc.TokamakSource._extract_and_map_imas_profiles(ods)


def test_empty_optional_fields_handling():
    """Verify that empty optional arrays in IMAS structures do not trigger interpolation errors."""
    n_grid = 15
    r_in = np.linspace(3.1, 1.0, n_grid)
    r_out = np.linspace(3.1, 5.0, n_grid)
    ti = np.full(n_grid, 10.0e3)
    nd = np.full(n_grid, 4.0e19)
    nt = np.full(n_grid, 4.0e19)

    ods = {
        'core_profiles': {
            'profiles_1d': [{
                'grid': {'rho_tor_norm': np.linspace(0.0, 1.0, n_grid)},
                't_i_average': ti,
                'electrons': {'density': np.array([])},  # empty array
                'ion': [
                    {'name': 'D', 'density': nd},
                    {'name': 'T', 'density': nt},
                ]
            }]
        },
        'equilibrium': {
            'time_slice': [{
                'profiles_1d': {
                    'r_inboard': r_in,
                    'r_outboard': r_out,
                    'elongation': np.array([]),  # empty array
                    'triangularity_upper': np.array([]),  # empty array
                    'triangularity_lower': np.array([]),  # empty array
                },
                'global_quantities': {
                    'magnetic_axis': {'r': np.array([])}  # empty array
                }
            }]
        }
    }

    data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=31)
    assert len(data['temperature']) == 31
    assert np.all(data['elongation'] == 1.0)
    assert np.all(data['triangularity'] == 0.0)
    assert data['shafranov_shift'] == 0.0


def test_tokamak_source_ensemble_kl_info_attribute():
    """Verify kl_info attribute is initialized to None and preserved across slicing."""
    ens = openmc.TokamakSourceEnsemble(
        major_radius=300.0,
        minor_radius=100.0,
        elongation=1.5,
        triangularity=0.3,
        r_over_a=np.linspace(0.0, 1.0, 10),
        temperature=np.full((3, 10), 10.0e3),
        density_D=np.full((3, 10), 4.0e19),
        density_T=np.full((3, 10), 4.0e19),
    )
    assert ens.kl_info is None
    sub = ens[0:2]
    assert sub.kl_info is None


def test_tokamak_source_ensemble_single_species_sampling():
    """Verify ensemble sampling and reaction dispatch for single-species (D-only) plasmas."""
    n_samples = 2
    n_points = 10
    r_over_a = np.linspace(0.0, 1.0, n_points)
    temperature = np.full((n_samples, n_points), 1.0e4)
    density_D = np.full((n_samples, n_points), 4.0e19)
    density_T = np.zeros((n_samples, n_points))

    ens = openmc.TokamakSourceEnsemble(
        major_radius=300.0,
        minor_radius=100.0,
        elongation=1.5,
        triangularity=0.3,
        r_over_a=r_over_a,
        temperature=temperature,
        density_D=density_D,
        density_T=density_T,
    )

    # 1. sample() should return only active channels (DD only)
    sources_raw = ens.sample(0, normalize_strength=False)
    assert len(sources_raw) == 1
    assert np.isclose(sources_raw[0].strength, ens.strength_DD[0])

    sources_norm = ens.sample(0, normalize_strength=True)
    assert len(sources_norm) == 1
    assert np.isclose(sources_norm[0].strength, 1.0)

    # 2. Verify zero-density ensemble raises ValueError on sample()
    ens_zero = openmc.TokamakSourceEnsemble(
        major_radius=300.0,
        minor_radius=100.0,
        elongation=1.5,
        triangularity=0.3,
        r_over_a=r_over_a,
        temperature=temperature,
        density_D=np.zeros((n_samples, n_points)),
        density_T=np.zeros((n_samples, n_points)),
    )
    with pytest.raises(ValueError, match="No active fusion reaction channels"):
        ens_zero.sample(0)


def test_from_imas_multislice_hdf5(tmp_path):
    """Verify loading multi-slice native IMAS HDF5 dataset via time_s and time_index."""
    h5py = pytest.importorskip("h5py")

    # Create synthetic multi-slice IMAS HDF5 files
    eq_file = tmp_path / 'equilibrium.h5'
    cp_file = tmp_path / 'core_profiles.h5'

    with h5py.File(eq_file, 'w') as f:
        f.create_dataset('equilibrium/time', data=np.array([0.5, 1.0]))
        f.create_dataset('equilibrium/code&output_flag', data=np.array([0, -1], dtype=np.int32))
        f.create_dataset('equilibrium/time_slice[]&boundary&minor_radius', data=np.array([1.0, 1.1]))
        f.create_dataset('equilibrium/time_slice[]&boundary&geometric_axis&r', data=np.array([3.0, 3.2]))
        f.create_dataset('equilibrium/time_slice[]&boundary&elongation', data=np.array([1.4, 1.6]))
        f.create_dataset('equilibrium/time_slice[]&boundary&triangularity', data=np.array([0.3, 0.4]))
        f.create_dataset('equilibrium/time_slice[]&global_quantities&magnetic_axis&r', data=np.array([3.05, 3.28]))
        f.create_dataset('equilibrium/time_slice[]&profiles_1d&rho_tor_norm', data=np.array([
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 11)
        ]))

    with h5py.File(cp_file, 'w') as f:
        f.create_dataset('core_profiles/time', data=np.array([0.5, 1.0]))
        f.create_dataset('core_profiles/profiles_1d[]&grid&rho_tor_norm', data=np.array([
            np.linspace(0, 1, 11),
            np.linspace(0, 1, 11)
        ]))
        f.create_dataset('core_profiles/profiles_1d[]&electrons&density', data=np.array([
            np.ones(11) * 1e20,
            np.ones(11) * 2e20
        ]))
        f.create_dataset('core_profiles/profiles_1d[]&electrons&temperature', data=np.array([
            np.linspace(5000, 500, 11),
            np.linspace(8000, 800, 11)
        ]))

    # 1. Load slice at t = 1.0 s; verify time_s selection, boundary fallbacks, and output_flag warning
    with pytest.warns(UserWarning) as record:
        source_s1 = openmc.TokamakSource.from_imas(tmp_path, time_s=1.0, reaction='DT')

    messages = [str(w.message) for w in record]
    assert any("boundary scalar dimensions" in m for m in messages)
    assert any("output_flag=-1" in m for m in messages)
    assert any("Assuming thermal equilibrium T_i = T_e" in m for m in messages)
    assert any("Assuming 50:50 D-T plasma" in m for m in messages)

    # In cm: R0=3.2m -> 320cm, a=1.1m -> 110cm
    assert np.isclose(source_s1.major_radius, 320.0)
    assert np.isclose(source_s1.minor_radius, 110.0)
    assert np.isclose(source_s1.elongation[-1], 1.6)
    assert np.isclose(source_s1.triangularity[-1], 0.4)
    assert np.isclose(source_s1.shafranov_shift, 8.0)

    # 2. Loading via exact slice index 1 should match time_s = 1.0 s
    source_idx1 = openmc.TokamakSource.from_imas(tmp_path, time_index=1, reaction='DT')
    assert np.isclose(source_s1.major_radius, source_idx1.major_radius)
    assert np.isclose(source_s1.minor_radius, source_idx1.minor_radius)
    assert np.isclose(source_s1.shafranov_shift, source_idx1.shafranov_shift)

    # 3. Load slice 0 at t = 0.5 s (R0=3.0m -> 300cm, a=1.0m -> 100cm, kappa=1.4)
    source_s0 = openmc.TokamakSource.from_imas(tmp_path, time_s=0.5, reaction='DT')
    assert np.isclose(source_s0.major_radius, 300.0)
    assert np.isclose(source_s0.minor_radius, 100.0)
    assert np.isclose(source_s0.elongation[-1], 1.4)

    # 4. Out-of-bounds time_index should raise IndexError
    with pytest.raises(IndexError, match="exceeds available slices"):
        openmc.TokamakSource.from_imas(tmp_path, time_index=99)

    # 5. TokamakSourceEnsemble.from_imas without covariance should raise ValueError
    with pytest.raises(ValueError, match="No covariance data found in IMAS core_profiles"):
        openmc.TokamakSourceEnsemble.from_imas(tmp_path, time_s=1.0)

    # 6. Loading via time_norm: 0.0 -> t=0.5s (slice 0), 1.0 -> t=1.0s (slice 1)
    source_tn0 = openmc.TokamakSource.from_imas(tmp_path, time_norm=0.0, reaction='DT')
    assert np.isclose(source_tn0.major_radius, source_s0.major_radius)
    assert np.isclose(source_tn0.elongation[-1], 1.4)

    source_tn1 = openmc.TokamakSource.from_imas(tmp_path, time_norm=1.0, reaction='DT')
    assert np.isclose(source_tn1.major_radius, source_s1.major_radius)
    assert np.isclose(source_tn1.elongation[-1], 1.6)

    # 7. Intermediate time_norm values
    source_tn_mid1 = openmc.TokamakSource.from_imas(tmp_path, time_norm=0.8, reaction='DT')
    assert np.isclose(source_tn_mid1.elongation[-1], 1.6)

    source_tn_mid0 = openmc.TokamakSource.from_imas(tmp_path, time_norm=0.2, reaction='DT')
    assert np.isclose(source_tn_mid0.elongation[-1], 1.4)

    # 8. Out-of-bounds time_norm should raise ValueError
    with pytest.raises(ValueError, match="'time_norm' must be a float between 0.0 and 1.0"):
        openmc.TokamakSource.from_imas(tmp_path, time_norm=-0.1)

    with pytest.raises(ValueError, match="'time_norm' must be a float between 0.0 and 1.0"):
        openmc.TokamakSource.from_imas(tmp_path, time_norm=1.05)


def test_from_imas_vertical_shift_autodetect_and_override():
    """Verify automatic vertical_shift extraction and user override."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['core_profiles.profiles_1d.0.electrons.density'] = np.full(10, 2e20)
    ods['core_profiles.profiles_1d.0.electrons.temperature'] = np.full(10, 3000.0)

    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['equilibrium.time_slice.0.boundary.geometric_axis.r'] = 0.68
    ods['equilibrium.time_slice.0.boundary.minor_radius'] = 0.22
    ods['equilibrium.time_slice.0.boundary.elongation'] = 1.6
    ods['equilibrium.time_slice.0.boundary.triangularity'] = 0.4
    ods['equilibrium.time_slice.0.global_quantities.magnetic_axis.r'] = 0.70
    ods['equilibrium.time_slice.0.global_quantities.magnetic_axis.z'] = -0.05

    # 1. Auto-detected vertical shift (in cm: -0.05 m -> -5.0 cm)
    with pytest.warns(UserWarning):
        src_auto = openmc.TokamakSource.from_imas(ods)
    assert np.isclose(src_auto.vertical_shift, -5.0)

    # 2. Explicit float override
    with pytest.warns(UserWarning):
        src_override = openmc.TokamakSource.from_imas(ods, vertical_shift=3.5)
    assert np.isclose(src_override.vertical_shift, 3.5)

    # 3. Explicit zero override
    with pytest.warns(UserWarning):
        src_zero = openmc.TokamakSource.from_imas(ods, vertical_shift=0.0)
    assert np.isclose(src_zero.vertical_shift, 0.0)


def test_from_imas_2d_equilibrium_geometric_mapping():
    """Verify geometric midplane radial mapping extracted from 2D equilibrium grid,
    including square grid layout (R, Z) and non-monotonic psi turnover outside boundary."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = np.linspace(0, 1, 11)
    ods['core_profiles.profiles_1d.0.electrons.density'] = np.full(11, 2e20)
    ods['core_profiles.profiles_1d.0.electrons.temperature'] = np.full(11, 3000.0)

    ods['equilibrium.time_slice.0.global_quantities.magnetic_axis.r'] = 3.0
    ods['equilibrium.time_slice.0.global_quantities.magnetic_axis.z'] = 0.0
    ods['equilibrium.time_slice.0.global_quantities.psi_magnetic_axis'] = 0.0
    ods['equilibrium.time_slice.0.boundary.psi'] = 1.0
    ods['equilibrium.time_slice.0.boundary.geometric_axis.r'] = 3.0
    ods['equilibrium.time_slice.0.boundary.minor_radius'] = 1.0
    ods['equilibrium.time_slice.0.boundary.elongation'] = 1.0
    ods['equilibrium.time_slice.0.boundary.triangularity'] = 0.0

    # Square grid (31 x 31) with IMAS (dim1=R, dim2=Z) layout
    r_grid = np.linspace(1.5, 4.5, 31)
    z_grid = np.linspace(-1.5, 1.5, 31)
    R, Z = np.meshgrid(r_grid, z_grid, indexing='ij')
    # Circular flux surfaces inside plasma, with artificial turnover outside r > 1.2
    psi_2d = (R - 3.0)**2 + Z**2
    psi_2d = np.where(np.abs(R - 3.0) > 1.2, 2 * 1.44 - psi_2d, psi_2d)

    ods['equilibrium.time_slice.0.profiles_2d.0.grid.dim1'] = r_grid
    ods['equilibrium.time_slice.0.profiles_2d.0.grid.dim2'] = z_grid
    ods['equilibrium.time_slice.0.profiles_2d.0.psi'] = psi_2d
    psi_norm = np.linspace(0, 1, 11)
    ods['equilibrium.time_slice.0.profiles_1d.psi_norm'] = psi_norm
    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = np.sqrt(psi_norm)

    data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=11)
    assert np.isclose(data['a_minor'], 1.0, atol=1e-2)
    assert np.isclose(data['major_radius'], 300.0, atol=1e-1)
    assert data['r_inboard'] is not None and data['r_outboard'] is not None

    # For circular flux surfaces: R_out - R_axis = a * sqrt(psi_N), R_axis - R_in = a * sqrt(psi_N)
    # In cm: R_axis = 300 cm, a = 100 cm
    expected_r_geom = 100.0 * np.sqrt(psi_norm)
    actual_r_geom = 0.5 * (data['r_outboard'] - data['r_inboard'])
    assert np.allclose(actual_r_geom, expected_r_geom, atol=1.5)
    assert np.all(np.diff(actual_r_geom) > 0)


def test_from_imas_independent_time_bases(tmp_path):
    """Verify independent time-slice selection when equilibrium and core_profiles have different time grids."""
    h5py = pytest.importorskip("h5py")
    eq_file = tmp_path / 'equilibrium.h5'
    cp_file = tmp_path / 'core_profiles.h5'

    # Equilibrium has 4 slices at [0.0, 0.5, 1.0, 1.5]
    with h5py.File(eq_file, 'w') as f:
        f.create_dataset('equilibrium/time', data=np.array([0.0, 0.5, 1.0, 1.5]))
        f.create_dataset('equilibrium/time_slice[]&boundary&minor_radius', data=np.array([1.0, 1.1, 1.2, 1.3]))
        f.create_dataset('equilibrium/time_slice[]&boundary&geometric_axis&r', data=np.array([3.0, 3.1, 3.2, 3.3]))
        f.create_dataset('equilibrium/time_slice[]&global_quantities&magnetic_axis&r', data=np.array([3.0, 3.1, 3.2, 3.3]))
        f.create_dataset('equilibrium/time_slice[]&profiles_1d&rho_tor_norm', data=np.linspace(0, 1, 11)[None, :].repeat(4, axis=0))

    # Core profiles has 3 slices at [0.0, 0.9, 2.0] with distinct temperatures
    with h5py.File(cp_file, 'w') as f:
        f.create_dataset('core_profiles/time', data=np.array([0.0, 0.9, 2.0]))
        f.create_dataset('core_profiles/profiles_1d[]&grid&rho_tor_norm', data=np.linspace(0, 1, 11)[None, :].repeat(3, axis=0))
        f.create_dataset('core_profiles/profiles_1d[]&electrons&density', data=np.array([
            np.full(11, 1e20),
            np.full(11, 2e20),
            np.full(11, 3e20)
        ]))
        f.create_dataset('core_profiles/profiles_1d[]&electrons&temperature', data=np.array([
            np.full(11, 1000.0),
            np.full(11, 2500.0),
            np.full(11, 4000.0)
        ]))

    # Target t = 1.0 s: equilibrium should pick idx 2 (t=1.0, a=1.2m), core_profiles should pick idx 1 (t=0.9, Te=2500eV)
    ods = openmc.TokamakSource._load_ods(tmp_path, time_s=1.0)
    assert np.isclose(ods['core_profiles.profiles_1d.0.electrons.temperature'][0], 2500.0)
    assert np.isclose(ods['equilibrium.time_slice.0.boundary.minor_radius'], 1.2)

    src = openmc.TokamakSource.from_imas(tmp_path, time_s=1.0, reaction='DT')
    assert np.isclose(src.minor_radius, 120.0)
    assert np.isclose(src.major_radius, 320.0)


def test_from_imas_time_warnings():
    """Verify warnings when time parameters are passed to in-memory ODS or when multiple time parameters are given."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['core_profiles.profiles_1d.0.electrons.density'] = np.full(10, 2e20)
    ods['core_profiles.profiles_1d.0.electrons.temperature'] = np.full(10, 3000.0)
    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['equilibrium.time_slice.0.boundary.geometric_axis.r'] = 0.68
    ods['equilibrium.time_slice.0.boundary.minor_radius'] = 0.22

    with pytest.warns(UserWarning) as record:
        openmc.TokamakSource.from_imas(ods, time_s=1.0)
    messages = [str(w.message) for w in record]
    assert any("ignored when imas_input is an in-memory ODS or dict" in m for m in messages)

    with pytest.warns(UserWarning) as record:
        openmc.TokamakSource.from_imas(ods, time_norm=0.5)
    messages = [str(w.message) for w in record]
    assert any("ignored when imas_input is an in-memory ODS or dict" in m for m in messages)

    with pytest.warns(UserWarning) as record:
        openmc.TokamakSource._load_ods(ods, time_index=0, time_s=1.0)
    messages = [str(w.message) for w in record]
    assert any("Both 'time_index' and 'time_s' were provided" in m for m in messages)

    with pytest.warns(UserWarning) as record:
        openmc.TokamakSource._load_ods(ods, time_index=0, time_norm=0.5)
    messages = [str(w.message) for w in record]
    assert any("Both 'time_index' and 'time_norm' were provided" in m for m in messages)

    with pytest.warns(UserWarning) as record:
        openmc.TokamakSource._load_ods(ods, time_s=1.0, time_norm=0.5)
    messages = [str(w.message) for w in record]
    assert any("Both 'time_s' and 'time_norm' were provided" in m for m in messages)


def test_from_imas_time_norm_non_uniform_slices(tmp_path):
    """Verify that time_norm selects slices proportionally to physical time, not slice index."""
    h5py = pytest.importorskip("h5py")

    # Time grid: 5 slices closely spaced in [0.0, 0.4], 1 slice at 10.0 s
    # Total duration: 0.0 to 10.0 s
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 10.0])
    kappas = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 2.0])

    eq_file = tmp_path / 'equilibrium.h5'
    cp_file = tmp_path / 'core_profiles.h5'

    with h5py.File(eq_file, 'w') as f:
        f.create_dataset('equilibrium/time', data=times)
        f.create_dataset('equilibrium/time_slice[]&boundary&minor_radius', data=np.full(len(times), 1.0))
        f.create_dataset('equilibrium/time_slice[]&boundary&geometric_axis&r', data=np.full(len(times), 3.0))
        f.create_dataset('equilibrium/time_slice[]&boundary&elongation', data=kappas)
        f.create_dataset('equilibrium/time_slice[]&boundary&triangularity', data=np.full(len(times), 0.3))
        f.create_dataset('equilibrium/time_slice[]&profiles_1d&rho_tor_norm', data=np.tile(np.linspace(0, 1, 11), (len(times), 1)))

    with h5py.File(cp_file, 'w') as f:
        f.create_dataset('core_profiles/time', data=times)
        f.create_dataset('core_profiles/profiles_1d[]&grid&rho_tor_norm', data=np.tile(np.linspace(0, 1, 11), (len(times), 1)))
        f.create_dataset('core_profiles/profiles_1d[]&electrons&density', data=np.full((len(times), 11), 1e20))
        f.create_dataset('core_profiles/profiles_1d[]&electrons&temperature', data=np.full((len(times), 11), 5000.0))

    # time_norm = 0.6 -> t_target = 0.0 + 0.6 * (10.0 - 0.0) = 6.0 s.
    # Closest slice to 6.0 s is slice index 5 (t = 10.0 s, dist = 4.0), NOT slice index 3 or 4.
    src_prop = openmc.TokamakSource.from_imas(tmp_path, time_norm=0.6, reaction='DT')
    assert np.isclose(src_prop.elongation[-1], 2.0)

    # time_norm = 0.02 -> t_target = 0.2 s -> matches slice index 2 (t = 0.2 s, kappa = 1.3)
    src_early = openmc.TokamakSource.from_imas(tmp_path, time_norm=0.02, reaction='DT')
    assert np.isclose(src_early.elongation[-1], 1.3)


def test_dd_neutron_branching_and_tt_multiplicity():
    """Verify DD reactivity uses the neutron-producing branch DD_n and TT emissivity includes 2n multiplicity."""
    T_i = np.array([10.0e3, 20.0e3])
    nD = np.array([5.0e19, 5.0e19])
    nT = np.array([5.0e19, 5.0e19])

    sv_dd = openmc.TokamakSource._bosch_hale_reactivity(T_i, 'DD')
    sv_dd_n = openmc.TokamakSource._bosch_hale_reactivity(T_i, 'DD_n')
    sv_dd_p = openmc.TokamakSource._bosch_hale_reactivity(T_i, 'DD_p')
    sv_dd_tot = openmc.TokamakSource._bosch_hale_reactivity(T_i, 'DD_total')
    sv_tt = openmc.TokamakSource._bosch_hale_reactivity(T_i, 'TT')

    # DD neutron reactivity equals DD_n (not DD_n + DD_p)
    np.testing.assert_allclose(sv_dd, sv_dd_n)
    np.testing.assert_allclose(sv_dd_tot, sv_dd_n + sv_dd_p)
    assert np.all(sv_dd_tot > sv_dd)

    # Emissivities
    em_dd = openmc.TokamakSource._calculate_fusion_emissivity(nD, nT, T_i, reaction='DD')
    em_dd_n = openmc.TokamakSource._calculate_fusion_emissivity(nD, nT, T_i, reaction='DD_n')
    np.testing.assert_allclose(em_dd, em_dd_n)
    np.testing.assert_allclose(em_dd, 0.5 * (nD**2) * sv_dd_n)

    # TT produces 2 neutrons per reaction: 2 * 0.5 * n_T^2 * <sigma*v>_TT = n_T^2 * <sigma*v>_TT
    em_tt = openmc.TokamakSource._calculate_fusion_emissivity(nD, nT, T_i, reaction='TT')
    np.testing.assert_allclose(em_tt, (nT**2) * sv_tt)


def test_from_imas_single_species_error():
    """Verify that requesting DT reaction on a pure-D plasma raises a clear ValueError in from_imas."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = np.linspace(0, 1, 10)
    ods['core_profiles.profiles_1d.0.t_i_average'] = np.full(10, 5000.0)
    ods['core_profiles.profiles_1d.0.ion.0.name'] = 'D'
    ods['core_profiles.profiles_1d.0.ion.0.density'] = np.full(10, 1.0e20)

    ods['equilibrium.time_slice.0.profiles_1d.r_inboard'] = np.linspace(2.0, 1.0, 10)
    ods['equilibrium.time_slice.0.profiles_1d.r_outboard'] = np.linspace(2.0, 3.0, 10)
    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = np.linspace(0, 1, 10)

    # 1. Default reaction='DT' on pure-D plasma raises descriptive ValueError
    with pytest.warns(UserWarning, match="Assuming Tritium density is zero"):
        with pytest.raises(ValueError, match="Cannot construct a 'DT' TokamakSource because Tritium density is zero"):
            openmc.TokamakSource.from_imas(ods, reaction='DT')

    # 2. Specifying reaction='DD' succeeds cleanly
    src_dd = openmc.TokamakSource.from_imas(ods, reaction='DD')
    assert isinstance(src_dd, openmc.TokamakSource)
    assert np.any(src_dd.emission_density > 0.0)


def test_from_imas_species_ion_temperature_fallback():
    """Verify fallback to ion[].temperature when t_i_average is absent."""
    from omas import ODS

    ods = ODS(consistency_check=False)
    rho = np.linspace(0.0, 1.0, 11)

    ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = rho
    ods['core_profiles.profiles_1d.0.ion.0.name'] = 'D'
    ods['core_profiles.profiles_1d.0.ion.0.density'] = np.full(11, 5.0e19)
    ods['core_profiles.profiles_1d.0.ion.0.temperature'] = 8000.0 * (1.0 - rho**2) + 500.0
    ods['core_profiles.profiles_1d.0.ion.1.name'] = 'T'
    ods['core_profiles.profiles_1d.0.ion.1.density'] = np.full(11, 5.0e19)
    ods['core_profiles.profiles_1d.0.electrons.temperature'] = np.full(11, 2000.0)

    ods['equilibrium.time_slice.0.profiles_1d.r_inboard'] = np.linspace(3.0, 2.0, 11)
    ods['equilibrium.time_slice.0.profiles_1d.r_outboard'] = np.linspace(3.0, 4.0, 11)
    ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = rho

    with pytest.warns(UserWarning, match="using species ion temperature from 'ion\\[\\].temperature'"):
        data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=11)

    assert np.isclose(data['temperature'][0], 8500.0)


def test_tokamak_source_xml_roundtrip_method_and_primes():
    """Verify XML roundtrip preserves method ('bernstein' vs 'fourier') and derivative profiles."""
    r_over_a = np.linspace(0.0, 1.0, 11)
    emission = np.maximum(0.0, 1.0 - r_over_a**2)
    energy = openmc.stats.Discrete([14.08e6], [1.0])

    # 1. Bernstein scalar source roundtrip
    src_bern = openmc.TokamakSource(
        major_radius=620.0,
        minor_radius=200.0,
        elongation=1.7,
        triangularity=0.33,
        shafranov_shift=10.0,
        r_over_a=r_over_a,
        emission_density=emission,
        energy=energy,
        method='bernstein',
    )
    elem_bern = src_bern.to_xml_element()
    assert elem_bern.find('elongation_prime') is None
    src_bern_rt = openmc.TokamakSource.from_xml_element(elem_bern)
    assert src_bern_rt.method == 'bernstein'
    assert isinstance(src_bern_rt.elongation, float)
    assert isinstance(src_bern_rt.triangularity, float)

    # 2. Fourier profile source roundtrip
    kappa_prof = 1.4 + 0.4 * r_over_a**2
    delta_prof = 0.35 * r_over_a**2
    src_four = openmc.TokamakSource(
        major_radius=620.0,
        minor_radius=200.0,
        elongation=kappa_prof,
        triangularity=delta_prof,
        shafranov_shift=10.0,
        r_over_a=r_over_a,
        emission_density=emission,
        energy=energy,
        method='fourier',
    )
    elem_four = src_four.to_xml_element()
    assert elem_four.find('elongation_prime') is not None
    src_four_rt = openmc.TokamakSource.from_xml_element(elem_four)
    assert src_four_rt.method == 'fourier'
    np.testing.assert_allclose(src_four_rt._kappa_prime, src_four._kappa_prime)
    np.testing.assert_allclose(src_four_rt._delta_prime, src_four._delta_prime)


def test_tokamak_source_non_positive_jacobian():
    """Verify non-positive radial flux-surface Jacobian raises ValueError."""
    r_over_a = np.linspace(0.0, 1.0, 21)
    emission = np.maximum(0.0, 1.0 - r_over_a**2)
    energy = openmc.stats.Discrete([14.08e6], [1.0])
    # Steeply decreasing elongation where kappa + r_tilde * kappa' <= 0 near edge
    bad_elongation = 1.8 - 1.2 * r_over_a**2

    with pytest.raises(ValueError, match="Non-positive vertical Jacobian"):
        openmc.TokamakSource(
            major_radius=620.0,
            minor_radius=200.0,
            elongation=bad_elongation,
            triangularity=0.3,
            shafranov_shift=0.0,
            r_over_a=r_over_a,
            emission_density=emission,
            energy=energy,
            method='fourier',
        )


def test_integrated_source_strength_profile_derivatives():
    """Verify _calculate_integrated_source_strength matches 2D numerical quadrature when kappa' and delta' != 0."""
    from scipy.interpolate import CubicSpline

    r_tilde = np.linspace(0.0, 1.0, 201)
    R0_m = 6.2
    a_m = 2.0
    shift_m = 0.15
    kappa = 1.3 + 0.6 * r_tilde**2
    delta = 0.45 * r_tilde**2
    S_profile = 1.0e18 * (1.0 - r_tilde**2) ** 2

    strength_analytic = openmc.TokamakSourceEnsemble._calculate_integrated_source_strength(
        S_profile=S_profile,
        r_over_a=r_tilde,
        major_radius=R0_m * 100.0,
        minor_radius=a_m * 100.0,
        elongation=kappa,
        triangularity=delta,
        shafranov_shift=shift_m * 100.0,
    )

    # Direct 2D numerical quadrature in (r_tilde, alpha)
    kp = CubicSpline(r_tilde, kappa, bc_type=((1, 0.0), 'not-a-knot')).derivative()(r_tilde)
    dp = CubicSpline(r_tilde, delta, bc_type=((1, 0.0), 'not-a-knot')).derivative()(r_tilde)
    alpha = np.linspace(0.0, 2.0 * np.pi, 401)
    rr, aa = np.meshgrid(r_tilde, alpha, indexing='ij')
    kk = kappa[:, np.newaxis]
    dd = delta[:, np.newaxis]
    kkp = kp[:, np.newaxis]
    ddp = dp[:, np.newaxis]

    R_grid = R0_m + shift_m * (1.0 - rr**2) + a_m * rr * np.cos(aa + dd * np.sin(aa))
    dR_dr = -2.0 * shift_m * rr + a_m * np.cos(aa + dd * np.sin(aa)) - a_m * rr * ddp * np.sin(aa) * np.sin(aa + dd * np.sin(aa))
    dR_da = -a_m * rr * (1.0 + dd * np.cos(aa)) * np.sin(aa + dd * np.sin(aa))
    dz_dr = a_m * (kk + rr * kkp) * np.sin(aa)
    dz_da = a_m * rr * kk * np.cos(aa)
    J_pol = np.abs(dR_dr * dz_da - dR_da * dz_dr)

    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    dV_dr_num = 2.0 * np.pi * trapz_fn(R_grid * J_pol, alpha, axis=1)
    strength_num = float(trapz_fn(S_profile * dV_dr_num, r_tilde))

    assert np.isclose(strength_analytic, strength_num, rtol=1e-4)


def test_from_imas_ensemble_density_correlation_and_cross_covariance():
    """Verify TokamakSourceEnsemble.from_imas preserves D/T correlation (quasi-neutrality) and (T_i, n_e) cross-covariance."""
    from omas import ODS

    n_grid = 11
    rho = np.linspace(0.0, 1.0, n_grid)

    def make_ods(cross_corr: float = 0.0):
        ods = ODS(consistency_check=False)
        ods['core_profiles.profiles_1d.0.grid.rho_tor_norm'] = rho
        T_prof = 15000.0 * (1.0 - 0.8 * rho**2)
        ne_prof = 1.0e20 * (1.0 - 0.5 * rho**2)
        ods['core_profiles.profiles_1d.0.t_i_average'] = T_prof
        ods['core_profiles.profiles_1d.0.electrons.density'] = ne_prof
        ods['core_profiles.profiles_1d.0.ion.0.name'] = 'D'
        ods['core_profiles.profiles_1d.0.ion.0.density'] = 0.5 * ne_prof
        ods['core_profiles.profiles_1d.0.ion.1.name'] = 'T'
        ods['core_profiles.profiles_1d.0.ion.1.density'] = 0.5 * ne_prof

        ods['equilibrium.time_slice.0.profiles_1d.r_inboard'] = 6.2 - 2.0 * rho
        ods['equilibrium.time_slice.0.profiles_1d.r_outboard'] = 6.2 + 2.0 * rho
        ods['equilibrium.time_slice.0.profiles_1d.rho_tor_norm'] = rho

        # Construct smooth spatial kernel covariance for (T_i, n_e)
        dist_sq = (rho[:, None] - rho[None, :]) ** 2
        K = np.exp(-dist_sq / (2.0 * 0.25**2))
        sigma_T = 0.10 * T_prof
        sigma_ne = 0.10 * ne_prof
        cov_T = np.outer(sigma_T, sigma_T) * K
        cov_ne = np.outer(sigma_ne, sigma_ne) * K
        cov_T_ne = cross_corr * np.outer(sigma_T, sigma_ne) * K

        full_cov = np.block([[cov_T, cov_T_ne], [cov_T_ne.T, cov_ne]])
        rows_uri = (
            [f"core_profiles.profiles_1d[0].t_i_average[{i}]" for i in range(n_grid)]
            + [f"core_profiles.profiles_1d[0].electrons.density[{i}]" for i in range(n_grid)]
        )
        ods['core_profiles.covariance.data'] = full_cov
        ods['core_profiles.covariance.rows_uri'] = rows_uri
        return ods

    # 1. Block-diagonal covariance: verify D and T perturbations are perfectly correlated (quasi-neutrality)
    ods_diag = make_ods(cross_corr=0.0)
    ens_diag = openmc.TokamakSourceEnsemble.from_imas(
        ods_diag, n_samples=500, kl_components=0.999, n_points=n_grid, random_seed=42
    )
    corr_nD_nT_axis = np.corrcoef(ens_diag.density_D[:, 0], ens_diag.density_T[:, 0])[0, 1]
    assert corr_nD_nT_axis > 0.999
    # Verify Var(n_D + n_T) == 4 * Var(n_D) == Var(n_e)
    var_sum = np.var(ens_diag.density_D[:, 0] + ens_diag.density_T[:, 0], ddof=1)
    var_nD = np.var(ens_diag.density_D[:, 0], ddof=1)
    assert np.isclose(var_sum, 4.0 * var_nD, rtol=1e-5)

    # 2. Cross-correlated (T_i, n_e) covariance: verify joint KL expansion preserves T_i - n_D correlation
    ods_corr = make_ods(cross_corr=0.8)
    ens_corr = openmc.TokamakSourceEnsemble.from_imas(
        ods_corr, n_samples=1000, kl_components=0.999, n_points=n_grid, random_seed=42
    )
    corr_T_nD_axis = np.corrcoef(ens_corr.temperature[:, 0], ens_corr.density_D[:, 0])[0, 1]
    assert np.isclose(corr_T_nD_axis, 0.8, atol=0.08)


