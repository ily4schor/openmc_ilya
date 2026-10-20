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

    # Test reaction sampling
    src_dt = ensemble.sample_reaction(0, reaction='DT')
    assert isinstance(src_dt, openmc.TokamakSource)
    assert src_dt.strength > 0

    src_tot = ensemble.sample_reaction(0, reaction='total')
    assert isinstance(src_tot, openmc.TokamakSource)

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
            }]
        }
    }

    with pytest.warns(UserWarning, match="No 'r_inboard' and 'r_outboard' found in equilibrium"):
        data = openmc.TokamakSource._extract_and_map_imas_profiles(ods, n_points=51)
    assert len(data['elongation']) == 51
    assert np.allclose(data['elongation'], 1.6)


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

    # 2. sample_reaction('DD') and 'total' should succeed
    src_dd = ens.sample_reaction(0, reaction='DD')
    assert isinstance(src_dd, openmc.TokamakSource)
    assert np.isclose(src_dd.strength, ens.strength_DD[0])

    src_tot = ens.sample_reaction(0, reaction='total')
    assert isinstance(src_tot, openmc.TokamakSource)
    assert np.isclose(src_tot.strength, ens.strength_DD[0])

    # 3. Inactive reaction 'DT' should raise a clear ValueError
    with pytest.raises(ValueError, match="Cannot sample 'DT' reaction"):
        ens.sample_reaction(0, reaction='DT')

    # 4. Inactive reaction 'TT' should raise a clear ValueError
    with pytest.raises(ValueError, match="Cannot sample 'TT' reaction"):
        ens.sample_reaction(0, reaction='TT')

