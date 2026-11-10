"""Curated, nested vegetation-index presets for the wheat hyperspectral pipeline.

All names refer to formulas registered in INDEX_INFO, curated using the original GitHub repository and published band formulas.
Minimal, medium and maximum are nested; maximum avoids redundant aliases.
"""
from __future__ import annotations

from calculate_vegetation_indices import INDEX_INFO

# Distinct physical/biochemical questions, not necessarily independent statistics.
MINIMAL = (
    'NDVI',        # canopy greenness / NIR-red contrast
    'NDRE',        # red-edge chlorophyll sensitivity
    'PRI',         # original 531-nm photochemical reflectance
    'PSRI',        # senescence / chlorophyll-to-carotenoid balance
    'ARI2',        # anthocyanin-related pigmentation
    'CRI2',        # carotenoid-sensitive, calibrated reflectance required
    'WBI',         # water absorption near 970 nm (edge of 400-1000 sensor)
    'OSAVI',       # vegetation contrast with soil/background compensation
)

# Adds nonidentical spectral contrasts and additional pigment/canopy diagnostics.
MEDIUM_EXTRAS = (
    'GNDVI', 'CIgreen', 'CIrededge', 'NDVI705', 'MTCI',
    'SIPI', 'REP', 'mND705', 'TCARI_OSAVI', 'MSAVI',
)

MEDIUM = MINIMAL + MEDIUM_EXTRAS
MAXIMUM = tuple(INDEX_INFO)
PRESETS = {'minimal': MINIMAL, 'medium': MEDIUM, 'maximum': MAXIMUM}

# One-sentence interpretation for an auditable preset catalogue.
REASONS = {
    'NDVI': 'General vegetation greenness and structural NIR/red contrast',
    'NDRE': 'Red-edge sensitivity to chlorophyll when red absorption saturates',
    'PRI': 'Original PRI 531/570: photochemistry-related pigment variation',
    'PSRI': 'Senescence and relative chlorophyll/carotenoid changes',
    'ARI2': 'Anthocyanin-related coloration',
    'CRI2': 'Carotenoid-sensitive inverse reflectance contrast',
    'WBI': 'Original water index R970/R900; check 970-nm signal-to-noise',
    'OSAVI': 'Partial correction for visible substrate/soil background',
    'GNDVI': 'Green-band chlorophyll variation complementary to NDVI',
    'CIgreen': 'Green/NIR chlorophyll-sensitive ratio; pigment interference possible',
    'CIrededge': 'NIR-to-red-edge chlorophyll index',
    'NDVI705': 'Localized red-edge normalized difference, unlike NDRE band pair',
    'MTCI': 'Ratio of red-edge spectral slopes',
    'SIPI': 'Carotenoid/chlorophyll balance with reduced structure sensitivity',
    'REP': 'Red-edge position in nanometres rather than an index ratio',
    'mND705': 'Blue-adjusted chlorophyll red-edge contrast',
    'TCARI_OSAVI': 'Chlorophyll absorption adjusted by soil-sensitive index',
    'MSAVI': 'Alternative adaptive soil-background compensation',
    'NPCI': 'Blue/red pigment contrast (R680 and R430); validate blue SNR',
    'NPQI': 'Phaeophytinization-related contrast (415/435); interpret cautiously with 10-nm samples',
}


def resolve_indices(preset='medium', indices=None):
    """Return ordered, validated index names.

    `indices` explicitly overrides the preset. `preset` accepts minimal,
    medium, maximum (aliases min/default/all) or an iterable of names.
    Unknown names raise ValueError rather than silently being skipped.
    """
    if indices is not None:
        names = list(indices) if not isinstance(indices, str) else [indices]
    elif isinstance(preset, str):
        key = {'min':'minimal','default':'medium','all':'maximum','max':'maximum'}.get(
            preset.strip().lower(), preset.strip().lower())
        if key not in PRESETS:
            raise ValueError(f'Unknown preset {preset!r}; choose minimal, medium, maximum')
        names = list(PRESETS[key])
    else:
        names = list(preset)
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError('Selected indices must not be empty')
    unknown = [n for n in names if n not in INDEX_INFO]
    if unknown:
        raise ValueError(f'Unknown vegetation indices: {unknown}. Available: {list(INDEX_INFO)}')
    return names


def catalog_rows():
    """Document tier, intended meaning, and exact current implemented expression."""
    for name, (description, formula, bands, signal_ok) in INDEX_INFO.items():
        tier = 'minimal' if name in MINIMAL else 'medium' if name in MEDIUM else 'maximum'
        yield dict(index=name,first_included=tier,description=description,
                   formula=formula,required_nm=','.join(map(str,bands)),
                   scale_invariant=bool(signal_ok),reason=REASONS.get(name,
                   'Additional/alternative index; use for targeted exploratory comparisons'))


def validate_presets():
    # Extra verified formulas can later be registered without changing these presets.
    assert set(MINIMAL).issubset(MEDIUM)
    assert set(MEDIUM).issubset(MAXIMUM)
    assert set(MAXIMUM)==set(INDEX_INFO)
    assert len(MAXIMUM)==len(set(MAXIMUM))

validate_presets()

# Names emitted by previous releases but deliberately excluded from this
# catalogue. Only delete files bearing these known pipeline-generated names.
OBSOLETE_INDEX_NAMES = frozenset({
    'NDVI_670','PRI570_530','SR','SR705','ARI1','MSR705','NRI',
    'CARI_legacy','GM1','NPQI_420_440','SRPI','VREI2',
    'CIrededge_840_720','RENDVI_750_710','SIPI_450','WBI_970_900',
})

# Fingerprint the published formulas and their band centers. Results produced
# by older catalogs must not be reused under a new set of equations.
import hashlib as _hashlib
import json as _json
FORMULA_FINGERPRINT = _hashlib.sha256(_json.dumps(
    [(k,entry[1],list(entry[2]),bool(entry[3])) for k,entry in INDEX_INFO.items()],
    ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]
