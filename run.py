"""Dump classical phenotype data from MySQL into a row-major LMDB matrix.

Usage (inside a guix shell with the required deps):

    # dump phenotypes
    python run.py dump-phenotypes \
        "mysql://webqtlout:webqtlout@localhost/db_webqtl" \
        /home/kabui/test_lmdb_data

    # print the stored matrix
    python run.py print-phenotype-matrix \
        /home/kabui/test_lmdb_data

    # round-trip sanity check (dump → read → compare)
    python dump_phenos_matrix.py sanity-check \
        "mysql://webqtlout:webqtlout@localhost/db_webqtl" \
        /home/kabui/test_lmdb_data
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import click
import lmdb
import mysql.connector
import numpy as np
from numpy.testing import assert_array_equal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DTYPE: str = "<f8"                          # little-endian float64
MISSING_VALUE: str = "X"                    # placeholder for absent strains
DEFAULT_PORT: int = 3306
LMDB_MAP_SIZE: int = 100 * 1024 * 1024 * 1024  # 100 GB

FETCH_PHENOTYPES_QUERY: str = """
    SELECT PublishXRef.Id, Strain.Name, PublishData.Value, PublishSE.error
    FROM PublishData
    INNER JOIN Strain        ON PublishData.StrainId        = Strain.Id
    INNER JOIN PublishXRef   ON PublishData.Id               = PublishXRef.DataId
    INNER JOIN PublishFreeze ON PublishXRef.InbredSetId      = PublishFreeze.InbredSetId
    LEFT  JOIN PublishSE     ON PublishSE.DataId             = PublishData.Id
                            AND PublishSE.StrainId          = PublishData.StrainId
    LEFT  JOIN NStrain       ON NStrain.DataId               = PublishData.Id
                            AND NStrain.StrainId            = PublishData.StrainId
    WHERE  PublishFreeze.Name            = %s
      AND  PublishFreeze.public          > 0
      AND  PublishData.value             IS NOT NULL
      AND  PublishFreeze.confidentiality < 1
    ORDER BY LENGTH(Strain.Name), Strain.Name
"""


@dataclass(frozen=True)
class DBConnectionParams:
    """Parsed connection parameters — never printed in full."""
    host: str
    user: str
    password: str
    database: str
    port: int = DEFAULT_PORT


@dataclass(frozen=True)
class PhenoMatrix:
    """Immutable container for the phenotype matrix and its labels.

    Rows = traits, columns = strains (row-major / C-order in memory).
    """
    matrix: np.ndarray          # shape (n_traits, n_strains), dtype <f8
    traits: list[str]
    strains: list[str]
    se_matrix: Optional[np.ndarray] = None  # optional SE values, same shape


@dataclass(frozen=True)
class SerializedPhenoMatrix:
    """Byte-level representation ready to be written to LMDB."""
    matrix_bytes: bytes
    metadata_bytes: bytes
    se_matrix_bytes: Optional[bytes] = None


def parse_connection_params(sql_uri: str, port: int = DEFAULT_PORT) -> DBConnectionParams:
    """Parse a mysql:// URI into connection parameters.
    """
    parsed = urlparse(sql_uri)
    return DBConnectionParams(
        host=parsed.hostname or "localhost",
        user=parsed.username or "",
        password=parsed.password or "",
        database=parsed.path.lstrip("/"),
        port=port,
    )


def open_db_connection(params: DBConnectionParams) -> mysql.connector.MySQLConnection:
    """Open a MySQL connection from parsed parameters.
    """
    logger.info("Connecting to %s@%s/%s", params.user,
                params.host, params.database)
    return mysql.connector.connect(
        host=params.host,
        user=params.user,
        password=params.password,
        database=params.database,
        port=params.port,
    )


def open_lmdb(db_path: str, *, create: bool = False) -> lmdb.Environment:
    """Open (or create) an LMDB environment."""
    return lmdb.open(db_path, map_size=LMDB_MAP_SIZE, create=create)


def fetch_phenotypes(
    dataset_name: str, sql_uri: str
) -> tuple[set[str], dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Query MySQL for all phenotype values and SE belonging to *dataset_name*.

    Returns:
        strains  – set of all strain names encountered.
        datasets – {trait_id: {strain_name: value}}.
        se_data  – {trait_id: {strain_name: se_value}}.
    """
    params = parse_connection_params(sql_uri)
    datasets: dict[int, dict[str, float]] = defaultdict(dict)
    se_data: dict[int, dict[str, float]] = defaultdict(dict)
    strains: set[str] = set()

    with open_db_connection(params) as conn:
        cursor = conn.cursor()
        cursor.execute(FETCH_PHENOTYPES_QUERY, (dataset_name,))
        for trait_id, strain_name, value, se_error in cursor.fetchall():
            datasets[trait_id][strain_name] = value
            if se_error is not None:
                se_data[trait_id][strain_name] = se_error
            strains.add(strain_name)

    return strains, dict(datasets), dict(se_data)


def _to_float(value) -> float:
    """Cast a single value to float64, defaulting to NaN on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _fill_missing(
    trait_data: dict[str, float],
    strains: list[str],
    default: str = MISSING_VALUE,
) -> list[float]:
    """Return one row (all strains) for a trait, filling gaps with *default*.

    The default is passed through _to_float so missing values become NaN.
    """
    return [_to_float(trait_data.get(s, default)) for s in strains]


def build_pheno_matrix(
    datasets: dict[int, dict[str, float]],
    se_data: dict[int, dict[str, float]],
    strains: list[str],
) -> PhenoMatrix:
    """Convert the raw dataset dict into a PhenoMatrix with optional SE values.
    """
    sorted_strains = sorted(
        strains)                          # deterministic column order
    traits = list(datasets.keys())

    # Build value matrix
    value_rows = [_fill_missing(datasets[t], sorted_strains) for t in traits]

    # Build SE matrix if we have SE data
    se_matrix = None
    if se_data:
        se_rows = [_fill_missing(se_data.get(t, {}), sorted_strains)
                   for t in traits]
        se_matrix = np.array(se_rows, dtype=DTYPE)

    return PhenoMatrix(
        matrix=np.array(value_rows, dtype=DTYPE),
        traits=[str(t) for t in traits],
        strains=sorted_strains,
        se_matrix=se_matrix,
    )


def build_metadata(pheno: PhenoMatrix) -> dict:
    """Derive metadata purely from the PhenoMatrix.
    dtype is read from the array itself so it can never drift out of sync.
    """
    rows, columns = pheno.matrix.shape
    return {
        "rows": rows,
        "columns": columns,
        "dtype": str(pheno.matrix.dtype),
        "order": "C",
        "endian": "little",
        "traits": pheno.traits,
        "strains": pheno.strains,
        "transposed": False,
        "has_se": pheno.se_matrix is not None,
    }


def serialize(pheno: PhenoMatrix) -> SerializedPhenoMatrix:
    """Serialize a PhenoMatrix into the byte-blobs LMDB will store.
    """
    se_bytes = None
    if pheno.se_matrix is not None:
        se_bytes = pheno.se_matrix.tobytes(order="C")

    return SerializedPhenoMatrix(
        matrix_bytes=pheno.matrix.tobytes(order="C"),
        metadata_bytes=json.dumps(build_metadata(pheno)).encode("utf-8"),
        se_matrix_bytes=se_bytes,
    )


# ---------------------------------------------------------------------------
# LMDB read helpers (pure once the bytes are in hand)
# ---------------------------------------------------------------------------
def read_metadata(txn: lmdb.Transaction) -> dict:
    """Read and decode the metadata JSON from an open transaction."""
    raw = txn.get(b"pheno_metadata")
    if raw is None:
        raise KeyError("Key 'pheno_metadata' not found in LMDB")
    return json.loads(raw.decode("utf-8"))


def reconstruct_matrix(txn: lmdb.Transaction) -> PhenoMatrix:
    """Reconstruct a PhenoMatrix from an open LMDB transaction.
    """
    metadata = read_metadata(txn)
    raw_bytes = txn.get(b"pheno_matrix")
    if raw_bytes is None:
        raise KeyError("Key 'pheno_matrix' not found in LMDB")

    matrix = np.frombuffer(raw_bytes, dtype=metadata["dtype"]).reshape(
        metadata["rows"], metadata["columns"]
    )

    # Try to load SE matrix if it exists
    # Ideally I want to allow this for other metadata more like a check thing
    se_matrix = None
    if metadata.get("has_se", False):
        se_raw = txn.get(b"pheno_se_matrix")
        if se_raw is not None:
            se_matrix = np.frombuffer(se_raw, dtype=metadata["dtype"]).reshape(
                metadata["rows"], metadata["columns"]
            ).copy()

    return PhenoMatrix(
        matrix=matrix.copy(),           # frombuffer is read-only; copy to own the memory
        traits=metadata["traits"],
        strains=metadata["strains"],
        se_matrix=se_matrix,
    )


def fetch_single_trait(lmdb_path: str, trait_name: str) -> dict[str, float]:
    """Fetch one trait's values for all strains without rebuilding the full matrix.

    Slices the raw byte buffer at the exact row offset — only the bytes
    for the requested trait are decoded.
    """
    with open_lmdb(lmdb_path) as env:
        with env.begin() as txn:
            metadata = read_metadata(txn)
            traits = metadata["traits"]

            if trait_name not in traits:
                raise KeyError(
                    f"Trait '{trait_name}' not found in stored metadata")

            row_idx = traits.index(trait_name)
            dtype = np.dtype(metadata["dtype"])
            bytes_per_row = metadata["columns"] * dtype.itemsize
            start = row_idx * bytes_per_row

            raw = txn.get(b"pheno_matrix")
            row_values = np.frombuffer(
                raw, dtype=dtype, count=metadata["columns"], offset=start)

    return dict(zip(metadata["strains"], row_values.tolist()))


def write_to_lmdb(lmdb_path: str, serialized: SerializedPhenoMatrix) -> None:
    """Persist the serialized matrix + metadata into LMDB.
    """
    with open_lmdb(lmdb_path, create=True) as env:
        with env.begin(write=True) as txn:
            txn.put(b"pheno_matrix", serialized.matrix_bytes)
            txn.put(b"pheno_metadata", serialized.metadata_bytes)
            if serialized.se_matrix_bytes is not None:
                txn.put(b"pheno_se_matrix", serialized.se_matrix_bytes)
    logger.info("Matrix written to %s", lmdb_path)


def prepare_and_dump(dataset_name: str, sql_uri: str, lmdb_path: str) -> PhenoMatrix:
    """
    Returns the PhenoMatrix
    """
    strains, datasets, se_data = fetch_phenotypes(dataset_name, sql_uri)
    pheno = build_pheno_matrix(datasets, se_data, list(strains))
    write_to_lmdb(lmdb_path, serialize(pheno))
    return pheno


def load_from_lmdb(lmdb_path: str) -> PhenoMatrix:
    """Read and reconstruct the full PhenoMatrix from LMDB."""
    with open_lmdb(lmdb_path) as env:
        with env.begin() as txn:
            return reconstruct_matrix(txn)


def verify_roundtrip(dumped: PhenoMatrix, loaded: PhenoMatrix) -> None:
    """Assert that a dumped matrix round-trips perfectly through LMDB.

    Pure — raises AssertionError on mismatch, returns None on success.
    """
    assert_array_equal(dumped.matrix, loaded.matrix)
    assert dumped.traits == loaded.traits,   "traits mismatch after round-trip"
    assert dumped.strains == loaded.strains, "strains mismatch after round-trip"
    logger.info("Round-trip sanity check passed.")


@click.group()
def cli():
    """Phenotype cmd."""


@cli.command("dump-phenotypes")
@click.argument("sql_uri")
@click.argument(
    "lmdb_path",
    type=click.Path(file_okay=False, path_type=str),
)
def dump_phenotypes_cmd(sql_uri: str, lmdb_path: str) -> None:
    """Fetch phenotypes from MySQL and write the matrix to LMDB."""
    prepare_and_dump("BXDPublish", sql_uri, lmdb_path)


@cli.command("print-phenotype-matrix")
@click.argument(
    "lmdb_path",
    type=click.Path(exists=True, file_okay=False,
                    readable=True, path_type=str),
)
def print_phenotype_matrix_cmd(lmdb_path: str) -> None:
    """Read and print the phenotype matrix stored in LMDB."""
    pheno = load_from_lmdb(lmdb_path)
    rows, cols = pheno.matrix.shape
    logger.info("PhenoMatrix  rows(traits)=%d  cols(strains)=%d", rows, cols)
    print(pheno.matrix)


@cli.command("list-traits")
@click.argument(
    "lmdb_path",
    type=click.Path(exists=True, file_okay=False,
                    readable=True, path_type=str),
)
def list_traits_cmd(lmdb_path: str) -> None:
    """Print every trait name stored in the LMDB metadata, one per line."""
    with open_lmdb(lmdb_path) as env:
        with env.begin() as txn:
            metadata = read_metadata(txn)

    for trait in metadata["traits"]:
        click.echo(trait)


@cli.command("fetch-trait")
@click.argument(
    "lmdb_path",
    type=click.Path(exists=True, file_okay=False,
                    readable=True, path_type=str),
)
@click.argument("trait_name")
@click.option(
    "--json/--no-json",
    "as_json",
    default=False,
    help="Output as JSON instead of a plain table.",
)
def fetch_trait_cmd(lmdb_path: str, trait_name: str, as_json: bool) -> None:
    """Fetch and print values for a single trait across all strains.

    Only the bytes for the requested row are decoded — the full matrix is
    never reconstructed.

    Example:

        python run.py fetch-trait /path/ 10003
    """
    result = fetch_single_trait(lmdb_path, trait_name)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        # column-aligned table
        max_strain_len = max(len(s) for s in result)
        click.echo(f"{'Strain':<{max_strain_len}}  Value")
        click.echo(f"{'-' * max_strain_len}  -----")
        for strain, value in result.items():
            click.echo(f"{strain:<{max_strain_len}}  {value}")


@cli.command("sanity-check")
@click.argument("sql_uri")
@click.argument(
    "lmdb_path",
    type=click.Path(file_okay=False, path_type=str),
)
def sanity_check_cmd(sql_uri: str, lmdb_path: str) -> None:
    """Dump, reload, and compare — fails loudly on any mismatch."""
    dumped = prepare_and_dump("BXDPublish", sql_uri, lmdb_path) # I dont like this hardcoding thing 
    loaded = load_from_lmdb(lmdb_path)
    verify_roundtrip(dumped, loaded)


if __name__ == "__main__":
    cli()
