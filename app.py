"""Flask REST API for serving phenotype data from LMDB."""

# using guix load the dependencies: guix shell python-wrapper python-flask python-lmdb python-numpy

import json
import logging
import re
from typing import Optional

import lmdb
import numpy as np
from flask import Flask, jsonify, request


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)




app.config.update(
    LMDB_PATH="/home/kabui/test_lmdb_data",  # Default path
    DEBUG=False,
)


def open_lmdb(db_path: str) -> lmdb.Environment:
    """Open an LMDB environment for reading."""
    return lmdb.open(db_path, readonly=True, lock=False)


def read_metadata(txn: lmdb.Transaction) -> dict:
    """Read and decode the metadata JSON from an open transaction."""
    raw = txn.get(b"pheno_metadata")
    if raw is None:
        raise KeyError("Metadata not found in LMDB")
    return json.loads(raw.decode("utf-8"))


def fetch_trait_row(
    txn: lmdb.Transaction,
    trait_name: str,
    metadata: dict,
    matrix_key: bytes = b"pheno_matrix",
) -> np.ndarray:
    """Fetch a single trait row from the matrix by slicing the raw bytes.

    Args:
        txn: Open LMDB transaction
        trait_name: Name/ID of the trait to fetch
        metadata: Decoded metadata dict
        matrix_key: Key for the matrix in LMDB (pheno_matrix or pheno_se_matrix)

    Returns:
        1D numpy array of values for this trait across all strains
    """
    traits = metadata["traits"]

    if trait_name not in traits:
        raise KeyError(f"Trait '{trait_name}' not found")

    row_idx = traits.index(trait_name)
    dtype = np.dtype(metadata["dtype"])
    bytes_per_row = metadata["columns"] * dtype.itemsize
    start = row_idx * bytes_per_row

    raw = txn.get(matrix_key)
    if raw is None:
        raise KeyError(f"Matrix '{matrix_key.decode()}' not found in LMDB")

    return np.frombuffer(raw, dtype=dtype, count=metadata["columns"], offset=start)


def build_trait_response(
    strains: list[str],
    values: np.ndarray,
    se_values: Optional[np.ndarray] = None,
) -> dict:
    """Build the JSON response for a trait query.

    Returns a dict with strain names as keys and value/SE pairs as values.
    NaN values are converted to null in JSON.
    """
    result = {}

    for idx, strain in enumerate(strains):
        value = values[idx]

        # Convert numpy types to Python native types
        # Convert NaN to None (which becomes null in JSON)
        if np.isnan(value):
            py_value = None
        else:
            py_value = float(value)

        if se_values is not None:
            se_val = se_values[idx]
            py_se = None if np.isnan(se_val) else float(se_val)
            result[strain] = {"value": py_value, "se": py_se}
        else:
            result[strain] = {"value": py_value, "se": None}

    return result


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint - returns 200 if the service is running."""
    return jsonify({
        "status": "healthy",
        "service": "phenotype-api",
        "lmdb_path": app.config["LMDB_PATH"],
    })


@app.route("/dataset/phenotype/<path:trait_spec>", methods=["GET"])
def get_phenotype(trait_spec: str):
    """Fetch phenotype data for a specific trait.
   : http://127.0.0.1:5000/dataset/phenotype/BXDPublish_10007
    Path parameters:
        trait_spec: Format is <dataset>_<trait_id>
                    Example: BXDPublish_12315

    Query parameters:
        format: 'compact' or 'detailed' (default: detailed)
                - compact: only non-null values
                - detailed: all strains with metadata

    Returns:
        JSON object with strain names as keys and values:
        {
            "BXD1": 5.67,
            "BXD2": 4.32,
            ...
        }

        NaN/missing values are returned as null in JSON.

    Examples:
        GET /dataset/phenotype/BXDPublish_12315
        GET /dataset/phenotype/BXDPublish_12315?format=compact
    """
    # Parse the trait specification
    match = re.match(r"^(.+?)_(\d+)$", trait_spec)
    if not match:
        return jsonify({
            "error": "Invalid trait specification",
            "message": "Format should be <dataset>_<trait_id>, e.g., BXDPublish_12315",
        }), 400

    dataset, trait_id = match.groups()
    trait_id = str(int(trait_id))  # Normalize: "00123" -> "123"

    # Check format parameter
    response_format = request.args.get("format", "detailed")

    try:
        with open_lmdb(app.config["LMDB_PATH"]) as env:
            with env.begin() as txn:
                metadata = read_metadata(txn)

                # Fetch values
                values = fetch_trait_row(
                    txn, trait_id, metadata, b"pheno_matrix")

                # Build simple {strain: value} dict
                strains = metadata["strains"]
                data = {}

                for idx, strain in enumerate(strains):
                    value = values[idx]
                    # Convert NaN to None (becomes null in JSON)
                    py_value = None if np.isnan(value) else float(value)
                    data[strain] = py_value

                # Compact format: only return non-null values
                if response_format == "compact":
                    data = {k: v for k, v in data.items() if v is not None}

                return jsonify(data)

    except KeyError as e:
        logger.warning("Trait not found: %s (%s)", trait_spec, e)
        return jsonify({
            "error": "Trait not found",
            "message": str(e),
            "trait_spec": trait_spec,
        }), 404

    except Exception as e:
        logger.error("Error fetching trait %s: %s",
                     trait_spec, e, exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
        }), 500


@app.route("/dataset/phenotype/<path:trait_spec>/se", methods=["GET"])
def get_phenotype_with_se(trait_spec: str):
    """Fetch phenotype data WITH standard errors for a specific trait.

    Path parameters:
        trait_spec: Format is <dataset>_<trait_id>
                    Example: BXDPublish_12315

    Returns:
        JSON object with strain names as keys, each containing value and se:
        {
            "BXD1": {"value": 5.67, "se": 0.12},
            "BXD2": {"value": 4.32, "se": 0.15},
            ...
        }

    Example:
        GET /dataset/phenotype/BXDPublish_12315/se
    """
    # Parse the trait specification
    match = re.match(r"^(.+?)_(\d+)$", trait_spec)
    if not match:
        return jsonify({
            "error": "Invalid trait specification",
            "message": "Format should be <dataset>_<trait_id>, e.g., BXDPublish_12315",
        }), 400

    dataset, trait_id = match.groups()
    trait_id = str(int(trait_id))  # Normalize: "00123" -> "123"

    try:
        with open_lmdb(app.config["LMDB_PATH"]) as env:
            with env.begin() as txn:
                metadata = read_metadata(txn)

                # Fetch values
                values = fetch_trait_row(
                    txn, trait_id, metadata, b"pheno_matrix")

                # Try to fetch SE values
                se_values = None
                try:
                    se_values = fetch_trait_row(
                        txn, trait_id, metadata, b"pheno_se_matrix")
                except KeyError:
                    logger.debug(
                        "SE data not available for trait %s", trait_id)

                # Build response with both value and SE
                strains = metadata["strains"]
                data = {}

                for idx, strain in enumerate(strains):
                    value = values[idx]
                    py_value = None if np.isnan(value) else float(value)

                    if se_values is not None:
                        se_val = se_values[idx]
                        py_se = None if np.isnan(se_val) else float(se_val)
                    else:
                        py_se = None

                    data[strain] = {
                        "value": py_value,
                        "se": py_se
                    }

                return jsonify(data)

    except KeyError as e:
        logger.warning("Trait not found: %s (%s)", trait_spec, e)
        return jsonify({
            "error": "Trait not found",
            "message": str(e),
            "trait_spec": trait_spec,
        }), 404

    except Exception as e:
        logger.error("Error fetching trait %s: %s",
                     trait_spec, e, exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
        }), 500


@app.route("/dataset/phenotype/<path:trait_spec>/strains", methods=["GET"])
def get_phenotype_strains(trait_spec: str):
    """Get list of available strains for a trait.

    Returns only strain names without values.
    """
    try:
        with open_lmdb(app.config["LMDB_PATH"]) as env:
            with env.begin() as txn:
                metadata = read_metadata(txn)
                return jsonify({
                    "strains": metadata["strains"],
                    "count": len(metadata["strains"]),
                })
    except Exception as e:
        logger.error("Error fetching strains: %s", e, exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
        }), 500


@app.route("/datasets/<dataset>/traits", methods=["GET"])
def list_traits(dataset: str):
    """List all available trait IDs for a dataset.

    Query parameters:
        limit: Maximum number of traits to return (default: 100)
        offset: Number of traits to skip (default: 0)
    """
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)

    try:
        with open_lmdb(app.config["LMDB_PATH"]) as env:
            with env.begin() as txn:
                metadata = read_metadata(txn)
                all_traits = metadata["traits"]

                # Apply pagination
                paginated_traits = all_traits[offset:offset + limit]

                return jsonify({
                    "dataset": dataset,
                    "traits": paginated_traits,
                    "total": len(all_traits),
                    "limit": limit,
                    "offset": offset,
                })
    except Exception as e:
        logger.error("Error listing traits: %s", e, exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist",
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", error, exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
    }), 500


if __name__ == "__main__":
    import os

    # Allow configuration via environment variables
    lmdb_path = os.getenv("LMDB_PATH", app.config["LMDB_PATH"])
    app.config["LMDB_PATH"] = lmdb_path

    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"

    logger.info("Starting phenotype API server")
    logger.info("LMDB path: %s", lmdb_path)
    logger.info("Port: %d", port)

    app.run(host="0.0.0.0", port=port, debug=debug)
