#!/usr/bin/env bash
set -euo pipefail

FLUENT="/c/PROGRA~1/ANSYSI~1/v252/fluent/ntbin/win64/fluent.exe"

# ===========================================================
# Re sweep (single test: Re=150) with Dh computed from inlet area
# No base_setup. Fluent reads mesh directly in the journal.
# ===========================================================

RHO=998.2
MU=0.001003
NCORES=4

PROBE_JOU="probe_inlet_geom.jou"
TEMPLATE="bifurcation_template_robust.jou"

MESHES=("coronary_extracted_vessel.msh")

# Single test
RES=(150)

get_max_iters() {
  local re=$1
  if   (( re <= 200 )); then echo 2500
  elif (( re <= 500 )); then echo 3500
  else echo 5000
  fi
}

# ---- sanity checks
if [[ ! -f "$FLUENT" ]]; then
  echo "ERROR: Fluent executable not found:"
  echo "  $FLUENT"
  exit 1
fi

for f in "$PROBE_JOU" "$TEMPLATE"; do
  [[ -f "$f" ]] || { echo "ERROR: Missing file: $f"; exit 1; }
done

# Kinematic viscosity
NU=$(python -c "print(${MU}/${RHO})")

mkdir -p results
SUMMARY="results/batch_summary_Dh.log"
echo "Batch started: $(date)" > "$SUMMARY"
echo "Mesh,Re,Dh,Velocity,MaxIters,Status,Time" >> "$SUMMARY"

for MESH in "${MESHES[@]}"; do
  [[ -f "$MESH" ]] || { echo "Missing mesh: $MESH"; continue; }

  MESH_NAME=$(basename "$MESH")
  MESH_NAME=${MESH_NAME%.msh.h5}
  MESH_NAME=${MESH_NAME%.msh}

  echo ""
  echo "========================================="
  echo "Mesh: $MESH_NAME"
  echo "========================================="

  # ---------- Stage 1: Probe inlet geometry ----------
  PROBE_DIR="results/${MESH_NAME}/_geom"
  mkdir -p "$PROBE_DIR"

  PROBE_RUN_JOU="${PROBE_DIR}/probe_${MESH_NAME}.jou"
  sed -e "s|MESH_FILE|${MESH}|g" \
      -e "s|OUTDIR|${PROBE_DIR}|g" \
      "$PROBE_JOU" > "$PROBE_RUN_JOU"

  echo "[Probe] Computing inlet area..."
  "$FLUENT" 3ddp -g -t${NCORES} -i "$PROBE_RUN_JOU" 2>&1 | tee "${PROBE_DIR}/probe_console.log" || true

  TRANS="${PROBE_DIR}/inlet_geom_transcript.txt"
  [[ -f "$TRANS" ]] || { echo "ERROR: Missing $TRANS"; exit 1; }

  # ---- Robust area parsing (table row: "inlet  8.8200818e-06")
  AREA=$(
    awk '
      BEGIN{val=""}
      tolower($1)=="inlet" && $2 ~ /^[0-9.+-]+([eE][0-9+-]+)?$/ { val=$2 }
      END{ print val }
    ' "$TRANS"
  )

  # ---- Fallback (if Fluent prints "Area = <value>")
  if [[ -z "${AREA}" ]]; then
    AREA=$(grep -iE "area\s*=" "$TRANS" | tail -1 | awk -F'=' '{print $2}' | tr -d ' ')
  fi

  if [[ -z "${AREA}" ]]; then
    echo "ERROR: Could not parse inlet area from transcript."
    echo "Open: $TRANS"
    exit 1
  fi

  # Equivalent diameter from inlet area (fallback when perimeter not available)
  DH=$(python - <<PY
import math
A = float("${AREA}")
print(math.sqrt(4.0*A/math.pi))
PY
)

  echo "[Probe] inlet area A = ${AREA} m^2"
  echo "[Probe] Dh(eq from area) = ${DH} m"

  cat > "${PROBE_DIR}/inlet_geom.txt" <<EOF
mesh=${MESH}
A_m2=${AREA}
Dh_m=${DH}
nu_m2s=${NU}
EOF

  # ---------- Stage 2: Solve sweep using Dh ----------
  for Re in "${RES[@]}"; do
    OUTDIR="results/${MESH_NAME}/Re${Re}"
    mkdir -p "$OUTDIR"

    MAX_ITERS=$(get_max_iters "$Re")
    VELOCITY=$(python -c "print(${Re} * ${NU} / ${DH})")

    RUN_JOU="${OUTDIR}/run_${MESH_NAME}_Re${Re}.jou"
    sed -e "s|MESH_FILE|${MESH}|g" \
        -e "s|MESH_NAME|${MESH_NAME}|g" \
        -e "s|VALUE_RE|${Re}|g" \
        -e "s|VALUE_ITERS|${MAX_ITERS}|g" \
        -e "s|VALUE_VELOCITY|${VELOCITY}|g" \
        "$TEMPLATE" > "$RUN_JOU"

    START=$(date +%s)
    echo "Run: ${MESH_NAME} Re=${Re} Dh=${DH} U=${VELOCITY} iters=${MAX_ITERS}"

    if "$FLUENT" 3ddp -g -t${NCORES} -i "$RUN_JOU" 2>&1 | tee "${OUTDIR}/console.log" ; then
      END=$(date +%s)
      echo "${MESH_NAME},${Re},${DH},${VELOCITY},${MAX_ITERS},OK,$((END-START))s" >> "$SUMMARY"
    else
      END=$(date +%s)
      echo "${MESH_NAME},${Re},${DH},${VELOCITY},${MAX_ITERS},FAILED,$((END-START))s" >> "$SUMMARY"
    fi
  done
done

echo ""
echo "Done. Summary: $SUMMARY"