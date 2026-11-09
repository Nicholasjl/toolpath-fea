"""Shared CLI/desktop mesh defaults and conservative numerical screening limits."""
DEFAULT_MESH_MODE = 'tetra'
DEFAULT_PITCH_MM = 2.0
TETGEN_SWITCHES = 'pq1.4Q'  # lowercase q refines; uppercase Q only silences output.
MAX_VOLUME_RATIO = 4.0
VOLUME_P95_RATIO = 1.5
REFINED_VOLUME_RATIO = 0.85
